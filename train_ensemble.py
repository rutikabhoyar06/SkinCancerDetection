"""Training script for multi-model ensemble on skin lesion classification.

This script fine-tunes EfficientNet, ResNet, and DenseNet backbones and
combines their predictions via max voting. Designed to work with
pre-segmented lesion crops produced by a U-Net pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, f1_score, jaccard_score


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_transforms(input_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tfms, eval_tfms


def make_loaders(data_root: Path, batch_size: int, input_size: int) -> Tuple[DataLoader, DataLoader, DataLoader, Sequence[str]]:
    train_tfms, eval_tfms = build_transforms(input_size)
    train_ds = ImageFolder(data_root / "train", transform=train_tfms)
    val_ds = ImageFolder(data_root / "val", transform=eval_tfms)
    test_ds = ImageFolder(data_root / "test", transform=eval_tfms)

    loader_kwargs = dict(batch_size=batch_size, num_workers=4, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, train_ds.classes


def build_model(backbone: str, num_classes: int, dropout: float = 0.2) -> nn.Module:
    backbone = backbone.lower()
    if backbone == "efficientnet":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
    elif backbone == "resnet":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
    elif backbone == "densenet":
        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return model.to(DEVICE)


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> nn.Module:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_state = None
    best_acc = -np.inf

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(xb)
                loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        scheduler.step()

        val_acc = evaluate_single(model, val_loader)["accuracy"]
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch + 1}/{epochs} — val_acc: {val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_single(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    logits, labels = infer_logits(model, loader)
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "iou_macro": jaccard_score(labels, preds, average="macro"),
    }


def infer_logits(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            logits = model(xb)
            logits_list.append(logits.detach().cpu().numpy())
            labels_list.append(yb.numpy())
    return np.concatenate(logits_list, axis=0), np.concatenate(labels_list, axis=0)


def max_vote_ensemble(logits_collection: Sequence[np.ndarray]) -> np.ndarray:
    stacked = np.stack(logits_collection, axis=0)
    preds = np.argmax(stacked, axis=-1)
    max_vote = []
    for sample_votes in preds.transpose(1, 0):
        values, counts = np.unique(sample_votes, return_counts=True)
        max_vote.append(values[np.argmax(counts)])
    return np.array(max_vote)


def evaluate_ensemble(models_list: Sequence[nn.Module], loader: DataLoader) -> Dict[str, float]:
    logits_collection: List[np.ndarray] = []
    all_labels: np.ndarray | None = None
    for model in models_list:
        logits, labels = infer_logits(model, loader)
        logits_collection.append(logits)
        if all_labels is None:
            all_labels = labels
    assert all_labels is not None
    ensemble_preds = max_vote_ensemble(logits_collection)
    return {
        "accuracy": accuracy_score(all_labels, ensemble_preds),
        "f1_macro": f1_score(all_labels, ensemble_preds, average="macro"),
        "iou_macro": jaccard_score(all_labels, ensemble_preds, average="macro"),
    }


def save_metrics(output_path: Path, metrics: Dict[str, float]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ensemble of CNN classifiers on segmented lesions")
    parser.add_argument("--data-root", type=Path, default=Path("dataset/segmented"), help="Root directory with train/val/test folders")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--output-metrics", type=Path, default=Path("analysis_results/ensemble_metrics.json"))
    parser.add_argument("--backbones", nargs="*", default=["efficientnet", "resnet", "densenet"], help="List of backbones to train")
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_loader, val_loader, test_loader, classes = make_loaders(args.data_root, args.batch_size, args.input_size)

    models_trained: List[nn.Module] = []
    for backbone in args.backbones:
        print(f"Training backbone: {backbone}")
        model = build_model(backbone, args.num_classes, args.dropout)

        # Optionally freeze lower layers for first few epochs; here we unfreeze everything
        model = train_one_model(model, train_loader, val_loader, args.epochs, args.lr, args.weight_decay)

        metrics_val = evaluate_single(model, val_loader)
        print(f"Validation metrics {backbone}: {metrics_val}")
        models_trained.append(model)

    ensemble_metrics = evaluate_ensemble(models_trained, test_loader)
    print(f"Ensemble test metrics: {ensemble_metrics}")
    save_metrics(args.output_metrics, ensemble_metrics)


if __name__ == "__main__":
    main()




























