"""
Comprehensive Ensemble Training Script for Skin Cancer Detection
Trains EfficientNet, ResNet, and DenseNet with max voting ensemble
"""

import os
import json
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    jaccard_score
)

from ensemble_models import (
    build_efficientnet_model, build_resnet_model, build_densenet_model,
    compile_model, unfreeze_base_model
)
from ensemble_voting import MaxVotingEnsemble

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


def create_advanced_augmentation():
    """Create advanced data augmentation pipeline"""
    layers = [
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),
    ]
    # RandomShear might not be available in all TF versions
    try:
        layers.append(tf.keras.layers.RandomShear(0.1))
    except AttributeError:
        pass  # Skip if not available
    return tf.keras.Sequential(layers)


def load_data_from_split(
    data_root: str = "dataset_split",
    image_size: Tuple[int, int] = (300, 300),
    batch_size: int = 16
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    """Load data from pre-split directories"""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_root, "train"),
        labels="inferred",
        label_mode="int",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_root, "val"),
        labels="inferred",
        label_mode="int",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_root, "test"),
        labels="inferred",
        label_mode="int",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
    )
    
    return train_ds, val_ds, test_ds, train_ds.class_names


def compute_class_weights(data_root: str) -> Dict[int, float]:
    """Compute class weights for imbalanced dataset"""
    import glob
    
    benign_path = os.path.join(data_root, "train", "benign")
    malignant_path = os.path.join(data_root, "train", "malignant")
    
    benign_count = (
        len(glob.glob(os.path.join(benign_path, "*.jpg"))) +
        len(glob.glob(os.path.join(benign_path, "*.jpeg"))) +
        len(glob.glob(os.path.join(benign_path, "*.png")))
    )
    
    malignant_count = (
        len(glob.glob(os.path.join(malignant_path, "*.jpg"))) +
        len(glob.glob(os.path.join(malignant_path, "*.jpeg"))) +
        len(glob.glob(os.path.join(malignant_path, "*.png")))
    )
    
    total = benign_count + malignant_count
    weight_benign = total / (2.0 * benign_count)
    weight_malignant = total / (2.0 * malignant_count)
    
    print(f"Class distribution - Benign: {benign_count}, Malignant: {malignant_count}")
    print(f"Class weights - Benign: {weight_benign:.3f}, Malignant: {weight_malignant:.3f}")
    
    return {0: weight_benign, 1: weight_malignant}


def train_single_model(
    model: tf.keras.Model,
    model_name: str,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs_frozen: int,
    epochs_finetune: int,
    lr_frozen: float,
    lr_finetune: float,
    class_weight: Dict[int, float],
    model_dir: str,
    image_size: Tuple[int, int]
) -> tf.keras.Model:
    """Train a single model with frozen and fine-tuning phases"""
    
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}")
    
    # Phase 1: Frozen training
    print(f"\nPhase 1: Training with frozen base ({epochs_frozen} epochs)...")
    compile_model(model, learning_rate=lr_frozen)
    
    callbacks_frozen = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, f"{model_name}_frozen_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
    ]
    
    history_frozen = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_frozen,
        callbacks=callbacks_frozen,
        verbose=1,
        class_weight=class_weight,
    )
    
    # Phase 2: Fine-tuning
    print(f"\nPhase 2: Fine-tuning with unfrozen base ({epochs_finetune} epochs)...")
    
    # Unfreeze base model gradually
    unfreeze_base_model(model, unfreeze_ratio=0.5)
    
    compile_model(model, learning_rate=lr_finetune)
    
    callbacks_finetune = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, f"{model_name}_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.3,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
    ]
    
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_finetune,
        callbacks=callbacks_finetune,
        verbose=1,
        class_weight=class_weight,
    )
    
    # Load best model
    best_model_path = os.path.join(model_dir, f"{model_name}_best.keras")
    if os.path.exists(best_model_path):
        model = tf.keras.models.load_model(best_model_path)
        print(f"[OK] Loaded best model from {best_model_path}")
    
    return model


def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    model_name: str = "Model"
) -> Dict[str, float]:
    """Evaluate a single model"""
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    for batch_x, batch_y in test_ds:
        prob = model.predict(batch_x, verbose=0)
        y_true.extend(batch_y.numpy().tolist())
        y_pred.extend(np.argmax(prob, axis=1).tolist())
        y_pred_proba.extend(prob[:, 1].tolist())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0, average="weighted")
    rec = recall_score(y_true, y_pred, zero_division=0, average="weighted")
    f1 = f1_score(y_true, y_pred, zero_division=0, average="weighted")
    f1_macro = f1_score(y_true, y_pred, zero_division=0, average="macro")
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # Calculate IoU (Jaccard Score)
    iou = jaccard_score(y_true, y_pred, average="macro")
    
    return {
        "model_name": model_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "f1_macro": float(f1_macro),
        "auc": float(auc),
        "iou": float(iou),
    }


def evaluate_ensemble(
    ensemble: MaxVotingEnsemble,
    test_ds: tf.data.Dataset,
    class_names: List[str]
) -> Dict[str, float]:
    """Evaluate ensemble model"""
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    # Collect all test data
    all_x = []
    all_y = []
    for batch_x, batch_y in test_ds:
        all_x.append(batch_x.numpy())
        all_y.extend(batch_y.numpy().tolist())
    
    all_x = np.concatenate(all_x, axis=0)
    all_y = np.array(all_y)
    
    # Ensemble predictions
    y_pred = ensemble.predict(all_x, batch_size=32, verbose=0)
    y_pred_proba = ensemble.predict_proba(all_x, batch_size=32, verbose=0)
    
    acc = accuracy_score(all_y, y_pred)
    prec = precision_score(all_y, y_pred, zero_division=0, average="weighted")
    rec = recall_score(all_y, y_pred, zero_division=0, average="weighted")
    f1 = f1_score(all_y, y_pred, zero_division=0, average="weighted")
    f1_macro = f1_score(all_y, y_pred, zero_division=0, average="macro")
    auc = roc_auc_score(all_y, y_pred_proba[:, 1])
    iou = jaccard_score(all_y, y_pred, average="macro")
    
    cm = confusion_matrix(all_y, y_pred)
    report = classification_report(all_y, y_pred, target_names=class_names, output_dict=True)
    
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "f1_macro": float(f1_macro),
        "auc": float(auc),
        "iou": float(iou),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def plot_results(metrics_dict: Dict[str, Dict[str, float]], output_dir: str):
    """Plot comparison of individual models and ensemble"""
    models = list(metrics_dict.keys())
    metrics = ["accuracy", "f1_macro", "iou"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        values = [metrics_dict[m][metric] for m in models]
        axes[idx].bar(models, values, alpha=0.7)
        axes[idx].set_title(f"{metric.upper()}")
        axes[idx].set_ylabel("Score")
        axes[idx].set_ylim([0, 1])
        axes[idx].grid(axis="y", alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ensemble_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()


def train_ensemble(
    data_root: str = "dataset_split",
    image_size: Tuple[int, int] = (300, 300),
    batch_size: int = 16,
    epochs_frozen: int = 15,
    epochs_finetune: int = 50,
    lr_frozen: float = 1e-3,
    lr_finetune: float = 1e-4,
    dropout: float = 0.4,
    model_dir: str = "ensemble_checkpoints",
    use_efficientnet: bool = True,
    use_resnet: bool = True,
    use_densenet: bool = True,
) -> Dict:
    """Main training function for ensemble"""
    
    os.makedirs(model_dir, exist_ok=True)
    
    print("="*80)
    print("ENSEMBLE TRAINING FOR SKIN CANCER DETECTION")
    print("="*80)
    print(f"Image Size: {image_size}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs_frozen} frozen + {epochs_finetune} fine-tune")
    print(f"Learning Rates: {lr_frozen} -> {lr_finetune}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    train_ds, val_ds, test_ds, class_names = load_data_from_split(data_root, image_size, batch_size)
    print(f"Classes: {class_names}")
    
    # Compute class weights
    class_weight = compute_class_weights(data_root)
    
    # Data augmentation
    augmentation = create_advanced_augmentation()
    
    def augment_and_cast(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        x = augmentation(x, training=True)
        return x, y
    
    def cast_only(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        return x, y
    
    train_ds = train_ds.map(augment_and_cast, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(cast_only, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(cast_only, num_parallel_calls=tf.data.AUTOTUNE)
    
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)
    
    # Train individual models
    trained_models = []
    model_configs = []
    
    if use_efficientnet:
        print("\n" + "="*80)
        print("Building EfficientNetB3...")
        efficientnet = build_efficientnet_model(
            input_shape=(*image_size, 3),
            num_classes=2,
            dropout=dropout,
            base_trainable=False,
            model_size="B3"
        )
        efficientnet = train_single_model(
            efficientnet, "EfficientNetB3", train_ds, val_ds,
            epochs_frozen, epochs_finetune, lr_frozen, lr_finetune,
            class_weight, model_dir, image_size
        )
        trained_models.append(efficientnet)
        model_configs.append("EfficientNetB3")
    
    if use_resnet:
        print("\n" + "="*80)
        print("Building ResNet50...")
        resnet = build_resnet_model(
            input_shape=(*image_size, 3),
            num_classes=2,
            dropout=dropout,
            base_trainable=False,
            model_type="ResNet50"
        )
        resnet = train_single_model(
            resnet, "ResNet50", train_ds, val_ds,
            epochs_frozen, epochs_finetune, lr_frozen, lr_finetune,
            class_weight, model_dir, image_size
        )
        trained_models.append(resnet)
        model_configs.append("ResNet50")
    
    if use_densenet:
        print("\n" + "="*80)
        print("Building DenseNet121...")
        densenet = build_densenet_model(
            input_shape=(*image_size, 3),
            num_classes=2,
            dropout=dropout,
            base_trainable=False,
            model_type="DenseNet121"
        )
        densenet = train_single_model(
            densenet, "DenseNet121", train_ds, val_ds,
            epochs_frozen, epochs_finetune, lr_frozen, lr_finetune,
            class_weight, model_dir, image_size
        )
        trained_models.append(densenet)
        model_configs.append("DenseNet121")
    
    # Evaluate individual models
    print("\n" + "="*80)
    print("Evaluating Individual Models")
    print("="*80)
    
    individual_metrics = {}
    for model, name in zip(trained_models, model_configs):
        metrics = evaluate_model(model, test_ds, name)
        individual_metrics[name] = metrics
        print(f"\n{name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  IoU: {metrics['iou']:.4f}")
    
    # Create ensemble
    print("\n" + "="*80)
    print("Creating Max Voting Ensemble")
    print("="*80)
    
    ensemble = MaxVotingEnsemble(trained_models)
    
    # Evaluate ensemble
    print("\nEvaluating Ensemble...")
    ensemble_metrics = evaluate_ensemble(ensemble, test_ds, class_names)
    
    print("\n" + "="*80)
    print("ENSEMBLE RESULTS")
    print("="*80)
    print(f"Accuracy:  {ensemble_metrics['accuracy']:.4f} ({ensemble_metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {ensemble_metrics['precision']:.4f}")
    print(f"Recall:    {ensemble_metrics['recall']:.4f}")
    print(f"F1-Score:  {ensemble_metrics['f1_score']:.4f}")
    print(f"F1-Macro:  {ensemble_metrics['f1_macro']:.4f}")
    print(f"AUC:       {ensemble_metrics['auc']:.4f}")
    print(f"IoU:       {ensemble_metrics['iou']:.4f}")
    print("="*80)
    
    # Save ensemble model
    ensemble_dir = os.path.join(model_dir, "ensemble")
    os.makedirs(ensemble_dir, exist_ok=True)
    
    # Save individual models
    for model, name in zip(trained_models, model_configs):
        model.save(os.path.join(ensemble_dir, f"{name}_final.keras"))
    
    # Save metrics
    all_metrics = {
        "individual_models": individual_metrics,
        "ensemble": ensemble_metrics,
        "class_names": class_names,
    }
    
    with open(os.path.join(model_dir, "ensemble_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    # Plot results
    plot_results({**individual_metrics, "Ensemble": ensemble_metrics}, model_dir)
    
    # Save confusion matrix for ensemble
    cm = np.array(ensemble_metrics["confusion_matrix"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Ensemble Confusion Matrix\nAccuracy: {ensemble_metrics['accuracy']*100:.2f}%")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "ensemble_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\n[OK] All results saved to: {model_dir}")
    
    return {
        "models": trained_models,
        "ensemble": ensemble,
        "metrics": all_metrics
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ensemble of CNN models for skin cancer detection")
    parser.add_argument("--data_root", type=str, default="dataset_split")
    parser.add_argument("--image_size", type=int, nargs=2, default=[300, 300])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs_frozen", type=int, default=15)
    parser.add_argument("--epochs_finetune", type=int, default=50)
    parser.add_argument("--lr_frozen", type=float, default=1e-3)
    parser.add_argument("--lr_finetune", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--model_dir", type=str, default="ensemble_checkpoints")
    parser.add_argument("--no_efficientnet", action="store_true", help="Skip EfficientNet")
    parser.add_argument("--no_resnet", action="store_true", help="Skip ResNet")
    parser.add_argument("--no_densenet", action="store_true", help="Skip DenseNet")
    
    args = parser.parse_args()
    
    train_ensemble(
        data_root=args.data_root,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        epochs_frozen=args.epochs_frozen,
        epochs_finetune=args.epochs_finetune,
        lr_frozen=args.lr_frozen,
        lr_finetune=args.lr_finetune,
        dropout=args.dropout,
        model_dir=args.model_dir,
        use_efficientnet=not args.no_efficientnet,
        use_resnet=not args.no_resnet,
        use_densenet=not args.no_densenet,
    )

