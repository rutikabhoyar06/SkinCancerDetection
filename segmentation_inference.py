from typing import Tuple, Optional, Dict, Any
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from segmentation_inference import run_segmentation_inference, evaluate_on_split, _prepare_model
from data_utils import load_dataset
from data_utils import load_dataset
from unet import build_unet
from evaluation import compute_pixel_metrics


def _prepare_model(model_path: Optional[str], input_shape: Tuple[int, int, int]) -> tf.keras.Model:
	"""Load a trained U-Net model or create a new one if path not provided."""
	if model_path and os.path.isfile(model_path):
		return tf.keras.models.load_model(model_path, compile=False)
	# Fallback to uninitialized U-Net (useful for exporting code paths)
	return build_unet(input_shape=input_shape, num_classes=1)


def _predict_mask(model: tf.keras.Model, image: np.ndarray) -> np.ndarray:
	"""Run model inference on a single RGB image (H,W,3) -> prob mask (H,W)."""
	img = image.astype(np.float32) / 255.0
	pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
	if pred.ndim == 4 and pred.shape[-1] == 1:
		pred = pred[0, ..., 0]
	else:
		pred = pred[0]
	return np.clip(pred, 0.0, 1.0)


def _draw_contours_on_image(image: np.ndarray, mask_prob: np.ndarray, threshold: float = 0.5,
							 color: Tuple[int, int, int] = (255, 0, 0), thickness: int = 2) -> np.ndarray:
	"""Return a copy of the image with lesion contours drawn from predicted mask.

	- image: RGB uint8 (H,W,3)
	- mask_prob: float32 (H,W) in [0,1]
	- threshold: binarization threshold
	"""
	binary = (mask_prob >= threshold).astype(np.uint8) * 255
	contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	out = image.copy()
	# Draw contours in red by default
	cv2.drawContours(out, contours, -1, color, thickness)
	return out


def visualize_prediction_with_contours(image: np.ndarray, mask_prob: np.ndarray, threshold: float = 0.5,
										 title: str = "Segmentation with Contours",
										 show_binary: bool = True,
										 overlay_cmap: str = "Reds") -> None:
	"""Show side-by-side: original, overlay with boundary, and optional binary mask."""
	outlined = _draw_contours_on_image(image, mask_prob, threshold)
	binary = (mask_prob >= threshold).astype(np.uint8)

	cols = 3 if show_binary else 2
	plt.figure(figsize=(14 if show_binary else 10, 5))
	plt.subplot(1, cols, 1)
	plt.imshow(image)
	plt.title("Original Image")
	plt.axis("off")

	plt.subplot(1, cols, 2)
	plt.imshow(outlined)
	plt.imshow(binary, cmap=overlay_cmap, alpha=0.35)
	plt.title(title)
	plt.axis("off")

	if show_binary:
		plt.subplot(1, cols, 3)
		plt.imshow(binary, cmap="gray")
		plt.title("Binary Mask")
		plt.axis("off")

	plt.tight_layout()
	plt.show()


def evaluate_on_split(model: tf.keras.Model, split: Dict[str, Any], threshold: float = 0.5) -> Dict[str, float]:
    """Compute overall pixel metrics (incl. accuracy, dice, IoU/F1) if masks available in split."""
	if "masks" not in split or split["masks"].size == 0:
		return {}

	images = split["x"].astype(np.float32)
	masks_true = split["masks"].astype(np.float32)

	# Predict in batches to avoid OOM
	batch_size = 16
	preds = []
	for i in range(0, len(images), batch_size):
		batch = images[i:i + batch_size] / 255.0
		p = model.predict(batch, verbose=0)
		if p.ndim == 4 and p.shape[-1] == 1:
			p = p[..., 0]
		preds.append(p)
	mask_prob = np.concatenate(preds, axis=0)

    metrics = compute_pixel_metrics(y_true=masks_true, y_pred_prob=mask_prob[..., np.newaxis], threshold=threshold)
    # Derive IoU and F1 (for binary segmentation, F1 == Dice)
    dice = metrics.get("dice", 0.0)
    iou = float(dice / (2.0 - max(dice, 1e-12)))  # IoU = Dice / (2 - Dice)
    f1 = float(dice)
    metrics["iou"] = iou
    metrics["f1"] = f1
    return metrics


def run_segmentation_inference(
	model_path: str = "checkpoints/unet_best.keras",
	data_root: str = "dataset",
	image_size: Tuple[int, int] = (224, 224),
	threshold: float = 0.5,
	show_examples: int = 4,
	use_split: str = "val",
) -> None:
	"""Compute overall pixel accuracy on a split and visualize contours on samples."""
	input_shape = (image_size[0], image_size[1], 3)
	model = _prepare_model(model_path, input_shape)

	splits = load_dataset(
		root_dir=data_root,
		image_size=image_size,
		test_ratio=0.15,
		val_ratio=0.15,
		random_state=42,
		masks_root=os.path.join(data_root, "masks"),  # adjust if masks live elsewhere
	)

	if use_split not in ("val", "test"):
		use_split = "val"
	split = splits[use_split]

    metrics = evaluate_on_split(model, split, threshold=threshold)
    if metrics:
        # Pretty print as percentages where meaningful
        acc_pct = metrics['accuracy'] * 100.0
        prec_pct = metrics['precision'] * 100.0
        rec_pct = metrics['recall'] * 100.0
        dice_pct = metrics['dice'] * 100.0
        iou_pct = metrics['iou'] * 100.0
        f1_pct = metrics['f1'] * 100.0

        print(f"\nSegmentation metrics on {use_split} split (threshold={threshold}):")
        print(f"  Accuracy: {acc_pct:.2f}%")
        print(f"  Precision: {prec_pct:.2f}%")
        print(f"  Recall: {rec_pct:.2f}%")
        print(f"  Dice: {dice_pct:.2f}%")
        print(f"  IoU (Jaccard): {iou_pct:.2f}%")
        print(f"  F1-score: {f1_pct:.2f}%")

        # Save a small bar chart for the metrics
        try:
            import matplotlib.pyplot as plt
            labels = ["Accuracy", "Precision", "Recall", "Dice", "IoU", "F1"]
            values = [acc_pct, prec_pct, rec_pct, dice_pct, iou_pct, f1_pct]
            plt.figure(figsize=(8, 4))
            bars = plt.bar(labels, values, color=["skyblue", "lightgreen", "gold", "plum", "salmon", "cornflowerblue"])
            plt.ylim(0, 100)
            for b, v in zip(bars, values):
                plt.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
            plt.title(f"Segmentation Metrics ({use_split} split)")
            plt.ylabel("Percentage")
            os.makedirs("evaluation_results", exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join("evaluation_results", f"segmentation_metrics_{use_split}.png"), dpi=200)
            plt.close()
        except Exception:
            pass
	else:
		print(f"No ground-truth masks found for {use_split} split; skipping metric computation.")

	# Visualize a few predictions with contours
	images = split["x"]
	count = min(show_examples, len(images))
	for idx in range(count):
		img = images[idx].astype(np.uint8)
		mask_prob = _predict_mask(model, img)
		visualize_prediction_with_contours(img, mask_prob, threshold=threshold,
											title=f"Predicted Mask with Lesion Contours (idx={idx})")


if __name__ == "__main__":
	run_segmentation_inference()


