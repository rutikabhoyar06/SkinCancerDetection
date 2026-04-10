from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

from losses import dice_coefficient as dice_tf


def _threshold_preds(preds: np.ndarray, threshold: float = 0.5) -> np.ndarray:
	return (preds >= threshold).astype(np.uint8)


def compute_pixel_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
	"""Compute pixel-wise Accuracy, Precision, Recall, and Dice for binary masks.
	Expects shapes (N,H,W,1) or (N,H,W). y_pred_prob in [0,1].
	"""
	if y_true.ndim == 4 and y_true.shape[-1] == 1:
		y_true = y_true[..., 0]
	if y_pred_prob.ndim == 4 and y_pred_prob.shape[-1] == 1:
		y_pred_prob = y_pred_prob[..., 0]

	y_true_bin = (y_true >= 0.5).astype(np.uint8)
	y_pred_bin = _threshold_preds(y_pred_prob, threshold)

	# Flatten
	y_true_f = y_true_bin.reshape(-1)
	y_pred_f = y_pred_bin.reshape(-1)

	TP = int(np.sum((y_true_f == 1) & (y_pred_f == 1)))
	TN = int(np.sum((y_true_f == 0) & (y_pred_f == 0)))
	FP = int(np.sum((y_true_f == 0) & (y_pred_f == 1)))
	FN = int(np.sum((y_true_f == 1) & (y_pred_f == 0)))

	accuracy = (TP + TN) / max(1, (TP + TN + FP + FN))
	precision = TP / max(1, (TP + FP))
	recall = TP / max(1, (TP + FN))
	dice = (2 * TP) / max(1, (2 * TP + FP + FN))

	return {
		"accuracy": float(accuracy),
		"precision": float(precision),
		"recall": float(recall),
		"dice": float(dice),
	}


def plot_training_curves(history: Dict[str, Any], save_path: Optional[str] = None) -> None:
	"""Plot training/validation loss and metric curves from Keras history dict."""
	plt.figure(figsize=(10, 4))

	plt.subplot(1, 2, 1)
	plt.plot(history.get("loss", []), label="train_loss")
	plt.plot(history.get("val_loss", []), label="val_loss")
	plt.title("Loss")
	plt.xlabel("Epoch")
	plt.legend()

	# Try common metric keys
	metric_keys = [k for k in history.keys() if k not in ("loss", "val_loss")]
	if metric_keys:
		m = metric_keys[0]
		vm = f"val_{m}"
		plt.subplot(1, 2, 2)
		plt.plot(history.get(m, []), label=m)
		plt.plot(history.get(vm, []), label=vm)
		plt.title(m)
		plt.xlabel("Epoch")
		plt.legend()

	plt.tight_layout()
	if save_path:
		plt.savefig(save_path, dpi=150, bbox_inches="tight")
	else:
		plt.show()















































