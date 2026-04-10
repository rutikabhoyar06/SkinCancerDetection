import numpy as np
from typing import Dict, Tuple


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
	cm = np.zeros((num_classes, num_classes), dtype=np.int64)
	for t, p in zip(y_true, y_pred):
		cm[int(t), int(p)] += 1
	return cm


def precision_recall_f1(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
	# Per-class precision, recall, f1 and macro f1
	TP = np.diag(cm).astype(np.float64)
	FP = np.sum(cm, axis=0) - TP
	FN = np.sum(cm, axis=1) - TP
	precision = np.divide(TP, TP + FP + 1e-12)
	recall = np.divide(TP, TP + FN + 1e-12)
	f1 = 2 * precision * recall / (precision + recall + 1e-12)
	macro_f1 = float(np.mean(f1))
	return precision, recall, f1, macro_f1


def topk_accuracy(y_true: np.ndarray, prob: np.ndarray, k: int = 1) -> float:
	pred_topk = np.argsort(-prob, axis=1)[:, :k]
	correct = 0
	for i, t in enumerate(y_true):
		if t in pred_topk[i]:
			correct += 1
	return correct / len(y_true)


def ece(prob: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
	# Expected Calibration Error (multiclass)
	conf = np.max(prob, axis=1)
	pred = np.argmax(prob, axis=1)
	acc = (pred == y_true).astype(np.float32)
	bins = np.linspace(0.0, 1.0, n_bins + 1)
	ece_val = 0.0
	for i in range(n_bins):
		m = (conf >= bins[i]) & (conf < bins[i + 1])
		if np.any(m):
			bin_acc = np.mean(acc[m])
			bin_conf = np.mean(conf[m])
			w = np.mean(m)
			ece_val += w * abs(bin_acc - bin_conf)
	return float(ece_val)

































