import os
import math
import json
import random
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, Optional, List


AUTOTUNE = tf.data.AUTOTUNE


def set_seed(seed: int = 42) -> None:
	random.seed(seed)
	np.random.seed(seed)
	tf.keras.utils.set_random_seed(seed)


def parse_csv(
	csv_path: str,
	image_root: str,
	class_names: Optional[List[str]] = None,
) -> Tuple[List[str], List[int], Dict[str, Dict[str, float]]]:
	"""Read CSV with columns: filepath,label[,age,sex,localization].
	Returns image paths, label indices, and metadata dict (per image).
	"""
	paths, labels, meta = [], [], {}
	with open(csv_path, "r", encoding="utf-8") as f:
		header = f.readline().strip().split(",")
		col_idx = {h: i for i, h in enumerate(header)}
		for line in f:
			parts = line.strip().split(",")
			rel = parts[col_idx["filepath"]]
			label = parts[col_idx["label"]]
			p = os.path.join(image_root, rel) if not os.path.isabs(rel) else rel
			paths.append(p)
			labels.append(label)
			# metadata (optional)
			m = {}
			if "age" in col_idx:
				try:
					m["age"] = float(parts[col_idx["age"]])
				except Exception:
					m["age"] = 0.0
			if "sex" in col_idx:
				sex = parts[col_idx["sex"]].lower()
				m["sex_f"] = 1.0 if sex == "female" else 0.0
				m["sex_m"] = 1.0 if sex == "male" else 0.0
				m["sex_u"] = 1.0 if sex not in ("male", "female") else 0.0
			if "localization" in col_idx:
				loc = parts[col_idx["localization"]].lower()
				for key in [
					"back","lower extremity","upper extremity","torso","abdomen","face","chest","foot","hand","ear","neck","scalp","genital","unknown"
				]:
					m[f"loc_{key}"] = 1.0 if loc == key else 0.0
			meta[p] = m

	# normalize labels to indices
	if class_names is None:
		class_names = sorted(list({l for l in labels}))
	name_to_idx = {n: i for i, n in enumerate(class_names)}
	label_indices = [name_to_idx[l] for l in labels]
	return paths, label_indices, meta


def compute_class_weights(labels: List[int], num_classes: int) -> Dict[int, float]:
	counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
	weights = (np.sum(counts) / (counts + 1e-6))
	weights = weights / np.max(weights)
	return {i: float(w) for i, w in enumerate(weights)}


def decode_image(path: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:
	image = tf.io.read_file(path)
	image = tf.io.decode_jpeg(image, channels=3)
	image = tf.image.resize(image, image_size)
	image = tf.cast(image, tf.float32) / 255.0
	return image


def augment(image: tf.Tensor) -> tf.Tensor:
	# Color
	image = tf.image.random_brightness(image, max_delta=0.1)
	image = tf.image.random_contrast(image, 0.9, 1.1)
	image = tf.image.random_saturation(image, 0.9, 1.1)
	# Geometric
	image = tf.image.random_flip_left_right(image)
	image = tf.image.random_flip_up_down(image)
	image = tf.image.rot90(image, k=tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32))
	# Cutout-like dropout
	if tf.random.uniform(()) < 0.5:
		h = tf.shape(image)[0]
		w = tf.shape(image)[1]
		ch = tf.cast(0.2 * tf.cast(h, tf.float32), tf.int32)
		cw = tf.cast(0.2 * tf.cast(w, tf.float32), tf.int32)
		y = tf.random.uniform([], 0, h - ch, dtype=tf.int32)
		x = tf.random.uniform([], 0, w - cw, dtype=tf.int32)
		mask = tf.ones([ch, cw, 3], dtype=image.dtype)
		padded = tf.image.pad_to_bounding_box(mask, y, x, h, w)
		image = tf.where(padded > 0, tf.zeros_like(image), image)
	return image


def make_dataset(
	image_paths: List[str],
	labels: List[int],
	meta: Dict[str, Dict[str, float]],
	image_size: Tuple[int, int] = (300, 300),
	batch_size: int = 16,
	augment_train: bool = True,
	shuffle: bool = True,
) -> tf.data.Dataset:
	paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)
	labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int32))

	def load_with_meta(p, y):
		img = decode_image(p, image_size)
		# metadata vector (stable ordering by sorted keys)
		m = meta[tf.compat.as_str_any(p.numpy().decode("utf-8"))]
		keys = sorted(list(m.keys()))
		vec = np.array([m[k] for k in keys], dtype=np.float32)
		return img, y, vec

	def load_py(p, y):
		img, yy, vec = tf.py_function(load_with_meta, [p, y], [tf.float32, tf.int32, tf.float32])
		img.set_shape([image_size[0], image_size[1], 3])
		yy.set_shape([])
		vec.set_shape([None])
		return img, yy, vec

	ds = tf.data.Dataset.zip((paths_ds, labels_ds)).map(load_py, num_parallel_calls=AUTOTUNE)
	if shuffle:
		ds = ds.shuffle(buffer_size=min(8192, len(image_paths)))
	if augment_train:
		ds = ds.map(lambda x, y, m: (augment(x), y, m), num_parallel_calls=AUTOTUNE)
	ds = ds.batch(batch_size).prefetch(AUTOTUNE)
	return ds


def build_classifier(
	num_classes: int,
	image_size: Tuple[int, int] = (300, 300),
	backbone: str = "B3",
	metadata_dim: int = 0,
	dropout: float = 0.3,
) -> tf.keras.Model:
	variant = {
		"B0": tf.keras.applications.EfficientNetB0,
		"B1": tf.keras.applications.EfficientNetB1,
		"B2": tf.keras.applications.EfficientNetB2,
		"B3": tf.keras.applications.EfficientNetB3,
		"B4": tf.keras.applications.EfficientNetB4,
	}[backbone]
	inputs = tf.keras.Input(shape=(*image_size, 3))
	meta_in = tf.keras.Input(shape=(metadata_dim,), name="metadata") if metadata_dim > 0 else None

	base = variant(include_top=False, weights="imagenet", input_tensor=inputs)
	x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
	if meta_in is not None:
		m = tf.keras.layers.Dense(64, activation="relu")(meta_in)
		m = tf.keras.layers.Dropout(0.2)(m)
		x = tf.keras.layers.Concatenate()([x, m])
	x = tf.keras.layers.Dropout(dropout)(x)
	x = tf.keras.layers.Dense(256, activation="relu")(x)
	x = tf.keras.layers.Dropout(dropout)(x)
	outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
	model = tf.keras.Model(inputs=[inputs, meta_in] if meta_in is not None else inputs, outputs=outputs)
	return model


def train(
	csv_train: str,
	csv_val: str,
	image_root: str,
	class_names: Optional[List[str]] = None,
	image_size: Tuple[int, int] = (300, 300),
	batch_size: int = 16,
	backbone: str = "B3",
	initial_lr: float = 3e-4,
	max_epochs: int = 40,
	output_dir: str = "checkpoints_classifier",
):
	os.makedirs(output_dir, exist_ok=True)
	# Load data
	tr_paths, tr_labels, tr_meta = parse_csv(csv_train, image_root, class_names)
	va_paths, va_labels, va_meta = parse_csv(csv_val, image_root, class_names)
	all_keys = sorted({k for d in [*tr_meta.values(), *va_meta.values()] for k in d.keys()})
	# Ensure vectors contain same keys
	for d in tr_meta.values():
		for k in all_keys:
			if k not in d:
				d[k] = 0.0
	for d in va_meta.values():
		for k in all_keys:
			if k not in d:
				d[k] = 0.0

	num_classes = len(sorted(list({l for l in tr_labels + va_labels})))
	class_weights = compute_class_weights(tr_labels, num_classes)

	# Datasets
	train_ds = make_dataset(tr_paths, tr_labels, tr_meta, image_size, batch_size, augment_train=True, shuffle=True)
	val_ds = make_dataset(va_paths, va_labels, va_meta, image_size, batch_size, augment_train=False, shuffle=False)

	# Build model
	model = build_classifier(
		num_classes=num_classes,
		image_size=image_size,
		backbone=backbone,
		metadata_dim=len(all_keys),
		dropout=0.3,
	)

	# Freeze encoder for warmup
	for layer in model.layers:
		if isinstance(layer, tf.keras.Model) and hasattr(layer, "name") and layer.name.startswith("efficientnet"):
			for l in layer.layers:
				l.trainable = False

	optimizer = tf.keras.optimizers.AdamW(learning_rate=initial_lr, weight_decay=1e-4)
	loss = tf.keras.losses.SparseCategoricalCrossentropy()
	metrics = [
		tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
		tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2"),
	]
	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

	callbacks = [
		tf.keras.callbacks.ModelCheckpoint(
			os.path.join(output_dir, "best_warmup.keras"),
			monitor="val_acc",
			save_best_only=True,
			save_weights_only=False,
		),
		tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=6, restore_best_weights=True),
	]

	model.fit(
		train_ds,
		epochs=min(8, max_epochs // 4),
		validation_data=val_ds,
		class_weight=class_weights,
		callbacks=callbacks,
		verbose=1,
	)

	# Unfreeze for fine-tuning
	for layer in model.layers:
		layer.trainable = True

	# Cosine decay schedule
	steps_per_epoch = math.ceil(len(tr_paths) / batch_size)
	total_steps = steps_per_epoch * max(1, max_epochs - min(8, max_epochs // 4))
	lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_lr, first_decay_steps=max(steps_per_epoch * 3, 1000))
	model.compile(
		optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=5e-5),
		loss=loss,
		metrics=metrics,
	)
	callbacks = [
		tf.keras.callbacks.ModelCheckpoint(
			os.path.join(output_dir, "best_finetune.keras"),
			monitor="val_acc",
			save_best_only=True,
			save_weights_only=False,
		),
		tf.keras.callbacks.ReduceLROnPlateau(monitor="val_acc", factor=0.5, patience=3, min_lr=1e-6),
		tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=10, restore_best_weights=True),
	]

	history = model.fit(
		train_ds,
		epochs=max_epochs,
		validation_data=val_ds,
		class_weight=class_weights,
		callbacks=callbacks,
		verbose=1,
	)

	# Save label map and metadata keys for inference
	with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
		json.dump({"class_names": class_names or [], "metadata_keys": all_keys}, f, indent=2)

	return model, history


if __name__ == "__main__":
	# Example CLI usage (adjust paths):
	# python train_classifier.py -- expects CSVs prepared externally
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--csv_train", type=str, required=True)
	parser.add_argument("--csv_val", type=str, required=True)
	parser.add_argument("--image_root", type=str, required=True)
	parser.add_argument("--image_size", type=int, nargs=2, default=(300, 300))
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--backbone", type=str, default="B3")
	parser.add_argument("--initial_lr", type=float, default=3e-4)
	parser.add_argument("--max_epochs", type=int, default=40)
	parser.add_argument("--output_dir", type=str, default="checkpoints_classifier")
	args = parser.parse_args()

	set_seed(42)
	train(
		csv_train=args.csv_train,
		csv_val=args.csv_val,
		image_root=args.image_root,
		image_size=tuple(args.image_size),
		batch_size=args.batch_size,
		backbone=args.backbone,
		initial_lr=args.initial_lr,
		max_epochs=args.max_epochs,
		output_dir=args.output_dir,
	)

import os
import json
import argparse
from typing import Tuple
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

from classifier import build_classifier, compile_classifier


def _plot_and_save_curves(history: tf.keras.callbacks.History, out_path: str) -> None:
	h = history.history
	plt.figure(figsize=(10, 4))
	plt.subplot(1, 2, 1)
	plt.plot(h.get("loss", []), label="train_loss")
	plt.plot(h.get("val_loss", []), label="val_loss")
	plt.title("Loss")
	plt.xlabel("Epoch")
	plt.legend()
	plt.subplot(1, 2, 2)
	plt.plot(h.get("accuracy", []), label="train_acc")
	plt.plot(h.get("val_accuracy", []), label="val_acc")
	plt.title("Accuracy")
	plt.xlabel("Epoch")
	plt.legend()
	plt.tight_layout()
	plt.savefig(out_path, dpi=150, bbox_inches="tight")
	plt.close()


def _count_class_files(data_root: str) -> Tuple[int, int]:
	"""Return counts for benign (0) and malignant (1) images in directory tree."""
	benign_dir = os.path.join(data_root, "benign")
	malignant_dir = os.path.join(data_root, "malignant")
	benign = sum(
		1 for f in os.listdir(benign_dir)
		if os.path.isfile(os.path.join(benign_dir, f)) and f.lower().endswith((".jpg", ".jpeg", ".png"))
	)
	malignant = sum(
		1 for f in os.listdir(malignant_dir)
		if os.path.isfile(os.path.join(malignant_dir, f)) and f.lower().endswith((".jpg", ".jpeg", ".png"))
	)
	return benign, malignant


def train_classifier(
	data_root: str = "dataset",
	image_size: Tuple[int, int] = (224, 224),
	batch_size: int = 32,
	epochs_frozen: int = 2,
	epochs_finetune: int = 3,
	lr_base: float = 1e-3,
	lr_finetune: float = 1e-4,
	base_trainable: bool = True,
	dropout: float = 0.3,
	model_dir: str = "checkpoints",
) -> tf.keras.callbacks.History:
	os.makedirs(model_dir, exist_ok=True)

	train_ds = tf.keras.utils.image_dataset_from_directory(
		data_root,
		labels="inferred",
		label_mode="int",
		validation_split=0.15,
		subset="training",
		seed=42,
		image_size=image_size,
		batch_size=batch_size,
	)

	val_ds = tf.keras.utils.image_dataset_from_directory(
		data_root,
		labels="inferred",
		label_mode="int",
		validation_split=0.15,
		subset="validation",
		seed=42,
		image_size=image_size,
		batch_size=batch_size,
	)

	class_names = train_ds.class_names

	# Basic augmentation pipeline (on-the-fly)
	augmentation = tf.keras.Sequential([
		tf.keras.layers.RandomFlip("horizontal_and_vertical"),
		tf.keras.layers.RandomRotation(0.08),
		tf.keras.layers.RandomZoom(0.1),
		tf.keras.layers.RandomContrast(0.1),
	])

	def augment_and_cast(x, y):
		x = tf.cast(x, tf.float32)
		x = augmentation(x)
		return x, y

	train_ds = train_ds.map(augment_and_cast, num_parallel_calls=tf.data.AUTOTUNE)
	val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)

	train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
	val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

	model = build_classifier(input_shape=(image_size[0], image_size[1], 3), base_trainable=False, dropout=dropout)
	compile_classifier(model, learning_rate=lr_base)

	callbacks = [
		tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
		tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir, "bm_classifier_best.keras"), monitor="val_accuracy", save_best_only=True),
	]

	# Compute class weights to mitigate imbalance
	try:
		c0, c1 = _count_class_files(data_root)
		total = max(1, c0 + c1)
		w0 = total / (2.0 * max(1, c0))
		w1 = total / (2.0 * max(1, c1))
		class_weight = {0: w0, 1: w1}
	except Exception:
		class_weight = None

	# Phase 1: train with base frozen
	history = model.fit(
		train_ds,
		validation_data=val_ds,
		epochs=epochs_frozen,
		callbacks=callbacks,
		verbose=1,
		class_weight=class_weight,
	)

	# Phase 2: unfreeze backbone and fine-tune with lower LR
	try:
		base_layer = None
		for lyr in model.layers:
			if isinstance(lyr, tf.keras.Model) and "efficientnet" in lyr.name:
				base_layer = lyr
				break
		if base_layer is None:
			# Fallback by name
			base_layer = model.get_layer("efficientnetb0")
		base_layer.trainable = True
	except Exception:
		# If we can't find the base explicitly, enable trainable on all but head layers
		for lyr in model.layers:
			if hasattr(lyr, "trainable"):
				lyr.trainable = True

	compile_classifier(model, learning_rate=lr_finetune)

	finetune_history = model.fit(
		train_ds,
		validation_data=val_ds,
		epochs=epochs_finetune,
		callbacks=callbacks,
		verbose=1,
		class_weight=class_weight,
	)

	# Save history JSON and curves
	hist_json_path = os.path.join(model_dir, "bm_classifier_history.json")
	with open(hist_json_path, "w", encoding="utf-8") as f:
		json.dump(history.history, f)
	_plot_and_save_curves(history, os.path.join(model_dir, "bm_classifier_training_curves.png"))

	# Evaluate on validation split: predictions and metrics
	y_true = []
	y_pred = []
	for batch_x, batch_y in val_ds:
		prob = model.predict(batch_x, verbose=0)
		y_true.extend(batch_y.numpy().tolist())
		y_pred.extend(np.argmax(prob, axis=1).tolist())
		
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)

	acc = accuracy_score(y_true, y_pred)
	prec = precision_score(y_true, y_pred, zero_division=0)
	rec = recall_score(y_true, y_pred, zero_division=0)
	cm = confusion_matrix(y_true, y_pred)
	report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

	metrics_json = {
		"accuracy": float(acc),
		"precision": float(prec),
		"recall": float(rec),
		"confusion_matrix": cm.tolist(),
		"classification_report": report,
		"class_names": class_names,
	}
	with open(os.path.join(model_dir, "bm_classifier_val_metrics.json"), "w", encoding="utf-8") as f:
		json.dump(metrics_json, f)

	# Save confusion matrix plot
	plt.figure(figsize=(4, 4))
	plt.imshow(cm, cmap="Blues")
	plt.title("Confusion Matrix (Val)")
	plt.xlabel("Predicted")
	plt.ylabel("True")
	for (i, j), v in np.ndenumerate(cm):
		plt.text(j, i, str(v), ha="center", va="center")
	plt.xticks([0, 1], class_names)
	plt.yticks([0, 1], class_names)
	plt.tight_layout()
	plt.savefig(os.path.join(model_dir, "bm_classifier_confusion_matrix.png"), dpi=150, bbox_inches="tight")
	plt.close()

	return finetune_history


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train benign/malignant classifier with fine-tuning")
	parser.add_argument("--data_root", type=str, default="dataset")
	parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--epochs_frozen", type=int, default=2)
	parser.add_argument("--epochs_finetune", type=int, default=3)
	parser.add_argument("--lr_base", type=float, default=1e-3)
	parser.add_argument("--lr_finetune", type=float, default=1e-4)
	parser.add_argument("--dropout", type=float, default=0.3)
	parser.add_argument("--model_dir", type=str, default="checkpoints")
	args = parser.parse_args()

	train_classifier(
		data_root=args.data_root,
		image_size=(args.image_size[0], args.image_size[1]),
		batch_size=args.batch_size,
		epochs_frozen=args.epochs_frozen,
		epochs_finetune=args.epochs_finetune,
		lr_base=args.lr_base,
		lr_finetune=args.lr_finetune,
		dropout=args.dropout,
		model_dir=args.model_dir,
	)
