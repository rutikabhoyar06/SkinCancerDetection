from typing import Tuple
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


# ------------------------------
# Losses & Metrics (Segmentation)
# ------------------------------
def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
	"""Compute Dice coefficient for binary or multiclass masks.

	Expects shapes [..., C]. If C==1, treated as binary.
	"""
	y_true = tf.cast(y_true, tf.float32)
	y_pred = tf.cast(y_pred, tf.float32)
	if y_pred.shape.rank is not None and y_pred.shape[-1] == 1:
		# Binary: ensure probabilities
		y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
		intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
		sums = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
		dice = (2.0 * intersection + smooth) / (sums + smooth)
		return tf.reduce_mean(dice)
	# Multiclass: compute per-class dice and average
	y_true = tf.one_hot(tf.cast(tf.squeeze(y_true, axis=-1), tf.int32), depth=tf.shape(y_pred)[-1]) if y_true.shape[-1] == 1 else y_true
	y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
	intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
	sums = tf.reduce_sum(y_true + y_pred, axis=[1, 2])
	dice = (2.0 * intersection + smooth) / (sums + smooth)
	return tf.reduce_mean(dice)


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
	return 1.0 - dice_coefficient(y_true, y_pred)


def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, bce_weight: float = 0.5) -> tf.Tensor:
	"""Hybrid BCE + Dice loss for stable optimization and overlap quality."""
	if y_pred.shape[-1] == 1:
		bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
	else:
		bce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
	return bce_weight * tf.reduce_mean(bce) + (1.0 - bce_weight) * dice_loss(y_true, y_pred)


def conv_block(x: tf.Tensor, filters: int, kernel_size: int = 3, dropout: float = 0.0, l2: float = 0.0) -> tf.Tensor:
	x = layers.Conv2D(filters, kernel_size, padding="same", activation="relu", kernel_regularizer=regularizers.l2(l2) if l2 > 0.0 else None)(x)
	x = layers.Conv2D(filters, kernel_size, padding="same", activation="relu", kernel_regularizer=regularizers.l2(l2) if l2 > 0.0 else None)(x)
	if dropout > 0.0:
		x = layers.Dropout(dropout)(x)
	return x


def down_block(x: tf.Tensor, filters: int, dropout: float = 0.0, l2: float = 0.0) -> Tuple[tf.Tensor, tf.Tensor]:
	skip = conv_block(x, filters, dropout=dropout, l2=l2)
	down = layers.MaxPooling2D(pool_size=(2, 2))(skip)
	return down, skip


def up_block(x: tf.Tensor, skip: tf.Tensor, filters: int, dropout: float = 0.0, l2: float = 0.0) -> tf.Tensor:
	x = layers.UpSampling2D(size=(2, 2))(x)
	x = layers.Concatenate()([x, skip])
	x = conv_block(x, filters, dropout=dropout, l2=l2)
	return x


def build_unet(
	input_shape: Tuple[int, int, int] = (224, 224, 3),
	base_filters: int = 32,
	dropout: float = 0.0,
	l2_reg: float = 0.0,
	num_classes: int = 1,
) -> tf.keras.Model:
	"""Build a standard U-Net with 4 down/4 up blocks and sigmoid output for segmentation."""
	inputs = layers.Input(shape=input_shape)

	# Encoder
	d1, s1 = down_block(inputs, base_filters * 1, dropout=dropout, l2=l2_reg)
	d2, s2 = down_block(d1, base_filters * 2, dropout=dropout, l2=l2_reg)
	d3, s3 = down_block(d2, base_filters * 4, dropout=dropout, l2=l2_reg)
	d4, s4 = down_block(d3, base_filters * 8, dropout=dropout, l2=l2_reg)

	# Bridge
	b = conv_block(d4, base_filters * 16, dropout=dropout, l2=l2_reg)

	# Decoder
	u1 = up_block(b, s4, base_filters * 8, dropout=dropout, l2=l2_reg)
	u2 = up_block(u1, s3, base_filters * 4, dropout=dropout, l2=l2_reg)
	u3 = up_block(u2, s2, base_filters * 2, dropout=dropout, l2=l2_reg)
	u4 = up_block(u3, s1, base_filters * 1, dropout=dropout, l2=l2_reg)

	# Output
	activation = "sigmoid" if num_classes == 1 else "softmax"
	outputs = layers.Conv2D(num_classes, kernel_size=1, padding="same", activation=activation)(u4)

	model = models.Model(inputs, outputs, name="UNet")
	return model


def build_unet_with_efficientnet_encoder(
	input_shape: Tuple[int, int, int] = (224, 224, 3),
	encoder_variant: str = "B0",
	trainable_at: int = -1,
	dropout: float = 0.0,
	l2_reg: float = 0.0,
	num_classes: int = 1,
) -> tf.keras.Model:
	"""U-Net with EfficientNet encoder (transfer learning) and custom decoder.

	- encoder_variant: one of {"B0","B1","B2","B3"}
	- trainable_at: freeze all layers before this index; -1 freezes all
	"""
	variant_map = {
		"B0": tf.keras.applications.EfficientNetB0,
		"B1": tf.keras.applications.EfficientNetB1,
		"B2": tf.keras.applications.EfficientNetB2,
		"B3": tf.keras.applications.EfficientNetB3,
	}
	if encoder_variant not in variant_map:
		raise ValueError(f"Unsupported encoder_variant: {encoder_variant}")
	Encoder = variant_map[encoder_variant]

	# Build encoder
	inputs = layers.Input(shape=input_shape)
	base = Encoder(include_top=False, weights="imagenet", input_tensor=inputs)

	# Identify skip connections by common EfficientNet endpoints
	# Names vary slightly across variants but blocks exist consistently.
	# We select feature maps at reduction levels 1..4.
	skips = [
		base.get_layer(name).output
		for name in [
			"block2a_expand_activation",  # 1/2
			"block3a_expand_activation",  # 1/4
			"block4a_expand_activation",  # 1/8
			"block6a_expand_activation",  # 1/16
		]
	]
	bottleneck = base.output  # 1/32

	# Optionally set trainable layers
	for i, layer in enumerate(base.layers):
		layer.trainable = (i >= trainable_at) if trainable_at >= 0 else False

	# Decoder: mirror with UpSampling, Concatenate, and conv blocks
	filters_seq = [256, 128, 64, 32]
	x = conv_block(bottleneck, filters=512, dropout=dropout, l2=l2_reg)
	for filters, skip in zip(filters_seq, reversed(skips)):
		x = layers.UpSampling2D(size=(2, 2))(x)
		x = layers.Concatenate()([x, skip])
		x = conv_block(x, filters=filters, dropout=dropout, l2=l2_reg)

	# Final upsampling to reach input resolution if needed
	if x.shape[1] != input_shape[0] or x.shape[2] != input_shape[1]:
		x = layers.UpSampling2D(size=(2, 2))(x)
		x = conv_block(x, filters=32, dropout=dropout, l2=l2_reg)

	activation = "sigmoid" if num_classes == 1 else "softmax"
	outputs = layers.Conv2D(num_classes, kernel_size=1, padding="same", activation=activation)(x)

	model = models.Model(inputs, outputs, name=f"UNet_EfficientNet{encoder_variant}")
	return model


def summarize_model(model: tf.keras.Model) -> str:
	"""Return model summary as a string."""
	stream = []
	model.summary(print_fn=lambda s: stream.append(s))
	return "\n".join(stream)


def visualize_model(model: tf.keras.Model, filepath: str = "unet.png") -> None:
	"""Save model architecture visualization to the given filepath.
	Requires pydot and Graphviz installed on the system.
	"""
	tf.keras.utils.plot_model(model, to_file=filepath, show_shapes=True, show_dtype=False, show_layer_names=True, expand_nested=False)


def _evaluate_test_split_accuracy(
	model_path: str = "checkpoints/unet_best.keras",
	data_root: str = "dataset",
	image_size: Tuple[int, int] = (224, 224),
	threshold: float = 0.5,
):
	"""Load U-Net and compute pixel-wise test metrics (prints accuracy as %)."""
	from data_utils import load_dataset
	from evaluation import compute_pixel_metrics

	if not os.path.isfile(model_path):
		print(f"Model not found: {model_path}. Train first (python train.py).")
		return
	model = tf.keras.models.load_model(model_path, compile=False)

	splits = load_dataset(
		root_dir=data_root,
		image_size=image_size,
		test_ratio=0.15,
		val_ratio=0.15,
		random_state=42,
		masks_root=os.path.join(data_root, "masks"),
	)
	if "masks" not in splits["test"] or splits["test"]["masks"].size == 0:
		print("No ground-truth masks found for the test split; cannot compute accuracy.")
		return

	images = splits["test"]["x"].astype(np.float32) / 255.0
	masks_true = splits["test"]["masks"].astype(np.float32)

	# Predict in batches
	batch_size = 16
	preds = []
	for i in range(0, len(images), batch_size):
		p = model.predict(images[i:i + batch_size], verbose=0)
		if p.ndim == 4 and p.shape[-1] == 1:
			p = p[..., 0]
		preds.append(p)
	mask_prob = np.concatenate(preds, axis=0)

	metrics = compute_pixel_metrics(y_true=masks_true, y_pred_prob=mask_prob[..., np.newaxis], threshold=threshold)
	dice = metrics.get("dice", 0.0)
	iou = float(dice / (2.0 - max(dice, 1e-12)))
	f1 = float(dice)

	acc_pct = metrics["accuracy"] * 100.0
	prec_pct = metrics["precision"] * 100.0
	rec_pct = metrics["recall"] * 100.0
	dice_pct = dice * 100.0
	iou_pct = iou * 100.0
	f1_pct = f1 * 100.0

	print("\nSegmentation metrics on TEST split:")
	print(f"  Accuracy: {acc_pct:.2f}%")
	print(f"  Precision: {prec_pct:.2f}%")
	print(f"  Recall: {rec_pct:.2f}%")
	print(f"  Dice: {dice_pct:.2f}%")
	print(f"  IoU (Jaccard): {iou_pct:.2f}%")
	print(f"  F1-score: {f1_pct:.2f}%")


def tta_predict_segmentation(
	model: tf.keras.Model,
	images: np.ndarray,
	num_classes: int = 1,
	aggregation: str = "mean",
) -> np.ndarray:
	"""Simple TTA for segmentation: flips and 90-degree rotations.

	- aggregation: "mean" or "median"
	Returns probabilities with shape [N, H, W, C].
	"""
	def apply_aug(x, k, flip_lr, flip_ud):
		# rotate k*90
		y = np.rot90(x, k=k, axes=(1, 2))
		if flip_lr:
			y = np.flip(y, axis=2)
		if flip_ud:
			y = np.flip(y, axis=1)
		return y

	def invert_aug(y, k, flip_lr, flip_ud):
		z = y
		if flip_ud:
			z = np.flip(z, axis=1)
		if flip_lr:
			z = np.flip(z, axis=2)
		# inverse rotation
		z = np.rot90(z, k=(4 - k) % 4, axes=(1, 2))
		return z

	images = images.astype(np.float32)
	aug_params = []
	for k in range(4):
		for flip_lr in [False, True]:
			for flip_ud in [False, True]:
				aug_params.append((k, flip_lr, flip_ud))

	preds = []
	batch_size = 8
	for k, flip_lr, flip_ud in aug_params:
		img_aug = apply_aug(images, k, flip_lr, flip_ud)
		out = []
		for i in range(0, len(img_aug), batch_size):
			p = model.predict(img_aug[i:i + batch_size], verbose=0)
			out.append(p)
		out = np.concatenate(out, axis=0)
		out = invert_aug(out, k, flip_lr, flip_ud)
		preds.append(out)

	stack = np.stack(preds, axis=0)
	if aggregation == "median":
		prob = np.median(stack, axis=0)
	else:
		prob = np.mean(stack, axis=0)
	return prob


if __name__ == "__main__":
	# Print segmentation accuracy on test split (if masks available)
	_evaluate_test_split_accuracy()
