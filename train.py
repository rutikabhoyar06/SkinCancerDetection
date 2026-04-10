from typing import Optional, Dict, Any
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from data_utils import load_dataset
from unet import build_unet
from losses import bce_dice_loss, dice_coefficient
from optimizers import create_lr_schedule, create_adam_optimizer
from augmentations import AugmentConfig, augmented_batch_generator


def _draw_contours_on_image(image: np.ndarray, mask_prob: np.ndarray, threshold: float = 0.5,
						 color=(0, 0, 255), thickness: int = 2) -> np.ndarray:
	"""Return image copy with lesion contours drawn from predicted mask.

	- image: RGB uint8 (H,W,3)
	- mask_prob: float32 (H,W) in [0,1]
	"""
	binary = (mask_prob >= threshold).astype(np.uint8) * 255
	contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	out = image.copy()
	cv2.drawContours(out, contours, -1, color, thickness)
	return out


def _save_example_contours(model: tf.keras.Model, images: np.ndarray, out_dir: str,
						  threshold: float = 0.5, max_examples: int = 4) -> None:
	"""Save side-by-side visualizations with contour overlays for a few samples."""
	os.makedirs(out_dir, exist_ok=True)
	count = min(max_examples, len(images))
	for idx in range(count):
		img = images[idx].astype(np.uint8)
		pred = model.predict(np.expand_dims(img.astype(np.float32) / 255.0, axis=0), verbose=0)
		if pred.ndim == 4 and pred.shape[-1] == 1:
			mask_prob = pred[0, ..., 0]
		else:
			mask_prob = pred[0]
		outlined = _draw_contours_on_image(img, mask_prob, threshold=threshold, color=(0, 0, 255), thickness=2)
		binary = (mask_prob >= threshold).astype(np.uint8)

		plt.figure(figsize=(10, 5))
		plt.subplot(1, 2, 1); plt.imshow(img); plt.title("Original"); plt.axis("off")
		plt.subplot(1, 2, 2); plt.imshow(outlined); plt.imshow(binary, cmap="Reds", alpha=0.35)
		plt.title("Prediction + Contour"); plt.axis("off"); plt.tight_layout()
		plt.savefig(os.path.join(out_dir, f"example_{idx:02d}.png"), dpi=200, bbox_inches="tight")
		plt.close()


def _prepare_masks_array(array: Optional[np.ndarray]) -> Optional[np.ndarray]:
	if array is None:
		return None
	arr = np.asarray(array)
	if arr.size == 0:
		return None
	if arr.ndim == 3:
		arr = arr[..., np.newaxis]
	return arr.astype(np.float32)


def train_unet(
	data_root: str = "dataset",
	image_size=(224, 224),
	epochs: int = 10,
	batch_size: int = 16,
	base_filters: int = 32,
	dropout: float = 0.2,
	l2_reg: float = 1e-4,
	initial_lr: float = 1e-3,
	decay_steps: int = 2000,
	decay_rate: float = 0.96,
	patience: int = 5,
	model_dir: str = "checkpoints",
	use_augmentation: bool = True,
	masks_root: Optional[str] = None,
	save_example_contours: bool = True,
	contour_threshold: float = 0.5,
	max_example_visuals: int = 4,
	history_filename: str = "unet_history.json",
	max_samples: Optional[int] = None,
) -> Dict[str, Any]:
	"""Train U-Net on the dataset with early stopping and checkpoints."""
	os.makedirs(model_dir, exist_ok=True)

	splits = load_dataset(
		root_dir=data_root,
		image_size=image_size,
		test_ratio=0.15,
		val_ratio=0.15,
		random_state=42,
		masks_root=masks_root,
		max_samples=max_samples,
	)

	model = build_unet(input_shape=(image_size[0], image_size[1], 3), base_filters=base_filters, dropout=dropout, l2_reg=l2_reg, num_classes=1)

	lr_schedule = create_lr_schedule(initial_lr=initial_lr, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
	optimizer = create_adam_optimizer(lr_or_schedule=lr_schedule)

	model.compile(
		optimizer=optimizer,
		loss=bce_dice_loss,
		metrics=[
			dice_coefficient,
			tf.keras.metrics.BinaryAccuracy(name="accuracy"),
		],
	)

	callbacks = [
		tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
		tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir, "unet_best.keras"), monitor="val_loss", save_best_only=True),
	]

	train_masks = _prepare_masks_array(splits["train"].get("masks"))
	val_masks = _prepare_masks_array(splits["val"].get("masks"))
	test_masks = _prepare_masks_array(splits["test"].get("masks"))

	if train_masks is None or val_masks is None or test_masks is None:
		raise ValueError(
			"Segmentation masks are required for UNet training. "
			"Provide mask images via the 'masks_root' argument when calling train_unet."
		)

	if use_augmentation:
		aug_config = AugmentConfig()
		train_gen = augmented_batch_generator(
			x=splits["train"]["x"], y=train_masks,
			batch_size=batch_size, shuffle=True, augment=True, seed=42, config=aug_config
		)
		val_gen = augmented_batch_generator(
			x=splits["val"]["x"], y=val_masks,
			batch_size=batch_size, shuffle=False, augment=False, seed=42, config=aug_config
		)

		steps_per_epoch = max(1, int(len(splits["train"]["x"]) // batch_size))
		validation_steps = max(1, int(len(splits["val"]["x"]) // batch_size))

		history = model.fit(
			train_gen,
			epochs=epochs,
			steps_per_epoch=steps_per_epoch,
			validation_data=val_gen,
			validation_steps=validation_steps,
			callbacks=callbacks,
			verbose=1,
		)
	else:
		history = model.fit(
			x=splits["train"]["x"], y=train_masks,
			batch_size=batch_size,
			epochs=epochs,
			validation_data=(splits["val"]["x"], val_masks),
			callbacks=callbacks,
			verbose=1,
		)

	# Evaluate on the test split
	test_results = model.evaluate(splits["test"]["x"], test_masks, verbose=1, return_dict=True)

	# Persist training history for downstream visualization scripts
	history_path = os.path.join(model_dir, history_filename)
	try:
		with open(history_path, "w", encoding="utf-8") as f:
			json.dump(history.history, f, indent=2)
		print(f"Saved training history to {history_path}")
	except OSError as exc:
		print(f"Warning: Unable to save training history to {history_path}: {exc}")

	# Save contour visualizations on a few validation samples
	if save_example_contours:
		vis_dir = os.path.join(model_dir, "segmentation_examples")
		_save_example_contours(
			model=model,
			images=splits["val"]["x"],
			out_dir=vis_dir,
			threshold=contour_threshold,
			max_examples=max_example_visuals,
		)
	return {"model": model, "history": history.history, "test": test_results}


if __name__ == "__main__":
	# Quick start training with defaults
	train_unet()














