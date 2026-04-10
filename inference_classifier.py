import os
import json
import numpy as np
import tensorflow as tf
from typing import Tuple, List, Dict


def load_meta_keys(checkpoint_dir: str) -> List[str]:
	meta_path = os.path.join(checkpoint_dir, "meta.json")
	if os.path.isfile(meta_path):
		with open(meta_path, "r", encoding="utf-8") as f:
			obj = json.load(f)
			return obj.get("metadata_keys", [])
	return []


def decode_image(path: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:
	image = tf.io.read_file(path)
	image = tf.io.decode_jpeg(image, channels=3)
	image = tf.image.resize(image, image_size)
	image = tf.cast(image, tf.float32) / 255.0
	return image


def tta_augment(image: tf.Tensor) -> tf.Tensor:
	return image


def tta_predictions(
	model: tf.keras.Model,
	images: np.ndarray,
	metadata: np.ndarray = None,
	n_transforms: int = 8,
) -> np.ndarray:
	"""Test-time augmentation for classification: flips and rotations.
	Returns averaged probabilities.
	"""
	probs = []
	for k in range(n_transforms):
		aug = images
		if k % 2 == 1:
			aug = np.flip(aug, axis=2)
		rot_k = (k // 2) % 4
		if rot_k:
			aug = np.rot90(aug, k=rot_k, axes=(1, 2))
		p = model.predict([aug, metadata] if metadata is not None else aug, verbose=0)
		# invert rotation for consistency not required in classification
		probs.append(p)
	return np.mean(np.stack(probs, axis=0), axis=0)


def predict_paths(
	checkpoint_path: str,
	image_paths: List[str],
	metadata_list: List[Dict[str, float]] = None,
	image_size: Tuple[int, int] = (300, 300),
	use_tta: bool = True,
	batch_size: int = 16,
) -> np.ndarray:
	model = tf.keras.models.load_model(checkpoint_path, compile=False)
	# Build consistent metadata vectors
	meta_keys = load_meta_keys(os.path.dirname(checkpoint_path))
	if metadata_list is not None and meta_keys:
		meta_arr = np.stack([[m.get(k, 0.0) for k in meta_keys] for m in metadata_list], axis=0).astype(np.float32)
	else:
		meta_arr = None

	# Load images
	imgs = []
	for p in image_paths:
		img = tf.keras.preprocessing.image.load_img(p, target_size=image_size)
		img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
		imgs.append(img)
	imgs = np.stack(imgs, axis=0)

	if use_tta:
		return tta_predictions(model, imgs, meta_arr)
	else:
		return model.predict([imgs, meta_arr] if meta_arr is not None else imgs, batch_size=batch_size, verbose=0)


def ensemble_predict(
	checkpoint_paths: List[str],
	image_paths: List[str],
	metadata_list: List[Dict[str, float]] = None,
	image_size: Tuple[int, int] = (300, 300),
	weights: Optional[List[float]] = None,
	use_tta: bool = True,
) -> np.ndarray:
	probs = []
	for ckpt in checkpoint_paths:
		p = predict_paths(ckpt, image_paths, metadata_list, image_size, use_tta)
		probs.append(p)
	stack = np.stack(probs, axis=0)
	if weights is None:
		weights = np.ones((stack.shape[0],), dtype=np.float32) / stack.shape[0]
	weights = np.array(weights).reshape([-1, 1, 1])
	return np.sum(stack * weights, axis=0)

































