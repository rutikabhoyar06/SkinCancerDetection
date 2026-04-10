import math
import random
from typing import Optional, Tuple, Dict, Iterator

import numpy as np
import cv2


class AugmentConfig:
	def __init__(
		self,
		rotation_degrees: float = 30.0,
		flip_horizontal_prob: float = 0.5,
		flip_vertical_prob: float = 0.2,
		zoom_range: Tuple[float, float] = (0.9, 1.1),
		brightness_limit: float = 0.15,  # fraction of 255
		contrast_limit: float = 0.15,    # fraction factor
		apply_prob: float = 1.0,
	):
		self.rotation_degrees = rotation_degrees
		self.flip_horizontal_prob = flip_horizontal_prob
		self.flip_vertical_prob = flip_vertical_prob
		self.zoom_range = zoom_range
		self.brightness_limit = brightness_limit
		self.contrast_limit = contrast_limit
		self.apply_prob = apply_prob


def _to_uint8(image01: np.ndarray) -> np.ndarray:
	if image01.dtype != np.float32 and image01.dtype != np.float64:
		return image01
	return np.clip(image01 * 255.0, 0, 255).astype(np.uint8)


def _to_float01(image8u: np.ndarray) -> np.ndarray:
	if image8u.dtype == np.float32 or image8u.dtype == np.float64:
		return np.clip(image8u, 0.0, 1.0).astype(np.float32)
	return (image8u.astype(np.float32) / 255.0).astype(np.float32)


def _random_rotation_matrix(h: int, w: int, max_degrees: float) -> np.ndarray:
	angle = random.uniform(-max_degrees, max_degrees)
	center = (w / 2.0, h / 2.0)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	return M


def _apply_affine(image: np.ndarray, M: np.ndarray, size: Tuple[int, int], is_mask: bool = False) -> np.ndarray:
	interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
	border = cv2.BORDER_REFLECT_101
	return cv2.warpAffine(image, M, size, flags=interp, borderMode=border)


def _random_flip(image: np.ndarray, do_h: bool, do_v: bool) -> np.ndarray:
	res = image
	if do_h:
		res = cv2.flip(res, 1)
	if do_v:
		res = cv2.flip(res, 0)
	return res


def _random_zoom(image: np.ndarray, zoom_range: Tuple[float, float]) -> np.ndarray:
	h, w = image.shape[:2]
	zoom = random.uniform(zoom_range[0], zoom_range[1])
	if zoom == 1.0:
		return image
	# scale and then center-crop or pad back to original size
	new_w = max(1, int(round(w * zoom)))
	new_h = max(1, int(round(h * zoom)))
	resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
	if zoom >= 1.0:
		# center crop
		start_x = (new_w - w) // 2
		start_y = (new_h - h) // 2
		return resized[start_y:start_y + h, start_x:start_x + w]
	else:
		# pad to original size
		pad_x = (w - new_w)
		pad_y = (h - new_h)
		pad_left = pad_x // 2
		pad_right = pad_x - pad_left
		pad_top = pad_y // 2
		pad_bottom = pad_y - pad_top
		border = cv2.BORDER_REFLECT_101
		return cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, border)


def _random_brightness_contrast(image: np.ndarray, brightness_limit: float, contrast_limit: float) -> np.ndarray:
	# image expected uint8
	alpha = 1.0 + random.uniform(-contrast_limit, contrast_limit)  # contrast factor
	beta = random.uniform(-brightness_limit, brightness_limit) * 255.0  # brightness shift
	result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
	return result


def augment_image(
	image01: np.ndarray,
	mask01: Optional[np.ndarray] = None,
	config: Optional[AugmentConfig] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
	"""Apply a sequence of random augmentations to an image (and mask if provided).
	Input/Output image in float32 [0,1]. Mask if present is float32 [0,1].
	"""
	if config is None:
		config = AugmentConfig()

	if random.random() > config.apply_prob:
		return image01, mask01

	img = _to_uint8(image01)
	msk = None if mask01 is None else _to_uint8((mask01 * 255.0).astype(np.uint8))

	h, w = img.shape[:2]

	# Rotation
	M = _random_rotation_matrix(h, w, config.rotation_degrees)
	img = _apply_affine(img, M, (w, h), is_mask=False)
	if msk is not None:
		msk = _apply_affine(msk, M, (w, h), is_mask=True)

	# Zoom
	img = _random_zoom(img, config.zoom_range)
	if msk is not None:
		msk = _random_zoom(msk, config.zoom_range)

	# Flips
	do_h = random.random() < config.flip_horizontal_prob
	do_v = random.random() < config.flip_vertical_prob
	img = _random_flip(img, do_h, do_v)
	if msk is not None:
		msk = _random_flip(msk, do_h, do_v)

	# Brightness/Contrast (image only)
	img = _random_brightness_contrast(img, config.brightness_limit, config.contrast_limit)

	img01 = _to_float01(img)
	if msk is not None:
		msk01 = (msk > 127).astype(np.float32)
		return img01, msk01
	return img01, None


def augmented_batch_generator(
	x: np.ndarray,
	y: np.ndarray,
	batch_size: int = 32,
	shuffle: bool = True,
	augment: bool = True,
	seed: Optional[int] = None,
	config: Optional[AugmentConfig] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
	"""Yield mini-batches (inputs, targets) applying augmentation when enabled."""
	N = x.shape[0]
	indices = np.arange(N)
	if seed is not None:
		random.seed(seed)
		np.random.seed(seed)

	while True:
		if shuffle:
			np.random.shuffle(indices)
		for start in range(0, N, batch_size):
			end = min(start + batch_size, N)
			batch_idx = indices[start:end]

			bx = x[batch_idx].copy()
			by = y[batch_idx].copy()

			if augment:
				aug_x = []
				aug_y = []
				for i in range(bx.shape[0]):
					img = bx[i]
					mask = by[i]
					img_aug, mask_aug = augment_image(img, mask, config)
					aug_x.append(img_aug)
					aug_y.append(mask_aug if mask_aug is not None else mask)
				bx = np.stack(aug_x, axis=0).astype(np.float32)
				by = np.stack(aug_y, axis=0).astype(np.float32)

			if by.ndim == 3:
				by = by[..., np.newaxis]

			yield bx, by

