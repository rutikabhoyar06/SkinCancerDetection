"""
Generate a small synthetic segmentation dataset for quickly exercising the UNet
training and visualization pipeline.

The dataset imitates benign vs malignant lesions with paired binary masks. It is
lightweight (~200 samples by default) so it fits into memory-constrained
environments and is ideal for demos/tests when the real mask annotations are not
available locally.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


@dataclass
class SyntheticConfig:
	root: Path = Path("synthetic_dataset")
	image_size: Tuple[int, int] = (224, 224)
	samples_per_class: int = 100
	class_names: Tuple[str, str] = ("benign", "malignant")
	random_seed: int = 13


def _ensure_dirs(cfg: SyntheticConfig) -> Tuple[Path, Path]:
	"""Create image class directories and mask directory."""
	img_root = cfg.root
	mask_root = cfg.root / "masks"
	for class_name in cfg.class_names:
		(img_root / class_name).mkdir(parents=True, exist_ok=True)
	mask_root.mkdir(parents=True, exist_ok=True)
	return img_root, mask_root


def _generate_background(size: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
	h, w = size
	base = rng.normal(loc=190, scale=18, size=(h, w, 3)).astype(np.float32)
	noise = rng.normal(loc=0, scale=12, size=(h, w, 3)).astype(np.float32)
	img = np.clip(base + noise, 0, 255).astype(np.uint8)
	return img


def _draw_lesion(mask: np.ndarray, rng: np.random.Generator, class_idx: int) -> None:
	h, w = mask.shape
	center = (
		rng.integers(low=int(0.3 * w), high=int(0.7 * w)),
		rng.integers(low=int(0.3 * h), high=int(0.7 * h)),
	)
	axes = (
		rng.integers(low=int(0.15 * w), high=int(0.25 * w)),
		rng.integers(low=int(0.15 * h), high=int(0.25 * h)),
	)
	angle = rng.integers(low=0, high=180)
	cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
	# malignant samples get some irregular perturbations
	if class_idx == 1:
		for _ in range(3):
			offset = (
				center[0] + rng.integers(-axes[0] // 2, axes[0] // 2),
				center[1] + rng.integers(-axes[1] // 2, axes[1] // 2),
			)
			r = rng.integers(low=5, high=15)
			cv2.circle(mask, offset, r, 255, -1)


def _render_sample(cfg: SyntheticConfig, class_idx: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
	img = _generate_background(cfg.image_size, rng)
	mask = np.zeros(cfg.image_size, dtype=np.uint8)
	_draw_lesion(mask, rng, class_idx)

	color_shift = -30 if class_idx == 0 else -80
	img = np.clip(img.astype(np.int16) + color_shift, 0, 255).astype(np.uint8)

	# Blend lesion color
	colored_mask = np.stack([mask] * 3, axis=-1)
	img = np.where(colored_mask > 0, (0.3 * img + 0.7 * colored_mask).astype(np.uint8), img)

	return img, mask


def generate_dataset(cfg: SyntheticConfig = SyntheticConfig()) -> None:
	rng = np.random.default_rng(cfg.random_seed)
	img_root, mask_root = _ensure_dirs(cfg)

	total = cfg.samples_per_class * len(cfg.class_names)
	print(f"Generating synthetic dataset under '{cfg.root}' ({total} samples)...")

	counter = 0
	for class_idx, class_name in enumerate(cfg.class_names):
		target_dir = img_root / class_name
		for idx in range(cfg.samples_per_class):
			img, mask = _render_sample(cfg, class_idx, rng)
			fname = f"{class_name}_{idx:04d}.png"
			cv2.imwrite(str(target_dir / fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
			cv2.imwrite(str(mask_root / fname), mask)
			counter += 1
	print(f"Finished generating {counter} samples.")


if __name__ == "__main__":
	generate_dataset()




