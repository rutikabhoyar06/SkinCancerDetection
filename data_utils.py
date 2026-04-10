import os
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def list_image_files(root_dir: str, class_names: Tuple[str, str] = ("benign", "malignant")) -> List[Tuple[str, int]]:
	"""Return list of (filepath, label) pairs for the two classes.
	label: 0 for benign, 1 for malignant.
	"""
	items: List[Tuple[str, int]] = []
	for label, class_name in enumerate(class_names):
		class_dir = os.path.join(root_dir, class_name)
		if not os.path.isdir(class_dir):
			raise FileNotFoundError(f"Class directory not found: {class_dir}")
		for fname in os.listdir(class_dir):
			fpath = os.path.join(class_dir, fname)
			if not os.path.isfile(fpath):
				continue
			lower = fname.lower()
			if lower.endswith((".jpg", ".jpeg", ".png")):
				items.append((fpath, label))
	return items


def load_image(path: str, image_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
	"""Load an image as RGB and resize to image_size, return float32 array in [0,1]."""
	with Image.open(path) as im:
		im = im.convert("RGB").resize(image_size)
		arr = np.asarray(im, dtype=np.float32) / 255.0
	return arr


def load_optional_mask(path: str, masks_root: Optional[str], image_size: Tuple[int, int]) -> Optional[np.ndarray]:
	"""If masks_root is provided, try to find a mask with the same filename under it."""
	if not masks_root:
		return None
	fname = os.path.basename(path)
	candidate = os.path.join(masks_root, fname)
	if os.path.isfile(candidate):
		with Image.open(candidate) as m:
			m = m.convert("L").resize(image_size)
			mask = np.asarray(m, dtype=np.float32) / 255.0
			# Binarize softly
			mask = (mask >= 0.5).astype(np.float32)
			return mask
	return None


def load_dataset(
	root_dir: str,
	image_size: Tuple[int, int] = (224, 224),
	test_ratio: float = 0.15,
	val_ratio: float = 0.15,
	random_state: int = 42,
	masks_root: Optional[str] = None,
	max_samples: Optional[int] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
	"""
	Load images from benign/ and malignant/ under root_dir, preprocess, and split.

	Returns a dict with keys 'train', 'val', 'test', each containing:
	- 'x': images as float32 (N, H, W, 3)
	- 'y': labels as int64 (N,)
	- 'masks': optional masks as float32 (N, H, W) if masks exist; otherwise empty array with shape (0,)
	- If max_samples is provided, randomly subsample that many images (before splitting) for quicker experiments.
	"""
	pairs = list_image_files(root_dir)
	if len(pairs) == 0:
		raise ValueError(f"No images found under {root_dir}")

	if max_samples is not None and len(pairs) > max_samples:
		rng = np.random.default_rng(random_state)
		indices = rng.choice(len(pairs), size=max_samples, replace=False)
		indices.sort()
		pairs = [pairs[i] for i in indices]

	paths = [p for p, _ in pairs]
	labels = np.array([y for _, y in pairs], dtype=np.int64)

	# First split off test set
	paths_trainval, paths_test, y_trainval, y_test = train_test_split(
		paths, labels, test_size=test_ratio, random_state=random_state, stratify=labels
	)

	# Compute validation proportion from the remaining set
	val_size_relative = val_ratio / (1.0 - test_ratio)
	paths_train, paths_val, y_train, y_val = train_test_split(
		paths_trainval, y_trainval, test_size=val_size_relative,
		random_state=random_state, stratify=y_trainval
	)

	def load_batch(batch_paths: List[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
		images: List[np.ndarray] = []
		masks: List[np.ndarray] = []
		for p in batch_paths:
			img = load_image(p, image_size)
			images.append(img)
			mask = load_optional_mask(p, masks_root, image_size)
			if mask is not None:
				masks.append(mask)
		return (
			np.stack(images, axis=0).astype(np.float32),
			np.stack(masks, axis=0).astype(np.float32) if len(masks) == len(images) and len(masks) > 0 else None,
		)

	x_train, m_train = load_batch(paths_train)
	x_val, m_val = load_batch(paths_val)
	x_test, m_test = load_batch(paths_test)

	result: Dict[str, Dict[str, np.ndarray]] = {
		"train": {"x": x_train, "y": y_train},
		"val": {"x": x_val, "y": y_val},
		"test": {"x": x_test, "y": y_test},
	}
	# Attach masks if consistently available
	if m_train is not None and m_val is not None and m_test is not None:
		result["train"]["masks"] = m_train
		result["val"]["masks"] = m_val
		result["test"]["masks"] = m_test
	else:
		result["train"]["masks"] = np.array(())
		result["val"]["masks"] = np.array(())
		result["test"]["masks"] = np.array(())

	return result


def describe_split(split: Dict[str, np.ndarray]) -> str:
	"""Return a brief textual description of the split sizes and class balance."""
	y = split["y"]
	count_total = int(y.shape[0])
	count_benign = int((y == 0).sum())
	count_malignant = int((y == 1).sum())
	return (
		f"N={count_total}  |  Benign={count_benign}  |  Malignant={count_malignant}"
	)

