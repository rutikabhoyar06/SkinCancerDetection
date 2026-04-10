from typing import Optional, Tuple
import numpy as np
import tensorflow as tf
import cv2


def grad_cam(
	model: tf.keras.Model,
	image01: np.ndarray,
	target_layer_name: Optional[str] = None,
	class_index: Optional[int] = None,
	upsample_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
	"""Compute Grad-CAM heatmap for the given image and model.
	- image01: float32 [0,1] shaped (H,W,3)
	- For segmentation, we treat `class_index` as the output channel (0 for foreground in binary case).
	- target_layer_name: name of a convolutional layer near the end (e.g., last encoder/decoder conv)
	"""
	img = image01.astype(np.float32)
	# For EfficientNet, we need to pass the raw image (0-255) and let the model preprocess it
	img_input = np.expand_dims(img, axis=0)

	if target_layer_name is None:
		# Heuristic: choose the last Conv2D layer or EfficientNet block
		for layer in reversed(model.layers):
			if isinstance(layer, tf.keras.layers.Conv2D):
				target_layer_name = layer.name
				break
			# For EfficientNet, look for the last block with conv layers
			elif hasattr(layer, 'layers') and any(isinstance(sublayer, tf.keras.layers.Conv2D) for sublayer in layer.layers):
				# Find the last Conv2D in this block
				for sublayer in reversed(layer.layers):
					if isinstance(sublayer, tf.keras.layers.Conv2D):
						target_layer_name = f"{layer.name}/{sublayer.name}"
						break
				if target_layer_name:
					break
	
	# If still no layer found, try to find any layer with spatial dimensions
	if target_layer_name is None:
		for layer in reversed(model.layers):
			if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:  # (batch, height, width, channels)
				target_layer_name = layer.name
				break
	
	if target_layer_name is None:
		raise ValueError("No suitable layer found for Grad-CAM. Model may not have convolutional layers.")

	# Handle nested layer names (e.g., "block7a_expand_conv/Conv2D")
	if '/' in target_layer_name:
		parent_name, child_name = target_layer_name.split('/', 1)
		parent_layer = model.get_layer(parent_name)
		target_layer = parent_layer.get_layer(child_name)
	else:
		target_layer = model.get_layer(target_layer_name)
	
	grad_model = tf.keras.models.Model(
		inputs=model.inputs,
		outputs=[target_layer.output, model.output],
	)

	with tf.GradientTape() as tape:
		conv_outputs, predictions = grad_model(img_input)
		# Choose target class: if not provided, use argmax for classification models
		if predictions.shape[-1] == 1:
			target_idx = 0
		else:
			if class_index is None:
				target_idx = int(tf.argmax(predictions[0]).numpy())
			else:
				target_idx = int(class_index)
		loss = predictions[..., target_idx]

		# For segmentation, aggregate spatially (mean over HxW)
		loss = tf.reduce_mean(loss)

	grads = tape.gradient(loss, conv_outputs)
	pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

	conv_outputs = conv_outputs[0].numpy()
	pooled_grads = pooled_grads.numpy()

	for i in range(pooled_grads.shape[-1]):
		conv_outputs[..., i] *= pooled_grads[i]

	heatmap = np.mean(conv_outputs, axis=-1)
	heatmap = np.maximum(heatmap, 0)
	if np.max(heatmap) > 0:
		heatmap = heatmap / np.max(heatmap)
	else:
		heatmap = np.zeros_like(heatmap)

	H, W = image01.shape[:2]
	if upsample_size is None:
		upsample_size = (W, H)
	heatmap = cv2.resize(heatmap, upsample_size, interpolation=cv2.INTER_LINEAR)
	return heatmap.astype(np.float32)


def overlay_heatmap(image01: np.ndarray, heatmap01: np.ndarray, alpha: float = 0.4) -> np.ndarray:
	"""Overlay the Grad-CAM heatmap onto the original image.
	- image01: float32 [0,1] (H,W,3)
	- heatmap01: float32 [0,1] (H,W)
	"""
	image8u = (np.clip(image01, 0, 1) * 255).astype(np.uint8)
	heat8u = (np.clip(heatmap01, 0, 1) * 255).astype(np.uint8)
	colormap = cv2.applyColorMap(heat8u, cv2.COLORMAP_JET)
	colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
	overlay = cv2.addWeighted(image8u, 1.0, colormap, alpha, 0)
	return overlay

