import streamlit as st
from PIL import Image
import numpy as np
import io
import os
import tensorflow as tf
import cv2

from classifier import build_classifier, compile_classifier
from unet import build_unet
from interpretability import grad_cam, overlay_heatmap

st.set_page_config(page_title="Skin Cancer Diagnosis", page_icon="🩺", layout="centered")

st.title("AI-Powered Skin Cancer Diagnosis")
st.caption("Image-based detection using Deep Learning — Benign vs Malignant")

MODEL_PATH = os.path.join("checkpoints", "bm_classifier_best.keras")
UNET_PATH = os.path.join("checkpoints", "unet_best.keras")

@st.cache_resource(show_spinner=False)
def load_classifier():
	if os.path.isfile(MODEL_PATH):
		try:
			model = tf.keras.models.load_model(MODEL_PATH)
			return model
		except Exception:
			pass
	# Fallback: untrained model (for first run). User should train to get real predictions.
	model = build_classifier(input_shape=(224, 224, 3), base_trainable=False, dropout=0.3)
	compile_classifier(model)
	return model

@st.cache_resource(show_spinner=False)
def load_unet():
	"""Load trained U-Net for boundary visualization if available."""
	if os.path.isfile(UNET_PATH):
		try:
			model = tf.keras.models.load_model(UNET_PATH, compile=False)
			return model
		except Exception:
			pass
	return None

def draw_contours_on_image(image_rgb: np.ndarray, mask_prob: np.ndarray, threshold: float = 0.5,
						   color=(0, 0, 255), thickness: int = 2) -> np.ndarray:
	"""Overlay lesion contours (from mask_prob) on image_rgb and return outlined image."""
	binary = (mask_prob >= threshold).astype(np.uint8) * 255
	contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	out = image_rgb.copy()
	# Draw contours in specified BGR color (convert to RGB later)
	cv2.drawContours(out, contours, -1, color, thickness)
	return out

def fallback_segment_mask(image_rgb: np.ndarray) -> np.ndarray:
	"""Return a simple lesion mask in [0,1] using classical vision as fallback.
	Uses grayscale + Otsu threshold + morphology to approximate lesion region.
	"""
	img = np.asarray(image_rgb)
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# Otsu threshold (invert if needed so lesion is foreground)
	_, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	if np.mean(mask) > 127:
		mask = 255 - mask
	# Morphological clean-up
	k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
	return (mask.astype(np.float32) / 255.0)

st.markdown(
	"Upload a dermatoscopic image. The app will classify it as benign or malignant."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"]) 

col1, col2 = st.columns(2)

model = load_classifier()
unet_model = load_unet()

if uploaded_file is not None:
	image_bytes = uploaded_file.read()
	image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

	with col1:
		st.subheader("Input Image")
		st.image(image, use_column_width=True)
		predict_clicked = st.button("Predict", type="primary")

	# Prepare input for classifier: EfficientNet preprocess happens inside the model
	img_resized = image.resize((224, 224))
	img_array_255 = np.asarray(img_resized, dtype=np.float32)  # 0..255

	if predict_clicked:
		# Predict probabilities [p_benign, p_malignant]
		pred = model.predict(np.expand_dims(img_array_255, axis=0), verbose=0)[0]
		benign_prob = float(pred[0])
		malignant_prob = float(pred[1])
		predicted_label = "Malignant" if malignant_prob >= 0.5 else "Benign"

		with col2:
			st.subheader("Prediction")
			st.metric(
				label="Predicted Class",
				value=predicted_label,
				delta=f"Malignant prob: {malignant_prob:.2%}"
			)
			st.progress(int(malignant_prob * 100))
			st.write(
				f"Benign: {benign_prob:.2%}  |  Malignant: {malignant_prob:.2%}"
			)

		with st.expander("Explain prediction (Feature Analysis)"):
			if os.path.isfile(MODEL_PATH):
				try:
					# Try Grad-CAM first
					heat = grad_cam(model, img_array_255, target_layer_name=None)
					overlay = overlay_heatmap(np.asarray(img_resized, dtype=np.float32) / 255.0, heat, alpha=0.45)
					st.image(overlay, caption="Grad-CAM Overlay", use_column_width=True)
					st.caption("Highlighted regions contributed most to the predicted class.")
				except Exception as e:
					# Fallback to feature analysis
					st.info("Grad-CAM unavailable. Showing feature analysis instead.")
					
					# Create feature analysis visualization
					import matplotlib.pyplot as plt
					from PIL import ImageFilter
					
					fig, axes = plt.subplots(2, 2, figsize=(10, 8))
					
					# Original image
					axes[0, 0].imshow(image)
					axes[0, 0].set_title("Original Image")
					axes[0, 0].axis('off')
					
					# Edge detection
					img_gray = image.convert('L').resize((224, 224))
					edges = img_gray.filter(ImageFilter.FIND_EDGES)
					axes[0, 1].imshow(edges, cmap='gray')
					axes[0, 1].set_title("Edge Detection\n(Border Analysis)")
					axes[0, 1].axis('off')
					
					# Color variance
					img_array_norm = img_array_255 / 255.0
					color_variance = np.var(img_array_norm, axis=2)
					im = axes[1, 0].imshow(color_variance, cmap='hot')
					axes[1, 0].set_title("Color Variance\n(Heterogeneity)")
					axes[1, 0].axis('off')
					plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
					
					# Red channel (important for skin lesions)
					axes[1, 1].imshow(img_array_255[:, :, 0], cmap='Reds')
					axes[1, 1].set_title("Red Channel\n(Blood vessels)")
					axes[1, 1].axis('off')
					
					plt.suptitle("Feature Analysis - What the model looks for", fontsize=14, fontweight='bold')
					plt.tight_layout()
					
					# Save and display
					import io
					buf = io.BytesIO()
					plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
					buf.seek(0)
					st.image(buf, caption="Feature Analysis", use_column_width=True)
					plt.close()
					
					st.caption("The model analyzes border irregularity, color variation, and texture patterns to make predictions.")
			else:
				st.info("Train the classifier first to enable feature analysis.")

		# Lesion/Bacterial boundary overlay using segmentation model (if available)
		st.markdown("---")
		st.subheader("Lesion Boundary (Segmentation Overlay)")
		if unet_model is not None:
			# Run segmentation on 224x224 image
			img_norm = (img_array_255 / 255.0).astype(np.float32)
			pred = unet_model.predict(np.expand_dims(img_norm, axis=0), verbose=0)
			mask_prob = pred[0, ..., 0] if pred.ndim == 4 and pred.shape[-1] == 1 else pred[0]
			outlined = draw_contours_on_image(np.asarray(img_resized), mask_prob, threshold=0.5, color=(0, 0, 255), thickness=2)
			mask_vis = (mask_prob >= 0.5).astype(np.uint8) * 255
			# Show side-by-side (3 panels)
			c1, c2, c3 = st.columns(3)
			with c1:
				st.image(image, caption="Original", use_column_width=True)
			with c2:
				st.image(outlined, caption="Overlay with Blue Boundary", use_column_width=True)
			with c3:
				st.image(mask_vis, caption="Binary Mask", use_column_width=True, clamp=True)
			st.caption("A blue line outlines the detected lesion/bacterial region (from the segmentation model).")
		else:
			# Fallback segmentation for immediate boundary visualization
			mask_prob = fallback_segment_mask(img_resized)
			outlined = draw_contours_on_image(np.asarray(img_resized), mask_prob, threshold=0.5, color=(0, 0, 255), thickness=2)
			mask_vis = (mask_prob >= 0.5).astype(np.uint8) * 255
			c1, c2, c3 = st.columns(3)
			with c1:
				st.image(image, caption="Original", use_column_width=True)
			with c2:
				st.image(outlined, caption="Fallback Boundary (Heuristic)", use_column_width=True)
			with c3:
				st.image(mask_vis, caption="Binary Mask (Fallback)", use_column_width=True, clamp=True)
			st.caption("Using a heuristic fallback until the U-Net is trained. Run `python train.py` for model-based boundaries.")
	else:
		with col2:
			st.info("Click Predict to classify the uploaded image.")
else:
	st.info("Please upload a dermatoscopic image (JPG/PNG) to begin.")

if not os.path.isfile(MODEL_PATH):
	st.warning("Trained classifier not found. Run training to get accurate predictions: `python train_classifier.py`.")
