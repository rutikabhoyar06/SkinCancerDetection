# Skin Cancer Detection (Benign vs Malignant)

This project provides a Streamlit UI and scaffolding for an AI-powered skin cancer diagnosis app that classifies dermatoscopic images as **Benign** or **Malignant**.

## 1. Project Setup

- Python 3.10+ recommended
- Frameworks and libs: TensorFlow or PyTorch (pick one), NumPy, OpenCV, matplotlib, Pillow, Streamlit

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## 2. Dataset Organization

Organize your dataset as follows inside the `dataset/` directory:

```
dataset/
  benign/
    image_001.jpg
    image_002.jpg
    ...
  malignant/
    image_101.jpg
    image_102.jpg
    ...
```

- Each subfolder contains images for that class.
- Ensure images are reasonably sized or plan to resize during preprocessing.

## 3. Data Loading and Preprocessing

Use `data_utils.py` to load, preprocess, and split your dataset into train/val/test (70/15/15 by default). Images are resized to 224x224 and normalized to [0,1]. Optional segmentation masks can be loaded if provided under a separate masks folder with matching filenames.

Example:

```python
from data_utils import load_dataset, describe_split

splits = load_dataset(
    root_dir="dataset",
    image_size=(224, 224),
    test_ratio=0.15,
    val_ratio=0.15,
    random_state=42,
    masks_root=None  # or e.g., "dataset_masks" if available
)

print("Train:", describe_split(splits["train"]))
print("Val:", describe_split(splits["val"]))
print("Test:", describe_split(splits["test"]))

x_train, y_train = splits["train"]["x"], splits["train"]["y"]
```

## 4. Data Augmentation

Real-time augmentations are provided in `augmentations.py`. Supported transforms: rotation (±30°), horizontal/vertical flips, zoom (0.9–1.1), brightness, and contrast changes. Masks are transformed consistently when provided.

## 5. U-Net Model (Segmentation)

A U-Net architecture is provided in `unet.py` with encoder/decoder paths and skip connections. The final layer uses a sigmoid activation for binary segmentation (1 channel). For multi-class masks, set `num_classes>1` to use softmax.

## 6. Loss Functions and Optimizer

BCE + Dice combined loss and Adam optimizer with an exponential decay LR schedule are provided.

## 7. Training (Segmentation)

Use `train.py` for segmentation training if you have masks.

## 8. Evaluation Metrics

Compute pixel-wise accuracy, precision, recall, and Dice coefficient. Optionally, plot training/validation curves.

## 9. Model Interpretability (Grad-CAM)

Use Grad-CAM to visualize model decisions on selected predictions.

## 10. Classification Path (Benign vs Malignant)

If you only need classification, train the EfficientNet-based classifier and run the app:

```bash
python train_classifier.py  # saves checkpoints/bm_classifier_best.keras
streamlit run app.py       # app will load the checkpoint automatically
```

- The app shows predicted class and probabilities with an optional Grad-CAM overlay.
- Tip: For higher accuracy, allow fine-tuning by setting `base_trainable=True` inside `train_classifier.py`, and train for more epochs.
