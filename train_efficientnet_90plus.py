"""
EfficientNetB3 Training Script for 90%+ Accuracy
Includes transfer learning, data augmentation, batch normalization, and memory optimization
"""

import os
import json
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ============================================================================
# GPU MEMORY CONFIGURATION
# ============================================================================
print("Configuring GPU memory growth...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[OK] GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"[WARNING] GPU memory growth configuration error: {e}")
else:
    print("[INFO] No GPU detected, using CPU")

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
IMAGE_SIZE = (300, 300)
BATCH_SIZE = 8  # Start with smaller batch size to avoid memory issues
EPOCHS_FROZEN = 20
EPOCHS_FINETUNE = 60
LR_FROZEN = 1e-3
LR_FINETUNE = 1e-4
DROPOUT = 0.4
MODEL_DIR = "efficientnet_checkpoints"
DATA_ROOT = "dataset_split"

os.makedirs(MODEL_DIR, exist_ok=True)

print("="*80)
print("EFFICIENTNETB3 TRAINING FOR 90%+ ACCURACY")
print("="*80)
print(f"Image Size: {IMAGE_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS_FROZEN} frozen + {EPOCHS_FINETUNE} fine-tune")
print(f"Learning Rates: {LR_FROZEN} -> {LR_FINETUNE}")
print("="*80)

# ============================================================================
# DATA LOADING
# ============================================================================
print("\n[INFO] Loading data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_ROOT, "train"),
    labels="inferred",
    label_mode="int",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_ROOT, "val"),
    labels="inferred",
    label_mode="int",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_ROOT, "test"),
    labels="inferred",
    label_mode="int",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Classes: {class_names}")
print(f"Number of classes: {num_classes}")

# Compute class weights for imbalanced dataset
import glob
benign_path = os.path.join(DATA_ROOT, "train", "benign")
malignant_path = os.path.join(DATA_ROOT, "train", "malignant")

benign_count = (
    len(glob.glob(os.path.join(benign_path, "*.jpg"))) +
    len(glob.glob(os.path.join(benign_path, "*.jpeg"))) +
    len(glob.glob(os.path.join(benign_path, "*.png")))
)

malignant_count = (
    len(glob.glob(os.path.join(malignant_path, "*.jpg"))) +
    len(glob.glob(os.path.join(malignant_path, "*.jpeg"))) +
    len(glob.glob(os.path.join(malignant_path, "*.png")))
)

total = benign_count + malignant_count
weight_benign = total / (2.0 * benign_count)
weight_malignant = total / (2.0 * malignant_count)

class_weight = {0: weight_benign, 1: weight_malignant}
print(f"Class distribution - Benign: {benign_count}, Malignant: {malignant_count}")
print(f"Class weights - Benign: {weight_benign:.3f}, Malignant: {weight_malignant:.3f}")

# ============================================================================
# DATA AUGMENTATION
# ============================================================================
print("\n[INFO] Setting up data augmentation...")
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2),
])

def augment_and_cast(x, y):
    """Apply augmentation and normalize for training"""
    x = tf.cast(x, tf.float32) / 255.0
    x = augmentation(x, training=True)
    return x, y

def cast_only(x, y):
    """Normalize for validation/test (no augmentation)"""
    x = tf.cast(x, tf.float32) / 255.0
    return x, y

# Apply preprocessing
train_ds = train_ds.map(augment_and_cast, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(cast_only, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(cast_only, num_parallel_calls=tf.data.AUTOTUNE)

# Optimize data pipeline
train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

# ============================================================================
# MODEL BUILDING
# ============================================================================
print("\n[INFO] Building EfficientNetB3 model with transfer learning...")

# Load EfficientNetB3 base model with ImageNet weights
base_model = tf.keras.applications.EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(*IMAGE_SIZE, 3),
    pooling=None
)

# Freeze base model for Phase 1
base_model.trainable = False

# Build model with additional dense layers and batch normalization
inputs = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3))

# Preprocess input (EfficientNet preprocessing)
x = tf.keras.applications.efficientnet.preprocess_input(inputs)

# Base model
x = base_model(x, training=False)

# Global Average Pooling
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Batch Normalization
x = tf.keras.layers.BatchNormalization()(x)

# First dense layer with dropout
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(DROPOUT)(x)

# Second dense layer with dropout
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(DROPOUT * 0.5)(x)

# Third dense layer (optional, for extra capacity)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(DROPOUT * 0.25)(x)

# Output layer
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs, name="EfficientNetB3_90Plus")

print("[OK] Model built successfully!")
print(f"Total parameters: {model.count_params():,}")
print(f"Trainable parameters (Phase 1): {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# ============================================================================
# PHASE 1: FROZEN TRAINING
# ============================================================================
print(f"\n[PHASE 1] Training with frozen base ({EPOCHS_FROZEN} epochs)...")

# Compile with Adam optimizer and categorical crossentropy
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_FROZEN),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks for Phase 1
callbacks_frozen = [
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "efficientnet_frozen_best.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
        mode="max"
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
        mode="max"
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
        verbose=1,
        mode="max"
    ),
    tf.keras.callbacks.CSVLogger(
        os.path.join(MODEL_DIR, "training_frozen.csv")
    )
]

# Train Phase 1
history_frozen = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FROZEN,
    callbacks=callbacks_frozen,
    verbose=1,
    class_weight=class_weight
)

print("[OK] Phase 1 completed!")

# ============================================================================
# PHASE 2: FINE-TUNING
# ============================================================================
print(f"\n[PHASE 2] Fine-tuning with unfrozen base ({EPOCHS_FINETUNE} epochs)...")

# Unfreeze base model gradually (unfreeze top layers first)
base_model.trainable = True

# Fine-tune from a certain layer (unfreeze top 50% of layers)
num_layers = len(base_model.layers)
unfreeze_from = int(num_layers * 0.5)

for layer in base_model.layers[:unfreeze_from]:
    layer.trainable = False

print(f"Unfrozen {num_layers - unfreeze_from} out of {num_layers} base layers")

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_FINETUNE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print(f"Trainable parameters (Phase 2): {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# Callbacks for Phase 2
callbacks_finetune = [
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "efficientnet_best.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
        mode="max"
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.3,
        patience=7,
        min_lr=1e-7,
        verbose=1,
        mode="max"
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=15,
        restore_best_weights=True,
        verbose=1,
        mode="max"
    ),
    tf.keras.callbacks.CSVLogger(
        os.path.join(MODEL_DIR, "training_finetune.csv")
    )
]

# Train Phase 2
history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINETUNE,
    callbacks=callbacks_finetune,
    verbose=1,
    class_weight=class_weight
)

print("[OK] Phase 2 completed!")

# ============================================================================
# LOAD BEST MODEL
# ============================================================================
best_model_path = os.path.join(MODEL_DIR, "efficientnet_best.keras")
if os.path.exists(best_model_path):
    print(f"\n[INFO] Loading best model from {best_model_path}")
    model = tf.keras.models.load_model(best_model_path)
    print("[OK] Best model loaded!")
else:
    print("[WARNING] Best model checkpoint not found, using current model")

# ============================================================================
# EVALUATION
# ============================================================================
print("\n[INFO] Evaluating model on test set...")

# Collect predictions
y_true = []
y_pred = []
y_pred_proba = []

for batch_x, batch_y in test_ds:
    prob = model.predict(batch_x, verbose=0)
    y_true.extend(batch_y.numpy().tolist())
    y_pred.extend(np.argmax(prob, axis=1).tolist())
    y_pred_proba.extend(prob.tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_pred_proba = np.array(y_pred_proba)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Classification report
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

# ============================================================================
# PRINT RESULTS
# ============================================================================
print("\n" + "="*80)
print("TEST SET EVALUATION RESULTS")
print("="*80)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("="*80)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ============================================================================
# SAVE RESULTS
# ============================================================================
results = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "class_names": class_names,
    "model_path": best_model_path
}

results_path = os.path.join(MODEL_DIR, "evaluation_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n[OK] Results saved to {results_path}")

# Save final model
final_model_path = os.path.join(MODEL_DIR, "efficientnet_final.keras")
model.save(final_model_path)
print(f"[OK] Final model saved to {final_model_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TRAINING SUMMARY")
print("="*80)
print(f"Final Test Accuracy: {accuracy*100:.2f}%")
if accuracy >= 0.90:
    print("[SUCCESS] TARGET ACHIEVED: Accuracy is above 90%!")
else:
    print(f"[INFO] Target not yet reached. Current: {accuracy*100:.2f}%, Target: 90%")
    print("[TIP] Try:")
    print("   - Increasing training epochs")
    print("   - Using larger image size (e.g., 384x384)")
    print("   - Adjusting learning rates")
    print("   - Using ensemble methods")
print("="*80)

