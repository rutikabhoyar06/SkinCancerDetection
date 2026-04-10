"""
Optimized Training Script - Direct execution for 90%+ accuracy
"""

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score

# Set seeds
tf.random.set_seed(42)
np.random.seed(42)

print("="*80)
print("OPTIMIZED TRAINING FOR 90%+ ACCURACY")
print("="*80)

# Configuration
IMAGE_SIZE = (300, 300)
BATCH_SIZE = 16
EPOCHS_FROZEN = 20
EPOCHS_FINETUNE = 60
LR_FROZEN = 1e-3
LR_FINETUNE = 1e-4
DROPOUT = 0.4
MODEL_DIR = "checkpoints"

os.makedirs(MODEL_DIR, exist_ok=True)

# Load data from split directories
print("\n📂 Loading data from dataset_split...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset_split/train",
    labels="inferred",
    label_mode="int",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset_split/val",
    labels="inferred",
    label_mode="int",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset_split/test",
    labels="inferred",
    label_mode="int",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
print(f"Classes: {class_names}")

# Compute class weights
import glob
benign_count = len(glob.glob("dataset_split/train/benign/*.jpg")) + \
               len(glob.glob("dataset_split/train/benign/*.jpeg")) + \
               len(glob.glob("dataset_split/train/benign/*.png"))
malignant_count = len(glob.glob("dataset_split/train/malignant/*.jpg")) + \
                  len(glob.glob("dataset_split/train/malignant/*.jpeg")) + \
                  len(glob.glob("dataset_split/train/malignant/*.png"))
total = benign_count + malignant_count
class_weight = {
    0: total / (2.0 * benign_count),
    1: total / (2.0 * malignant_count)
}
print(f"Class weights: {class_weight}")

# Advanced augmentation
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2),
])

def augment_and_cast(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = augmentation(x, training=True)
    return x, y

def cast_only(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    return x, y

train_ds = train_ds.map(augment_and_cast, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(cast_only, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(cast_only, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

# Build model with EfficientNetB3 for better accuracy
print("\n🏗️  Building EfficientNetB3 model...")
try:
    base = tf.keras.applications.EfficientNetB3(
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
        weights="imagenet"
    )
    base.trainable = False
    
    inputs = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(DROPOUT * 0.5)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="HighAccuracyClassifier")
    print("✅ EfficientNetB3 model built successfully!")
except Exception as e:
    print(f"⚠️  Could not build EfficientNetB3, using EfficientNetB0: {e}")
    from classifier import build_classifier, compile_classifier
    model = build_classifier(input_shape=(*IMAGE_SIZE, 3), base_trainable=False, dropout=DROPOUT)

# Phase 1: Frozen training
print(f"\n🚂 Phase 1: Training with frozen base ({EPOCHS_FROZEN} epochs)...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_FROZEN),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_frozen = [
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "bm_classifier_frozen_best.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
]

history_frozen = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FROZEN,
    callbacks=callbacks_frozen,
    verbose=1,
    class_weight=class_weight,
)

# Phase 2: Fine-tuning
print(f"\n🔧 Phase 2: Fine-tuning with unfrozen base ({EPOCHS_FINETUNE} epochs)...")

# Unfreeze base layers gradually
try:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
            layer.trainable = True
            # Freeze first 70% of base layers
            num_layers = len(layer.layers)
            for i, l in enumerate(layer.layers):
                if i < int(num_layers * 0.7):
                    l.trainable = False
except:
    for layer in model.layers:
        if hasattr(layer, "trainable"):
            layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_FINETUNE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_finetune = [
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "bm_classifier_best.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.3,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
]

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINETUNE,
    callbacks=callbacks_finetune,
    verbose=1,
    class_weight=class_weight,
)

# Evaluate on test set
print("\n📊 Evaluating on test set...")
y_true_test = []
y_pred_test = []
y_pred_proba_test = []

for batch_x, batch_y in test_ds:
    prob = model.predict(batch_x, verbose=0)
    y_true_test.extend(batch_y.numpy().tolist())
    y_pred_test.extend(np.argmax(prob, axis=1).tolist())
    y_pred_proba_test.extend(prob[:, 1].tolist())

y_true_test = np.array(y_true_test)
y_pred_test = np.array(y_pred_test)
y_pred_proba_test = np.array(y_pred_proba_test)

test_acc = accuracy_score(y_true_test, y_pred_test)
test_prec = precision_score(y_true_test, y_pred_test, zero_division=0)
test_rec = recall_score(y_true_test, y_pred_test, zero_division=0)
test_f1 = f1_score(y_true_test, y_pred_test, zero_division=0)
test_auc = roc_auc_score(y_true_test, y_pred_proba_test)
cm = confusion_matrix(y_true_test, y_pred_test)
report = classification_report(y_true_test, y_pred_test, target_names=class_names, output_dict=True)

print("\n" + "="*80)
print("🎯 FINAL TEST SET RESULTS")
print("="*80)
print(f"Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Precision: {test_prec:.4f}")
print(f"Recall:    {test_rec:.4f}")
print(f"F1-Score:  {test_f1:.4f}")
print(f"AUC:       {test_auc:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_true_test, y_pred_test, target_names=class_names))
print("="*80)

# Save metrics
test_metrics = {
    "accuracy": float(test_acc),
    "precision": float(test_prec),
    "recall": float(test_rec),
    "f1_score": float(test_f1),
    "auc": float(test_auc),
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "class_names": class_names,
}

with open(os.path.join(MODEL_DIR, "bm_classifier_test_metrics.json"), "w") as f:
    json.dump(test_metrics, f, indent=2)

print(f"\n✅ Model saved to: {os.path.join(MODEL_DIR, 'bm_classifier_best.keras')}")
print(f"✅ Metrics saved to: {os.path.join(MODEL_DIR, 'bm_classifier_test_metrics.json')}")



























