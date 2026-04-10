"""
High-Performance Training Script for 90%+ Accuracy
Optimized settings for maximum performance
"""

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from classifier import build_classifier, compile_classifier

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def create_advanced_augmentation():
    """Create advanced augmentation pipeline"""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),
    ])

def load_data_from_split(data_root="dataset_split", image_size=(384, 384), batch_size=16):
    """Load data from pre-split directories"""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_root, "train"),
        labels="inferred",
        label_mode="int",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_root, "val"),
        labels="inferred",
        label_mode="int",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_root, "test"),
        labels="inferred",
        label_mode="int",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
    )
    
    return train_ds, val_ds, test_ds, train_ds.class_names

def compute_class_weights(data_root):
    """Compute class weights for imbalanced dataset"""
    import glob
    benign_path = os.path.join(data_root, "train", "benign")
    malignant_path = os.path.join(data_root, "train", "malignant")
    
    benign_count = len(glob.glob(os.path.join(benign_path, "*.jpg"))) + \
                   len(glob.glob(os.path.join(benign_path, "*.jpeg"))) + \
                   len(glob.glob(os.path.join(benign_path, "*.png")))
    
    malignant_count = len(glob.glob(os.path.join(malignant_path, "*.jpg"))) + \
                      len(glob.glob(os.path.join(malignant_path, "*.jpeg"))) + \
                      len(glob.glob(os.path.join(malignant_path, "*.png")))
    
    total = benign_count + malignant_count
    weight_benign = total / (2.0 * benign_count)
    weight_malignant = total / (2.0 * malignant_count)
    
    print(f"Class distribution - Benign: {benign_count}, Malignant: {malignant_count}")
    print(f"Class weights - Benign: {weight_benign:.3f}, Malignant: {weight_malignant:.3f}")
    
    return {0: weight_benign, 1: weight_malignant}

def train_high_accuracy_model(
    data_root="dataset_split",
    image_size=(384, 384),
    batch_size=16,
    epochs_frozen=15,
    epochs_finetune=50,
    lr_frozen=1e-3,
    lr_finetune=1e-4,
    model_dir="checkpoints",
    dropout=0.4,
):
    """Train model with optimized settings for 90%+ accuracy"""
    
    os.makedirs(model_dir, exist_ok=True)
    
    print("="*80)
    print("HIGH-PERFORMANCE TRAINING FOR 90%+ ACCURACY")
    print("="*80)
    print(f"Image Size: {image_size}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs_frozen} frozen + {epochs_finetune} fine-tune")
    print(f"Learning Rates: {lr_frozen} → {lr_finetune}")
    print("="*80)
    
    # Load data
    print("\n📂 Loading data...")
    train_ds, val_ds, test_ds, class_names = load_data_from_split(data_root, image_size, batch_size)
    print(f"Classes: {class_names}")
    
    # Compute class weights
    class_weight = compute_class_weights(data_root)
    
    # Advanced augmentation
    augmentation = create_advanced_augmentation()
    
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
    
    # Build model with EfficientNetB0 (will upgrade to B3 for better accuracy)
    print("\n🏗️  Building model...")
    # Use larger EfficientNet for better accuracy
    try:
        # Try to use EfficientNetB3 for better accuracy
        base = tf.keras.applications.EfficientNetB3(
            include_top=False,
            input_shape=(*image_size, 3),
            weights="imagenet"
        )
        base.trainable = False
        
        inputs = tf.keras.layers.Input(shape=(*image_size, 3))
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)
        x = base(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout * 0.5)(x)
        outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs, name="HighAccuracyClassifier")
    except:
        # Fallback to standard build_classifier
        model = build_classifier(input_shape=(*image_size, 3), base_trainable=False, dropout=dropout)
    
    # Phase 1: Frozen training
    print("\n🚂 Phase 1: Training with frozen base...")
    compile_classifier(model, learning_rate=lr_frozen)
    
    callbacks_frozen = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, "bm_classifier_frozen_best.keras"),
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
        epochs=epochs_frozen,
        callbacks=callbacks_frozen,
        verbose=1,
        class_weight=class_weight,
    )
    
    # Phase 2: Fine-tuning
    print("\n🔧 Phase 2: Fine-tuning with unfrozen base...")
    
    # Unfreeze base
    try:
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
                layer.trainable = True
                # Use lower learning rate for early layers
                for i, l in enumerate(layer.layers):
                    if i < len(layer.layers) * 0.7:  # Freeze first 70% of base layers
                        l.trainable = False
    except:
        # Fallback: make all trainable
        for layer in model.layers:
            if hasattr(layer, "trainable"):
                layer.trainable = True
    
    # Use lower learning rate for fine-tuning
    compile_classifier(model, learning_rate=lr_finetune)
    
    callbacks_finetune = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, "bm_classifier_best.keras"),
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
        epochs=epochs_finetune,
        callbacks=callbacks_finetune,
        verbose=1,
        class_weight=class_weight,
    )
    
    # Combine histories
    combined_history = {
        "accuracy": history_frozen.history.get("accuracy", []) + history_finetune.history.get("accuracy", []),
        "val_accuracy": history_frozen.history.get("val_accuracy", []) + history_finetune.history.get("val_accuracy", []),
        "loss": history_frozen.history.get("loss", []) + history_finetune.history.get("loss", []),
        "val_loss": history_frozen.history.get("val_loss", []) + history_finetune.history.get("val_loss", []),
    }
    
    # Save history
    with open(os.path.join(model_dir, "bm_classifier_history.json"), "w") as f:
        json.dump(combined_history, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(combined_history["loss"], label="Train Loss")
    plt.plot(combined_history["val_loss"], label="Val Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(combined_history["accuracy"], label="Train Accuracy")
    plt.plot(combined_history["val_accuracy"], label="Val Accuracy")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "bm_classifier_training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # Evaluate on validation set
    print("\n📊 Evaluating on validation set...")
    y_true_val = []
    y_pred_val = []
    y_pred_proba_val = []
    
    for batch_x, batch_y in val_ds:
        prob = model.predict(batch_x, verbose=0)
        y_true_val.extend(batch_y.numpy().tolist())
        y_pred_val.extend(np.argmax(prob, axis=1).tolist())
        y_pred_proba_val.extend(prob[:, 1].tolist())
    
    y_true_val = np.array(y_true_val)
    y_pred_val = np.array(y_pred_val)
    y_pred_proba_val = np.array(y_pred_proba_val)
    
    val_acc = accuracy_score(y_true_val, y_pred_val)
    val_prec = precision_score(y_true_val, y_pred_val, zero_division=0)
    val_rec = recall_score(y_true_val, y_pred_val, zero_division=0)
    val_f1 = f1_score(y_true_val, y_pred_val, zero_division=0)
    val_auc = roc_auc_score(y_true_val, y_pred_proba_val)
    
    print(f"\n✅ Validation Results:")
    print(f"   Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"   Precision: {val_prec:.4f}")
    print(f"   Recall: {val_rec:.4f}")
    print(f"   F1-Score: {val_f1:.4f}")
    print(f"   AUC: {val_auc:.4f}")
    
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
    
    print(f"\n🎯 TEST SET RESULTS:")
    print(f"   Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Precision: {test_prec:.4f}")
    print(f"   Recall: {test_rec:.4f}")
    print(f"   F1-Score: {test_f1:.4f}")
    print(f"   AUC: {test_auc:.4f}")
    
    # Save test metrics
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
    
    with open(os.path.join(model_dir, "bm_classifier_test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Test Set Confusion Matrix\nAccuracy: {test_acc*100:.2f}%")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "bm_classifier_test_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # Save validation metrics
    val_metrics = {
        "accuracy": float(val_acc),
        "precision": float(val_prec),
        "recall": float(val_rec),
        "f1_score": float(val_f1),
        "auc": float(val_auc),
    }
    
    with open(os.path.join(model_dir, "bm_classifier_val_metrics.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)
    
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE!")
    print("="*80)
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")
    print(f"Model saved to: {os.path.join(model_dir, 'bm_classifier_best.keras')}")
    print("="*80)
    
    return model, combined_history, test_metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train high-accuracy model (90%+ target)")
    parser.add_argument("--data_root", type=str, default="dataset_split")
    parser.add_argument("--image_size", type=int, nargs=2, default=[384, 384])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs_frozen", type=int, default=15)
    parser.add_argument("--epochs_finetune", type=int, default=50)
    parser.add_argument("--lr_frozen", type=float, default=1e-3)
    parser.add_argument("--lr_finetune", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--model_dir", type=str, default="checkpoints")
    
    args = parser.parse_args()
    
    train_high_accuracy_model(
        data_root=args.data_root,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        epochs_frozen=args.epochs_frozen,
        epochs_finetune=args.epochs_finetune,
        lr_frozen=args.lr_frozen,
        lr_finetune=args.lr_finetune,
        dropout=args.dropout,
        model_dir=args.model_dir,
    )



























