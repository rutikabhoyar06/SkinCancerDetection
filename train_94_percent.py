"""
Advanced Training Script to Achieve 94%+ Accuracy
Uses focal loss, advanced augmentation, larger images, and ensemble methods
"""

import os
import json
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    jaccard_score
)

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[OK] GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"[WARNING] GPU configuration error: {e}")

# Set seeds
tf.random.set_seed(42)
np.random.seed(42)

# ============================================================================
# FOCAL LOSS IMPLEMENTATION
# ============================================================================

def focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal Loss for handling class imbalance
    Paper: https://arxiv.org/abs/1708.02002
    """
    def loss_fn(y_true, y_pred):
        # Convert to one-hot if needed
        y_true = tf.cast(y_true, tf.float32)
        if len(y_true.shape) == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=2)
        
        # Clip predictions to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Calculate cross entropy
        ce = -y_true * tf.math.log(y_pred)
        
        # Calculate p_t
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        
        # Calculate alpha_t
        alpha_t = tf.ones_like(y_true) * alpha
        alpha_t = tf.where(tf.equal(y_true, 1), alpha_t, 1 - alpha_t)
        
        # Calculate focal weight
        focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
        
        # Calculate focal loss
        focal_loss = focal_weight * ce
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))
    
    return loss_fn


# ============================================================================
# ADVANCED DATA AUGMENTATION
# ============================================================================

def create_advanced_augmentation():
    """Create advanced data augmentation pipeline"""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomTranslation(0.15, 0.15),
        tf.keras.layers.RandomContrast(0.3),
        tf.keras.layers.RandomBrightness(0.3),
    ])


def mixup(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = tf.shape(x)[0]
    index = tf.random.shuffle(tf.range(batch_size))
    
    mixed_x = lam * x + (1 - lam) * tf.gather(x, index)
    y_a, y_b = y, tf.gather(y, index)
    
    return mixed_x, y_a, y_b, lam


def apply_mixup(x, y, alpha=0.2):
    """Apply mixup to a batch"""
    mixed_x, y_a, y_b, lam = mixup(x, y, alpha)
    
    # Convert to one-hot for mixup
    y_a_onehot = tf.one_hot(tf.cast(y_a, tf.int32), depth=2)
    y_b_onehot = tf.one_hot(tf.cast(y_b, tf.int32), depth=2)
    mixed_y = lam * y_a_onehot + (1 - lam) * y_b_onehot
    
    return mixed_x, mixed_y


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data_from_split(
    data_root: str = "dataset_split",
    image_size: Tuple[int, int] = (384, 384),
    batch_size: int = 16
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
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


def compute_class_weights(data_root: str) -> Dict[int, float]:
    """Compute class weights for imbalanced dataset"""
    import glob
    
    benign_path = os.path.join(data_root, "train", "benign")
    malignant_path = os.path.join(data_root, "train", "malignant")
    
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
    
    print(f"Class distribution - Benign: {benign_count}, Malignant: {malignant_count}")
    print(f"Class weights - Benign: {weight_benign:.3f}, Malignant: {weight_malignant:.3f}")
    
    return {0: weight_benign, 1: weight_malignant}


# ============================================================================
# MODEL BUILDING
# ============================================================================

def build_enhanced_efficientnet(
    input_shape: Tuple[int, int, int] = (384, 384, 3),
    num_classes: int = 2,
    dropout: float = 0.5,
    base_trainable: bool = False,
    model_size: str = "B4"
) -> tf.keras.Model:
    """Build enhanced EfficientNet with better architecture"""
    from ensemble_models import build_efficientnet_model
    
    model = build_efficientnet_model(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout=dropout,
        base_trainable=base_trainable,
        model_size=model_size
    )
    
    # Enhance the model with additional layers
    # Get the output before the final dense layer
    base_output = model.layers[-2].output
    
    # Add batch normalization and additional dense layers
    x = tf.keras.layers.BatchNormalization()(base_output)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout * 0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    enhanced_model = tf.keras.Model(model.input, outputs, name=f"EnhancedEfficientNet{model_size}")
    return enhanced_model


def build_enhanced_resnet(
    input_shape: Tuple[int, int, int] = (384, 384, 3),
    num_classes: int = 2,
    dropout: float = 0.5,
    base_trainable: bool = False,
    model_type: str = "ResNet50"
) -> tf.keras.Model:
    """Build enhanced ResNet"""
    from ensemble_models import build_resnet_model
    
    model = build_resnet_model(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout=dropout,
        base_trainable=base_trainable,
        model_type=model_type
    )
    
    # Enhance similar to EfficientNet
    base_output = model.layers[-2].output
    x = tf.keras.layers.BatchNormalization()(base_output)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    enhanced_model = tf.keras.Model(model.input, outputs, name=f"Enhanced{model_type}")
    return enhanced_model


def build_enhanced_densenet(
    input_shape: Tuple[int, int, int] = (384, 384, 3),
    num_classes: int = 2,
    dropout: float = 0.5,
    base_trainable: bool = False,
    model_type: str = "DenseNet201"
) -> tf.keras.Model:
    """Build enhanced DenseNet"""
    from ensemble_models import build_densenet_model
    
    model = build_densenet_model(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout=dropout,
        base_trainable=base_trainable,
        model_type=model_type
    )
    
    # Enhance similar to others
    base_output = model.layers[-2].output
    x = tf.keras.layers.BatchNormalization()(base_output)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    enhanced_model = tf.keras.Model(model.input, outputs, name=f"Enhanced{model_type}")
    return enhanced_model


# ============================================================================
# TRAINING
# ============================================================================

def train_single_model_advanced(
    model: tf.keras.Model,
    model_name: str,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs_frozen: int,
    epochs_finetune: int,
    lr_frozen: float,
    lr_finetune: float,
    class_weight: Dict[int, float],
    model_dir: str,
    use_focal_loss: bool = True,
    use_mixup: bool = True,
) -> tf.keras.Model:
    """Train a single model with advanced techniques"""
    
    print(f"\n{'='*80}")
    print(f"Training {model_name} with Advanced Techniques")
    print(f"{'='*80}")
    
    # Phase 1: Frozen training
    print(f"\nPhase 1: Training with frozen base ({epochs_frozen} epochs)...")
    
    # Use focal loss or standard loss
    if use_focal_loss:
        loss_fn = focal_loss(alpha=0.25, gamma=2.0)
    else:
        loss_fn = "sparse_categorical_crossentropy"
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_frozen),
        loss=loss_fn,
        metrics=["accuracy"]
    )
    
    callbacks_frozen = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, f"{model_name}_frozen_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
            mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
            mode="max"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=12,
            restore_best_weights=True,
            verbose=1,
            mode="max"
        ),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: lr_frozen * (0.95 ** epoch)
        ),
    ]
    
    history_frozen = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_frozen,
        callbacks=callbacks_frozen,
        verbose=1,
        class_weight=class_weight if not use_focal_loss else None,
    )
    
    # Phase 2: Fine-tuning
    print(f"\nPhase 2: Fine-tuning with unfrozen base ({epochs_finetune} epochs)...")
    
    # Unfreeze base model gradually
    from ensemble_models import unfreeze_base_model
    unfreeze_base_model(model, unfreeze_ratio=0.6)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_finetune),
        loss=loss_fn,
        metrics=["accuracy"]
    )
    
    callbacks_finetune = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, f"{model_name}_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
            mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.3,
            patience=7,
            min_lr=1e-8,
            verbose=1,
            mode="max"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=20,
            restore_best_weights=True,
            verbose=1,
            mode="max"
        ),
    ]
    
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_finetune,
        callbacks=callbacks_finetune,
        verbose=1,
        class_weight=class_weight if not use_focal_loss else None,
    )
    
    # Load best model
    best_model_path = os.path.join(model_dir, f"{model_name}_best.keras")
    if os.path.exists(best_model_path):
        try:
            if use_focal_loss:
                model = tf.keras.models.load_model(best_model_path, custom_objects={'focal_loss': focal_loss})
            else:
                model = tf.keras.models.load_model(best_model_path)
            print(f"[OK] Loaded best model from {best_model_path}")
        except Exception as e:
            print(f"[WARNING] Could not load best model: {e}, using current model")
    
    return model


# ============================================================================
# TEST-TIME AUGMENTATION
# ============================================================================

def predict_with_tta(model, x, num_augmentations=5):
    """Test-time augmentation for better predictions"""
    predictions = []
    
    # Original prediction
    pred = model.predict(x, verbose=0)
    predictions.append(pred)
    
    # Augmented predictions
    augmentation = create_advanced_augmentation()
    for _ in range(num_augmentations - 1):
        x_aug = augmentation(x, training=True)
        pred_aug = model.predict(x_aug, verbose=0)
        predictions.append(pred_aug)
    
    # Average predictions
    avg_pred = np.mean(predictions, axis=0)
    return avg_pred


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model_advanced(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    model_name: str = "Model",
    use_tta: bool = True
) -> Dict[str, float]:
    """Evaluate model with optional TTA"""
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    all_x = []
    all_y = []
    for batch_x, batch_y in test_ds:
        all_x.append(batch_x.numpy())
        all_y.extend(batch_y.numpy().tolist())
    
    all_x = np.concatenate(all_x, axis=0)
    all_y = np.array(all_y)
    
    if use_tta:
        print(f"  Using Test-Time Augmentation for {model_name}...")
        proba = predict_with_tta(model, all_x, num_augmentations=5)
    else:
        proba = model.predict(all_x, verbose=0, batch_size=32)
    
    y_pred = np.argmax(proba, axis=1)
    y_pred_proba = proba[:, 1]
    
    acc = accuracy_score(all_y, y_pred)
    prec = precision_score(all_y, y_pred, zero_division=0, average="weighted")
    rec = recall_score(all_y, y_pred, zero_division=0, average="weighted")
    f1 = f1_score(all_y, y_pred, zero_division=0, average="weighted")
    f1_macro = f1_score(all_y, y_pred, zero_division=0, average="macro")
    auc = roc_auc_score(all_y, y_pred_proba)
    iou = jaccard_score(all_y, y_pred, average="macro")
    
    return {
        "model_name": model_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "f1_macro": float(f1_macro),
        "auc": float(auc),
        "iou": float(iou),
    }


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_for_94_percent(
    data_root: str = "dataset_split",
    image_size: Tuple[int, int] = (384, 384),
    batch_size: int = 12,
    epochs_frozen: int = 25,
    epochs_finetune: int = 75,
    lr_frozen: float = 1e-3,
    lr_finetune: float = 5e-5,
    dropout: float = 0.5,
    model_dir: str = "checkpoints_94",
    use_focal_loss: bool = True,
    use_tta: bool = True,
) -> Dict:
    """Main training function optimized for 94%+ accuracy"""
    
    os.makedirs(model_dir, exist_ok=True)
    
    print("="*80)
    print("ADVANCED TRAINING FOR 94%+ ACCURACY")
    print("="*80)
    print(f"Image Size: {image_size}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs_frozen} frozen + {epochs_finetune} fine-tune")
    print(f"Learning Rates: {lr_frozen} -> {lr_finetune}")
    print(f"Focal Loss: {use_focal_loss}")
    print(f"Test-Time Augmentation: {use_tta}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    train_ds, val_ds, test_ds, class_names = load_data_from_split(data_root, image_size, batch_size)
    print(f"Classes: {class_names}")
    
    # Compute class weights
    class_weight = compute_class_weights(data_root)
    
    # Data augmentation
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
    
    # Train models
    trained_models = []
    model_configs = []
    
    # EfficientNetB4
    print("\n" + "="*80)
    print("Building Enhanced EfficientNetB4...")
    efficientnet = build_enhanced_efficientnet(
        input_shape=(*image_size, 3),
        num_classes=2,
        dropout=dropout,
        base_trainable=False,
        model_size="B4"
    )
    efficientnet = train_single_model_advanced(
        efficientnet, "EfficientNetB4", train_ds, val_ds,
        epochs_frozen, epochs_finetune, lr_frozen, lr_finetune,
        class_weight, model_dir, use_focal_loss=use_focal_loss
    )
    trained_models.append(efficientnet)
    model_configs.append("EfficientNetB4")
    
    # ResNet50
    print("\n" + "="*80)
    print("Building Enhanced ResNet50...")
    resnet = build_enhanced_resnet(
        input_shape=(*image_size, 3),
        num_classes=2,
        dropout=dropout,
        base_trainable=False,
        model_type="ResNet50"
    )
    resnet = train_single_model_advanced(
        resnet, "ResNet50", train_ds, val_ds,
        epochs_frozen, epochs_finetune, lr_frozen, lr_finetune,
        class_weight, model_dir, use_focal_loss=use_focal_loss
    )
    trained_models.append(resnet)
    model_configs.append("ResNet50")
    
    # DenseNet201
    print("\n" + "="*80)
    print("Building Enhanced DenseNet201...")
    densenet = build_enhanced_densenet(
        input_shape=(*image_size, 3),
        num_classes=2,
        dropout=dropout,
        base_trainable=False,
        model_type="DenseNet201"
    )
    densenet = train_single_model_advanced(
        densenet, "DenseNet201", train_ds, val_ds,
        epochs_frozen, epochs_finetune, lr_frozen, lr_finetune,
        class_weight, model_dir, use_focal_loss=use_focal_loss
    )
    trained_models.append(densenet)
    model_configs.append("DenseNet201")
    
    # Evaluate individual models
    print("\n" + "="*80)
    print("Evaluating Individual Models")
    print("="*80)
    
    individual_metrics = {}
    for model, name in zip(trained_models, model_configs):
        metrics = evaluate_model_advanced(model, test_ds, name, use_tta=use_tta)
        individual_metrics[name] = metrics
        print(f"\n{name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
    
    # Create weighted ensemble
    print("\n" + "="*80)
    print("Creating Weighted Ensemble")
    print("="*80)
    
    from ensemble_voting import MaxVotingEnsemble
    
    # Weight models by their individual accuracy
    weights = [individual_metrics[name]['accuracy'] for name in model_configs]
    ensemble = MaxVotingEnsemble(trained_models, weights=weights)
    
    # Evaluate ensemble
    print("\nEvaluating Ensemble...")
    y_true = []
    all_x = []
    for batch_x, batch_y in test_ds:
        all_x.append(batch_x.numpy())
        y_true.extend(batch_y.numpy().tolist())
    
    all_x = np.concatenate(all_x, axis=0)
    y_true = np.array(y_true)
    
    if use_tta:
        print("  Using Test-Time Augmentation for ensemble...")
        # TTA for each model, then ensemble
        all_preds = []
        for model in trained_models:
            pred = predict_with_tta(model, all_x, num_augmentations=5)
            all_preds.append(pred)
        
        # Weighted average
        ensemble_proba = np.zeros_like(all_preds[0])
        for pred, weight in zip(all_preds, weights):
            ensemble_proba += pred * weight
        ensemble_proba = ensemble_proba / sum(weights)
    else:
        ensemble_proba = ensemble.predict_proba(all_x, batch_size=32, verbose=0)
    
    y_pred = np.argmax(ensemble_proba, axis=1)
    y_pred_proba = ensemble_proba[:, 1]
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0, average="weighted")
    rec = recall_score(y_true, y_pred, zero_division=0, average="weighted")
    f1 = f1_score(y_true, y_pred, zero_division=0, average="weighted")
    f1_macro = f1_score(y_true, y_pred, zero_division=0, average="macro")
    auc = roc_auc_score(y_true, y_pred_proba)
    iou = jaccard_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    ensemble_metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "f1_macro": float(f1_macro),
        "auc": float(auc),
        "iou": float(iou),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    
    print("\n" + "="*80)
    print("ENSEMBLE RESULTS")
    print("="*80)
    print(f"Accuracy:  {ensemble_metrics['accuracy']:.4f} ({ensemble_metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {ensemble_metrics['precision']:.4f}")
    print(f"Recall:    {ensemble_metrics['recall']:.4f}")
    print(f"F1-Score:  {ensemble_metrics['f1_score']:.4f}")
    print(f"F1-Macro:  {ensemble_metrics['f1_macro']:.4f}")
    print(f"AUC:       {ensemble_metrics['auc']:.4f}")
    print(f"IoU:       {ensemble_metrics['iou']:.4f}")
    print("="*80)
    
    if ensemble_metrics['accuracy'] >= 0.94:
        print("\n✅ SUCCESS! Target of 94%+ accuracy achieved!")
    else:
        print(f"\n⚠️  Current accuracy: {ensemble_metrics['accuracy']*100:.2f}%, Target: 94%")
        print("   Consider:")
        print("   - Training for more epochs")
        print("   - Using larger image size (448x448)")
        print("   - Adding more models to ensemble")
    
    # Save results
    ensemble_dir = os.path.join(model_dir, "ensemble")
    os.makedirs(ensemble_dir, exist_ok=True)
    
    for model, name in zip(trained_models, model_configs):
        model.save(os.path.join(ensemble_dir, f"{name}_final.keras"))
    
    all_metrics = {
        "individual_models": individual_metrics,
        "ensemble": ensemble_metrics,
        "class_names": class_names,
        "config": {
            "image_size": image_size,
            "epochs_frozen": epochs_frozen,
            "epochs_finetune": epochs_finetune,
            "use_focal_loss": use_focal_loss,
            "use_tta": use_tta,
        }
    }
    
    with open(os.path.join(model_dir, "ensemble_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Ensemble Confusion Matrix\nAccuracy: {ensemble_metrics['accuracy']*100:.2f}%")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "ensemble_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\n[OK] All results saved to: {model_dir}")
    
    return {
        "models": trained_models,
        "ensemble": ensemble,
        "metrics": all_metrics
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train models for 94%+ accuracy")
    parser.add_argument("--data_root", type=str, default="dataset_split")
    parser.add_argument("--image_size", type=int, nargs=2, default=[384, 384])
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--epochs_frozen", type=int, default=25)
    parser.add_argument("--epochs_finetune", type=int, default=75)
    parser.add_argument("--lr_frozen", type=float, default=1e-3)
    parser.add_argument("--lr_finetune", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--model_dir", type=str, default="checkpoints_94")
    parser.add_argument("--no_focal_loss", action="store_true")
    parser.add_argument("--no_tta", action="store_true")
    
    args = parser.parse_args()
    
    train_for_94_percent(
        data_root=args.data_root,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        epochs_frozen=args.epochs_frozen,
        epochs_finetune=args.epochs_finetune,
        lr_frozen=args.lr_frozen,
        lr_finetune=args.lr_finetune,
        dropout=args.dropout,
        model_dir=args.model_dir,
        use_focal_loss=not args.no_focal_loss,
        use_tta=not args.no_tta,
    )

