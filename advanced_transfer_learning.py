"""
Advanced Transfer Learning Implementation for Skin Cancer Detection
Using Xception and EfficientNet V2 with comprehensive fine-tuning techniques
"""

import os
import json
import argparse
from typing import Tuple, Dict, List, Optional, Union
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


class AdvancedTransferLearningClassifier:
    """
    Advanced transfer learning classifier with support for Xception and EfficientNet V2
    Includes comprehensive fine-tuning, advanced optimizers, and evaluation metrics
    """
    
    def __init__(self, 
                 model_name: str = "efficientnet_v2",
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 2,
                 dropout_rate: float = 0.3,
                 use_imagenet_weights: bool = True):
        """
        Initialize the advanced transfer learning classifier
        
        Args:
            model_name: 'xception' or 'efficientnet_v2'
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
            use_imagenet_weights: Whether to use ImageNet pretrained weights
        """
        self.model_name = model_name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_imagenet_weights = use_imagenet_weights
        self.model = None
        self.history = None
        self.class_names = ['benign', 'malignant']
        
    def _get_base_model(self) -> tf.keras.Model:
        """Get the base model (Xception or EfficientNet V2)"""
        weights = "imagenet" if self.use_imagenet_weights else None
        
        if self.model_name == "xception":
            base_model = tf.keras.applications.Xception(
                include_top=False,
                weights=weights,
                input_shape=self.input_shape,
                pooling=None
            )
        elif self.model_name == "efficientnet_v2":
            # Use EfficientNetV2B0 for better performance
            base_model = tf.keras.applications.EfficientNetV2B0(
                include_top=False,
                weights=weights,
                input_shape=self.input_shape,
                pooling=None
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
            
        return base_model
    
    def build_model(self, 
                   base_trainable: bool = False,
                   use_advanced_head: bool = True) -> tf.keras.Model:
        """
        Build the complete model with advanced architecture
        
        Args:
            base_trainable: Whether base model layers are trainable
            use_advanced_head: Whether to use advanced classification head
        """
        base_model = self._get_base_model()
        base_model.trainable = base_trainable
        
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        
        # Preprocessing
        if self.model_name == "xception":
            x = tf.keras.applications.xception.preprocess_input(inputs)
        else:  # efficientnet_v2
            x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
        
        # Base model
        x = base_model(x, training=False)
        
        # Global pooling
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        if use_advanced_head:
            # Advanced classification head with multiple layers
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
            
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(self.dropout_rate * 0.5)(x)
            
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dropout(self.dropout_rate * 0.3)(x)
        else:
            # Simple head
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs, name=f"{self.model_name}_classifier")
        self.model = model
        return model
    
    def compile_model(self, 
                     optimizer: str = "adamw",
                     learning_rate: float = 1e-3,
                     weight_decay: float = 1e-4,
                     use_cosine_decay: bool = False,
                     warmup_epochs: int = 5) -> None:
        """
        Compile the model with advanced optimizer and learning rate schedule
        
        Args:
            optimizer: Optimizer type ('adam', 'adamw', 'sgd', 'rmsprop')
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
            use_cosine_decay: Whether to use cosine learning rate decay
            warmup_epochs: Number of warmup epochs for cosine decay
        """
        if optimizer == "adamw":
            opt = tf.keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == "sgd":
            opt = tf.keras.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=0.9,
                nesterov=True
            )
        elif optimizer == "rmsprop":
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        # Loss function with label smoothing
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            label_smoothing=0.1
        )
        
        # Metrics
        metrics = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
    
    def get_callbacks(self, 
                     model_dir: str,
                     patience: int = 10,
                     min_delta: float = 1e-4,
                     factor: float = 0.5,
                     min_lr: float = 1e-7,
                     monitor: str = 'val_auc') -> List[tf.keras.callbacks.Callback]:
        """
        Get comprehensive callbacks for training
        
        Args:
            model_dir: Directory to save model checkpoints
            patience: Patience for early stopping and LR reduction
            min_delta: Minimum change to qualify as improvement
            factor: Factor by which learning rate will be reduced
            min_lr: Minimum learning rate
            monitor: Metric to monitor
        """
        os.makedirs(model_dir, exist_ok=True)
        
        callbacks = [
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                min_delta=min_delta,
                verbose=1
            ),
            
            # Model checkpointing
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(model_dir, f"{self.model_name}_best.keras"),
                monitor=monitor,
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=factor,
                patience=patience // 2,
                min_lr=min_lr,
                verbose=1
            ),
            
            # Learning rate scheduler
            tf.keras.callbacks.LearningRateScheduler(
                self._cosine_decay_schedule,
                verbose=0
            ),
            
            # CSV logger
            tf.keras.callbacks.CSVLogger(
                os.path.join(model_dir, f"{self.model_name}_training.log")
            )
        ]
        
        return callbacks
    
    def _cosine_decay_schedule(self, epoch: int, lr: float) -> float:
        """Cosine learning rate decay schedule"""
        initial_lr = 1e-3
        epochs = 50  # Total epochs
        return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))
    
    def create_data_generators(self, 
                              data_dir: str,
                              batch_size: int = 32,
                              validation_split: float = 0.2,
                              use_advanced_augmentation: bool = True) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Create advanced data generators with comprehensive augmentation
        
        Args:
            data_dir: Directory containing 'benign' and 'malignant' folders
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            use_advanced_augmentation: Whether to use advanced augmentation
        """
        # Create datasets
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            labels='inferred',
            label_mode='int',
            validation_split=validation_split,
            subset='training',
            seed=42,
            image_size=self.input_shape[:2],
            batch_size=batch_size,
            shuffle=True
        )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            labels='inferred',
            label_mode='int',
            validation_split=validation_split,
            subset='validation',
            seed=42,
            image_size=self.input_shape[:2],
            batch_size=batch_size,
            shuffle=False
        )
        
        # Advanced augmentation pipeline
        if use_advanced_augmentation:
            augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomContrast(0.1),
                tf.keras.layers.RandomBrightness(0.1),
                tf.keras.layers.RandomTranslation(0.1, 0.1),
                tf.keras.layers.RandomShear(0.1),
            ])
        else:
            # Basic augmentation
            augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.05),
                tf.keras.layers.RandomZoom(0.05),
            ])
        
        def augment_and_normalize(x, y):
            x = tf.cast(x, tf.float32) / 255.0
            x = augmentation(x, training=True)
            return x, y
        
        def normalize(x, y):
            x = tf.cast(x, tf.float32) / 255.0
            return x, y
        
        # Apply augmentation to training data
        train_ds = train_ds.map(augment_and_normalize, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Optimize performance
        train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds
    
    def compute_class_weights(self, data_dir: str) -> Dict[int, float]:
        """Compute class weights to handle class imbalance"""
        benign_dir = os.path.join(data_dir, "benign")
        malignant_dir = os.path.join(data_dir, "malignant")
        
        benign_count = len([f for f in os.listdir(benign_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        malignant_count = len([f for f in os.listdir(malignant_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        total = benign_count + malignant_count
        weight_benign = total / (2 * benign_count)
        weight_malignant = total / (2 * malignant_count)
        
        return {0: weight_benign, 1: weight_malignant}
    
    def train(self,
              data_dir: str,
              model_dir: str,
              batch_size: int = 32,
              epochs_frozen: int = 10,
              epochs_finetune: int = 20,
              lr_frozen: float = 1e-3,
              lr_finetune: float = 1e-4,
              use_class_weights: bool = True,
              use_advanced_augmentation: bool = True) -> tf.keras.callbacks.History:
        """
        Train the model with two-phase fine-tuning
        
        Args:
            data_dir: Directory containing training data
            model_dir: Directory to save model and logs
            batch_size: Batch size for training
            epochs_frozen: Epochs for frozen base training
            epochs_finetune: Epochs for fine-tuning
            lr_frozen: Learning rate for frozen phase
            lr_finetune: Learning rate for fine-tuning phase
            use_class_weights: Whether to use class weights
            use_advanced_augmentation: Whether to use advanced augmentation
        """
        # Create data generators
        train_ds, val_ds = self.create_data_generators(
            data_dir, batch_size, use_advanced_augmentation=use_advanced_augmentation
        )
        
        # Compute class weights
        class_weights = self.compute_class_weights(data_dir) if use_class_weights else None
        
        # Phase 1: Train with frozen base
        print("Phase 1: Training with frozen base model...")
        self.build_model(base_trainable=False, use_advanced_head=True)
        self.compile_model(learning_rate=lr_frozen, optimizer="adamw")
        
        callbacks = self.get_callbacks(model_dir, patience=5)
        
        history_frozen = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_frozen,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Phase 2: Fine-tune with unfrozen base
        print("Phase 2: Fine-tuning with unfrozen base model...")
        self.build_model(base_trainable=True, use_advanced_head=True)
        self.compile_model(learning_rate=lr_finetune, optimizer="adamw")
        
        # Update callbacks for fine-tuning
        callbacks_finetune = self.get_callbacks(model_dir, patience=8)
        
        history_finetune = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_finetune,
            callbacks=callbacks_finetune,
            class_weight=class_weights,
            verbose=1
        )
        
        # Combine histories
        combined_history = self._combine_histories(history_frozen, history_finetune)
        self.history = combined_history
        
        return combined_history
    
    def _combine_histories(self, history1, history2):
        """Combine two training histories"""
        combined = {}
        for key in history1.history.keys():
            combined[key] = history1.history[key] + history2.history[key]
        
        # Create a new History object
        combined_history = tf.keras.callbacks.History()
        combined_history.history = combined
        combined_history.epoch = list(range(len(combined['loss'])))
        
        return combined_history
    
    def evaluate(self, test_data_dir: str, model_path: Optional[str] = None) -> Dict:
        """
        Comprehensive evaluation of the model
        
        Args:
            test_data_dir: Directory containing test data
            model_path: Path to saved model (if None, uses current model)
        """
        if model_path:
            self.model = tf.keras.models.load_model(model_path)
        
        # Create test dataset
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_data_dir,
            labels='inferred',
            label_mode='int',
            seed=42,
            image_size=self.input_shape[:2],
            batch_size=32,
            shuffle=False
        )
        
        # Normalize test data
        test_ds = test_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
        
        # Get predictions
        y_true = []
        y_pred = []
        y_pred_proba = []
        
        for batch_x, batch_y in test_ds:
            predictions = self.model.predict(batch_x, verbose=0)
            y_true.extend(batch_y.numpy())
            y_pred.extend(np.argmax(predictions, axis=1))
            y_pred_proba.extend(predictions[:, 1])  # Probability of malignant class
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'auc': roc_auc_score(y_true, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, 
                                                         target_names=self.class_names, 
                                                         output_dict=True)
        }
        
        return metrics, y_true, y_pred, y_pred_proba
    
    def plot_training_history(self, save_path: str) -> None:
        """Plot comprehensive training history"""
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
        
        history = self.history.history
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Loss
        axes[0, 0].plot(history['loss'], label='Training Loss')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[0, 2].plot(history['precision'], label='Training Precision')
        axes[0, 2].plot(history['val_precision'], label='Validation Precision')
        axes[0, 2].set_title('Model Precision')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Recall
        axes[1, 0].plot(history['recall'], label='Training Recall')
        axes[1, 0].plot(history['val_recall'], label='Validation Recall')
        axes[1, 0].set_title('Model Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # AUC
        axes[1, 1].plot(history['auc'], label='Training AUC')
        axes[1, 1].plot(history['val_auc'], label='Validation AUC')
        axes[1, 1].set_title('Model AUC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Learning Rate (if available)
        if 'lr' in history:
            axes[1, 2].plot(history['lr'])
            axes[1, 2].set_title('Learning Rate')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].set_yscale('log')
            axes[1, 2].grid(True)
        else:
            axes[1, 2].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Learning Rate')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path: str) -> None:
        """Plot confusion matrix with detailed annotations"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_path: str) -> None:
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def hyperparameter_tuning_suggestions():
    """
    Provide comprehensive hyperparameter tuning suggestions
    """
    suggestions = {
        "learning_rates": {
            "frozen_phase": [1e-3, 5e-4, 1e-4],
            "finetune_phase": [1e-4, 5e-5, 1e-5]
        },
        "batch_sizes": [16, 32, 64],
        "dropout_rates": [0.2, 0.3, 0.4, 0.5],
        "optimizers": ["adamw", "adam", "sgd"],
        "weight_decay": [1e-4, 1e-3, 1e-2],
        "augmentation_strength": ["light", "medium", "heavy"],
        "model_architectures": ["xception", "efficientnet_v2"],
        "input_sizes": [(224, 224), (299, 299), (384, 384)],
        "regularization": {
            "label_smoothing": [0.0, 0.1, 0.2],
            "mixup_alpha": [0.0, 0.2, 0.4],
            "cutmix_alpha": [0.0, 1.0, 2.0]
        }
    }
    
    return suggestions


def advanced_regularization_techniques():
    """
    Advanced regularization techniques to improve accuracy beyond 80%
    """
    techniques = {
        "data_augmentation": {
            "mixup": "Blend images and labels to create synthetic training examples",
            "cutmix": "Cut and paste patches between images",
            "autoaugment": "Use learned augmentation policies",
            "randaugment": "Random augmentation with magnitude control"
        },
        "model_regularization": {
            "dropout_scheduling": "Gradually reduce dropout during training",
            "weight_decay": "L2 regularization on model weights",
            "label_smoothing": "Smooth hard labels to prevent overconfidence",
            "stochastic_depth": "Randomly skip layers during training"
        },
        "training_techniques": {
            "cosine_annealing": "Cosine learning rate schedule",
            "warmup": "Gradual learning rate warmup",
            "gradient_clipping": "Clip gradients to prevent exploding gradients",
            "exponential_moving_average": "Use EMA of model weights"
        },
        "ensemble_methods": {
            "model_ensemble": "Combine multiple models",
            "test_time_augmentation": "Apply augmentations at inference",
            "multi_scale_inference": "Test on multiple image scales"
        }
    }
    
    return techniques


def main():
    """Main training function with comprehensive configuration"""
    parser = argparse.ArgumentParser(description="Advanced Transfer Learning for Skin Cancer Detection")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Directory containing training data")
    parser.add_argument("--model_name", type=str, default="efficientnet_v2", 
                       choices=["xception", "efficientnet_v2"], help="Base model architecture")
    parser.add_argument("--input_size", type=int, nargs=2, default=[224, 224], 
                       help="Input image size (height width)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs_frozen", type=int, default=10, help="Epochs for frozen training")
    parser.add_argument("--epochs_finetune", type=int, default=20, help="Epochs for fine-tuning")
    parser.add_argument("--lr_frozen", type=float, default=1e-3, help="Learning rate for frozen phase")
    parser.add_argument("--lr_finetune", type=float, default=1e-4, help="Learning rate for fine-tuning")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--model_dir", type=str, default="advanced_checkpoints", help="Model save directory")
    parser.add_argument("--use_class_weights", action="store_true", help="Use class weights for imbalance")
    parser.add_argument("--advanced_augmentation", action="store_true", help="Use advanced augmentation")
    
    args = parser.parse_args()
    
    # Create classifier
    classifier = AdvancedTransferLearningClassifier(
        model_name=args.model_name,
        input_shape=(args.input_size[0], args.input_size[1], 3),
        dropout_rate=args.dropout
    )
    
    # Train model
    print(f"Training {args.model_name} model...")
    history = classifier.train(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        epochs_frozen=args.epochs_frozen,
        epochs_finetune=args.epochs_finetune,
        lr_frozen=args.lr_frozen,
        lr_finetune=args.lr_finetune,
        use_class_weights=args.use_class_weights,
        use_advanced_augmentation=args.advanced_augmentation
    )
    
    # Evaluate model
    print("Evaluating model...")
    metrics, y_true, y_pred, y_pred_proba = classifier.evaluate(args.data_dir)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print("="*50)
    
    # Save plots
    os.makedirs(args.model_dir, exist_ok=True)
    classifier.plot_training_history(os.path.join(args.model_dir, "training_history.png"))
    classifier.plot_confusion_matrix(y_true, y_pred, 
                                   os.path.join(args.model_dir, "confusion_matrix.png"))
    classifier.plot_roc_curve(y_true, y_pred_proba, 
                            os.path.join(args.model_dir, "roc_curve.png"))
    
    # Save metrics
    with open(os.path.join(args.model_dir, "evaluation_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print hyperparameter tuning suggestions
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING SUGGESTIONS")
    print("="*50)
    suggestions = hyperparameter_tuning_suggestions()
    for category, params in suggestions.items():
        print(f"\n{category.upper()}:")
        for param, values in params.items():
            print(f"  {param}: {values}")
    
    # Print regularization techniques
    print("\n" + "="*50)
    print("ADVANCED REGULARIZATION TECHNIQUES")
    print("="*50)
    techniques = advanced_regularization_techniques()
    for category, methods in techniques.items():
        print(f"\n{category.upper()}:")
        for method, description in methods.items():
            print(f"  {method}: {description}")


if __name__ == "__main__":
    main()






































