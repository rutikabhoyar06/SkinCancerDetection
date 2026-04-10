#!/usr/bin/env python3
"""
Model Evaluation Script for Skin Cancer Detection
Evaluates the trained model on test dataset and provides comprehensive metrics.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self, model_path="checkpoints/bm_classifier_best.keras", data_root="dataset"):
        self.model_path = model_path
        self.data_root = data_root
        self.model = None
        self.class_names = ['benign', 'malignant']
        self.image_size = (224, 224)
        
    def load_model(self):
        """Load the trained model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        print(f"Loading model from {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)
        print("Model loaded successfully!")
        return True
        
    def load_validation_metrics(self):
        """Load validation metrics if available."""
        metrics_path = "checkpoints/bm_classifier_val_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                return json.load(f)
        return None
        
    def create_test_dataset(self, validation_split=0.15, test_split=0.15, batch_size=32):
        """Create test dataset from the remaining data after train/validation split."""
        print("Creating test dataset...")
        
        # Create full dataset
        full_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_root,
            labels="inferred",
            label_mode="int",
            seed=42,
            image_size=self.image_size,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Calculate split sizes
        total_batches = tf.data.experimental.cardinality(full_ds).numpy()
        val_size = int(validation_split * total_batches)
        test_size = int(test_split * total_batches)
        train_size = total_batches - val_size - test_size
        
        # Split the dataset
        train_ds = full_ds.take(train_size)
        remaining = full_ds.skip(train_size)
        val_ds = remaining.take(val_size)
        test_ds = remaining.skip(val_size)
        
        # Preprocess test dataset
        def preprocess(x, y):
            x = tf.cast(x, tf.float32)
            return x, y
            
        test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)
        
        print(f"Test dataset created with {tf.data.experimental.cardinality(test_ds).numpy()} batches")
        return test_ds
        
    def evaluate_on_test_data(self, test_ds):
        """Evaluate model on test dataset and return detailed metrics."""
        print("Evaluating model on test dataset...")
        
        # Collect all predictions and true labels
        y_true = []
        y_pred_probs = []
        y_pred_classes = []
        
        batch_count = 0
        total_batches = tf.data.experimental.cardinality(test_ds).numpy()
        
        for batch_x, batch_y in test_ds:
            # Get predictions
            pred_probs = self.model.predict(batch_x, verbose=0)
            pred_classes = np.argmax(pred_probs, axis=1)
            
            # Store results
            y_true.extend(batch_y.numpy())
            y_pred_probs.extend(pred_probs[:, 1])  # Malignant probability
            y_pred_classes.extend(pred_classes)
            
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"Processed {batch_count}/{total_batches} batches")
        
        y_true = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)
        y_pred_classes = np.array(y_pred_classes)
        
        return y_true, y_pred_probs, y_pred_classes
        
    def calculate_comprehensive_metrics(self, y_true, y_pred_probs, y_pred_classes):
        """Calculate comprehensive evaluation metrics."""
        print("Calculating comprehensive metrics...")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred_classes)
        precision = precision_score(y_true, y_pred_classes, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred_classes, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred_classes, average='weighted', zero_division=0)
        
        # Class-specific metrics
        precision_per_class = precision_score(y_true, y_pred_classes, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred_classes, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred_classes, average=None, zero_division=0)
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_true, y_pred_probs)
        except:
            roc_auc = 0.0
            
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate total cases
        total_cases = len(y_true)
        correct_predictions = np.sum(y_true == y_pred_classes)
        
        metrics = {
            'total_cases': int(total_cases),
            'correct_predictions': int(correct_predictions),
            'accuracy': float(accuracy),
            'precision_weighted': float(precision),
            'recall_weighted': float(recall),
            'f1_weighted': float(f1),
            'roc_auc': float(roc_auc),
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'class_specific': {
                'benign': {
                    'precision': float(precision_per_class[0]),
                    'recall': float(recall_per_class[0]),
                    'f1_score': float(f1_per_class[0]),
                    'support': int(np.sum(y_true == 0))
                },
                'malignant': {
                    'precision': float(precision_per_class[1]),
                    'recall': float(recall_per_class[1]),
                    'f1_score': float(f1_per_class[1]),
                    'support': int(np.sum(y_true == 1))
                }
            }
        }
        
        return metrics, cm
        
    def print_evaluation_report(self, metrics):
        """Print comprehensive evaluation report."""
        print("\n" + "="*70)
        print("SKIN CANCER DETECTION MODEL EVALUATION REPORT")
        print("="*70)
        
        # Overall Performance
        print(f"\nOVERALL PERFORMANCE:")
        print(f"{'Total Test Cases:':<25} {metrics['total_cases']:,}")
        print(f"{'Correct Predictions:':<25} {metrics['correct_predictions']:,}")
        print(f"{'ACCURACY:':<25} {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"{'Weighted Precision:':<25} {metrics['precision_weighted']:.4f}")
        print(f"{'Weighted Recall:':<25} {metrics['recall_weighted']:.4f}")
        print(f"{'Weighted F1-Score:':<25} {metrics['f1_weighted']:.4f}")
        print(f"{'ROC AUC Score:':<25} {metrics['roc_auc']:.4f}")
        
        # Confusion Matrix Components
        cm = metrics['confusion_matrix']
        print(f"\nCONFUSION MATRIX COMPONENTS:")
        print(f"{'True Negatives (TN):':<25} {cm['true_negatives']:,}")
        print(f"{'False Positives (FP):':<25} {cm['false_positives']:,}")
        print(f"{'False Negatives (FN):':<25} {cm['false_negatives']:,}")
        print(f"{'True Positives (TP):':<25} {cm['true_positives']:,}")
        
        # Class-specific Performance
        print(f"\nCLASS-SPECIFIC PERFORMANCE:")
        for class_name, class_metrics in metrics['class_specific'].items():
            print(f"\n{class_name.upper()} CLASS:")
            print(f"  {'Support:':<20} {class_metrics['support']:,}")
            print(f"  {'Precision:':<20} {class_metrics['precision']:.4f}")
            print(f"  {'Recall:':<20} {class_metrics['recall']:.4f}")
            print(f"  {'F1-Score:':<20} {class_metrics['f1_score']:.4f}")
        
        # Additional Insights
        total = cm['true_negatives'] + cm['false_positives'] + cm['false_negatives'] + cm['true_positives']
        benign_accuracy = cm['true_negatives'] / (cm['true_negatives'] + cm['false_positives']) if (cm['true_negatives'] + cm['false_positives']) > 0 else 0
        malignant_accuracy = cm['true_positives'] / (cm['true_positives'] + cm['false_negatives']) if (cm['true_positives'] + cm['false_negatives']) > 0 else 0
        
        print(f"\nADDITIONAL INSIGHTS:")
        print(f"{'Benign Detection Rate:':<25} {benign_accuracy:.4f} ({benign_accuracy*100:.2f}%)")
        print(f"{'Malignant Detection Rate:':<25} {malignant_accuracy:.4f} ({malignant_accuracy*100:.2f}%)")
        print(f"{'False Positive Rate:':<25} {cm['false_positives']/total:.4f} ({cm['false_positives']/total*100:.2f}%)")
        print(f"{'False Negative Rate:':<25} {cm['false_negatives']/total:.4f} ({cm['false_negatives']/total*100:.2f}%)")
        
    def visualize_results(self, cm, metrics, save_path="evaluation_results"):
        """Create visualizations of the evaluation results."""
        os.makedirs(save_path, exist_ok=True)
        
        # Create a comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted Class')
        axes[0, 0].set_ylabel('True Class')
        
        # Accuracy breakdown
        categories = ['Overall\nAccuracy', 'Benign\nPrecision', 'Benign\nRecall', 
                     'Malignant\nPrecision', 'Malignant\nRecall']
        values = [
            metrics['accuracy'],
            metrics['class_specific']['benign']['precision'],
            metrics['class_specific']['benign']['recall'],
            metrics['class_specific']['malignant']['precision'],
            metrics['class_specific']['malignant']['recall']
        ]
        
        bars = axes[0, 1].bar(categories, values, color=['skyblue', 'lightgreen', 'lightgreen', 'lightcoral', 'lightcoral'])
        axes[0, 1].set_title('Performance Metrics')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Class distribution
        class_counts = [metrics['class_specific']['benign']['support'],
                       metrics['class_specific']['malignant']['support']]
        axes[1, 0].pie(class_counts, labels=self.class_names, autopct='%1.1f%%', 
                      colors=['lightblue', 'lightcoral'], startangle=90)
        axes[1, 0].set_title('Test Dataset Class Distribution')
        
        # Confusion matrix breakdown
        cm_values = [metrics['confusion_matrix']['true_negatives'],
                    metrics['confusion_matrix']['false_positives'],
                    metrics['confusion_matrix']['false_negatives'],
                    metrics['confusion_matrix']['true_positives']]
        cm_labels = ['True\nNegatives', 'False\nPositives', 'False\nNegatives', 'True\nPositives']
        colors = ['lightgreen', 'lightcoral', 'orange', 'lightblue']
        
        bars = axes[1, 1].bar(cm_labels, cm_values, color=colors)
        axes[1, 1].set_title('Confusion Matrix Components')
        axes[1, 1].set_ylabel('Count')
        
        # Add value labels on bars
        for bar, value in zip(bars, cm_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cm_values)*0.01, 
                           f'{value:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'evaluation_report.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_metrics_to_file(self, metrics, save_path="evaluation_results"):
        """Save detailed metrics to JSON file."""
        os.makedirs(save_path, exist_ok=True)
        
        with open(os.path.join(save_path, 'test_evaluation_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nDetailed metrics saved to: {save_path}/test_evaluation_metrics.json")
        
    def run_evaluation(self):
        """Run complete model evaluation."""
        try:
            # Load model
            self.load_model()
            
            # Check if validation metrics are available
            val_metrics = self.load_validation_metrics()
            if val_metrics:
                print("\nValidation metrics from training found:")
                print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"Validation Precision: {val_metrics['precision']:.4f}")
                print(f"Validation Recall: {val_metrics['recall']:.4f}")
            
            # Create test dataset
            test_ds = self.create_test_dataset()
            
            # Evaluate on test data
            y_true, y_pred_probs, y_pred_classes = self.evaluate_on_test_data(test_ds)
            
            # Calculate metrics
            metrics, cm = self.calculate_comprehensive_metrics(y_true, y_pred_probs, y_pred_classes)
            
            # Print report
            self.print_evaluation_report(metrics)
            
            # Create visualizations
            self.visualize_results(cm, metrics)
            
            # Save results
            self.save_metrics_to_file(metrics)
            
            print("\n" + "="*70)
            print("EVALUATION COMPLETE!")
            print("="*70)
            
            return metrics
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return None

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    results = evaluator.run_evaluation()









































