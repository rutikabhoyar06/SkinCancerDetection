#!/usr/bin/env python3
"""
Comprehensive Skin Lesion Analysis Script
Analyzes all images in the dataset, provides classifications with confidence scores,
generates confusion matrix, and analyzes misclassifications.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import warnings
warnings.filterwarnings('ignore')

from classifier import build_classifier, compile_classifier
from train_classifier import train_classifier

class SkinLesionAnalyzer:
    def __init__(self, data_root="dataset", model_path="checkpoints/bm_classifier_best.keras"):
        self.data_root = data_root
        self.model_path = model_path
        self.model = None
        self.class_names = ['benign', 'malignant']
        self.image_size = (224, 224)
        
    def ensure_model_exists(self):
        """Train model if it doesn't exist, otherwise load it."""
        if not os.path.exists(self.model_path):
            print("No trained model found. Training a new model...")
            print("This may take some time depending on your hardware.")
            
            # Train the model
            train_classifier(
                data_root=self.data_root,
                image_size=self.image_size,
                batch_size=16,  # Smaller batch size for stability
                epochs_frozen=3,
                epochs_finetune=5,
                lr_base=1e-3,
                lr_finetune=1e-4,
                dropout=0.3,
                model_dir="checkpoints"
            )
        
        # Load the trained model
        print(f"Loading model from {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)
        print("Model loaded successfully!")
        
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image."""
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize(self.image_size)
            image_array = np.asarray(image, dtype=np.float32)
            return image_array
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def get_all_image_paths(self):
        """Get all image paths with their true labels."""
        image_paths = []
        true_labels = []
        
        # Benign images (label 0)
        benign_dir = os.path.join(self.data_root, "benign")
        if os.path.exists(benign_dir):
            for filename in os.listdir(benign_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(benign_dir, filename))
                    true_labels.append(0)
        
        # Malignant images (label 1)
        malignant_dir = os.path.join(self.data_root, "malignant")
        if os.path.exists(malignant_dir):
            for filename in os.listdir(malignant_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(malignant_dir, filename))
                    true_labels.append(1)
        
        return image_paths, true_labels
    
    def classify_all_images(self, batch_size=32, max_images=None):
        """Classify all images and return predictions with confidence scores."""
        image_paths, true_labels = self.get_all_image_paths()
        
        if max_images:
            # For testing purposes, limit the number of images
            indices = np.random.choice(len(image_paths), min(max_images, len(image_paths)), replace=False)
            image_paths = [image_paths[i] for i in indices]
            true_labels = [true_labels[i] for i in indices]
        
        print(f"Classifying {len(image_paths)} images...")
        
        results = []
        
        # Process images in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_labels = true_labels[i:i+batch_size]
            
            # Load and preprocess batch
            batch_images = []
            valid_indices = []
            
            for j, path in enumerate(batch_paths):
                img = self.load_and_preprocess_image(path)
                if img is not None:
                    batch_images.append(img)
                    valid_indices.append(j)
            
            if not batch_images:
                continue
                
            # Convert to numpy array and predict
            batch_array = np.array(batch_images)
            predictions = self.model.predict(batch_array, verbose=0)
            
            # Store results
            for j, pred_idx in enumerate(valid_indices):
                path = batch_paths[pred_idx]
                true_label = batch_labels[pred_idx]
                pred_probs = predictions[j]
                
                benign_prob = float(pred_probs[0])
                malignant_prob = float(pred_probs[1])
                predicted_class = 1 if malignant_prob >= 0.5 else 0
                
                results.append({
                    'image_path': path,
                    'true_label': true_label,
                    'true_class': self.class_names[true_label],
                    'predicted_label': predicted_class,
                    'predicted_class': self.class_names[predicted_class],
                    'benign_prob': benign_prob,
                    'malignant_prob': malignant_prob,
                    'confidence': max(benign_prob, malignant_prob),
                    'correct': true_label == predicted_class
                })
            
            if (i + batch_size) % (batch_size * 10) == 0:
                print(f"Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images")
        
        return pd.DataFrame(results)
    
    def generate_confusion_matrix(self, results_df, save_path="analysis_results"):
        """Generate and save confusion matrix."""
        os.makedirs(save_path, exist_ok=True)
        
        y_true = results_df['true_label'].values
        y_pred = results_df['predicted_label'].values
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix - Skin Lesion Classification')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def analyze_performance(self, results_df, save_path="analysis_results"):
        """Comprehensive performance analysis."""
        os.makedirs(save_path, exist_ok=True)
        
        y_true = results_df['true_label'].values
        y_pred = results_df['predicted_label'].values
        y_prob = results_df['malignant_prob'].values
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        
        # ROC AUC
        roc_auc = roc_auc_score(y_true, y_prob)
        
        # Average Precision
        avg_precision = average_precision_score(y_true, y_prob)
        
        # Print summary
        print("\n" + "="*60)
        print("CLASSIFICATION PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Total Images Analyzed: {len(results_df)}")
        print(f"Benign Images: {sum(y_true == 0)} ({sum(y_true == 0)/len(y_true)*100:.1f}%)")
        print(f"Malignant Images: {sum(y_true == 1)} ({sum(y_true == 1)/len(y_true)*100:.1f}%)")
        print(f"\nOverall Accuracy: {report['accuracy']:.3f}")
        print(f"ROC AUC Score: {roc_auc:.3f}")
        print(f"Average Precision: {avg_precision:.3f}")
        
        print(f"\nBenign Classification:")
        print(f"  Precision: {report['benign']['precision']:.3f}")
        print(f"  Recall: {report['benign']['recall']:.3f}")
        print(f"  F1-Score: {report['benign']['f1-score']:.3f}")
        
        print(f"\nMalignant Classification:")
        print(f"  Precision: {report['malignant']['precision']:.3f}")
        print(f"  Recall: {report['malignant']['recall']:.3f}")
        print(f"  F1-Score: {report['malignant']['f1-score']:.3f}")
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'performance_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return report, roc_auc, avg_precision
    
    def analyze_misclassifications(self, results_df, save_path="analysis_results"):
        """Analyze misclassified images and provide insights."""
        os.makedirs(save_path, exist_ok=True)
        
        # Separate correct and incorrect predictions
        correct_preds = results_df[results_df['correct'] == True]
        incorrect_preds = results_df[results_df['correct'] == False]
        
        print("\n" + "="*60)
        print("MISCLASSIFICATION ANALYSIS")
        print("="*60)
        
        # False Positives (Benign classified as Malignant)
        false_positives = incorrect_preds[incorrect_preds['true_label'] == 0]
        print(f"\nFalse Positives (Benign → Malignant): {len(false_positives)}")
        if len(false_positives) > 0:
            print(f"  Average confidence: {false_positives['malignant_prob'].mean():.3f}")
            print(f"  Confidence range: {false_positives['malignant_prob'].min():.3f} - {false_positives['malignant_prob'].max():.3f}")
        
        # False Negatives (Malignant classified as Benign)
        false_negatives = incorrect_preds[incorrect_preds['true_label'] == 1]
        print(f"\nFalse Negatives (Malignant → Benign): {len(false_negatives)}")
        if len(false_negatives) > 0:
            print(f"  Average confidence: {false_negatives['benign_prob'].mean():.3f}")
            print(f"  Confidence range: {false_negatives['benign_prob'].min():.3f} - {false_negatives['benign_prob'].max():.3f}")
        
        # Confidence distribution analysis
        plt.figure(figsize=(15, 10))
        
        # Confidence distributions
        plt.subplot(2, 3, 1)
        plt.hist(correct_preds['confidence'], bins=30, alpha=0.7, label='Correct', color='green')
        plt.hist(incorrect_preds['confidence'], bins=30, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.title('Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Benign probability distributions
        plt.subplot(2, 3, 2)
        benign_correct = results_df[(results_df['true_label'] == 0) & (results_df['correct'] == True)]
        benign_incorrect = results_df[(results_df['true_label'] == 0) & (results_df['correct'] == False)]
        plt.hist(benign_correct['benign_prob'], bins=30, alpha=0.7, label='Correct Benign', color='lightblue')
        plt.hist(benign_incorrect['benign_prob'], bins=30, alpha=0.7, label='Misclassified Benign', color='red')
        plt.xlabel('Benign Probability')
        plt.ylabel('Count')
        plt.title('Benign Images - Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Malignant probability distributions
        plt.subplot(2, 3, 3)
        malignant_correct = results_df[(results_df['true_label'] == 1) & (results_df['correct'] == True)]
        malignant_incorrect = results_df[(results_df['true_label'] == 1) & (results_df['correct'] == False)]
        plt.hist(malignant_correct['malignant_prob'], bins=30, alpha=0.7, label='Correct Malignant', color='orange')
        plt.hist(malignant_incorrect['malignant_prob'], bins=30, alpha=0.7, label='Misclassified Malignant', color='red')
        plt.xlabel('Malignant Probability')
        plt.ylabel('Count')
        plt.title('Malignant Images - Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Threshold analysis
        thresholds = np.arange(0.1, 0.9, 0.05)
        accuracies = []
        benign_recalls = []
        malignant_recalls = []
        
        for threshold in thresholds:
            pred_with_threshold = (results_df['malignant_prob'] >= threshold).astype(int)
            accuracy = (results_df['true_label'] == pred_with_threshold).mean()
            
            # Benign recall (sensitivity for benign class)
            benign_true = results_df['true_label'] == 0
            benign_pred = pred_with_threshold == 0
            benign_recall = (benign_true & benign_pred).sum() / benign_true.sum()
            
            # Malignant recall (sensitivity for malignant class)
            malignant_true = results_df['true_label'] == 1
            malignant_pred = pred_with_threshold == 1
            malignant_recall = (malignant_true & malignant_pred).sum() / malignant_true.sum()
            
            accuracies.append(accuracy)
            benign_recalls.append(benign_recall)
            malignant_recalls.append(malignant_recall)
        
        plt.subplot(2, 3, 4)
        plt.plot(thresholds, accuracies, label='Overall Accuracy', marker='o')
        plt.plot(thresholds, benign_recalls, label='Benign Recall', marker='s')
        plt.plot(thresholds, malignant_recalls, label='Malignant Recall', marker='^')
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Current Threshold')
        plt.xlabel('Decision Threshold')
        plt.ylabel('Score')
        plt.title('Performance vs Decision Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Find optimal threshold for balanced accuracy
        balanced_scores = [(b + m) / 2 for b, m in zip(benign_recalls, malignant_recalls)]
        optimal_idx = np.argmax(balanced_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        plt.subplot(2, 3, 5)
        plt.plot(thresholds, balanced_scores, label='Balanced Accuracy', marker='o', color='purple')
        plt.axvline(x=optimal_threshold, color='green', linestyle='--', alpha=0.7, 
                   label=f'Optimal Threshold: {optimal_threshold:.2f}')
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Current Threshold')
        plt.xlabel('Decision Threshold')
        plt.ylabel('Balanced Accuracy')
        plt.title('Optimal Threshold Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Class imbalance visualization
        plt.subplot(2, 3, 6)
        class_counts = results_df['true_class'].value_counts()
        plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Dataset Class Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'misclassification_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return optimal_threshold, false_positives, false_negatives
    
    def generate_recommendations(self, results_df, optimal_threshold, false_positives, false_negatives):
        """Generate recommendations for improving model performance."""
        print("\n" + "="*60)
        print("RECOMMENDATIONS FOR IMPROVEMENT")
        print("="*60)
        
        total_benign = sum(results_df['true_label'] == 0)
        total_malignant = sum(results_df['true_label'] == 1)
        
        print(f"\n1. DATASET IMBALANCE:")
        print(f"   - Current ratio: {total_benign:,} benign : {total_malignant:,} malignant")
        print(f"   - Imbalance ratio: {total_benign/total_malignant:.1f}:1")
        print(f"   - Recommendation: Consider data augmentation for malignant class")
        print(f"     or undersampling benign class to achieve better balance")
        
        print(f"\n2. DECISION THRESHOLD:")
        print(f"   - Current threshold: 0.50")
        print(f"   - Optimal threshold for balanced accuracy: {optimal_threshold:.3f}")
        if optimal_threshold != 0.5:
            print(f"   - Recommendation: Adjust threshold to {optimal_threshold:.3f}")
            print(f"     This should improve benign detection while maintaining")
            print(f"     reasonable malignant detection performance")
        
        print(f"\n3. MISCLASSIFICATION PATTERNS:")
        fp_rate = len(false_positives) / total_benign * 100
        fn_rate = len(false_negatives) / total_malignant * 100
        
        print(f"   - False Positive Rate: {fp_rate:.1f}% ({len(false_positives)}/{total_benign})")
        print(f"   - False Negative Rate: {fn_rate:.1f}% ({len(false_negatives)}/{total_malignant})")
        
        if fp_rate > fn_rate:
            print(f"   - Primary issue: Too many benign lesions classified as malignant")
            print(f"   - Recommendation: Increase decision threshold or retrain with")
            print(f"     more emphasis on benign class specificity")
        elif fn_rate > fp_rate:
            print(f"   - Primary issue: Missing malignant lesions (high risk!)")
            print(f"   - Recommendation: Decrease decision threshold or retrain with")
            print(f"     higher recall for malignant class")
        
        print(f"\n4. MODEL RETRAINING SUGGESTIONS:")
        print(f"   - Use class weights to handle imbalance:")
        print(f"     benign_weight = {total_malignant/(total_benign + total_malignant) * 2:.3f}")
        print(f"     malignant_weight = {total_benign/(total_benign + total_malignant) * 2:.3f}")
        print(f"   - Consider focal loss to focus on hard examples")
        print(f"   - Implement stratified cross-validation")
        print(f"   - Add more data augmentation for minority class")
        
        print(f"\n5. CLINICAL CONSIDERATIONS:")
        print(f"   - In medical diagnosis, false negatives (missing cancer) are")
        print(f"     typically more dangerous than false positives")
        print(f"   - Consider using a lower threshold (e.g., 0.3-0.4) to catch")
        print(f"     more potential malignant cases for further examination")
        print(f"   - Implement confidence-based referral system:")
        print(f"     - High confidence (>0.8): Direct classification")
        print(f"     - Medium confidence (0.5-0.8): Specialist review")
        print(f"     - Low confidence (<0.5): Additional testing required")
    
    def save_detailed_results(self, results_df, save_path="analysis_results"):
        """Save detailed results to CSV files."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save all results
        results_df.to_csv(os.path.join(save_path, 'all_predictions.csv'), index=False)
        
        # Save misclassified cases
        misclassified = results_df[results_df['correct'] == False]
        misclassified.to_csv(os.path.join(save_path, 'misclassified_cases.csv'), index=False)
        
        # Save low confidence predictions
        low_confidence = results_df[results_df['confidence'] < 0.7]
        low_confidence.to_csv(os.path.join(save_path, 'low_confidence_predictions.csv'), index=False)
        
        print(f"\nDetailed results saved to '{save_path}' directory:")
        print(f"  - all_predictions.csv: Complete results for all images")
        print(f"  - misclassified_cases.csv: Only incorrectly classified images")
        print(f"  - low_confidence_predictions.csv: Predictions with confidence < 0.7")
    
    def run_complete_analysis(self, max_images=None):
        """Run the complete analysis pipeline."""
        print("Starting Comprehensive Skin Lesion Analysis...")
        print("="*60)
        
        # Ensure model exists and load it
        self.ensure_model_exists()
        
        # Classify all images
        print("\nClassifying all images...")
        results_df = self.classify_all_images(max_images=max_images)
        
        # Generate confusion matrix
        print("\nGenerating confusion matrix...")
        cm = self.generate_confusion_matrix(results_df)
        
        # Analyze performance
        print("\nAnalyzing performance...")
        report, roc_auc, avg_precision = self.analyze_performance(results_df)
        
        # Analyze misclassifications
        print("\nAnalyzing misclassifications...")
        optimal_threshold, false_positives, false_negatives = self.analyze_misclassifications(results_df)
        
        # Generate recommendations
        self.generate_recommendations(results_df, optimal_threshold, false_positives, false_negatives)
        
        # Save detailed results
        self.save_detailed_results(results_df)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("Check the 'analysis_results' directory for detailed outputs.")
        
        return results_df


if __name__ == "__main__":
    # Initialize analyzer
    analyzer = SkinLesionAnalyzer()
    
    # Run complete analysis
    # For testing, you can limit the number of images with max_images parameter
    results = analyzer.run_complete_analysis(max_images=2000)  # Test with 2000 images first
    # results = analyzer.run_complete_analysis()  # Analyze all images
