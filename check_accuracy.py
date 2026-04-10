"""
Quick script to check model accuracy
"""

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score

def check_accuracy():
    """Check accuracy from saved metrics or evaluate model"""
    
    # Check for test metrics
    test_metrics_path = "checkpoints/bm_classifier_test_metrics.json"
    if os.path.exists(test_metrics_path):
        print("📊 Loading saved test metrics...")
        with open(test_metrics_path, "r") as f:
            metrics = json.load(f)
        
        print("\n" + "="*80)
        print("🎯 MODEL ACCURACY RESULTS")
        print("="*80)
        print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"AUC:       {metrics['auc']:.4f}")
        print("="*80)
        
        if metrics['accuracy'] >= 0.90:
            print("\n✅ SUCCESS! Model achieved 90%+ accuracy!")
        else:
            print(f"\n⚠️  Model accuracy is {metrics['accuracy']*100:.2f}%. Target: 90%+")
            print("   Training may still be in progress or needs more epochs.")
        
        return metrics
    
    # Check for validation metrics
    val_metrics_path = "checkpoints/bm_classifier_val_metrics.json"
    if os.path.exists(val_metrics_path):
        print("📊 Loading saved validation metrics...")
        with open(val_metrics_path, "r") as f:
            metrics = json.load(f)
        
        print("\n" + "="*80)
        print("📈 VALIDATION SET RESULTS")
        print("="*80)
        print(f"Accuracy:  {metrics.get('accuracy', 0):.4f} ({metrics.get('accuracy', 0)*100:.2f}%)")
        print("="*80)
        
        return metrics
    
    # Check if model exists and evaluate
    model_path = "checkpoints/bm_classifier_best.keras"
    if os.path.exists(model_path):
        print("🔍 Model found. Evaluating on test set...")
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Load test data
        test_ds = tf.keras.utils.image_dataset_from_directory(
            "dataset_split/test",
            labels="inferred",
            label_mode="int",
            seed=42,
            image_size=(300, 300),
            batch_size=32,
        )
        
        def cast_only(x, y):
            x = tf.cast(x, tf.float32) / 255.0
            return x, y
        
        test_ds = test_ds.map(cast_only, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Evaluate
        y_true = []
        y_pred = []
        y_pred_proba = []
        
        for batch_x, batch_y in test_ds:
            prob = model.predict(batch_x, verbose=0)
            y_true.extend(batch_y.numpy().tolist())
            y_pred.extend(np.argmax(prob, axis=1).tolist())
            y_pred_proba.extend(prob[:, 1].tolist())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        print("\n" + "="*80)
        print("🎯 TEST SET RESULTS")
        print("="*80)
        print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC:       {auc:.4f}")
        print("="*80)
        
        if acc >= 0.90:
            print("\n✅ SUCCESS! Model achieved 90%+ accuracy!")
        else:
            print(f"\n⚠️  Model accuracy is {acc*100:.2f}%. Target: 90%+")
        
        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "auc": float(auc)
        }
    
    else:
        print("❌ No trained model found.")
        print("   Please run: python train_optimized.py")
        return None

if __name__ == "__main__":
    check_accuracy()



























