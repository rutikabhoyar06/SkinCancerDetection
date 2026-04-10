"""
Check the progress of 94% accuracy training
"""

import os
import json
import glob
from pathlib import Path

def check_progress():
    """Check training progress"""
    model_dir = "checkpoints_94"
    
    if not os.path.exists(model_dir):
        print("❌ Training not started yet. No checkpoints directory found.")
        return
    
    print("="*80)
    print("TRAINING PROGRESS CHECK")
    print("="*80)
    
    # Check for individual model checkpoints
    models = ["EfficientNetB4", "ResNet50", "DenseNet201"]
    completed = []
    in_progress = []
    
    for model_name in models:
        frozen_path = os.path.join(model_dir, f"{model_name}_frozen_best.keras")
        best_path = os.path.join(model_dir, f"{model_name}_best.keras")
        
        if os.path.exists(best_path):
            completed.append(model_name)
            print(f"✅ {model_name}: Training complete")
        elif os.path.exists(frozen_path):
            in_progress.append(model_name)
            print(f"🔄 {model_name}: Fine-tuning in progress")
        else:
            print(f"⏳ {model_name}: Not started yet")
    
    # Check for ensemble results
    metrics_path = os.path.join(model_dir, "ensemble_metrics.json")
    if os.path.exists(metrics_path):
        print("\n" + "="*80)
        print("FINAL RESULTS FOUND!")
        print("="*80)
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        
        if "ensemble" in metrics:
            acc = metrics["ensemble"]["accuracy"]
            print(f"\n🎯 Final Ensemble Accuracy: {acc*100:.2f}%")
            
            if acc >= 0.94:
                print("✅ SUCCESS! 94%+ accuracy achieved!")
            else:
                print(f"⚠️  Current: {acc*100:.2f}%, Target: 94%")
            
            print("\nIndividual Model Results:")
            if "individual_models" in metrics:
                for name, model_metrics in metrics["individual_models"].items():
                    print(f"  {name}: {model_metrics['accuracy']*100:.2f}%")
        else:
            print("Training still in progress...")
    else:
        print("\n📊 Final ensemble evaluation not yet completed.")
    
    print("\n" + "="*80)
    print(f"Checkpoint directory: {os.path.abspath(model_dir)}")
    print("="*80)

if __name__ == "__main__":
    check_progress()



















