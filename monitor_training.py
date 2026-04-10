"""
Training Progress Monitor for Ensemble Learning
Checks training status and displays progress
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
import glob


def check_training_status(model_dir="ensemble_checkpoints"):
    """Check the current status of ensemble training"""
    
    print("="*80)
    print("ENSEMBLE TRAINING MONITOR")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if directory exists
    if not os.path.exists(model_dir):
        print(f"[ERROR] Model directory '{model_dir}' not found.")
        print("   Training may not have started yet.")
        return
    
    # Model names to check
    models = ["EfficientNetB3", "ResNet50", "DenseNet121"]
    
    print("Training Progress:")
    print("-" * 80)
    
    training_status = {
        "EfficientNetB3": {"frozen": False, "finetuned": False, "complete": False},
        "ResNet50": {"frozen": False, "finetuned": False, "complete": False},
        "DenseNet121": {"frozen": False, "finetuned": False, "complete": False},
    }
    
    for model_name in models:
        frozen_path = os.path.join(model_dir, f"{model_name}_frozen_best.keras")
        finetuned_path = os.path.join(model_dir, f"{model_name}_best.keras")
        final_path = os.path.join(model_dir, "ensemble", f"{model_name}_final.keras")
        
        status = []
        
        if os.path.exists(frozen_path):
            training_status[model_name]["frozen"] = True
            status.append("[OK] Frozen phase complete")
        else:
            status.append("[...] Frozen phase in progress...")
        
        if os.path.exists(finetuned_path):
            training_status[model_name]["finetuned"] = True
            status.append("[OK] Fine-tuning complete")
        else:
            status.append("[...] Fine-tuning pending...")
        
        if os.path.exists(final_path):
            training_status[model_name]["complete"] = True
            status.append("[OK] Final model saved")
        
        print(f"\n{model_name}:")
        for s in status:
            print(f"  {s}")
        
        # Check file sizes to see if training is active
        if os.path.exists(frozen_path):
            file_size = os.path.getsize(frozen_path) / (1024 * 1024)  # MB
            mod_time = datetime.fromtimestamp(os.path.getmtime(frozen_path))
            print(f"  [FILE] Frozen model: {file_size:.2f} MB (modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
        
        if os.path.exists(finetuned_path):
            file_size = os.path.getsize(finetuned_path) / (1024 * 1024)  # MB
            mod_time = datetime.fromtimestamp(os.path.getmtime(finetuned_path))
            print(f"  [FILE] Fine-tuned model: {file_size:.2f} MB (modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
    
    # Check for ensemble results
    print("\n" + "-" * 80)
    print("Ensemble Status:")
    print("-" * 80)
    
    metrics_path = os.path.join(model_dir, "ensemble_metrics.json")
    if os.path.exists(metrics_path):
        print("[OK] Ensemble evaluation complete!")
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            
            if "ensemble" in metrics:
                ensemble_metrics = metrics["ensemble"]
                print(f"\nFinal Ensemble Results:")
                print(f"   Accuracy:  {ensemble_metrics.get('accuracy', 0):.4f} ({ensemble_metrics.get('accuracy', 0)*100:.2f}%)")
                print(f"   F1-Score:  {ensemble_metrics.get('f1_score', 0):.4f}")
                print(f"   F1-Macro:  {ensemble_metrics.get('f1_macro', 0):.4f}")
                print(f"   IoU:       {ensemble_metrics.get('iou', 0):.4f}")
                print(f"   Precision: {ensemble_metrics.get('precision', 0):.4f}")
                print(f"   Recall:    {ensemble_metrics.get('recall', 0):.4f}")
                print(f"   AUC:       {ensemble_metrics.get('auc', 0):.4f}")
                
                if ensemble_metrics.get('accuracy', 0) >= 0.90:
                    print("\n[SUCCESS] Model achieved 90%+ accuracy!")
                else:
                    print(f"\n[WARNING] Accuracy is {ensemble_metrics.get('accuracy', 0)*100:.2f}%. Target: 90%+")
            
            if "individual_models" in metrics:
                print(f"\nIndividual Model Results:")
                for model_name, model_metrics in metrics["individual_models"].items():
                    acc = model_metrics.get('accuracy', 0)
                    print(f"   {model_name}: {acc:.4f} ({acc*100:.2f}%)")
        
        except Exception as e:
            print(f"   [ERROR] Could not read metrics: {e}")
    else:
        print("[...] Ensemble evaluation pending...")
        print("   (Will be created after all models are trained)")
    
    # Check for visualization files
    print("\n" + "-" * 80)
    print("Generated Files:")
    print("-" * 80)
    
    files_to_check = [
        ("ensemble_comparison.png", "Model comparison chart"),
        ("ensemble_confusion_matrix.png", "Confusion matrix"),
        ("ensemble_metrics.json", "Complete metrics JSON"),
    ]
    
    for filename, description in files_to_check:
        filepath = os.path.join(model_dir, filename)
        if os.path.exists(filepath):
            print(f"  [OK] {description}: {filename}")
        else:
            print(f"  [...] {description}: Not yet generated")
    
    # Overall progress
    print("\n" + "=" * 80)
    print("Overall Progress:")
    print("=" * 80)
    
    total_phases = 0
    completed_phases = 0
    
    for model_name, status in training_status.items():
        if status["frozen"]:
            completed_phases += 1
        total_phases += 1
        
        if status["finetuned"]:
            completed_phases += 1
        total_phases += 1
    
    if total_phases > 0:
        progress_pct = (completed_phases / total_phases) * 100
        print(f"Training Progress: {completed_phases}/{total_phases} phases complete ({progress_pct:.1f}%)")
        
        # Estimate remaining time
        if completed_phases > 0 and completed_phases < total_phases:
            # Rough estimate: each phase takes ~1-2 hours
            remaining_phases = total_phases - completed_phases
            estimated_hours = remaining_phases * 1.5
            print(f"Estimated time remaining: ~{estimated_hours:.1f} hours")
    
    if os.path.exists(metrics_path):
        print("\n[OK] Training Complete!")
    else:
        print("\n[...] Training in progress...")
        print("   Run this script again to check updated progress.")
    
    print("=" * 80)


def check_python_processes():
    """Check if Python training processes are running"""
    import subprocess
    
    try:
        result = subprocess.run(
            ["powershell", "-Command", "Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, StartTime"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.stdout.strip():
            print("\nRunning Python Processes:")
            print("-" * 80)
            print(result.stdout)
        else:
            print("\n[INFO] No Python processes found.")
            print("   Training may have completed or stopped.")
    except Exception as e:
        print(f"\n[WARNING] Could not check processes: {e}")


def main():
    """Main monitoring function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor ensemble training progress")
    parser.add_argument("--model_dir", type=str, default="ensemble_checkpoints",
                       help="Model checkpoint directory")
    parser.add_argument("--watch", action="store_true",
                       help="Watch mode: continuously monitor (updates every 60s)")
    parser.add_argument("--interval", type=int, default=60,
                       help="Update interval in seconds (for watch mode)")
    
    args = parser.parse_args()
    
    if args.watch:
        print("Watch mode: Monitoring training progress...")
        print("   Press Ctrl+C to stop\n")
        
        try:
            while True:
                os.system("cls" if os.name == "nt" else "clear")  # Clear screen
                check_training_status(args.model_dir)
                check_python_processes()
                print(f"\nNext update in {args.interval} seconds...")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
    else:
        check_training_status(args.model_dir)
        check_python_processes()


if __name__ == "__main__":
    main()
