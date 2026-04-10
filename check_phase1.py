"""
Quick script to check Phase 1 (frozen training) progress
"""

import os
import time
from datetime import datetime

def check_phase1_progress():
    """Check if Phase 1 checkpoints are being created"""
    model_dir = "ensemble_checkpoints"
    models = ["EfficientNetB3", "ResNet50", "DenseNet121"]
    
    print("="*80)
    print("PHASE 1 (FROZEN TRAINING) STATUS")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if not os.path.exists(model_dir):
        print(f"[INFO] Directory '{model_dir}' not created yet.")
        print("       Training is initializing...")
        return
    
    all_started = False
    for model_name in models:
        frozen_path = os.path.join(model_dir, f"{model_name}_frozen_best.keras")
        
        if os.path.exists(frozen_path):
            file_size = os.path.getsize(frozen_path) / (1024 * 1024)  # MB
            mod_time = datetime.fromtimestamp(os.path.getmtime(frozen_path))
            age = datetime.now() - mod_time
            
            print(f"{model_name}:")
            print(f"  [OK] Phase 1 checkpoint exists")
            print(f"  Size: {file_size:.2f} MB")
            print(f"  Last updated: {age.total_seconds():.0f} seconds ago")
            print(f"  Time: {mod_time.strftime('%H:%M:%S')}")
            all_started = True
        else:
            print(f"{model_name}:")
            print(f"  [...] Phase 1 in progress (no checkpoint yet)")
            print(f"  (Checkpoints saved after first epoch completes)")
    
    print("\n" + "="*80)
    if all_started:
        print("[OK] Phase 1 is running - checkpoints are being created!")
    else:
        print("[INFO] Phase 1 is running - waiting for first epoch to complete...")
        print("       This can take 10-30 minutes depending on your hardware.")
    print("="*80)

if __name__ == "__main__":
    check_phase1_progress()

























