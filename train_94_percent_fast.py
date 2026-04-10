"""
Fast and Memory-Efficient Training for 94%+ Accuracy
Optimized for CPU with limited RAM
"""

from train_94_percent_simple import train_for_94_percent

if __name__ == "__main__":
    print("="*80)
    print("FAST TRAINING FOR 94%+ ACCURACY (MEMORY OPTIMIZED)")
    print("="*80)
    print("\nUsing optimized settings to avoid memory issues:")
    print("  - Smaller batch size (4)")
    print("  - Slightly smaller images (320x320)")
    print("  - EfficientNetB3 instead of B4")
    print("\nStarting training...\n")
    
    # Memory-optimized settings
    results = train_for_94_percent(
        data_root="dataset_split",
        image_size=(320, 320),  # Smaller to save memory
        batch_size=4,  # Much smaller batch size
        epochs_frozen=20,  # Slightly fewer epochs
        epochs_finetune=60,  # Still good fine-tuning
        lr_frozen=1e-3,
        lr_finetune=5e-5,
        dropout=0.5,
        model_dir="checkpoints_94",
        use_tta=True,
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nFinal Ensemble Accuracy: {results['metrics']['ensemble']['accuracy']*100:.2f}%")
    
    if results['metrics']['ensemble']['accuracy'] >= 0.94:
        print("\n✅ SUCCESS! 94%+ accuracy achieved!")
    else:
        print("\n⚠️  Target not reached. Consider:")
        print("   - Training for more epochs")
        print("   - Using larger image size (if memory allows)")
    
    print(f"\nResults saved to: checkpoints_94/ensemble_metrics.json")



















