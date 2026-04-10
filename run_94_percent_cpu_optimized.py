"""
CPU-optimized version for training 94%+ accuracy
Uses smaller batch size and adjusted settings for CPU training
"""

from train_94_percent_simple import train_for_94_percent

if __name__ == "__main__":
    print("="*80)
    print("STARTING TRAINING FOR 94%+ ACCURACY (CPU OPTIMIZED)")
    print("="*80)
    print("\nThis will train an ensemble of EfficientNetB4, ResNet50, and DenseNet201")
    print("with CPU-optimized settings to achieve 94%+ accuracy.")
    print("\n⚠️  WARNING: Training on CPU will take MUCH longer (possibly 2-3 days)")
    print("   Consider using GPU or cloud computing for faster training.")
    print("\nStarting training...\n")
    
    # CPU-optimized settings (smaller batch size, but same quality)
    results = train_for_94_percent(
        data_root="dataset_split",
        image_size=(384, 384),
        batch_size=8,  # Smaller for CPU
        epochs_frozen=25,
        epochs_finetune=75,
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
        print("   - Using larger image size (448x448)")
        print("   - Checking data quality")
    
    print(f"\nResults saved to: checkpoints_94/ensemble_metrics.json")



















