"""
Fast training script with memory optimizations
Uses smaller batch size, smaller models, and no caching
"""

from train_94_percent_simple import train_for_94_percent

if __name__ == "__main__":
    print("="*80)
    print("FAST & MEMORY-EFFICIENT TRAINING FOR 94%+ ACCURACY")
    print("="*80)
    print("\nOptimizations:")
    print("  - Batch size: 4 (reduced from 12)")
    print("  - Image size: 320x320 (reduced from 384x384)")
    print("  - EfficientNetB3 (instead of B4)")
    print("  - No dataset caching (saves RAM)")
    print("  - Still uses ensemble + TTA for 94%+ accuracy")
    print("\nStarting training...\n")
    
    results = train_for_94_percent(
        data_root="dataset_split",
        image_size=(320, 320),  # Smaller to save memory
        batch_size=4,  # Much smaller batch size
        epochs_frozen=20,  # Slightly fewer but still good
        epochs_finetune=60,  # Good fine-tuning
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
        print("\n⚠️  Current: {results['metrics']['ensemble']['accuracy']*100:.2f}%, Target: 94%")
    
    print(f"\nResults saved to: checkpoints_94/ensemble_metrics.json")

