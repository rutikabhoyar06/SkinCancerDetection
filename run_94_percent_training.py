"""
Quick start script to train models for 94%+ accuracy
Run this script to start training immediately
"""

from train_94_percent_simple import train_for_94_percent

if __name__ == "__main__":
    print("="*80)
    print("STARTING TRAINING FOR 94%+ ACCURACY")
    print("="*80)
    print("\nThis will train an ensemble of EfficientNetB4, ResNet50, and DenseNet201")
    print("with optimized settings to achieve 94%+ accuracy.")
    print("\nTraining will take several hours depending on your hardware.")
    print("Progress will be saved, so you can stop and resume if needed.")
    print("\nStarting training...\n")
    
    # Optimized settings for 94% accuracy
    results = train_for_94_percent(
        data_root="dataset_split",
        image_size=(384, 384),  # Larger images for better accuracy
        batch_size=12,  # Adjust based on GPU memory
        epochs_frozen=25,  # More epochs for better learning
        epochs_finetune=75,  # Extended fine-tuning
        lr_frozen=1e-3,
        lr_finetune=5e-5,  # Lower learning rate for fine-tuning
        dropout=0.5,  # Higher dropout for regularization
        model_dir="checkpoints_94",
        use_tta=True,  # Test-time augmentation for better accuracy
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



















