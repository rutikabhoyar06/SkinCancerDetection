"""
Quick Start Script for Advanced Transfer Learning
Run this script to quickly train and evaluate a model with optimal settings
"""

import os
import sys
from advanced_transfer_learning import AdvancedTransferLearningClassifier
from comprehensive_evaluation import ComprehensiveEvaluator


def quick_start():
    """Quick start with optimal settings for best performance"""
    
    print("🚀 QUICK START: Advanced Transfer Learning for Skin Cancer Detection")
    print("="*80)
    
    # Check if dataset exists
    if not os.path.exists("dataset"):
        print("❌ Error: Dataset directory not found!")
        print("Please ensure you have a 'dataset' directory with 'benign' and 'malignant' subdirectories")
        return
    
    # Check dataset structure
    benign_dir = os.path.join("dataset", "benign")
    malignant_dir = os.path.join("dataset", "malignant")
    
    if not os.path.exists(benign_dir) or not os.path.exists(malignant_dir):
        print("❌ Error: Dataset structure incorrect!")
        print("Please ensure your dataset has the following structure:")
        print("dataset/")
        print("├── benign/")
        print("└── malignant/")
        return
    
    # Count images
    benign_count = len([f for f in os.listdir(benign_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    malignant_count = len([f for f in os.listdir(malignant_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"📊 Dataset Info:")
    print(f"   Benign images: {benign_count}")
    print(f"   Malignant images: {malignant_count}")
    print(f"   Total images: {benign_count + malignant_count}")
    print(f"   Class ratio: {benign_count/malignant_count:.2f}:1")
    
    if benign_count == 0 or malignant_count == 0:
        print("❌ Error: No images found in dataset directories!")
        return
    
    print("\n🎯 Training Configuration:")
    print("   Model: EfficientNet V2 (optimal for this task)")
    print("   Input Size: 224x224 (good balance of speed and accuracy)")
    print("   Batch Size: 32 (optimal for most GPUs)")
    print("   Epochs: 10 frozen + 20 fine-tuning")
    print("   Learning Rates: 1e-3 → 1e-4")
    print("   Features: Class weights, advanced augmentation, AdamW optimizer")
    
    # Create classifier with optimal settings
    print("\n🏗️  Building model...")
    classifier = AdvancedTransferLearningClassifier(
        model_name="efficientnet_v2",
        input_shape=(224, 224, 3),
        dropout_rate=0.3
    )
    
    # Train model
    print("\n🚂 Training model...")
    print("   Phase 1: Training with frozen base (10 epochs)")
    print("   Phase 2: Fine-tuning with unfrozen base (20 epochs)")
    
    try:
        history = classifier.train(
            data_dir="dataset",
            model_dir="quick_start_results",
            batch_size=32,
            epochs_frozen=10,
            epochs_finetune=20,
            lr_frozen=1e-3,
            lr_finetune=1e-4,
            use_class_weights=True,
            use_advanced_augmentation=True
        )
        
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    try:
        metrics, y_true, y_pred, y_pred_proba = classifier.evaluate("dataset")
        
        print("✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        return
    
    # Print results
    print("\n" + "="*60)
    print("🎉 RESULTS SUMMARY")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.1f}%)")
    print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.1f}%)")
    print(f"F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.1f}%)")
    print(f"AUC:       {metrics['auc']:.4f} ({metrics['auc']*100:.1f}%)")
    
    # Performance assessment
    print(f"\n📈 Performance Assessment:")
    if metrics['accuracy'] >= 0.85:
        print("   🎯 Excellent! Accuracy ≥ 85%")
    elif metrics['accuracy'] >= 0.80:
        print("   ✅ Good! Accuracy ≥ 80%")
    elif metrics['accuracy'] >= 0.75:
        print("   ⚠️  Fair. Consider hyperparameter tuning")
    else:
        print("   ❌ Poor. Check dataset and training parameters")
    
    if metrics['auc'] >= 0.90:
        print("   🎯 Excellent! AUC ≥ 0.90")
    elif metrics['auc'] >= 0.85:
        print("   ✅ Good! AUC ≥ 0.85")
    else:
        print("   ⚠️  Consider hyperparameter tuning for better AUC")
    
    # Save results
    print(f"\n💾 Results saved to:")
    print(f"   Model: quick_start_results/efficientnet_v2_best.keras")
    print(f"   Training curves: quick_start_results/training_history.png")
    print(f"   Confusion matrix: quick_start_results/confusion_matrix.png")
    print(f"   ROC curve: quick_start_results/roc_curve.png")
    print(f"   Metrics: quick_start_results/evaluation_metrics.json")
    
    # Next steps
    print(f"\n🚀 Next Steps to Improve Performance:")
    print(f"   1. Run hyperparameter tuning:")
    print(f"      python hyperparameter_tuning.py --search_type random --n_trials 20")
    print(f"   2. Try different model architectures:")
    print(f"      python train_advanced_model.py --model_name xception --input_size 299 299")
    print(f"   3. Use larger input sizes:")
    print(f"      python train_advanced_model.py --input_size 384 384")
    print(f"   4. Run comprehensive evaluation:")
    print(f"      python comprehensive_evaluation.py --model_path quick_start_results/efficientnet_v2_best.keras")
    
    # Advanced features
    print(f"\n🔧 Advanced Features Available:")
    print(f"   - Hyperparameter tuning with grid/random search")
    print(f"   - Model architecture comparison (Xception vs EfficientNet V2)")
    print(f"   - Comprehensive evaluation with clinical metrics")
    print(f"   - Advanced regularization techniques")
    print(f"   - Ensemble methods and test-time augmentation")
    
    print(f"\n📚 Documentation:")
    print(f"   - README: ADVANCED_TRANSFER_LEARNING_README.md")
    print(f"   - Demo: python demo_advanced_features.py")
    print(f"   - Training: python train_advanced_model.py --help")
    
    print("\n" + "="*80)
    print("🎉 QUICK START COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    quick_start()






































