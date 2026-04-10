"""
Training Script for Advanced Transfer Learning Models
Demonstrates usage of the advanced transfer learning implementation
"""

import os
import argparse
from advanced_transfer_learning import AdvancedTransferLearningClassifier
from comprehensive_evaluation import ComprehensiveEvaluator


def main():
    """Main training function with comprehensive configuration"""
    parser = argparse.ArgumentParser(description="Train Advanced Transfer Learning Model for Skin Cancer Detection")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="efficientnet_v2", 
                       choices=["xception", "efficientnet_v2"], 
                       help="Base model architecture")
    parser.add_argument("--input_size", type=int, nargs=2, default=[224, 224], 
                       help="Input image size (height width)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    
    # Training configuration
    parser.add_argument("--data_dir", type=str, default="dataset", help="Directory containing training data")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs_frozen", type=int, default=10, help="Epochs for frozen training")
    parser.add_argument("--epochs_finetune", type=int, default=20, help="Epochs for fine-tuning")
    parser.add_argument("--lr_frozen", type=float, default=1e-3, help="Learning rate for frozen phase")
    parser.add_argument("--lr_finetune", type=float, default=1e-4, help="Learning rate for fine-tuning")
    
    # Advanced options
    parser.add_argument("--use_class_weights", action="store_true", help="Use class weights for imbalance")
    parser.add_argument("--advanced_augmentation", action="store_true", help="Use advanced augmentation")
    parser.add_argument("--optimizer", type=str, default="adamw", 
                       choices=["adamw", "adam", "sgd", "rmsprop"], help="Optimizer type")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for regularization")
    
    # Output configuration
    parser.add_argument("--model_dir", type=str, default="advanced_checkpoints", help="Model save directory")
    parser.add_argument("--evaluate", action="store_true", help="Run comprehensive evaluation after training")
    
    args = parser.parse_args()
    
    print("="*80)
    print("ADVANCED TRANSFER LEARNING FOR SKIN CANCER DETECTION")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Input Size: {args.input_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs (Frozen/Finetune): {args.epochs_frozen}/{args.epochs_finetune}")
    print(f"Learning Rates: {args.lr_frozen}/{args.lr_finetune}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Class Weights: {args.use_class_weights}")
    print(f"Advanced Augmentation: {args.advanced_augmentation}")
    print("="*80)
    
    # Create classifier
    classifier = AdvancedTransferLearningClassifier(
        model_name=args.model_name,
        input_shape=(args.input_size[0], args.input_size[1], 3),
        dropout_rate=args.dropout
    )
    
    # Train model
    print(f"\nTraining {args.model_name} model...")
    history = classifier.train(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        epochs_frozen=args.epochs_frozen,
        epochs_finetune=args.epochs_finetune,
        lr_frozen=args.lr_frozen,
        lr_finetune=args.lr_finetune,
        use_class_weights=args.use_class_weights,
        use_advanced_augmentation=args.advanced_augmentation
    )
    
    # Plot training history
    print("Saving training history plots...")
    classifier.plot_training_history(os.path.join(args.model_dir, "training_history.png"))
    
    # Evaluate model
    print("Evaluating model...")
    metrics, y_true, y_pred, y_pred_proba = classifier.evaluate(args.data_dir)
    
    # Print results
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print("="*60)
    
    # Save plots
    classifier.plot_confusion_matrix(y_true, y_pred, 
                                   os.path.join(args.model_dir, "confusion_matrix.png"))
    classifier.plot_roc_curve(y_true, y_pred_proba, 
                            os.path.join(args.model_dir, "roc_curve.png"))
    
    # Save metrics
    import json
    with open(os.path.join(args.model_dir, "evaluation_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Run comprehensive evaluation if requested
    if args.evaluate:
        print("\nRunning comprehensive evaluation...")
        model_path = os.path.join(args.model_dir, f"{args.model_name}_best.keras")
        
        if os.path.exists(model_path):
            evaluator = ComprehensiveEvaluator(model_path)
            comprehensive_metrics = evaluator.evaluate(
                test_data_dir=args.data_dir,
                output_dir=os.path.join(args.model_dir, "comprehensive_evaluation"),
                input_shape=tuple(args.input_size)
            )
            
            print("\n" + "="*60)
            print("COMPREHENSIVE EVALUATION RESULTS")
            print("="*60)
            
            basic = comprehensive_metrics['basic_metrics']
            clinical = comprehensive_metrics['clinical_metrics']
            
            print(f"Accuracy: {basic['accuracy']:.4f}")
            print(f"Precision (Weighted): {basic['precision_weighted']:.4f}")
            print(f"Recall (Weighted): {basic['recall_weighted']:.4f}")
            print(f"F1-Score (Weighted): {basic['f1_weighted']:.4f}")
            print(f"AUC: {basic['auc']:.4f}")
            print(f"Average Precision: {basic['average_precision']:.4f}")
            print(f"Cohen Kappa: {basic['cohen_kappa']:.4f}")
            print(f"Matthews Correlation Coefficient: {basic['matthews_corrcoef']:.4f}")
            
            print(f"\nClinical Metrics:")
            print(f"Sensitivity: {clinical['sensitivity']:.4f}")
            print(f"Specificity: {clinical['specificity']:.4f}")
            print(f"Positive Predictive Value: {clinical['positive_predictive_value']:.4f}")
            print(f"Negative Predictive Value: {clinical['negative_predictive_value']:.4f}")
            
            print(f"\nPer-Class F1 Scores:")
            for class_name, f1_score in comprehensive_metrics['per_class_metrics']['f1_score'].items():
                print(f"  {class_name}: {f1_score:.4f}")
            
            print("="*60)
        else:
            print(f"Model file not found: {model_path}")
    
    print(f"\nTraining complete! Model and results saved to: {args.model_dir}")
    
    # Print hyperparameter tuning suggestions
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING SUGGESTIONS")
    print("="*60)
    print("To improve accuracy beyond 80%, consider:")
    print("1. Run hyperparameter tuning: python hyperparameter_tuning.py")
    print("2. Try different model architectures (Xception vs EfficientNet V2)")
    print("3. Experiment with input sizes: 224x224, 299x299, 384x384")
    print("4. Adjust learning rates and training epochs")
    print("5. Use advanced regularization techniques")
    print("6. Implement ensemble methods")
    print("7. Apply test-time augmentation")
    print("="*60)


if __name__ == "__main__":
    main()






































