"""
Demonstration Script for Advanced Transfer Learning Features
Shows how to use all the advanced features for skin cancer detection
"""

import os
import argparse
from advanced_transfer_learning import AdvancedTransferLearningClassifier, hyperparameter_tuning_suggestions, advanced_regularization_techniques
from comprehensive_evaluation import ComprehensiveEvaluator


def demo_basic_training():
    """Demonstrate basic training with default parameters"""
    print("="*60)
    print("DEMO 1: Basic Training with EfficientNet V2")
    print("="*60)
    
    # Create classifier
    classifier = AdvancedTransferLearningClassifier(
        model_name="efficientnet_v2",
        input_shape=(224, 224, 3),
        dropout_rate=0.3
    )
    
    # Train with basic parameters
    print("Training model with basic parameters...")
    history = classifier.train(
        data_dir="dataset",
        model_dir="demo_basic",
        batch_size=32,
        epochs_frozen=5,
        epochs_finetune=10,
        lr_frozen=1e-3,
        lr_finetune=1e-4,
        use_class_weights=True,
        use_advanced_augmentation=False
    )
    
    # Evaluate
    metrics, y_true, y_pred, y_pred_proba = classifier.evaluate("dataset")
    
    print(f"Basic Training Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    
    return classifier, metrics


def demo_advanced_training():
    """Demonstrate advanced training with all features enabled"""
    print("\n" + "="*60)
    print("DEMO 2: Advanced Training with All Features")
    print("="*60)
    
    # Create classifier with advanced settings
    classifier = AdvancedTransferLearningClassifier(
        model_name="xception",
        input_shape=(299, 299, 3),  # Xception optimal size
        dropout_rate=0.4
    )
    
    # Train with advanced parameters
    print("Training model with advanced parameters...")
    history = classifier.train(
        data_dir="dataset",
        model_dir="demo_advanced",
        batch_size=16,  # Smaller batch for larger input
        epochs_frozen=8,
        epochs_finetune=15,
        lr_frozen=5e-4,
        lr_finetune=1e-5,
        use_class_weights=True,
        use_advanced_augmentation=True
    )
    
    # Evaluate
    metrics, y_true, y_pred, y_pred_proba = classifier.evaluate("dataset")
    
    print(f"Advanced Training Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    
    return classifier, metrics


def demo_comprehensive_evaluation():
    """Demonstrate comprehensive evaluation"""
    print("\n" + "="*60)
    print("DEMO 3: Comprehensive Evaluation")
    print("="*60)
    
    # Check if we have a trained model
    model_path = "demo_advanced/xception_best.keras"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please run demo_advanced_training() first")
        return
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(model_path)
    
    # Run comprehensive evaluation
    print("Running comprehensive evaluation...")
    metrics = evaluator.evaluate(
        test_data_dir="dataset",
        output_dir="demo_comprehensive_eval",
        input_shape=(299, 299)
    )
    
    # Print detailed results
    basic = metrics['basic_metrics']
    clinical = metrics['clinical_metrics']
    
    print(f"Comprehensive Evaluation Results:")
    print(f"  Accuracy: {basic['accuracy']:.4f}")
    print(f"  Precision: {basic['precision_weighted']:.4f}")
    print(f"  Recall: {basic['recall_weighted']:.4f}")
    print(f"  F1-Score: {basic['f1_weighted']:.4f}")
    print(f"  AUC: {basic['auc']:.4f}")
    print(f"  Cohen Kappa: {basic['cohen_kappa']:.4f}")
    print(f"  Matthews CC: {basic['matthews_corrcoef']:.4f}")
    
    print(f"\nClinical Metrics:")
    print(f"  Sensitivity: {clinical['sensitivity']:.4f}")
    print(f"  Specificity: {clinical['specificity']:.4f}")
    print(f"  PPV: {clinical['positive_predictive_value']:.4f}")
    print(f"  NPV: {clinical['negative_predictive_value']:.4f}")
    
    return metrics


def demo_hyperparameter_suggestions():
    """Demonstrate hyperparameter tuning suggestions"""
    print("\n" + "="*60)
    print("DEMO 4: Hyperparameter Tuning Suggestions")
    print("="*60)
    
    # Get suggestions
    suggestions = hyperparameter_tuning_suggestions()
    
    print("Hyperparameter Tuning Suggestions:")
    for category, params in suggestions.items():
        print(f"\n{category.upper()}:")
        for param, values in params.items():
            print(f"  {param}: {values}")
    
    # Get regularization techniques
    techniques = advanced_regularization_techniques()
    
    print(f"\nAdvanced Regularization Techniques:")
    for category, methods in techniques.items():
        print(f"\n{category.upper()}:")
        for method, description in methods.items():
            print(f"  {method}: {description}")


def demo_model_comparison():
    """Demonstrate comparing different models"""
    print("\n" + "="*60)
    print("DEMO 5: Model Architecture Comparison")
    print("="*60)
    
    models_to_compare = [
        {"name": "efficientnet_v2", "input_size": (224, 224, 3)},
        {"name": "xception", "input_size": (299, 299, 3)}
    ]
    
    results = {}
    
    for model_config in models_to_compare:
        print(f"\nTraining {model_config['name']}...")
        
        classifier = AdvancedTransferLearningClassifier(
            model_name=model_config["name"],
            input_shape=model_config["input_size"],
            dropout_rate=0.3
        )
        
        # Quick training for demo
        history = classifier.train(
            data_dir="dataset",
            model_dir=f"demo_comparison_{model_config['name']}",
            batch_size=32,
            epochs_frozen=3,
            epochs_finetune=5,
            lr_frozen=1e-3,
            lr_finetune=1e-4,
            use_class_weights=True,
            use_advanced_augmentation=True
        )
        
        # Evaluate
        metrics, _, _, _ = classifier.evaluate("dataset")
        results[model_config["name"]] = metrics
        
        print(f"  {model_config['name']} Results:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    AUC: {metrics['auc']:.4f}")
    
    # Compare results
    print(f"\nModel Comparison Summary:")
    print(f"{'Model':<20} {'Accuracy':<10} {'AUC':<10}")
    print("-" * 40)
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['accuracy']:<10.4f} {metrics['auc']:<10.4f}")
    
    return results


def demo_advanced_regularization():
    """Demonstrate advanced regularization techniques"""
    print("\n" + "="*60)
    print("DEMO 6: Advanced Regularization Techniques")
    print("="*60)
    
    # Create classifier with high regularization
    classifier = AdvancedTransferLearningClassifier(
        model_name="efficientnet_v2",
        input_shape=(224, 224, 3),
        dropout_rate=0.5  # High dropout
    )
    
    # Build model with advanced head
    model = classifier.build_model(
        base_trainable=False,
        use_advanced_head=True
    )
    
    # Compile with weight decay
    classifier.compile_model(
        optimizer="adamw",
        learning_rate=1e-3,
        weight_decay=1e-3,  # High weight decay
        use_cosine_decay=True
    )
    
    print("Model compiled with advanced regularization:")
    print("  - High dropout (0.5)")
    print("  - Weight decay (1e-3)")
    print("  - Cosine learning rate decay")
    print("  - Advanced classification head")
    
    # Train with regularization
    history = classifier.train(
        data_dir="dataset",
        model_dir="demo_regularization",
        batch_size=32,
        epochs_frozen=5,
        epochs_finetune=10,
        lr_frozen=1e-3,
        lr_finetune=1e-4,
        use_class_weights=True,
        use_advanced_augmentation=True
    )
    
    # Evaluate
    metrics, _, _, _ = classifier.evaluate("dataset")
    
    print(f"Regularized Model Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    
    return metrics


def main():
    """Run all demonstrations"""
    parser = argparse.ArgumentParser(description="Demonstrate Advanced Transfer Learning Features")
    parser.add_argument("--demo", type=str, default="all", 
                       choices=["all", "basic", "advanced", "evaluation", "suggestions", "comparison", "regularization"],
                       help="Which demo to run")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Dataset directory")
    
    args = parser.parse_args()
    
    print("ADVANCED TRANSFER LEARNING DEMONSTRATION")
    print("="*80)
    print("This demo shows various features of the advanced transfer learning implementation")
    print("="*80)
    
    if args.demo in ["all", "basic"]:
        demo_basic_training()
    
    if args.demo in ["all", "advanced"]:
        demo_advanced_training()
    
    if args.demo in ["all", "evaluation"]:
        demo_comprehensive_evaluation()
    
    if args.demo in ["all", "suggestions"]:
        demo_hyperparameter_suggestions()
    
    if args.demo in ["all", "comparison"]:
        demo_model_comparison()
    
    if args.demo in ["all", "regularization"]:
        demo_advanced_regularization()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("Check the generated directories for results:")
    print("  - demo_basic/: Basic training results")
    print("  - demo_advanced/: Advanced training results")
    print("  - demo_comprehensive_eval/: Comprehensive evaluation")
    print("  - demo_comparison_*/: Model comparison results")
    print("  - demo_regularization/: Regularization demo results")
    print("="*80)


if __name__ == "__main__":
    main()






































