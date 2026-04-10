"""
Main script to run ensemble training with optional GA feature selection
"""

import os
import argparse
from train_ensemble_tf import train_ensemble
from ga_feature_selection_tf import apply_feature_selection
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(
        description="Train ensemble model with optional GA feature selection"
    )
    
    # Data arguments
    parser.add_argument("--data_root", type=str, default="dataset_split",
                       help="Root directory with train/val/test folders")
    parser.add_argument("--image_size", type=int, nargs=2, default=[300, 300],
                       help="Image size (height width)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs_frozen", type=int, default=15,
                       help="Epochs for frozen base training")
    parser.add_argument("--epochs_finetune", type=int, default=50,
                       help="Epochs for fine-tuning")
    parser.add_argument("--lr_frozen", type=float, default=1e-3,
                       help="Learning rate for frozen phase")
    parser.add_argument("--lr_finetune", type=float, default=1e-4,
                       help="Learning rate for fine-tuning")
    parser.add_argument("--dropout", type=float, default=0.4)
    
    # Model arguments
    parser.add_argument("--model_dir", type=str, default="ensemble_checkpoints")
    parser.add_argument("--no_efficientnet", action="store_true",
                       help="Skip EfficientNet")
    parser.add_argument("--no_resnet", action="store_true",
                       help="Skip ResNet")
    parser.add_argument("--no_densenet", action="store_true",
                       help="Skip DenseNet")
    
    # GA feature selection arguments
    parser.add_argument("--use_ga", action="store_true",
                       help="Use genetic algorithm for feature selection")
    parser.add_argument("--ga_population", type=int, default=60,
                       help="GA population size")
    parser.add_argument("--ga_generations", type=int, default=40,
                       help="GA generations")
    parser.add_argument("--ga_output_dir", type=str, default="ga_results")
    
    args = parser.parse_args()
    
    print("="*80)
    print("ENSEMBLE TRAINING FOR SKIN CANCER DETECTION")
    print("="*80)
    print(f"Data: {args.data_root}")
    print(f"Image Size: {args.image_size}")
    print(f"Models: EfficientNet={not args.no_efficientnet}, "
          f"ResNet={not args.no_resnet}, DenseNet={not args.no_densenet}")
    print(f"GA Feature Selection: {args.use_ga}")
    print("="*80)
    
    # Train ensemble
    results = train_ensemble(
        data_root=args.data_root,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        epochs_frozen=args.epochs_frozen,
        epochs_finetune=args.epochs_finetune,
        lr_frozen=args.lr_frozen,
        lr_finetune=args.lr_finetune,
        dropout=args.dropout,
        model_dir=args.model_dir,
        use_efficientnet=not args.no_efficientnet,
        use_resnet=not args.no_resnet,
        use_densenet=not args.no_densenet,
    )
    
    # Optional: Apply GA feature selection
    if args.use_ga:
        print("\n" + "="*80)
        print("APPLYING GENETIC ALGORITHM FEATURE SELECTION")
        print("="*80)
        
        # Reload datasets for GA
        from train_ensemble_tf import load_data_from_split
        
        train_ds, val_ds, test_ds, _ = load_data_from_split(
            args.data_root, tuple(args.image_size), args.batch_size
        )
        
        def cast_only(x, y):
            x = tf.cast(x, tf.float32) / 255.0
            return x, y
        
        train_ds = train_ds.map(cast_only, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(cast_only, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(cast_only, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Apply GA feature selection
        ga_results = apply_feature_selection(
            results["models"],
            train_ds,
            val_ds,
            test_ds,
            output_dir=args.ga_output_dir,
            population_size=args.ga_population,
            generations=args.ga_generations
        )
        
        print("\n" + "="*80)
        print("FINAL RESULTS WITH GA FEATURE SELECTION")
        print("="*80)
        print(f"Test Accuracy: {ga_results['test_accuracy']:.4f} ({ga_results['test_accuracy']*100:.2f}%)")
        print(f"Test F1-Score: {ga_results['test_f1']:.4f}")
        print(f"Selected Features: {ga_results['num_features_selected']}/{ga_results['total_features']}")
        print("="*80)
    
    print("\n✅ Training complete!")
    print(f"   Results saved to: {args.model_dir}")
    if args.use_ga:
        print(f"   GA results saved to: {args.ga_output_dir}")


if __name__ == "__main__":
    main()


























