"""
Example: How to use the ensemble system for inference
"""

import numpy as np
import tensorflow as tf
from ensemble_voting import MaxVotingEnsemble
import os


def load_ensemble_models(model_dir="ensemble_checkpoints"):
    """Load trained models from checkpoint directory"""
    models = []
    model_names = []
    
    # Try to load each model
    for model_name in ["EfficientNetB3", "ResNet50", "DenseNet121"]:
        model_path = os.path.join(model_dir, "ensemble", f"{model_name}_final.keras")
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            models.append(model)
            model_names.append(model_name)
            print(f"✅ Loaded {model_name}")
        else:
            # Try alternative path
            alt_path = os.path.join(model_dir, f"{model_name}_best.keras")
            if os.path.exists(alt_path):
                model = tf.keras.models.load_model(alt_path)
                models.append(model)
                model_names.append(model_name)
                print(f"✅ Loaded {model_name} from alternative path")
    
    if len(models) == 0:
        raise ValueError(f"No models found in {model_dir}")
    
    return models, model_names


def predict_single_image(ensemble, image_path, image_size=(300, 300)):
    """
    Predict on a single image
    
    Args:
        ensemble: MaxVotingEnsemble instance
        image_path: Path to image file
        image_size: Target image size
        
    Returns:
        Tuple of (prediction, confidence, probabilities)
    """
    from PIL import Image
    
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img = img.resize(image_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction, confidence = ensemble.predict_with_confidence(img_array)
    probabilities = ensemble.predict_proba(img_array)
    
    return prediction[0], confidence[0], probabilities[0]


def predict_batch(ensemble, images, batch_size=32):
    """
    Predict on a batch of images
    
    Args:
        ensemble: MaxVotingEnsemble instance
        images: Numpy array of images (N, H, W, 3)
        batch_size: Batch size for prediction
        
    Returns:
        Tuple of (predictions, confidences, probabilities)
    """
    predictions = ensemble.predict(images, batch_size=batch_size)
    probabilities = ensemble.predict_proba(images, batch_size=batch_size)
    confidences = np.max(probabilities, axis=1)
    
    return predictions, confidences, probabilities


def main():
    """Example usage"""
    print("="*80)
    print("ENSEMBLE INFERENCE EXAMPLE")
    print("="*80)
    
    # Load models
    print("\n📂 Loading trained models...")
    try:
        models, model_names = load_ensemble_models("ensemble_checkpoints")
        print(f"Loaded {len(models)} models: {', '.join(model_names)}")
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("   Please train models first using: python train_ensemble_tf.py")
        return
    
    # Create ensemble
    print("\n🔗 Creating ensemble...")
    ensemble = MaxVotingEnsemble(models)
    print("✅ Ensemble created!")
    
    # Example: Predict on a single image
    print("\n📸 Example: Single image prediction")
    print("   (Replace with your image path)")
    # Uncomment to use:
    # prediction, confidence, probabilities = predict_single_image(
    #     ensemble, "path/to/image.jpg"
    # )
    # print(f"   Prediction: {prediction}")
    # print(f"   Confidence: {confidence:.4f}")
    # print(f"   Probabilities: Benign={probabilities[0]:.4f}, Malignant={probabilities[1]:.4f}")
    
    # Example: Predict on test dataset
    print("\n📊 Example: Batch prediction on test dataset")
    try:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            "dataset_split/test",
            labels="inferred",
            label_mode="int",
            seed=42,
            image_size=(300, 300),
            batch_size=32,
        )
        
        def cast_only(x, y):
            x = tf.cast(x, tf.float32) / 255.0
            return x, y
        
        test_ds = test_ds.map(cast_only, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Collect all test data
        all_x = []
        all_y = []
        for batch_x, batch_y in test_ds:
            all_x.append(batch_x.numpy())
            all_y.extend(batch_y.numpy().tolist())
        
        all_x = np.concatenate(all_x, axis=0)
        all_y = np.array(all_y)
        
        # Predict
        predictions, confidences, probabilities = predict_batch(ensemble, all_x)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == all_y)
        print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Average Confidence: {np.mean(confidences):.4f}")
        
    except Exception as e:
        print(f"   ⚠️  Could not load test dataset: {e}")
        print("   This is okay - you can still use the ensemble for your own images")
    
    print("\n✅ Example complete!")
    print("\nTo use the ensemble in your code:")
    print("  1. Load models: models, _ = load_ensemble_models('ensemble_checkpoints')")
    print("  2. Create ensemble: ensemble = MaxVotingEnsemble(models)")
    print("  3. Predict: predictions = ensemble.predict(images)")


if __name__ == "__main__":
    main()


























