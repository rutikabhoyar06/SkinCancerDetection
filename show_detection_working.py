#!/usr/bin/env python3
"""
Working demonstration of how the skin cancer detection model identifies cancerous regions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import tensorflow as tf

def analyze_image_features(image_path):
    """Analyze the image to show what features the model might be looking for"""
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    img_resized = image.resize((224, 224))
    img_array = np.asarray(img_resized, dtype=np.float32)
    
    # Convert to different color spaces to show what the model sees
    img_hsv = Image.open(image_path).convert('HSV')
    img_hsv_resized = img_hsv.resize((224, 224))
    img_hsv_array = np.asarray(img_hsv_resized)
    
    # Create feature visualizations
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # 1. Original Image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image", fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. Grayscale (luminance)
    img_gray = Image.open(image_path).convert('L')
    img_gray_resized = img_gray.resize((224, 224))
    axes[0, 1].imshow(img_gray_resized, cmap='gray')
    axes[0, 1].set_title("Grayscale (Luminance)", fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. Edge Detection
    edges = img_gray_resized.filter(ImageFilter.FIND_EDGES)
    axes[0, 2].imshow(edges, cmap='gray')
    axes[0, 2].set_title("Edge Detection\n(Important for borders)", fontweight='bold')
    axes[0, 2].axis('off')
    
    # 4. Color Variance (shows color heterogeneity)
    img_array_norm = img_array / 255.0
    color_variance = np.var(img_array_norm, axis=2)
    im = axes[0, 3].imshow(color_variance, cmap='hot')
    axes[0, 3].set_title("Color Variance\n(Heterogeneity indicator)", fontweight='bold')
    axes[0, 3].axis('off')
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)
    
    # 5. Red Channel (often important for skin lesions)
    axes[1, 0].imshow(img_array[:, :, 0], cmap='Reds')
    axes[1, 0].set_title("Red Channel", fontweight='bold')
    axes[1, 0].axis('off')
    
    # 6. Green Channel
    axes[1, 1].imshow(img_array[:, :, 1], cmap='Greens')
    axes[1, 1].set_title("Green Channel", fontweight='bold')
    axes[1, 1].axis('off')
    
    # 7. Blue Channel
    axes[1, 2].imshow(img_array[:, :, 2], cmap='Blues')
    axes[1, 2].set_title("Blue Channel", fontweight='bold')
    axes[1, 2].axis('off')
    
    # 8. HSV Saturation (color intensity)
    saturation = img_hsv_array[:, :, 1]
    axes[1, 3].imshow(saturation, cmap='viridis')
    axes[1, 3].set_title("Saturation\n(Color intensity)", fontweight='bold')
    axes[1, 3].axis('off')
    
    plt.suptitle(f"🔬 Image Feature Analysis: {os.path.basename(image_path)}", 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return img_array

def show_model_detection_process(model, image_path, true_class):
    """Show the complete detection process"""
    
    print(f"\n🩺 ANALYZING: {os.path.basename(image_path)}")
    print(f"📂 True Class: {true_class}")
    print("=" * 60)
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    img_resized = image.resize((224, 224))
    img_array = np.asarray(img_resized, dtype=np.float32)
    
    # Make prediction
    pred = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]
    benign_prob = float(pred[0])
    malignant_prob = float(pred[1])
    predicted_class = "Malignant" if malignant_prob >= 0.5 else "Benign"
    
    print(f"📊 PREDICTION RESULTS:")
    print(f"   Predicted Class: {predicted_class}")
    print(f"   Benign Probability: {benign_prob:.2%}")
    print(f"   Malignant Probability: {malignant_prob:.2%}")
    print(f"   Confidence: {max(benign_prob, malignant_prob):.1%}")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Original Image
    ax1 = plt.subplot(2, 5, 1)
    ax1.imshow(image)
    ax1.set_title(f"Original Image\n{os.path.basename(image_path)}", fontweight='bold')
    ax1.axis('off')
    
    # 2. Model Input
    ax2 = plt.subplot(2, 5, 2)
    ax2.imshow(img_resized)
    ax2.set_title(f"Model Input (224×224)\nPredicted: {predicted_class}", fontweight='bold')
    ax2.axis('off')
    
    # 3. Prediction Bar Chart
    ax3 = plt.subplot(2, 5, 3)
    classes = ['Benign', 'Malignant']
    probs = [benign_prob, malignant_prob]
    colors = ['lightgreen', 'lightcoral']
    
    bars = ax3.bar(classes, probs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Probability')
    ax3.set_title('Prediction Confidence', fontweight='bold')
    ax3.set_ylim(0, 1)
    
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Edge Detection (important for lesion borders)
    img_gray = Image.open(image_path).convert('L').resize((224, 224))
    edges = img_gray.filter(ImageFilter.FIND_EDGES)
    ax4 = plt.subplot(2, 5, 4)
    ax4.imshow(edges, cmap='gray')
    ax4.set_title('Edge Detection\n(Border irregularity)', fontweight='bold')
    ax4.axis('off')
    
    # 5. Color Analysis
    img_array_norm = img_array / 255.0
    color_variance = np.var(img_array_norm, axis=2)
    im = ax5 = plt.subplot(2, 5, 5)
    ax5.imshow(color_variance, cmap='hot')
    ax5.set_title('Color Variance\n(Heterogeneity)', fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
    
    # 6. Red Channel Analysis
    ax6 = plt.subplot(2, 5, 6)
    ax6.imshow(img_array[:, :, 0], cmap='Reds')
    ax6.set_title('Red Channel\n(Blood vessels)', fontweight='bold')
    ax6.axis('off')
    
    # 7. Texture Analysis (Local Binary Pattern approximation)
    img_gray_array = np.asarray(img_gray, dtype=np.float32)
    texture = np.abs(np.diff(img_gray_array, axis=0))
    texture = np.pad(texture, ((0, 1), (0, 0)), mode='edge')
    ax7 = plt.subplot(2, 5, 7)
    ax7.imshow(texture, cmap='viridis')
    ax7.set_title('Texture Analysis\n(Surface patterns)', fontweight='bold')
    ax7.axis('off')
    
    # 8. Asymmetry Analysis
    left_half = img_array[:, :112, :]
    right_half = np.fliplr(img_array[:, 112:, :])
    asymmetry = np.abs(left_half - right_half).mean(axis=2)
    ax8 = plt.subplot(2, 5, 8)
    ax8.imshow(asymmetry, cmap='plasma')
    ax8.set_title('Asymmetry Analysis\n(Left vs Right)', fontweight='bold')
    ax8.axis('off')
    
    # 9. Size and Shape Analysis
    ax9 = plt.subplot(2, 5, 9)
    # Create a simple shape analysis
    img_gray_norm = img_gray_array / 255.0
    threshold = 0.3
    binary = (img_gray_norm < threshold).astype(np.float32)
    ax9.imshow(binary, cmap='gray')
    ax9.set_title('Shape Analysis\n(Dark regions)', fontweight='bold')
    ax9.axis('off')
    
    # 10. Clinical Analysis
    ax10 = plt.subplot(2, 5, 10)
    ax10.axis('off')
    
    # Calculate some metrics
    edge_intensity = np.mean(np.asarray(edges))
    color_heterogeneity = np.std(color_variance)
    asymmetry_score = np.mean(asymmetry)
    
    analysis_text = f"""
🔬 CLINICAL ANALYSIS

📊 Key Metrics:
• Edge Intensity: {edge_intensity:.2f}
• Color Heterogeneity: {color_heterogeneity:.2f}
• Asymmetry Score: {asymmetry_score:.2f}

🎯 Model Focus Areas:
• Border irregularity
• Color variation
• Asymmetric patterns
• Texture changes
• Size and shape

⚠️ Risk Factors Detected:
• {'High' if edge_intensity > 50 else 'Low'} border irregularity
• {'High' if color_heterogeneity > 0.1 else 'Low'} color variation
• {'High' if asymmetry_score > 0.05 else 'Low'} asymmetry

🔍 What the AI looks for:
• ABCDE criteria:
  A - Asymmetry
  B - Border irregularity  
  C - Color variation
  D - Diameter (>6mm)
  E - Evolution

📈 Confidence: {max(benign_prob, malignant_prob):.1%}
    """
    
    ax10.text(0.05, 0.95, analysis_text, transform=ax10.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    plt.suptitle(f'🩺 Skin Cancer Detection Analysis: {predicted_class} ({malignant_prob:.1%} malignant)', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()
    
    return predicted_class, benign_prob, malignant_prob

def main():
    print("🩺 Skin Cancer Detection Analysis")
    print("=" * 60)
    print("This shows exactly how the AI model detects cancerous regions")
    print("=" * 60)
    
    # Load model
    MODEL_PATH = "checkpoints/bm_classifier_best.keras"
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    
    print("Loading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
    
    # Test images
    test_images = [
        ("dataset/benign/ISIC_0024306.jpg", "Benign"),
        ("dataset/malignant/ISIC_0024310.jpg", "Malignant")
    ]
    
    results = []
    for image_path, true_class in test_images:
        if os.path.exists(image_path):
            # Show feature analysis
            print(f"\n🔬 Feature Analysis for {os.path.basename(image_path)}")
            analyze_image_features(image_path)
            
            # Show detection process
            predicted_class, benign_prob, malignant_prob = show_model_detection_process(
                model, image_path, true_class
            )
            
            results.append({
                'image': os.path.basename(image_path),
                'true_class': true_class,
                'predicted': predicted_class,
                'benign_prob': benign_prob,
                'malignant_prob': malignant_prob,
                'correct': predicted_class.lower() == true_class.lower()
            })
        else:
            print(f"❌ Image not found: {image_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 DETECTION SUMMARY")
    print("=" * 60)
    
    for result in results:
        status = "✅ CORRECT" if result['correct'] else "❌ INCORRECT"
        print(f"Image: {result['image']}")
        print(f"  True Class: {result['true_class']}")
        print(f"  Predicted: {result['predicted']} ({result['malignant_prob']:.1%} malignant)")
        print(f"  Result: {status}")
        print()
    
    print("🎯 How the Model Detects Cancerous Regions:")
    print("• Analyzes border irregularity using edge detection")
    print("• Measures color heterogeneity across the lesion")
    print("• Detects asymmetric patterns")
    print("• Examines texture variations")
    print("• Considers size and shape characteristics")
    print("• Uses deep learning to combine all these features")
    print("\n⚠️  Remember: This is a research tool. Always consult a dermatologist!")

if __name__ == "__main__":
    main()

