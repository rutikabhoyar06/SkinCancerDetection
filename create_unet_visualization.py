"""
Simple U-Net Medical Image Segmentation Visualization
===================================================

This script creates a visual explanation showing how U-Net processes medical images
for skin cancer diagnosis with step-by-step segmentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import cv2
from PIL import Image
import os

def create_medical_image():
    """Create a synthetic medical image with a skin lesion."""
    # Create base skin-like background
    base = np.random.normal(180, 20, (450, 600, 3)).astype(np.uint8)
    
    # Add a circular lesion
    center = (300, 225)
    radius = 80
    y, x = np.ogrid[:450, :600]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    
    # Create lesion with irregular border
    lesion_color = np.array([120, 80, 60])  # Brownish lesion
    base[mask] = lesion_color
    
    # Add some texture
    noise = np.random.normal(0, 15, base.shape).astype(np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return base

def create_segmentation_masks(image):
    """Create segmentation masks for lesion and bacterial regions."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Create lesion mask
    _, lesion_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(lesion_mask) > 127:
        lesion_mask = 255 - lesion_mask
    
    # Clean up the mask
    kernel = np.ones((5,5), np.uint8)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, kernel)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)
    
    # Create bacterial mask (smaller, more irregular)
    bacterial_mask = cv2.erode(lesion_mask, kernel, iterations=2)
    bacterial_mask = cv2.dilate(bacterial_mask, kernel, iterations=1)
    
    return lesion_mask, bacterial_mask

def create_visualization():
    """Create the main visualization."""
    print("Creating U-Net medical image visualization...")
    
    # Create figure
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('U-Net Medical Image Segmentation for Skin Cancer Diagnosis', 
                fontsize=24, fontweight='bold', y=0.95)
    
    # Create medical image
    medical_image = create_medical_image()
    
    # Create segmentation masks
    lesion_mask, bacterial_mask = create_segmentation_masks(medical_image)
    
    # 1. Input Image
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(medical_image)
    ax1.set_title('1. Input Medical Image\n(Skin Lesion)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Add arrow pointing to lesion
    arrow = patches.FancyArrowPatch((400, 150), (300, 200),
                                  arrowstyle='->', mutation_scale=20, 
                                  color='red', linewidth=3)
    ax1.add_patch(arrow)
    ax1.text(420, 140, 'Lesion', fontsize=10, color='red', fontweight='bold')
    
    # 2. U-Net Architecture
    ax2 = plt.subplot(3, 4, 2)
    draw_unet_architecture(ax2)
    
    # 3. Feature Maps (simulated)
    feature_titles = ['Edge Detection', 'Texture Analysis', 'High-level Features', 'Bottleneck']
    for i, title in enumerate(feature_titles):
        ax = plt.subplot(3, 4, 3 + i)
        # Create simulated feature map
        feature_map = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        if i == 0:  # Edge detection
            gray = cv2.cvtColor(medical_image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            feature_map = cv2.resize(edges, (224, 224))
            feature_map = np.stack([feature_map] * 3, axis=-1)
        elif i == 1:  # Texture
            gray = cv2.cvtColor(medical_image, cv2.COLOR_RGB2GRAY)
            texture = cv2.Laplacian(gray, cv2.CV_64F)
            texture = np.abs(texture)
            feature_map = cv2.resize(texture, (224, 224))
            feature_map = np.stack([feature_map] * 3, axis=-1)
            feature_map = (feature_map / feature_map.max() * 255).astype(np.uint8)
        
        ax.imshow(feature_map)
        ax.set_title(f'Feature Map:\n{title}', fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # 5. Segmentation Results
    ax5 = plt.subplot(3, 4, 7)
    # Create overlay
    overlay = medical_image.copy()
    lesion_colored = np.zeros_like(overlay)
    lesion_colored[:, :, 0] = lesion_mask  # Red for lesion
    bacterial_colored = np.zeros_like(overlay)
    bacterial_colored[:, :, 1] = bacterial_mask  # Green for bacterial
    
    result = cv2.addWeighted(overlay, 0.7, lesion_colored, 0.3, 0)
    result = cv2.addWeighted(result, 0.8, bacterial_colored, 0.2, 0)
    
    ax5.imshow(result)
    ax5.set_title('5. Final Segmentation\n(Colored Overlay)', fontsize=14, fontweight='bold')
    ax5.axis('off')
    
    # Individual masks
    ax6 = plt.subplot(3, 4, 8)
    ax6.imshow(lesion_mask, cmap='Reds')
    ax6.set_title('Lesion Mask\n(Red)', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    ax7 = plt.subplot(3, 4, 9)
    ax7.imshow(bacterial_mask, cmap='Greens')
    ax7.set_title('Bacterial/Infected\nRegion (Green)', fontsize=12, fontweight='bold')
    ax7.axis('off')
    
    # Innovation highlights
    ax8 = plt.subplot(3, 4, 10)
    draw_innovation_highlights(ax8)
    
    # Process explanation
    ax9 = plt.subplot(3, 4, 11)
    draw_process_explanation(ax9)
    
    # U-Net benefits
    ax10 = plt.subplot(3, 4, 12)
    draw_benefits(ax10)
    
    plt.tight_layout()
    return fig

def draw_unet_architecture(ax):
    """Draw U-Net architecture."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(5, 7.5, 'U-Net Architecture', fontsize=16, fontweight='bold', ha='center')
    
    # Encoder path
    encoder_boxes = [
        (1, 6, 'Input\n224×224×3'),
        (1, 5, 'Conv\n224×224×32'),
        (1, 4, 'Conv\n112×112×64'),
        (1, 3, 'Conv\n56×56×128'),
        (1, 2, 'Conv\n28×28×256'),
        (1, 1, 'Bottleneck\n14×14×512')
    ]
    
    # Decoder path
    decoder_boxes = [
        (9, 1, 'Bottleneck\n14×14×512'),
        (9, 2, 'Up+Skip\n28×28×256'),
        (9, 3, 'Up+Skip\n56×56×128'),
        (9, 4, 'Up+Skip\n112×112×64'),
        (9, 5, 'Up+Skip\n224×224×32'),
        (9, 6, 'Output\n224×224×1')
    ]
    
    # Draw boxes
    for x, y, text in encoder_boxes:
        box = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, 
                           boxstyle="round,pad=0.05", 
                           facecolor='lightblue', edgecolor='blue')
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
    
    for x, y, text in decoder_boxes:
        box = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, 
                           boxstyle="round,pad=0.05", 
                           facecolor='lightgreen', edgecolor='green')
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Skip connections
    skip_connections = [(1.4, 5.3, 8.6, 5.3), (1.4, 4.3, 8.6, 4.3), 
                       (1.4, 3.3, 8.6, 3.3), (1.4, 2.3, 8.6, 2.3)]
    
    for x1, y1, x2, y2 in skip_connections:
        connection = ConnectionPatch((x1, y1), (x2, y2), "data", "data",
                                   arrowstyle="<->", shrinkA=5, shrinkB=5,
                                   mutation_scale=10, color='red', linewidth=2)
        ax.add_patch(connection)
    
    ax.text(0.5, 3.5, 'ENCODER', ha='center', va='center', fontsize=10, 
           fontweight='bold', rotation=90, color='blue')
    ax.text(9.5, 3.5, 'DECODER', ha='center', va='center', fontsize=10, 
           fontweight='bold', rotation=90, color='green')

def draw_innovation_highlights(ax):
    """Draw innovation highlights."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    ax.text(5, 5.5, 'U-Net Innovation', fontsize=14, fontweight='bold', ha='center')
    
    innovations = [
        "✓ Skip Connections",
        "✓ Multi-scale Analysis", 
        "✓ End-to-end Learning",
        "✓ Data Efficiency",
        "✓ Precise Boundaries"
    ]
    
    for i, innovation in enumerate(innovations):
        ax.text(0.5, 4.5 - i*0.8, innovation, fontsize=10, 
               ha='left', va='center', color='darkgreen')
    
    highlight_box = FancyBboxPatch((0, 0.5), 10, 5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='lightyellow', 
                                 edgecolor='orange', linewidth=2)
    ax.add_patch(highlight_box)

def draw_process_explanation(ax):
    """Draw process explanation."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(5, 7.5, 'Process Steps', fontsize=14, fontweight='bold', ha='center')
    
    steps = [
        "1. Input: Medical image",
        "2. Encoder: Feature extraction",
        "3. Bottleneck: Compressed features",
        "4. Decoder: Feature reconstruction",
        "5. Skip connections: Preserve details",
        "6. Output: Segmentation masks"
    ]
    
    for i, step in enumerate(steps):
        ax.text(0.5, 6.5 - i*0.8, step, fontsize=10, 
               ha='left', va='center', color='darkblue')

def draw_benefits(ax):
    """Draw U-Net benefits."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(5, 7.5, 'Medical Benefits', fontsize=14, fontweight='bold', ha='center')
    
    benefits = [
        "• Accurate lesion detection",
        "• Precise boundary segmentation",
        "• Multi-class segmentation",
        "• Works with limited data",
        "• Real-time processing",
        "• Clinician interpretability"
    ]
    
    for i, benefit in enumerate(benefits):
        ax.text(0.5, 6.5 - i*0.8, benefit, fontsize=10, 
               ha='left', va='center', color='darkred')

def main():
    """Main function."""
    print("Creating U-Net medical image visualization...")
    
    try:
        fig = create_visualization()
        filename = "unet_medical_visualization.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"Visualization saved as: {filename}")
        
        # Create explanation file
        explanation = """
U-Net Medical Image Segmentation: Complete Process
================================================

This visualization demonstrates how U-Net processes medical images for skin cancer diagnosis:

1. INPUT IMAGE: Medical image containing a visible skin lesion
2. U-NET ARCHITECTURE: Encoder-decoder structure with skip connections
3. FEATURE EXTRACTION: Multi-scale analysis at different resolutions
4. SEGMENTATION OUTPUT: Colored masks showing lesion and infected regions

KEY INNOVATIONS:
- Skip Connections: Preserve fine details during upsampling
- Multi-scale Analysis: Detect lesions at all sizes
- End-to-end Learning: Direct pixel-wise prediction
- Data Efficiency: Works with limited medical data
- Precise Boundaries: Accurate lesion segmentation

MEDICAL BENEFITS:
- Accurate lesion detection and classification
- Precise boundary segmentation for surgical planning
- Multi-class segmentation (lesion + bacterial regions)
- Works effectively with limited medical datasets
- Real-time processing capabilities
- Results are interpretable by clinicians

This makes U-Net revolutionary for medical image analysis, especially for skin cancer
diagnosis where precise lesion boundaries are crucial for treatment planning.
"""
        
        with open("unet_explanation.txt", "w") as f:
            f.write(explanation)
        
        print("Detailed explanation saved as: unet_explanation.txt")
        print("\nVisualization complete!")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

