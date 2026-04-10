"""
Enhanced U-Net Medical Image Segmentation Demonstration
======================================================

This script creates an advanced visual explanation with detailed annotations
showing how U-Net processes medical images for skin cancer diagnosis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Rectangle, Circle
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
from typing import Tuple, List

def create_realistic_medical_image():
    """Create a more realistic medical image with skin lesion."""
    # Create base skin texture
    base = np.random.normal(180, 15, (450, 600, 3)).astype(np.uint8)
    
    # Add skin texture variations
    for i in range(3):
        noise = np.random.normal(0, 10, (450, 600))
        base[:, :, i] = np.clip(base[:, :, i] + noise, 0, 255)
    
    # Create main lesion (irregular shape)
    center = (300, 225)
    y, x = np.ogrid[:450, :600]
    
    # Main lesion (elliptical with irregular border)
    lesion_mask = ((x - center[0])**2 / (90**2) + (y - center[1])**2 / (70**2)) <= 1
    
    # Add irregularity to lesion border
    irregular_mask = np.random.random((450, 600)) > 0.85
    lesion_mask = lesion_mask & ~irregular_mask
    
    # Color the lesion
    lesion_color = np.array([100, 70, 50])  # Dark brown
    base[lesion_mask] = lesion_color
    
    # Add some texture to lesion
    lesion_texture = np.random.normal(0, 20, base.shape).astype(np.int16)
    base = np.clip(base.astype(np.int16) + lesion_texture, 0, 255).astype(np.uint8)
    
    # Add smaller bacterial/infected regions
    bacterial_centers = [(280, 200), (320, 250), (290, 240)]
    for bc in bacterial_centers:
        bacterial_mask = ((x - bc[0])**2 + (y - bc[1])**2) <= 25
        bacterial_color = np.array([80, 120, 60])  # Greenish
        base[bacterial_mask] = bacterial_color
    
    return base

def create_detailed_segmentation_masks(image):
    """Create detailed segmentation masks with multiple regions."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Create lesion mask using adaptive thresholding
    lesion_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
    
    # Invert if needed
    if np.mean(lesion_mask) > 127:
        lesion_mask = 255 - lesion_mask
    
    # Clean up lesion mask
    kernel = np.ones((7,7), np.uint8)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, kernel)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)
    
    # Create bacterial mask (smaller regions)
    bacterial_mask = cv2.erode(lesion_mask, kernel, iterations=3)
    bacterial_mask = cv2.dilate(bacterial_mask, kernel, iterations=1)
    
    # Add some noise to make it more realistic
    noise = np.random.random(bacterial_mask.shape) > 0.7
    bacterial_mask[noise] = 0
    
    # Create healthy skin mask
    healthy_mask = 255 - lesion_mask
    
    return lesion_mask, bacterial_mask, healthy_mask

def create_enhanced_visualization():
    """Create enhanced visualization with detailed annotations."""
    print("Creating enhanced U-Net medical image visualization...")
    
    # Create figure with more space
    fig = plt.figure(figsize=(24, 18))
    fig.suptitle('U-Net Medical Image Segmentation: Complete Process for Skin Cancer Diagnosis', 
                fontsize=28, fontweight='bold', y=0.96)
    
    # Create medical image
    medical_image = create_realistic_medical_image()
    
    # Create segmentation masks
    lesion_mask, bacterial_mask, healthy_mask = create_detailed_segmentation_masks(medical_image)
    
    # Create grid layout
    gs = fig.add_gridspec(4, 6, hspace=0.4, wspace=0.3, 
                         height_ratios=[1.2, 1, 1, 1], width_ratios=[1, 1, 1, 1, 1, 1])
    
    # 1. Input Image with detailed annotations
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.imshow(medical_image)
    ax1.set_title('1. Input Medical Image: Skin Lesion Detection', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.axis('off')
    
    # Add detailed annotations
    # Main lesion annotation
    lesion_rect = Rectangle((210, 155), 180, 140, linewidth=3, 
                           edgecolor='red', facecolor='none', linestyle='--')
    ax1.add_patch(lesion_rect)
    ax1.text(400, 120, 'Primary Lesion\n(Asymmetrical, Irregular Border)', 
            fontsize=12, color='red', fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Bacterial regions
    bacterial_circles = [(280, 200), (320, 250), (290, 240)]
    for i, (x, y) in enumerate(bacterial_circles):
        circle = Circle((x, y), 25, linewidth=2, edgecolor='green', 
                       facecolor='none', linestyle=':')
        ax1.add_patch(circle)
        if i == 0:
            ax1.text(x+30, y-30, 'Bacterial/\nInfected Regions', 
                    fontsize=10, color='green', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # 2. U-Net Architecture with detailed flow
    ax2 = fig.add_subplot(gs[0, 2:])
    draw_detailed_unet_architecture(ax2)
    
    # 3. Feature Extraction Process
    feature_titles = ['Edge Detection\n(Level 1)', 'Texture Analysis\n(Level 2)', 
                     'High-level Features\n(Level 3)', 'Bottleneck\n(Level 4)']
    
    for i, title in enumerate(feature_titles):
        ax = fig.add_subplot(gs[1, i])
        feature_map = create_feature_map(medical_image, i)
        ax.imshow(feature_map)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Add feature description
        descriptions = [
            'Detects lesion\nboundaries and\nedges',
            'Analyzes skin\ntexture patterns\nand irregularities',
            'Combines multiple\nfeatures for\nlesion recognition',
            'Compressed\nrepresentation\nof all features'
        ]
        ax.text(0.02, 0.98, descriptions[i], transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    # 4. Decoder Process
    decoder_titles = ['Upsampling\n+ Skip Connection', 'Feature Fusion\n+ Refinement', 
                     'Boundary\nRefinement', 'Final\nSegmentation']
    
    for i, title in enumerate(decoder_titles):
        ax = fig.add_subplot(gs[2, i])
        decoder_map = create_decoder_map(medical_image, i)
        ax.imshow(decoder_map)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Add decoder description
        decoder_descriptions = [
            'Restores spatial\nresolution using\nskip connections',
            'Combines multi-scale\nfeatures for better\nsegmentation',
            'Refines lesion\nboundaries for\nprecision',
            'Produces final\npixel-wise\npredictions'
        ]
        ax.text(0.02, 0.98, decoder_descriptions[i], transform=ax.transAxes, 
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    # 5. Final Results
    ax5 = fig.add_subplot(gs[3, :2])
    # Create comprehensive overlay
    overlay = medical_image.copy()
    
    # Create colored masks
    lesion_colored = np.zeros_like(overlay)
    lesion_colored[:, :, 0] = lesion_mask  # Red for lesion
    
    bacterial_colored = np.zeros_like(overlay)
    bacterial_colored[:, :, 1] = bacterial_mask  # Green for bacterial
    
    healthy_colored = np.zeros_like(overlay)
    healthy_colored[:, :, 2] = healthy_mask  # Blue for healthy skin
    
    # Blend all masks
    result = cv2.addWeighted(overlay, 0.6, lesion_colored, 0.25, 0)
    result = cv2.addWeighted(result, 0.8, bacterial_colored, 0.2, 0)
    result = cv2.addWeighted(result, 0.9, healthy_colored, 0.1, 0)
    
    ax5.imshow(result)
    ax5.set_title('5. Final Segmentation Result: Multi-Class Output', 
                 fontsize=16, fontweight='bold')
    ax5.axis('off')
    
    # Add legend
    legend_elements = [
        patches.Patch(color='red', alpha=0.7, label='Lesion Region'),
        patches.Patch(color='green', alpha=0.7, label='Bacterial/Infected'),
        patches.Patch(color='blue', alpha=0.7, label='Healthy Skin')
    ]
    ax5.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Individual mask displays
    masks = [
        (lesion_mask, 'Reds', 'Lesion Mask\n(Red Channel)'),
        (bacterial_mask, 'Greens', 'Bacterial/Infected\n(Green Channel)'),
        (healthy_mask, 'Blues', 'Healthy Skin\n(Blue Channel)')
    ]
    
    for i, (mask, cmap, title) in enumerate(masks):
        ax = fig.add_subplot(gs[3, 2 + i])
        ax.imshow(mask, cmap=cmap)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Innovation and benefits
    ax9 = fig.add_subplot(gs[3, 5])
    draw_innovation_benefits(ax9)
    
    plt.tight_layout()
    return fig

def create_feature_map(image, level):
    """Create simulated feature maps for different U-Net levels."""
    resized = cv2.resize(image, (224, 224))
    
    if level == 0:  # Edge detection
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        feature_map = cv2.resize(edges, (224, 224))
        feature_map = np.stack([feature_map] * 3, axis=-1)
    elif level == 1:  # Texture analysis
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        texture = cv2.Laplacian(gray, cv2.CV_64F)
        texture = np.abs(texture)
        feature_map = cv2.resize(texture, (224, 224))
        feature_map = np.stack([feature_map] * 3, axis=-1)
        feature_map = (feature_map / feature_map.max() * 255).astype(np.uint8)
    elif level == 2:  # High-level features
        feature_map = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        # Add some structure
        center = 112
        cv2.circle(feature_map, (center, center), 40, (255, 0, 0), -1)
        cv2.circle(feature_map, (center, center), 20, (0, 255, 0), -1)
    else:  # Bottleneck
        feature_map = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        feature_map[:, :] = [255, 128, 0]  # Orange for bottleneck
    
    return feature_map

def create_decoder_map(image, level):
    """Create simulated decoder feature maps."""
    resized = cv2.resize(image, (224, 224))
    
    if level == 0:  # Upsampling
        # Simulate upsampling with some blur
        blurred = cv2.GaussianBlur(resized, (15, 15), 0)
        return blurred
    elif level == 1:  # Feature fusion
        # Simulate feature fusion
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = np.stack([edges] * 3, axis=-1)
        return cv2.addWeighted(resized, 0.7, edges_colored, 0.3, 0)
    elif level == 2:  # Boundary refinement
        # Simulate boundary refinement
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        refined = cv2.bilateralFilter(gray, 9, 75, 75)
        return np.stack([refined] * 3, axis=-1)
    else:  # Final output
        # Simulate final segmentation
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(mask) > 127:
            mask = 255 - mask
        return np.stack([mask] * 3, axis=-1)

def draw_detailed_unet_architecture(ax):
    """Draw detailed U-Net architecture with flow annotations."""
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(6, 9.5, 'U-Net Architecture: Encoder-Decoder with Skip Connections', 
           fontsize=18, fontweight='bold', ha='center')
    
    # Encoder path (left side)
    encoder_boxes = [
        (1, 8, 'Input\n224×224×3\nRGB Image'),
        (1, 7, 'Conv Block\n224×224×32\nEdge Detection'),
        (1, 6, 'Conv Block\n112×112×64\nTexture Analysis'),
        (1, 5, 'Conv Block\n56×56×128\nHigh-level Features'),
        (1, 4, 'Conv Block\n28×28×256\nComplex Patterns'),
        (1, 3, 'Conv Block\n14×14×512\nBottleneck')
    ]
    
    # Decoder path (right side)
    decoder_boxes = [
        (11, 3, 'Bottleneck\n14×14×512\nFeature Compression'),
        (11, 4, 'Up + Skip\n28×28×256\nUpsampling + Fusion'),
        (11, 5, 'Up + Skip\n56×56×128\nFeature Reconstruction'),
        (11, 6, 'Up + Skip\n112×112×64\nBoundary Refinement'),
        (11, 7, 'Up + Skip\n224×224×32\nDetail Recovery'),
        (11, 8, 'Output\n224×224×1\nSegmentation Mask')
    ]
    
    # Draw encoder boxes
    for x, y, text in encoder_boxes:
        box = FancyBboxPatch((x-0.5, y-0.4), 1.0, 0.8, 
                           boxstyle="round,pad=0.1", 
                           facecolor='lightblue', edgecolor='blue', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Draw decoder boxes
    for x, y, text in decoder_boxes:
        box = FancyBboxPatch((x-0.5, y-0.4), 1.0, 0.8, 
                           boxstyle="round,pad=0.1", 
                           facecolor='lightgreen', edgecolor='green', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Draw skip connections with labels
    skip_connections = [
        (1.5, 7.4, 10.5, 7.4, 'Skip 1'),
        (1.5, 6.4, 10.5, 6.4, 'Skip 2'),
        (1.5, 5.4, 10.5, 5.4, 'Skip 3'),
        (1.5, 4.4, 10.5, 4.4, 'Skip 4')
    ]
    
    for x1, y1, x2, y2, label in skip_connections:
        connection = ConnectionPatch((x1, y1), (x2, y2), "data", "data",
                                   arrowstyle="<->", shrinkA=5, shrinkB=5,
                                   mutation_scale=15, color='red', linewidth=3)
        ax.add_patch(connection)
        ax.text((x1+x2)/2, y1+0.2, label, ha='center', va='bottom', 
               fontsize=10, fontweight='bold', color='red')
    
    # Add section labels
    ax.text(0.5, 5.5, 'ENCODER\n(Feature\nExtraction)', 
           ha='center', va='center', fontsize=12, fontweight='bold', 
           rotation=90, color='blue')
    ax.text(11.5, 5.5, 'DECODER\n(Feature\nReconstruction)', 
           ha='center', va='center', fontsize=12, fontweight='bold', 
           rotation=90, color='green')
    
    # Add flow arrows
    for i in range(5):
        y_pos = 7.5 - i * 0.8
        arrow = patches.FancyArrowPatch((1.5, y_pos), (1.5, y_pos-0.6),
                                      arrowstyle='->', mutation_scale=15, 
                                      color='blue', linewidth=2)
        ax.add_patch(arrow)
        
        arrow2 = patches.FancyArrowPatch((10.5, y_pos-0.6), (10.5, y_pos),
                                       arrowstyle='->', mutation_scale=15, 
                                       color='green', linewidth=2)
        ax.add_patch(arrow2)

def draw_innovation_benefits(ax):
    """Draw innovation and benefits section."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(5, 7.5, 'U-Net Innovation & Benefits', fontsize=16, fontweight='bold', ha='center')
    
    innovations = [
        "✓ Skip Connections: Preserve fine details",
        "✓ Multi-scale Analysis: Detect all lesion sizes", 
        "✓ End-to-end Learning: Direct pixel prediction",
        "✓ Data Efficiency: Works with limited data",
        "✓ Precise Boundaries: Accurate segmentation",
        "✓ Real-time Processing: Fast inference"
    ]
    
    for i, innovation in enumerate(innovations):
        ax.text(0.5, 6.5 - i*0.8, innovation, fontsize=10, 
               ha='left', va='center', color='darkgreen')
    
    # Add highlight box
    highlight_box = FancyBboxPatch((0, 0.5), 10, 7, 
                                 boxstyle="round,pad=0.2", 
                                 facecolor='lightyellow', 
                                 edgecolor='orange', linewidth=3)
    ax.add_patch(highlight_box)
    ax.text(5, 0.2, 'Revolutionary for Medical Image Analysis', 
           ha='center', va='center', fontsize=12, fontweight='bold', 
           color='darkorange')

def main():
    """Main function to create enhanced visualization."""
    print("Creating enhanced U-Net medical image visualization...")
    
    try:
        fig = create_enhanced_visualization()
        filename = "enhanced_unet_medical_visualization.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"Enhanced visualization saved as: {filename}")
        
        # Create comprehensive explanation
        explanation = """
Enhanced U-Net Medical Image Segmentation: Complete Process
==========================================================

This comprehensive visualization demonstrates the complete U-Net process for skin cancer diagnosis:

INPUT PROCESSING:
- Medical image (224×224×3) containing visible skin lesion
- Lesion shows asymmetrical shape with irregular borders
- Multiple bacterial/infected regions identified
- Healthy skin regions clearly defined

U-NET ARCHITECTURE:
- Encoder Path: Progressive feature extraction at multiple scales
  * Level 1: Edge detection identifies lesion boundaries
  * Level 2: Texture analysis captures skin irregularities
  * Level 3: High-level features combine multiple properties
  * Level 4: Bottleneck creates compressed representation

- Decoder Path: Feature reconstruction with skip connections
  * Upsampling restores spatial resolution
  * Skip connections preserve fine details from encoder
  * Feature fusion combines multi-scale information
  * Boundary refinement creates precise segmentation

FEATURE EXTRACTION PROCESS:
1. Edge Detection: Identifies lesion boundaries and contours
2. Texture Analysis: Analyzes skin texture patterns and irregularities
3. High-level Features: Combines multiple features for lesion recognition
4. Bottleneck: Creates compressed representation of all features

DECODER RECONSTRUCTION:
1. Upsampling + Skip Connection: Restores spatial resolution using skip connections
2. Feature Fusion + Refinement: Combines multi-scale features for better segmentation
3. Boundary Refinement: Refines lesion boundaries for precision
4. Final Segmentation: Produces final pixel-wise predictions

OUTPUT GENERATION:
- Multi-class segmentation with distinct colored masks:
  * Red: Primary lesion region
  * Green: Bacterial/infected regions
  * Blue: Healthy skin regions
- Overlay shows all regions on original image
- Precise boundary detection for surgical planning

KEY INNOVATIONS:
✓ Skip Connections: Preserve fine details during upsampling
✓ Multi-scale Analysis: Detect lesions at all sizes
✓ End-to-end Learning: Direct pixel-wise prediction
✓ Data Efficiency: Works with limited medical data
✓ Precise Boundaries: Accurate lesion segmentation
✓ Real-time Processing: Fast inference for clinical use

MEDICAL BENEFITS:
- Accurate lesion detection and classification
- Precise boundary segmentation for surgical planning
- Multi-class segmentation (lesion + bacterial + healthy regions)
- Works effectively with limited medical datasets
- Real-time processing capabilities for clinical workflow
- Results are interpretable by clinicians
- Reduces diagnostic errors and improves patient outcomes

This makes U-Net revolutionary for medical image analysis, especially for skin cancer
diagnosis where precise lesion boundaries are crucial for treatment planning and
surgical intervention.
"""
        
        with open("enhanced_unet_explanation.txt", "w") as f:
            f.write(explanation)
        
        print("Enhanced explanation saved as: enhanced_unet_explanation.txt")
        print("\nEnhanced visualization complete!")
        print("\nThe visualization includes:")
        print("1. Detailed input image with lesion annotations")
        print("2. Complete U-Net architecture with flow arrows")
        print("3. Step-by-step feature extraction process")
        print("4. Decoder reconstruction with skip connections")
        print("5. Multi-class segmentation results")
        print("6. Innovation highlights and medical benefits")
        
    except Exception as e:
        print(f"Error creating enhanced visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

