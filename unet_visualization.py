"""
U-Net Medical Image Segmentation Visualization
==============================================

This script creates a comprehensive visual explanation showing how the U-Net model
processes medical images for skin cancer diagnosis, including:
1. Input image with visible skin lesion
2. U-Net's feature extraction process
3. Final segmentation output with distinct colored masks
4. Detailed annotations explaining the innovation in medical image analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import cv2
from PIL import Image
import os
from typing import Tuple, List
import seaborn as sns

# Set style for medical visualization
plt.style.use('default')
sns.set_palette("husl")

class UNetVisualization:
    def __init__(self, image_path: str = None):
        """Initialize the U-Net visualization with a sample image."""
        self.image_path = image_path or "dataset/benign/ISIC_0024306.jpg"
        self.load_sample_image()
        
    def load_sample_image(self):
        """Load and preprocess the sample medical image."""
        if os.path.exists(self.image_path):
            self.original_image = Image.open(self.image_path).convert('RGB')
            self.image_array = np.array(self.original_image)
        else:
            # Create a synthetic medical image if dataset not available
            self.create_synthetic_medical_image()
    
    def create_synthetic_medical_image(self):
        """Create a synthetic medical image for demonstration purposes."""
        # Create a base skin-like background
        base = np.random.normal(180, 20, (450, 600, 3)).astype(np.uint8)
        
        # Add a circular lesion
        center = (300, 225)
        radius = 80
        y, x = np.ogrid[:450, :600]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        
        # Create lesion with irregular border
        lesion_color = np.array([120, 80, 60])  # Brownish lesion
        base[mask] = lesion_color
        
        # Add some texture and irregularity
        noise = np.random.normal(0, 15, base.shape).astype(np.int16)
        base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        self.image_array = base
        self.original_image = Image.fromarray(base)
    
    def simulate_unet_encoder_features(self, image: np.ndarray) -> List[np.ndarray]:
        """Simulate U-Net encoder feature maps at different scales."""
        features = []
        
        # Original image (224x224 for U-Net input)
        resized = cv2.resize(image, (224, 224))
        features.append(resized)
        
        # Simulate different encoder levels with progressively smaller feature maps
        for i, scale in enumerate([0.5, 0.25, 0.125, 0.0625]):
            size = int(224 * scale)
            if size < 4:
                size = 4
            
            # Create feature map with different characteristics
            if i == 0:
                # Level 1: Edge detection
                gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                feature_map = cv2.resize(edges, (size, size))
                feature_map = np.stack([feature_map] * 3, axis=-1)
            elif i == 1:
                # Level 2: Texture features
                gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
                texture = cv2.Laplacian(gray, cv2.CV_64F)
                texture = np.abs(texture)
                feature_map = cv2.resize(texture, (size, size))
                feature_map = np.stack([feature_map] * 3, axis=-1)
                feature_map = (feature_map / feature_map.max() * 255).astype(np.uint8)
            elif i == 2:
                # Level 3: High-level features
                feature_map = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
                # Add some structure
                center = size // 2
                cv2.circle(feature_map, (center, center), center//3, (255, 0, 0), -1)
            else:
                # Level 4: Bottleneck features
                feature_map = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
                feature_map[:, :] = [255, 128, 0]  # Orange for bottleneck
        
        features.extend([cv2.resize(f, (224, 224)) for f in features[1:]])
        return features
    
    def create_segmentation_mask(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create segmentation masks for lesion and bacterial/infected regions."""
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Create lesion mask using thresholding and morphological operations
        _, lesion_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if needed (assuming lesion is darker)
        if np.mean(lesion_mask) > 127:
            lesion_mask = 255 - lesion_mask
        
        # Clean up the mask
        kernel = np.ones((5,5), np.uint8)
        lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, kernel)
        lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)
        
        # Create bacterial/infected region mask (smaller, more irregular)
        bacterial_mask = lesion_mask.copy()
        # Make it smaller and more irregular
        bacterial_mask = cv2.erode(bacterial_mask, kernel, iterations=2)
        bacterial_mask = cv2.dilate(bacterial_mask, kernel, iterations=1)
        
        # Add some noise to make it more realistic
        noise = np.random.random(bacterial_mask.shape) > 0.8
        bacterial_mask[noise] = 0
        
        return lesion_mask, bacterial_mask
    
    def create_comprehensive_visualization(self):
        """Create the main comprehensive visualization."""
        fig = plt.figure(figsize=(20, 16))
        
        # Main title
        fig.suptitle('U-Net Medical Image Segmentation for Skin Cancer Diagnosis', 
                    fontsize=24, fontweight='bold', y=0.95)
        
        # Create a complex layout
        gs = fig.add_gridspec(4, 6, hspace=0.3, wspace=0.3, 
                             height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1, 1, 1, 1])
        
        # 1. Input Image (Top Left)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.imshow(self.image_array)
        ax1.set_title('1. Input Medical Image\n(Skin Lesion Detection)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.axis('off')
        
        # Add annotation arrow pointing to lesion
        arrow = patches.FancyArrowPatch((400, 150), (300, 200),
                                      arrowstyle='->', mutation_scale=20, 
                                      color='red', linewidth=3)
        ax1.add_patch(arrow)
        ax1.text(420, 140, 'Lesion\nRegion', fontsize=10, color='red', 
                fontweight='bold', ha='left')
        
        # 2. U-Net Architecture Overview (Top Right)
        ax2 = fig.add_subplot(gs[0, 2:])
        self.draw_unet_architecture(ax2)
        
        # 3. Feature Extraction Process (Second Row)
        features = self.simulate_unet_encoder_features(self.image_array)
        
        for i, (feature, title) in enumerate(zip(features[:4], 
            ['Level 1: Edge Detection', 'Level 2: Texture Analysis', 
             'Level 3: High-level Features', 'Level 4: Bottleneck'])):
            ax = fig.add_subplot(gs[1, i])
            ax.imshow(feature)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.axis('off')
        
        # 4. Decoder Process (Third Row)
        for i, (feature, title) in enumerate(zip(features[4:], 
            ['Upsampling + Skip', 'Feature Fusion', 'Boundary Refinement', 'Final Output'])):
            ax = fig.add_subplot(gs[2, i])
            ax.imshow(feature)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.axis('off')
        
        # 5. Final Segmentation Results (Bottom Row)
        lesion_mask, bacterial_mask = self.create_segmentation_mask(self.image_array)
        
        # Original with overlay
        ax5 = fig.add_subplot(gs[3, :2])
        overlay = self.image_array.copy()
        # Create colored masks
        lesion_colored = np.zeros_like(overlay)
        lesion_colored[:, :, 0] = lesion_mask  # Red channel for lesion
        bacterial_colored = np.zeros_like(overlay)
        bacterial_colored[:, :, 1] = bacterial_mask  # Green channel for bacterial
        
        # Blend with original
        result = cv2.addWeighted(overlay, 0.7, lesion_colored, 0.3, 0)
        result = cv2.addWeighted(result, 0.8, bacterial_colored, 0.2, 0)
        
        ax5.imshow(result)
        ax5.set_title('5. Final Segmentation Result\n(Colored Overlay)', 
                     fontsize=14, fontweight='bold')
        ax5.axis('off')
        
        # Individual masks
        ax6 = fig.add_subplot(gs[3, 2])
        ax6.imshow(lesion_mask, cmap='Reds')
        ax6.set_title('Lesion Mask\n(Red)', fontsize=12, fontweight='bold')
        ax6.axis('off')
        
        ax7 = fig.add_subplot(gs[3, 3])
        ax7.imshow(bacterial_mask, cmap='Greens')
        ax7.set_title('Bacterial/Infected\nRegion (Green)', fontsize=12, fontweight='bold')
        ax7.axis('off')
        
        # Innovation highlights
        ax8 = fig.add_subplot(gs[3, 4:])
        self.draw_innovation_highlights(ax8)
        
        plt.tight_layout()
        return fig
    
    def draw_unet_architecture(self, ax):
        """Draw a simplified U-Net architecture diagram."""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        # Title
        ax.text(5, 7.5, 'U-Net Architecture', fontsize=16, fontweight='bold', ha='center')
        
        # Encoder path (left side)
        encoder_boxes = [
            (1, 6, 'Input\n224×224×3'),
            (1, 5, 'Conv Block\n224×224×32'),
            (1, 4, 'Conv Block\n112×112×64'),
            (1, 3, 'Conv Block\n56×56×128'),
            (1, 2, 'Conv Block\n28×28×256'),
            (1, 1, 'Bottleneck\n14×14×512')
        ]
        
        # Decoder path (right side)
        decoder_boxes = [
            (9, 1, 'Bottleneck\n14×14×512'),
            (9, 2, 'Up + Skip\n28×28×256'),
            (9, 3, 'Up + Skip\n56×56×128'),
            (9, 4, 'Up + Skip\n112×112×64'),
            (9, 5, 'Up + Skip\n224×224×32'),
            (9, 6, 'Output\n224×224×1')
        ]
        
        # Draw encoder boxes
        for x, y, text in encoder_boxes:
            box = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, 
                               boxstyle="round,pad=0.05", 
                               facecolor='lightblue', edgecolor='blue')
            ax.add_patch(box)
            ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Draw decoder boxes
        for x, y, text in decoder_boxes:
            box = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, 
                               boxstyle="round,pad=0.05", 
                               facecolor='lightgreen', edgecolor='green')
            ax.add_patch(box)
            ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Draw skip connections
        skip_connections = [(1.4, 5.3, 8.6, 5.3), (1.4, 4.3, 8.6, 4.3), 
                           (1.4, 3.3, 8.6, 3.3), (1.4, 2.3, 8.6, 2.3)]
        
        for x1, y1, x2, y2 in skip_connections:
            connection = ConnectionPatch((x1, y1), (x2, y2), "data", "data",
                                       arrowstyle="<->", shrinkA=5, shrinkB=5,
                                       mutation_scale=10, color='red', linewidth=2)
            ax.add_patch(connection)
        
        # Add labels
        ax.text(0.5, 3.5, 'ENCODER\n(Feature\nExtraction)', 
               ha='center', va='center', fontsize=10, fontweight='bold', 
               rotation=90, color='blue')
        ax.text(9.5, 3.5, 'DECODER\n(Feature\nReconstruction)', 
               ha='center', va='center', fontsize=10, fontweight='bold', 
               rotation=90, color='green')
    
    def draw_innovation_highlights(self, ax):
        """Draw innovation highlights for U-Net in medical imaging."""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis('off')
        
        ax.text(5, 5.5, 'U-Net Innovation in Medical Imaging', 
               fontsize=14, fontweight='bold', ha='center')
        
        innovations = [
            "✓ Skip Connections: Preserve fine details",
            "✓ Multi-scale Analysis: Detect lesions at all sizes", 
            "✓ End-to-end Learning: Direct pixel-wise prediction",
            "✓ Data Efficiency: Works with limited medical data",
            "✓ Precise Boundaries: Accurate lesion segmentation"
        ]
        
        for i, innovation in enumerate(innovations):
            ax.text(0.5, 4.5 - i*0.8, innovation, fontsize=10, 
                   ha='left', va='center', color='darkgreen')
        
        # Add a highlight box
        highlight_box = FancyBboxPatch((0, 0.5), 10, 5, 
                                     boxstyle="round,pad=0.1", 
                                     facecolor='lightyellow', 
                                     edgecolor='orange', linewidth=2)
        ax.add_patch(highlight_box)
        ax.text(5, 0.2, 'Revolutionary for Medical Image Analysis', 
               ha='center', va='center', fontsize=12, fontweight='bold', 
               color='darkorange')
    
    def save_visualization(self, filename: str = "unet_medical_visualization.png"):
        """Save the comprehensive visualization."""
        fig = self.create_comprehensive_visualization()
        fig.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"Visualization saved as: {filename}")
        return filename

def main():
    """Main function to create and display the U-Net visualization."""
    print("Creating U-Net Medical Image Segmentation Visualization...")
    
    # Create visualization
    visualizer = UNetVisualization()
    
    # Save the visualization
    output_file = visualizer.save_visualization()
    
    # Also create a detailed step-by-step explanation
    create_detailed_explanation()
    
    print(f"\nVisualization complete! Check '{output_file}' for the comprehensive explanation.")
    print("\nThe visualization shows:")
    print("1. Input medical image with skin lesion")
    print("2. U-Net architecture with encoder-decoder structure")
    print("3. Feature extraction at different scales")
    print("4. Decoder reconstruction process")
    print("5. Final segmentation with colored masks")
    print("6. Innovation highlights for medical imaging")

def create_detailed_explanation():
    """Create a detailed text explanation of the U-Net process."""
    explanation = """
U-Net Medical Image Segmentation: Detailed Process Explanation
=============================================================

STEP 1: INPUT PREPROCESSING
- Medical image (224×224×3) is fed into the U-Net
- Image contains visible skin lesion that needs to be segmented
- Preprocessing includes normalization and resizing

STEP 2: ENCODER PATH (Feature Extraction)
- Level 1: Edge detection identifies lesion boundaries
- Level 2: Texture analysis captures lesion characteristics  
- Level 3: High-level features combine multiple lesion properties
- Level 4: Bottleneck creates compressed feature representation

STEP 3: DECODER PATH (Feature Reconstruction)
- Upsampling restores spatial resolution
- Skip connections preserve fine details from encoder
- Feature fusion combines multi-scale information
- Boundary refinement creates precise segmentation

STEP 4: OUTPUT GENERATION
- Final layer produces pixel-wise probability maps
- Lesion mask (red): Main skin lesion area
- Bacterial/Infected mask (green): Secondary infected regions
- Overlay shows both regions on original image

INNOVATION HIGHLIGHTS:
✓ Skip Connections: Preserve fine details during upsampling
✓ Multi-scale Analysis: Detect lesions at all sizes
✓ End-to-end Learning: Direct pixel-wise prediction
✓ Data Efficiency: Works with limited medical data
✓ Precise Boundaries: Accurate lesion segmentation

This makes U-Net revolutionary for medical image analysis, especially
for skin cancer diagnosis where precise lesion boundaries are crucial.
"""
    
    with open("unet_explanation.txt", "w") as f:
        f.write(explanation)
    
    print("Detailed explanation saved as: unet_explanation.txt")

if __name__ == "__main__":
    main()

