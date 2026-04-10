# Skin Cancer Detection System - Architecture Documentation

## Overview
This document provides a comprehensive overview of the architecture of the Skin Cancer Detection system, which classifies dermatoscopic images as Benign or Malignant using deep learning techniques.

## System Architecture

### 1. Data Layer

#### 1.1 Dataset Structure
```
dataset/
├── benign/        # 8,061 images
└── malignant/     # 1,954 images
```

#### 1.2 Data Loading (`data_utils.py`)
- **Function**: `load_dataset()`
- **Responsibilities**:
  - Load images from class directories
  - Resize images to standard size (224x224 or 300x300)
  - Normalize pixel values to [0, 1]
  - Split data into train/validation/test sets (70/15/15)
  - Optional: Load segmentation masks if available
- **Output**: Dictionary with 'train', 'val', 'test' splits containing images, labels, and optional masks

### 2. Preprocessing Layer

#### 2.1 Data Augmentation (`augmentations.py`)
- **Real-time augmentations**:
  - Rotation (±30°)
  - Horizontal/Vertical flips
  - Zoom (0.9-1.1x)
  - Brightness adjustment
  - Contrast adjustment
- **Purpose**: Increase dataset diversity and improve model generalization
- **Application**: Applied during training, not during validation/test

### 3. Model Layer

#### 3.1 Classification Models

##### 3.1.1 EfficientNet-based Classifier (`classifier.py`)
- **Base Model**: EfficientNetB0 (ImageNet pretrained)
- **Architecture**:
  - Input: (224, 224, 3)
  - EfficientNetB0 encoder (frozen or trainable)
  - Global Average Pooling
  - Dropout (0.3)
  - Dense layer (2 classes) with softmax activation
- **Output**: Probability distribution [p_benign, p_malignant]

##### 3.1.2 Ensemble Models (`ensemble_models.py`)
Supports multiple architectures for ensemble learning:

**EfficientNet Models** (B0-B7):
- Transfer learning from ImageNet
- Configurable model size (B0 to B7)
- Custom classification head with dropout

**ResNet Models** (50/101/152):
- Residual network architecture
- ImageNet pretrained weights
- Deep feature extraction

**DenseNet Models** (121/169/201):
- Dense connection pattern
- Feature reuse
- Efficient parameter usage

#### 3.2 Segmentation Models (`unet.py`)

##### 3.2.1 Standard U-Net
- **Architecture**:
  - Encoder: 4 down-sampling blocks (32, 64, 128, 256 filters)
  - Bridge: Bottleneck with 512 filters
  - Decoder: 4 up-sampling blocks with skip connections
  - Output: Binary segmentation mask (sigmoid activation)
- **Loss Function**: BCE + Dice loss (hybrid)
- **Purpose**: Lesion boundary detection

##### 3.2.2 U-Net with EfficientNet Encoder
- **Encoder**: EfficientNet (B0-B3) pretrained on ImageNet
- **Decoder**: Custom upsampling blocks
- **Skip Connections**: From EfficientNet feature maps
- **Advantage**: Better feature extraction with transfer learning

#### 3.3 Ensemble Learning (`ensemble_voting.py`)

##### 3.3.1 Max Voting Ensemble
- **Method**: Combines predictions from multiple models
- **Voting**: Each model votes for a class, majority wins
- **Usage**: Improves robustness and accuracy

##### 3.3.2 Weighted Average Ensemble
- **Method**: Weighted combination of probability predictions
- **Weights**: Can be based on individual model performance
- **Output**: Smoothed probability distribution

### 4. Training Layer

#### 4.1 Training Pipeline (`train_classifier.py`)

##### Phase 1: Frozen Base Training
- **Base Model**: Frozen (non-trainable)
- **Learning Rate**: 1e-3
- **Epochs**: 2-10 (configurable)
- **Purpose**: Train classification head only
- **Optimizer**: Adam

##### Phase 2: Fine-tuning
- **Base Model**: Unfrozen (trainable)
- **Learning Rate**: 1e-4 (lower)
- **Epochs**: 3-50 (configurable)
- **Purpose**: Fine-tune entire network
- **Optimizer**: Adam with learning rate scheduling

##### Training Features:
- Class weight balancing (handles imbalanced dataset)
- Early stopping (prevents overfitting)
- Model checkpointing (saves best model)
- Learning rate reduction on plateau
- Data augmentation during training

#### 4.2 Advanced Training (`advanced_transfer_learning.py`)
- Two-phase training with advanced techniques
- Advanced augmentation pipeline
- Custom learning rate schedules
- Advanced classification head architectures

#### 4.3 Ensemble Training (`train_ensemble.py`, `train_ensemble_tf.py`)
- Train multiple models independently
- Supports EfficientNet, ResNet, DenseNet
- Individual model evaluation
- Ensemble combination strategies

#### 4.4 High-Accuracy Training (`train_94_percent.py`)
- Optimized for 94%+ accuracy
- Larger image sizes (384x384)
- Extended training epochs
- Test-time augmentation (TTA)
- Advanced optimization techniques

### 5. Evaluation Layer

#### 5.1 Metrics (`evaluation.py`, `eval_metrics.py`)
- **Classification Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - Confusion matrix
- **Segmentation Metrics**:
  - Pixel-wise accuracy
  - Dice coefficient
  - IoU (Intersection over Union)
  - Precision/Recall

#### 5.2 Visualization
- Training curves (loss, accuracy)
- Confusion matrices
- ROC curves
- Performance comparisons
- Ablation study results

### 6. Inference Layer

#### 6.1 Streamlit Web Application (`app.py`)
- **Features**:
  - Image upload interface
  - Real-time classification
  - Probability display
  - Grad-CAM visualization
  - Segmentation overlay
  - Lesion boundary visualization
- **Models Used**:
  - Classification model (EfficientNet-based)
  - Segmentation model (U-Net, optional)
- **Output**: Predicted class, probabilities, heatmaps, boundaries

#### 6.2 Interpretability (`interpretability.py`)
- **Grad-CAM**: Gradient-weighted Class Activation Mapping
- **Purpose**: Visualize which regions of the image influence the prediction
- **Output**: Heatmap overlay on original image
- **Usage**: Explain model decisions to users

### 7. Storage Layer

#### 7.1 Model Checkpoints
- **Location**: `checkpoints/`
- **Files**:
  - `bm_classifier_best.keras`: Best classification model
  - `unet_best.keras`: Best segmentation model (if trained)
  - Training history JSON files
  - Validation metrics

#### 7.2 Results Storage
- **Evaluation Results**: `evaluation_results/`
- **Visualization Results**: `visualization_results/`
- **Ablation Study**: `ablation_study_results/`

### 8. Data Flow

#### 8.1 Training Flow
```
Dataset → Data Loading → Augmentation → Model Training → Evaluation → Checkpoints
```

#### 8.2 Inference Flow
```
User Image → Preprocessing → Classification Model → Prediction
                                      ↓
                            Segmentation Model (optional)
                                      ↓
                            Interpretability (Grad-CAM)
                                      ↓
                            Visualization & Results
```

### 9. Key Components Interaction

1. **Data Pipeline**: `data_utils.py` → loads and preprocesses data
2. **Augmentation**: `augmentations.py` → applies transformations
3. **Models**: `classifier.py`, `unet.py`, `ensemble_models.py` → define architectures
4. **Training**: `train_classifier.py`, `train_ensemble.py` → train models
5. **Evaluation**: `evaluation.py` → compute metrics
6. **Inference**: `app.py` → serve predictions via web interface
7. **Interpretability**: `interpretability.py` → generate explanations

### 10. Technology Stack

- **Framework**: TensorFlow/Keras
- **Web Framework**: Streamlit
- **Image Processing**: OpenCV, PIL
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **ML Utilities**: scikit-learn

### 11. Model Performance

- **Classification Accuracy**: 94%+ (with ensemble)
- **Individual Models**: 90-93% accuracy
- **Ensemble**: Combines multiple models for improved performance
- **Segmentation**: Dice coefficient >0.85 (if masks available)

### 12. Deployment

#### 12.1 Local Deployment
```bash
streamlit run app.py
```

#### 12.2 Model Loading
- Models are loaded from checkpoints directory
- Automatic fallback to untrained models if checkpoints not found
- Caching for improved performance

### 13. Extensibility

The architecture supports:
- Adding new model architectures
- Custom loss functions
- Additional augmentation techniques
- New evaluation metrics
- Multiple ensemble strategies
- Integration with other frameworks

## File Structure

```
SkinCancerDetection/
├── app.py                          # Streamlit web application
├── classifier.py                   # Classification model builder
├── unet.py                         # Segmentation model
├── ensemble_models.py              # Ensemble model builders
├── ensemble_voting.py              # Ensemble voting strategies
├── data_utils.py                   # Data loading utilities
├── augmentations.py                # Data augmentation
├── interpretability.py             # Grad-CAM implementation
├── train_classifier.py             # Classification training
├── train_ensemble.py               # Ensemble training
├── evaluation.py                   # Evaluation metrics
├── checkpoints/                    # Saved models
├── dataset/                        # Training data
├── evaluation_results/             # Evaluation outputs
└── visualization_results/          # Visualization outputs
```

## Conclusion

The Skin Cancer Detection system employs a modular architecture that separates concerns across data processing, model definition, training, evaluation, and inference. The use of transfer learning, ensemble methods, and interpretability techniques ensures both high accuracy and explainability, making it suitable for medical imaging applications.
















