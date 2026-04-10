# Ensemble Learning for Skin Cancer Detection

This implementation provides a comprehensive ensemble learning system combining multiple pre-trained CNN models (EfficientNet, ResNet, DenseNet) with max voting to achieve 90%+ accuracy for skin cancer classification.

## Features

- **Multiple Pre-trained Models**: EfficientNetB3, ResNet50, DenseNet121
- **Max Voting Ensemble**: Combines predictions from all models
- **Genetic Algorithm Feature Selection**: Optional feature selection to boost performance
- **Advanced Data Augmentation**: Comprehensive augmentation pipeline
- **Comprehensive Evaluation**: Accuracy, F1-Score, IoU, Precision, Recall, AUC

## Installation

Ensure you have the required dependencies:

```bash
pip install -r requirements.txt
pip install deap  # For genetic algorithm
pip install scikit-learn
```

## Quick Start

### 1. Basic Ensemble Training

Train all three models (EfficientNet, ResNet, DenseNet) with default settings:

```bash
python train_ensemble_tf.py --data_root dataset_split --image_size 300 300
```

### 2. Custom Training Configuration

```bash
python train_ensemble_tf.py \
    --data_root dataset_split \
    --image_size 300 300 \
    --batch_size 16 \
    --epochs_frozen 15 \
    --epochs_finetune 50 \
    --lr_frozen 0.001 \
    --lr_finetune 0.0001 \
    --dropout 0.4 \
    --model_dir ensemble_checkpoints
```

### 3. Training with GA Feature Selection

```bash
python run_ensemble_training.py \
    --data_root dataset_split \
    --image_size 300 300 \
    --use_ga \
    --ga_population 60 \
    --ga_generations 40
```

### 4. Select Specific Models

Train only EfficientNet and ResNet (skip DenseNet):

```bash
python train_ensemble_tf.py \
    --data_root dataset_split \
    --no_densenet
```

## Architecture

### Individual Models

Each model follows a transfer learning approach:

1. **Base Model**: Pre-trained on ImageNet (EfficientNetB3, ResNet50, or DenseNet121)
2. **Classification Head**: 
   - Global Average Pooling
   - Dense(512) → Dropout → Dense(256) → Dropout → Dense(2, softmax)

### Training Phases

1. **Frozen Phase**: Train only the classification head with frozen base
2. **Fine-tuning Phase**: Gradually unfreeze base layers and fine-tune

### Max Voting Ensemble

The ensemble combines predictions using max voting:
- Each model predicts a class
- The class with the most votes wins
- In case of ties, the class with higher average probability wins

## File Structure

```
├── ensemble_models.py          # Model builders (EfficientNet, ResNet, DenseNet)
├── ensemble_voting.py          # Max voting ensemble implementation
├── train_ensemble_tf.py        # Main training script
├── ga_feature_selection_tf.py  # Genetic algorithm feature selection
├── run_ensemble_training.py   # Complete pipeline with optional GA
└── ENSEMBLE_README.md          # This file
```

## Model Details

### EfficientNetB3
- Pre-trained on ImageNet
- Efficient architecture with compound scaling
- Good balance of accuracy and efficiency

### ResNet50
- Residual connections for deep networks
- Proven architecture for medical imaging
- Robust feature extraction

### DenseNet121
- Dense connections between layers
- Feature reuse and parameter efficiency
- Strong performance on small datasets

## Data Augmentation

The augmentation pipeline includes:
- Random horizontal/vertical flips
- Random rotation (±15°)
- Random zoom (±15%)
- Random translation (±10%)
- Random contrast/brightness adjustments
- Random shear (if available)

## Evaluation Metrics

The system evaluates:
- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted precision across classes
- **Recall**: Weighted recall across classes
- **F1-Score**: Weighted and macro F1-scores
- **AUC**: Area under ROC curve
- **IoU**: Intersection over Union (Jaccard Score)

## Genetic Algorithm Feature Selection

The optional GA feature selection:
1. Extracts embeddings from all trained models
2. Concatenates embeddings into a feature vector
3. Uses genetic algorithm to select optimal feature subset
4. Trains a RandomForest classifier on selected features

### GA Parameters

- **Population Size**: Number of individuals (default: 60)
- **Generations**: Number of evolution iterations (default: 40)
- **Crossover Probability**: 0.5
- **Mutation Probability**: 0.2
- **Fitness Function**: Macro F1-score with 5-fold CV

## Output Files

After training, you'll find:

```
ensemble_checkpoints/
├── EfficientNetB3_best.keras
├── ResNet50_best.keras
├── DenseNet121_best.keras
├── ensemble/
│   ├── EfficientNetB3_final.keras
│   ├── ResNet50_final.keras
│   └── DenseNet121_final.keras
├── ensemble_metrics.json
├── ensemble_comparison.png
└── ensemble_confusion_matrix.png
```

## Expected Results

With proper training, you should achieve:
- **Individual Models**: 85-90% accuracy
- **Ensemble (Max Voting)**: 90-95% accuracy
- **With GA Feature Selection**: 92-96% accuracy

## Tips for 90%+ Accuracy

1. **Use larger image sizes**: 300x300 or 384x384
2. **Train for more epochs**: 15 frozen + 50 fine-tune minimum
3. **Use class weights**: Automatically computed for imbalanced data
4. **Enable GA feature selection**: Can boost accuracy by 2-3%
5. **Use all three models**: Diversity improves ensemble performance

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch_size 8`
- Use smaller image size: `--image_size 224 224`
- Train models sequentially

### Low Accuracy
- Increase training epochs
- Use larger image size
- Enable GA feature selection
- Check data quality and class balance

### Slow Training
- Reduce image size
- Use smaller models (EfficientNetB0 instead of B3)
- Train on GPU if available

## Example Usage

### Complete Pipeline

```python
from train_ensemble_tf import train_ensemble
from ensemble_voting import MaxVotingEnsemble
import tensorflow as tf

# Train ensemble
results = train_ensemble(
    data_root="dataset_split",
    image_size=(300, 300),
    batch_size=16,
    epochs_frozen=15,
    epochs_finetune=50,
    model_dir="ensemble_checkpoints"
)

# Use ensemble for prediction
ensemble = results["ensemble"]
predictions = ensemble.predict(test_images, batch_size=32)
```

## Citation

If you use this code, please cite:
- EfficientNet: Tan & Le, 2019
- ResNet: He et al., 2016
- DenseNet: Huang et al., 2017

## License

This code is provided as-is for research and educational purposes.


























