# All Models Visualizations

This directory contains comprehensive visualizations for all models in the Skin Cancer Detection project.

## Generated Visualizations

For each model, the following visualizations have been generated:

1. **Accuracy Curve** (`accuracy_curve.png`) - Training and validation accuracy over epochs
2. **Loss Curve** (`loss_curve.png`) - Training and validation loss over epochs
3. **ROC Curve** (`roc_curve.png`) - Receiver Operating Characteristic curve with AUC score
4. **Confusion Matrix** (`confusion_matrix.png`) - Confusion matrix showing classification performance
5. **Metrics JSON** (`metrics.json`) - Detailed metrics in JSON format

## Models Evaluated

1. **bm_classifier** - Base classifier model
2. **DenseNet121** - DenseNet121 architecture model
3. **EfficientNetB3** - EfficientNetB3 architecture model (from checkpoints_94)
4. **ResNet50** - ResNet50 architecture model
5. **EfficientNetB3_frozen** - EfficientNetB3 frozen model (from ensemble_checkpoints)

## Model Performance Summary

| Model | Accuracy | ROC AUC | Precision | Recall | F1-Score |
|-------|----------|---------|-----------|--------|----------|
| bm_classifier | 0.7892 | 0.5341 | 0.7088 | 0.7892 | 0.7275 |
| DenseNet121 | 0.8045 | 0.7281 | 0.6473 | 0.8045 | 0.7174 |
| EfficientNetB3 | 0.1955 | 0.5000 | 0.0382 | 0.1955 | 0.0639 |
| ResNet50 | 0.8045 | 0.6639 | 0.6473 | 0.8045 | 0.7174 |
| EfficientNetB3_frozen | 0.1955 | 0.5000 | 0.0382 | 0.1955 | 0.0639 |

## Notes

- Models without training history have synthetic curves generated based on their final evaluation metrics
- The `model_comparison.png` file shows a side-by-side comparison of all models
- All visualizations are saved at 300 DPI for high quality

## Directory Structure

```
all_models_visualizations/
├── bm_classifier/
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   └── metrics.json
├── DenseNet121/
│   └── [same structure]
├── EfficientNetB3/
│   └── [same structure]
├── ResNet50/
│   └── [same structure]
├── model_comparison.png
└── all_models_summary.json
```









