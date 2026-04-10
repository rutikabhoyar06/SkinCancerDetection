# Architecture Diagram - Quick Reference

## Generated Files

1. **architecture_diagram.png** - Visual architecture diagram (high-resolution PNG)
2. **ARCHITECTURE.md** - Comprehensive architecture documentation
3. **generate_architecture_diagram.py** - Python script to regenerate the diagram

## Viewing the Diagram

### Option 1: Direct View
Open `architecture_diagram.png` in any image viewer or browser.

### Option 2: Regenerate
To regenerate the diagram with modifications:
```bash
python generate_architecture_diagram.py
```

This will create:
- `architecture_diagram.png` (PNG format)
- `architecture_diagram.pdf` (PDF format, if dependencies available)

## Diagram Components

The architecture diagram shows:

1. **Data Layer** (Blue)
   - Dataset (Benign/Malignant images)
   - Optional segmentation masks

2. **Preprocessing Layer** (Light Blue)
   - Data loading and splitting
   - Data augmentation

3. **Model Layer** (Medium Blue)
   - Classification models (EfficientNet, ResNet, DenseNet)
   - Segmentation models (U-Net)
   - Ensemble models

4. **Training Layer** (Darker Blue)
   - Training pipelines
   - Advanced training techniques
   - Evaluation

5. **Inference Layer** (Bright Blue)
   - Streamlit web application
   - Interpretability (Grad-CAM)
   - Visualization

6. **Storage Layer** (Purple)
   - Model checkpoints
   - Results storage

## Architecture Overview

The system follows a modular architecture:

```
Data → Preprocessing → Models → Training → Evaluation → Inference
```

### Key Features:
- **Transfer Learning**: Uses pretrained models (ImageNet)
- **Two-Phase Training**: Frozen base → Fine-tuning
- **Ensemble Learning**: Multiple models combined
- **Interpretability**: Grad-CAM visualizations
- **Web Interface**: Streamlit app for predictions

## Documentation

For detailed architecture information, see **ARCHITECTURE.md** which includes:
- Detailed component descriptions
- Data flow diagrams
- Model architectures
- Training procedures
- Evaluation metrics
- Technology stack

## Usage

1. **View the diagram**: Open `architecture_diagram.png`
2. **Read documentation**: See `ARCHITECTURE.md`
3. **Regenerate**: Run `python generate_architecture_diagram.py`

## Dependencies

To regenerate the diagram, ensure you have:
- Python 3.7+
- matplotlib
- numpy

Install with:
```bash
pip install matplotlib numpy
```

## Notes

- The diagram uses color coding to distinguish different layers
- Arrows show data flow and component interactions
- Dashed lines indicate optional or indirect connections
- The diagram is designed to be printable and presentation-ready
















