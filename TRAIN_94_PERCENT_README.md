# Training for 94%+ Accuracy

This guide explains how to train models to achieve 94%+ accuracy for skin cancer detection.

## Quick Start

The easiest way to start training:

```bash
python run_94_percent_training.py
```

This will automatically use optimized settings to train an ensemble of models.

## What This Does

The training script uses several advanced techniques to achieve 94%+ accuracy:

1. **Larger Image Size**: Uses 384x384 images (instead of 224x224 or 300x300) for better feature extraction
2. **Enhanced Models**: Uses larger model variants:
   - EfficientNetB4 (instead of B3)
   - ResNet50
   - DenseNet201 (instead of DenseNet121)
3. **Extended Training**: 
   - 25 epochs with frozen base
   - 75 epochs for fine-tuning
4. **Advanced Augmentation**: Stronger data augmentation to improve generalization
5. **Test-Time Augmentation (TTA)**: Uses multiple augmented versions of test images for better predictions
6. **Weighted Ensemble**: Combines models with weights based on individual performance
7. **Class Weights**: Handles class imbalance automatically

## Manual Training

If you want to customize the training:

```bash
python train_94_percent_simple.py \
    --data_root dataset_split \
    --image_size 384 384 \
    --batch_size 12 \
    --epochs_frozen 25 \
    --epochs_finetune 75 \
    --lr_frozen 0.001 \
    --lr_finetune 0.00005 \
    --dropout 0.5 \
    --model_dir checkpoints_94
```

### Parameters

- `--data_root`: Path to dataset split directory (default: `dataset_split`)
- `--image_size`: Image dimensions (default: `384 384`)
- `--batch_size`: Batch size (default: `12`, adjust based on GPU memory)
- `--epochs_frozen`: Epochs for frozen base training (default: `25`)
- `--epochs_finetune`: Epochs for fine-tuning (default: `75`)
- `--lr_frozen`: Learning rate for frozen phase (default: `0.001`)
- `--lr_finetune`: Learning rate for fine-tuning (default: `0.00005`)
- `--dropout`: Dropout rate (default: `0.5`)
- `--model_dir`: Directory to save models (default: `checkpoints_94`)
- `--no_tta`: Disable test-time augmentation (not recommended)

## Expected Results

After training, you should see:

- **Individual Models**: 88-92% accuracy each
- **Ensemble**: 94%+ accuracy

## Training Time

- **Per Model**: 3-6 hours (depending on hardware)
- **All 3 Models**: 9-18 hours
- **Total with Evaluation**: 10-20 hours

## Output Files

After training completes, you'll find:

- `checkpoints_94/EfficientNetB4_best.keras` - Best EfficientNet model
- `checkpoints_94/ResNet50_best.keras` - Best ResNet model
- `checkpoints_94/DenseNet201_best.keras` - Best DenseNet model
- `checkpoints_94/ensemble/` - Final ensemble models
- `checkpoints_94/ensemble_metrics.json` - All metrics and results
- `checkpoints_94/ensemble_confusion_matrix.png` - Confusion matrix visualization

## Tips for Best Results

1. **GPU Memory**: If you get out-of-memory errors:
   - Reduce `--batch_size` to 8 or 6
   - Reduce `--image_size` to 320 320

2. **Higher Accuracy**: If you want even better results:
   - Increase `--image_size` to 448 448
   - Increase `--epochs_finetune` to 100
   - Add more models to the ensemble

3. **Faster Training**: If training is too slow:
   - Reduce `--epochs_frozen` to 15
   - Reduce `--epochs_finetune` to 50
   - Use smaller models (EfficientNetB3, DenseNet121)

## Troubleshooting

**Out of Memory?**
- Reduce batch size: `--batch_size 8`
- Use smaller images: `--image_size 320 320`

**Low Accuracy?**
- Train for more epochs
- Use larger image size
- Ensure data quality is good
- Check class balance in dataset

**Training Too Slow?**
- Use GPU if available
- Reduce image size
- Reduce number of epochs (but may affect accuracy)

## Using the Trained Model

After training, you can use the ensemble for inference:

```python
from ensemble_voting import MaxVotingEnsemble
import tensorflow as tf
import numpy as np

# Load models
models = []
for name in ["EfficientNetB4", "ResNet50", "DenseNet201"]:
    model = tf.keras.models.load_model(f"checkpoints_94/ensemble/{name}_final.keras")
    models.append(model)

# Create ensemble (with weights from training)
weights = [0.92, 0.90, 0.91]  # Update with actual weights
ensemble = MaxVotingEnsemble(models, weights=weights)

# Predict
predictions = ensemble.predict(images, batch_size=32)
```

## Next Steps

1. Run the training: `python run_94_percent_training.py`
2. Monitor progress in the console
3. Check results in `checkpoints_94/ensemble_metrics.json`
4. Use the trained models for inference

Good luck! 🚀



















