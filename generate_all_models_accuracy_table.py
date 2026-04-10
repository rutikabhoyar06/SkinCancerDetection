"""
Generate a comprehensive table of all model accuracies
"""

import json
import os
from pathlib import Path

# Collect all model metrics
models_data = []

# Base directory
base_dir = "all_models_visualizations"

# Models to check
model_dirs = [
    "bm_classifier",
    "DenseNet121",
    "EfficientNetB3",
    "ResNet50",
    "Ensemble",
    "MaxVotingEnsemble_94",
    "UNET"
]

# Load metrics from each model
for model_dir in model_dirs:
    metrics_path = os.path.join(base_dir, model_dir, "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                
            model_name = metrics.get('model_name', model_dir)
            accuracy = metrics.get('accuracy', 0.0)
            precision = metrics.get('precision', 0.0)
            recall = metrics.get('recall', 0.0)
            f1_score = metrics.get('f1_score', metrics.get('dice', 0.0))  # UNET uses dice instead of f1
            roc_auc = metrics.get('roc_auc', 0.0)
            
            # For UNET, also include dice and IoU
            dice = metrics.get('dice', None)
            iou = metrics.get('iou', None)
            
            models_data.append({
                'name': model_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'roc_auc': roc_auc,
                'dice': dice,
                'iou': iou
            })
        except Exception as e:
            print(f"Error loading {metrics_path}: {e}")

# Sort by accuracy (descending)
models_data.sort(key=lambda x: x['accuracy'], reverse=True)

# Generate markdown table
print("\n" + "="*100)
print("ALL MODELS ACCURACY TABLE")
print("="*100 + "\n")

# Markdown table header
print("| Model Name | Accuracy (%) | Precision | Recall | F1-Score | ROC-AUC | Dice | IoU |")
print("|" + "-"*12 + "|" + "-"*14 + "|" + "-"*11 + "|" + "-"*8 + "|" + "-"*10 + "|" + "-"*8 + "|" + "-"*6 + "|" + "-"*5 + "|")

# Print each model
for model in models_data:
    name = model['name']
    acc = model['accuracy'] * 100
    prec = model['precision'] * 100 if model['precision'] > 0 else "N/A"
    rec = model['recall'] * 100 if model['recall'] > 0 else "N/A"
    f1 = model['f1_score'] * 100 if model['f1_score'] > 0 else "N/A"
    roc = model['roc_auc'] * 100 if model['roc_auc'] > 0 else "N/A"
    dice = model['dice'] * 100 if model['dice'] is not None else "N/A"
    iou = model['iou'] * 100 if model['iou'] is not None else "N/A"
    
    # Format values
    acc_str = f"{acc:.2f}"
    prec_str = f"{prec:.2f}" if isinstance(prec, float) else str(prec)
    rec_str = f"{rec:.2f}" if isinstance(rec, float) else str(rec)
    f1_str = f"{f1:.2f}" if isinstance(f1, float) else str(f1)
    roc_str = f"{roc:.2f}" if isinstance(roc, float) else str(roc)
    dice_str = f"{dice:.2f}" if isinstance(dice, float) else str(dice)
    iou_str = f"{iou:.2f}" if isinstance(iou, float) else str(iou)
    
    print(f"| {name:<10} | {acc_str:>12} | {prec_str:>9} | {rec_str:>6} | {f1_str:>8} | {roc_str:>6} | {dice_str:>4} | {iou_str:>3} |")

print("\n" + "="*100)

# Save to file
output_file = os.path.join(base_dir, "all_models_accuracy_table.txt")
with open(output_file, 'w') as f:
    f.write("="*100 + "\n")
    f.write("ALL MODELS ACCURACY TABLE\n")
    f.write("="*100 + "\n\n")
    f.write("| Model Name | Accuracy (%) | Precision | Recall | F1-Score | ROC-AUC | Dice | IoU |\n")
    f.write("|" + "-"*12 + "|" + "-"*14 + "|" + "-"*11 + "|" + "-"*8 + "|" + "-"*10 + "|" + "-"*8 + "|" + "-"*6 + "|" + "-"*5 + "|\n")
    
    for model in models_data:
        name = model['name']
        acc = model['accuracy'] * 100
        prec = model['precision'] * 100 if model['precision'] > 0 else "N/A"
        rec = model['recall'] * 100 if model['recall'] > 0 else "N/A"
        f1 = model['f1_score'] * 100 if model['f1_score'] > 0 else "N/A"
        roc = model['roc_auc'] * 100 if model['roc_auc'] > 0 else "N/A"
        dice = model['dice'] * 100 if model['dice'] is not None else "N/A"
        iou = model['iou'] * 100 if model['iou'] is not None else "N/A"
        
        acc_str = f"{acc:.2f}"
        prec_str = f"{prec:.2f}" if isinstance(prec, float) else str(prec)
        rec_str = f"{rec:.2f}" if isinstance(rec, float) else str(rec)
        f1_str = f"{f1:.2f}" if isinstance(f1, float) else str(f1)
        roc_str = f"{roc:.2f}" if isinstance(roc, float) else str(roc)
        dice_str = f"{dice:.2f}" if isinstance(dice, float) else str(dice)
        iou_str = f"{iou:.2f}" if isinstance(iou, float) else str(iou)
        
        f.write(f"| {name:<10} | {acc_str:>12} | {prec_str:>9} | {rec_str:>6} | {f1_str:>8} | {roc_str:>6} | {dice_str:>4} | {iou_str:>3} |\n")
    
    f.write("\n" + "="*100 + "\n")

print(f"\nTable saved to: {output_file}")

# Also create a visual table using matplotlib
try:
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Prepare data for visualization
    model_names = [m['name'] for m in models_data]
    accuracies = [m['accuracy'] * 100 for m in models_data]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # Bar chart
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    bars = ax1.barh(model_names, accuracies, color=colors)
    ax1.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlim([0, 100])
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        width = bar.get_width()
        ax1.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{acc:.2f}%', ha='left', va='center', fontweight='bold')
    
    # Table visualization
    ax2.axis('tight')
    ax2.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Model', 'Accuracy (%)', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    for model in models_data:
        row = [
            model['name'],
            f"{model['accuracy']*100:.2f}%",
            f"{model['precision']*100:.2f}%" if model['precision'] > 0 else "N/A",
            f"{model['recall']*100:.2f}%" if model['recall'] > 0 else "N/A",
            f"{model['f1_score']*100:.2f}%" if model['f1_score'] > 0 else "N/A",
            f"{model['roc_auc']*100:.2f}%" if model['roc_auc'] > 0 else "N/A"
        ]
        table_data.append(row)
    
    table = ax2.table(cellText=table_data, colLabels=headers,
                      cellLoc='center', loc='center',
                      colWidths=[0.25, 0.15, 0.12, 0.12, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax2.set_title('All Models Performance Metrics', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_image = os.path.join(base_dir, "all_models_accuracy_table.png")
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visual table saved to: {output_image}")
    
except ImportError:
    print("matplotlib not available, skipping visual table generation")

print("\n" + "="*100)


