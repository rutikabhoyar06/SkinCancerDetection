"""
Ensemble Model Builder for Skin Cancer Detection
Supports EfficientNet, ResNet, and DenseNet with transfer learning
"""

import os
import tensorflow as tf
from typing import Tuple, Optional
import numpy as np


def build_efficientnet_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    dropout: float = 0.4,
    base_trainable: bool = False,
    model_size: str = "B3"
) -> tf.keras.Model:
    """
    Build EfficientNet model with transfer learning
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        dropout: Dropout rate
        base_trainable: Whether base model is trainable
        model_size: EfficientNet variant (B0-B7)
    """
    # Map model size to EfficientNet variant
    efficientnet_map = {
        "B0": tf.keras.applications.EfficientNetB0,
        "B1": tf.keras.applications.EfficientNetB1,
        "B2": tf.keras.applications.EfficientNetB2,
        "B3": tf.keras.applications.EfficientNetB3,
        "B4": tf.keras.applications.EfficientNetB4,
        "B5": tf.keras.applications.EfficientNetB5,
        "B6": tf.keras.applications.EfficientNetB6,
        "B7": tf.keras.applications.EfficientNetB7,
    }
    
    efficientnet_class = efficientnet_map.get(model_size, tf.keras.applications.EfficientNetB3)
    
    try:
        base_model = efficientnet_class(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling=None
        )
    except Exception:
        # Fallback if ImageNet weights unavailable
        base_model = efficientnet_class(
            include_top=False,
            weights=None,
            input_shape=input_shape,
            pooling=None
        )
    
    base_model.trainable = base_trainable
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout * 0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs, name=f"EfficientNet{model_size}")
    return model


def build_resnet_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    dropout: float = 0.4,
    base_trainable: bool = False,
    model_type: str = "ResNet50"
) -> tf.keras.Model:
    """
    Build ResNet model with transfer learning
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        dropout: Dropout rate
        base_trainable: Whether base model is trainable
        model_type: ResNet variant (ResNet50, ResNet101, ResNet152)
    """
    resnet_map = {
        "ResNet50": tf.keras.applications.ResNet50,
        "ResNet101": tf.keras.applications.ResNet101,
        "ResNet152": tf.keras.applications.ResNet152,
    }
    
    resnet_class = resnet_map.get(model_type, tf.keras.applications.ResNet50)
    
    try:
        base_model = resnet_class(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling=None
        )
    except Exception:
        base_model = resnet_class(
            include_top=False,
            weights=None,
            input_shape=input_shape,
            pooling=None
        )
    
    base_model.trainable = base_trainable
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.applications.resnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout * 0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs, name=model_type)
    return model


def build_densenet_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    dropout: float = 0.4,
    base_trainable: bool = False,
    model_type: str = "DenseNet121"
) -> tf.keras.Model:
    """
    Build DenseNet model with transfer learning
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        dropout: Dropout rate
        base_trainable: Whether base model is trainable
        model_type: DenseNet variant (DenseNet121, DenseNet169, DenseNet201)
    """
    densenet_map = {
        "DenseNet121": tf.keras.applications.DenseNet121,
        "DenseNet169": tf.keras.applications.DenseNet169,
        "DenseNet201": tf.keras.applications.DenseNet201,
    }
    
    densenet_class = densenet_map.get(model_type, tf.keras.applications.DenseNet121)
    
    try:
        base_model = densenet_class(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling=None
        )
    except Exception:
        base_model = densenet_class(
            include_top=False,
            weights=None,
            input_shape=input_shape,
            pooling=None
        )
    
    base_model.trainable = base_trainable
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.applications.densenet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout * 0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs, name=model_type)
    return model


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 1e-3,
    optimizer: Optional[tf.keras.optimizers.Optimizer] = None
) -> None:
    """
    Compile model with optimizer and loss
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate
        optimizer: Custom optimizer (if None, uses Adam)
    """
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


def unfreeze_base_model(model: tf.keras.Model, unfreeze_ratio: float = 0.5) -> None:
    """
    Unfreeze base model layers for fine-tuning
    
    Args:
        model: Keras model
        unfreeze_ratio: Ratio of layers to unfreeze (0.0 to 1.0)
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            # Find base model (EfficientNet, ResNet, DenseNet)
            if any(name in layer.name.lower() for name in ["efficientnet", "resnet", "densenet"]):
                num_layers = len(layer.layers)
                unfreeze_start = int(num_layers * (1 - unfreeze_ratio))
                
                for i, l in enumerate(layer.layers):
                    if i >= unfreeze_start:
                        l.trainable = True
                    else:
                        l.trainable = False
                break


























