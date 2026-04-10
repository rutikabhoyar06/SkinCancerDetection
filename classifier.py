from typing import Tuple
import tensorflow as tf


def build_classifier(
	input_shape: Tuple[int, int, int] = (224, 224, 3),
	base_trainable: bool = False,
	dropout: float = 0.3,
) -> tf.keras.Model:
	"""Build a transfer-learning classifier using EfficientNetB0.
	Binary output: [p_benign, p_malignant] via softmax.
	"""
	# Try to use ImageNet weights; fall back to random weights if download/cache is unavailable
	try:
		base = tf.keras.applications.EfficientNetB0(
			include_top=False, input_shape=input_shape, weights="imagenet"
		)
	except Exception:
		base = tf.keras.applications.EfficientNetB0(
			include_top=False, input_shape=input_shape, weights=None
		)
	base.trainable = base_trainable

	inputs = tf.keras.layers.Input(shape=input_shape)
	x = tf.keras.applications.efficientnet.preprocess_input(inputs)
	x = base(x, training=False)
	x = tf.keras.layers.GlobalAveragePooling2D()(x)
	if dropout and dropout > 0.0:
		x = tf.keras.layers.Dropout(dropout)(x)
	outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
	model = tf.keras.Model(inputs, outputs, name="BMClassifier")
	return model


def compile_classifier(model: tf.keras.Model, learning_rate: float = 1e-3) -> None:
	opt = tf.keras.optimizers.Adam(learning_rate)
	model.compile(
		optimizer=opt,
		loss="sparse_categorical_crossentropy",
		metrics=["accuracy"],
	)
