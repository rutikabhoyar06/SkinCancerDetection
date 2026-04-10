from typing import Tuple
import tensorflow as tf


def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
	"""Compute Dice coefficient for binary masks.
	Assumes y_true and y_pred are probabilities in [0,1].
	"""
	y_true = tf.cast(y_true, tf.float32)
	y_pred = tf.cast(y_pred, tf.float32)
	# Flatten over spatial dims
	y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
	y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
	intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
	sums = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
	dice = (2.0 * intersection + smooth) / (sums + smooth)
	return tf.reduce_mean(dice)


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
	return 1.0 - dice_coefficient(y_true, y_pred, smooth=smooth)


def bce_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
	bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
	return bce(y_true, y_pred)


def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, dice_weight: float = 1.0, smooth: float = 1.0) -> tf.Tensor:
	"""Combined BCE + Dice loss.
	`dice_weight` scales the dice component; set to 1.0 typically.
	"""
	return bce_loss(y_true, y_pred) + dice_weight * dice_loss(y_true, y_pred, smooth=smooth)

