from typing import Optional
import tensorflow as tf


def create_lr_schedule(
	initial_lr: float = 1e-3,
	decay_steps: int = 1000,
	decay_rate: float = 0.96,
	staircase: bool = True,
) -> tf.keras.optimizers.schedules.LearningRateSchedule:
	"""Exponential decay LR schedule."""
	return tf.keras.optimizers.schedules.ExponentialDecay(
		initial_learning_rate=initial_lr,
		decay_steps=decay_steps,
		decay_rate=decay_rate,
		staircase=staircase,
	)


def create_adam_optimizer(
	lr_or_schedule: Optional[tf.keras.optimizers.schedules.LearningRateSchedule] = None,
	beta_1: float = 0.9,
	beta_2: float = 0.999,
	epsilon: float = 1e-7,
) -> tf.keras.optimizers.Optimizer:
	"""Create Adam optimizer with either a float LR or a schedule."""
	if lr_or_schedule is None:
		lr_or_schedule = 1e-3
	return tf.keras.optimizers.Adam(
		learning_rate=lr_or_schedule,
		beta_1=beta_1,
		beta_2=beta_2,
		epsilon=epsilon,
	)

