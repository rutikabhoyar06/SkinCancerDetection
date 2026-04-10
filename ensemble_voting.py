"""
Max Voting Ensemble Implementation for Multiple CNN Models
"""

import numpy as np
from typing import List, Tuple, Optional
import tensorflow as tf


class MaxVotingEnsemble:
    """
    Max Voting Ensemble that combines predictions from multiple models
    """
    
    def __init__(self, models: List[tf.keras.Model], weights: Optional[List[float]] = None):
        """
        Initialize ensemble with models and optional weights
        
        Args:
            models: List of trained Keras models
            weights: Optional weights for each model (if None, equal weights)
        """
        self.models = models
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def predict(self, x: np.ndarray, batch_size: int = 32, verbose: int = 0) -> np.ndarray:
        """
        Predict using max voting ensemble
        
        Args:
            x: Input images
            batch_size: Batch size for prediction
            verbose: Verbosity level
            
        Returns:
            Predicted class indices
        """
        all_predictions = []
        
        for model in self.models:
            pred_proba = model.predict(x, batch_size=batch_size, verbose=verbose)
            all_predictions.append(pred_proba)
        
        # Stack predictions: (num_models, num_samples, num_classes)
        stacked = np.stack(all_predictions, axis=0)
        
        # Get class predictions from each model
        class_predictions = np.argmax(stacked, axis=-1)  # (num_models, num_samples)
        
        # Max voting: for each sample, count votes for each class
        num_samples = class_predictions.shape[1]
        num_classes = stacked.shape[-1]
        ensemble_predictions = []
        
        for i in range(num_samples):
            # Get votes for this sample from all models
            votes = class_predictions[:, i]
            
            # Count votes for each class
            vote_counts = np.bincount(votes, minlength=num_classes)
            
            # Get class with maximum votes
            predicted_class = np.argmax(vote_counts)
            ensemble_predictions.append(predicted_class)
        
        return np.array(ensemble_predictions)
    
    def predict_proba(self, x: np.ndarray, batch_size: int = 32, verbose: int = 0) -> np.ndarray:
        """
        Predict probabilities using weighted average of model probabilities
        
        Args:
            x: Input images
            batch_size: Batch size for prediction
            verbose: Verbosity level
            
        Returns:
            Predicted probabilities for each class
        """
        all_predictions = []
        
        for model, weight in zip(self.models, self.weights):
            pred_proba = model.predict(x, batch_size=batch_size, verbose=verbose)
            all_predictions.append(pred_proba * weight)
        
        # Weighted average of probabilities
        ensemble_proba = np.sum(all_predictions, axis=0)
        return ensemble_proba
    
    def predict_with_confidence(self, x: np.ndarray, batch_size: int = 32, verbose: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence scores
        
        Args:
            x: Input images
            batch_size: Batch size for prediction
            verbose: Verbosity level
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        proba = self.predict_proba(x, batch_size=batch_size, verbose=verbose)
        predictions = np.argmax(proba, axis=1)
        confidence = np.max(proba, axis=1)
        
        return predictions, confidence


def max_vote_predictions(predictions_list: List[np.ndarray]) -> np.ndarray:
    """
    Simple max voting function for class predictions
    
    Args:
        predictions_list: List of prediction arrays, each of shape (num_samples,)
        
    Returns:
        Ensemble predictions of shape (num_samples,)
    """
    # Stack predictions: (num_models, num_samples)
    stacked = np.stack(predictions_list, axis=0)
    
    num_samples = stacked.shape[1]
    num_classes = int(np.max(stacked)) + 1
    
    ensemble_predictions = []
    
    for i in range(num_samples):
        votes = stacked[:, i]
        vote_counts = np.bincount(votes.astype(int), minlength=num_classes)
        predicted_class = np.argmax(vote_counts)
        ensemble_predictions.append(predicted_class)
    
    return np.array(ensemble_predictions)


def weighted_average_predictions(
    predictions_list: List[np.ndarray],
    weights: Optional[List[float]] = None
) -> np.ndarray:
    """
    Weighted average of probability predictions
    
    Args:
        predictions_list: List of probability arrays, each of shape (num_samples, num_classes)
        weights: Optional weights for each model
        
    Returns:
        Ensemble probabilities of shape (num_samples, num_classes)
    """
    if weights is None:
        weights = [1.0 / len(predictions_list)] * len(predictions_list)
    
    # Normalize weights
    total = sum(weights)
    weights = [w / total for w in weights]
    
    # Weighted average
    ensemble_proba = np.zeros_like(predictions_list[0])
    for pred, weight in zip(predictions_list, weights):
        ensemble_proba += pred * weight
    
    return ensemble_proba


























