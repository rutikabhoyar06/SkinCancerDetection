"""
Genetic Algorithm Feature Selection for Ensemble Embeddings (TensorFlow version)
Extracts embeddings from trained CNN models and uses GA to select optimal features
"""

import os
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional, Dict
from deap import base, creator, tools, algorithms
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib


def extract_embeddings(
    models: List[tf.keras.Model],
    dataset: tf.data.Dataset,
    layer_name: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings from multiple models
    
    Args:
        models: List of trained Keras models
        dataset: TensorFlow dataset
        layer_name: Name of layer to extract from (if None, uses second-to-last layer)
        
    Returns:
        Tuple of (embeddings, labels)
    """
    embeddings_list = []
    labels_list = []
    
    for model in models:
        # Create a model that outputs embeddings (remove last classification layer)
        if layer_name:
            embedding_model = tf.keras.Model(
                inputs=model.input,
                outputs=model.get_layer(layer_name).output
            )
        else:
            # Get second-to-last layer (before classification head)
            embedding_model = tf.keras.Model(
                inputs=model.input,
                outputs=model.layers[-2].output
            )
        
        model_embeddings = []
        model_labels = []
        
        for batch_x, batch_y in dataset:
            emb = embedding_model.predict(batch_x, verbose=0)
            model_embeddings.append(emb)
            model_labels.extend(batch_y.numpy().tolist())
        
        embeddings_list.append(np.concatenate(model_embeddings, axis=0))
        labels_list = model_labels  # Labels are same for all models
    
    # Concatenate embeddings from all models
    combined_embeddings = np.concatenate(embeddings_list, axis=1)
    labels = np.array(labels_list)
    
    return combined_embeddings, labels


def evaluate_individual(
    individual: List[int],
    features: np.ndarray,
    labels: np.ndarray,
    cv_folds: int = 5
) -> Tuple[float]:
    """
    Evaluate fitness of an individual (feature subset)
    
    Args:
        individual: Binary list indicating which features to use
        features: Full feature matrix
        labels: Class labels
        cv_folds: Number of cross-validation folds
        
    Returns:
        Fitness score (macro F1)
    """
    mask = np.array(individual, dtype=bool)
    
    # Check if at least one feature is selected
    if mask.sum() == 0:
        return (0.0,)
    
    # Select features
    X_selected = features[:, mask]
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X_selected, labels):
        # Use RandomForest as classifier
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_selected[train_idx], labels[train_idx])
        preds = clf.predict(X_selected[val_idx])
        scores.append(f1_score(labels[val_idx], preds, average="macro"))
    
    return (float(np.mean(scores)),)


def run_genetic_algorithm(
    features: np.ndarray,
    labels: np.ndarray,
    population_size: int = 60,
    generations: int = 40,
    crossover_prob: float = 0.5,
    mutation_prob: float = 0.2,
    cv_folds: int = 5
) -> Tuple[np.ndarray, float]:
    """
    Run genetic algorithm for feature selection
    
    Args:
        features: Feature matrix (num_samples, num_features)
        labels: Class labels (num_samples,)
        population_size: Size of GA population
        generations: Number of GA generations
        crossover_prob: Crossover probability
        mutation_prob: Mutation probability
        cv_folds: Number of CV folds for evaluation
        
    Returns:
        Tuple of (selected_feature_mask, best_fitness_score)
    """
    num_features = features.shape[1]
    
    # Create DEAP types
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    # Create toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.randint, 0, 2)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register evaluation function
    toolbox.register("evaluate", evaluate_individual, features=features, labels=labels, cv_folds=cv_folds)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create initial population
    population = toolbox.population(n=population_size)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    print(f"\n🧬 Running Genetic Algorithm...")
    print(f"   Population: {population_size}, Generations: {generations}")
    print(f"   Features: {num_features}")
    
    # Run evolution
    for gen in range(generations):
        # Select and clone next generation
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < crossover_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Apply mutation
        for mutant in offspring:
            if np.random.random() < mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace population
        population[:] = offspring
        
        # Print progress
        fits = [ind.fitness.values[0] for ind in population]
        if (gen + 1) % 5 == 0:
            print(f"   Generation {gen + 1}/{generations}: Best F1 = {max(fits):.4f}, Mean F1 = {np.mean(fits):.4f}")
    
    # Get best individual
    best_ind = tools.selBest(population, k=1)[0]
    best_fitness = best_ind.fitness.values[0]
    best_mask = np.array(best_ind, dtype=bool)
    
    print(f"\n✅ GA Complete!")
    print(f"   Best F1-Score: {best_fitness:.4f}")
    print(f"   Selected Features: {best_mask.sum()}/{num_features} ({best_mask.sum()/num_features*100:.1f}%)")
    
    return best_mask, best_fitness


def apply_feature_selection(
    models: List[tf.keras.Model],
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    output_dir: str = "ga_results",
    population_size: int = 60,
    generations: int = 40
) -> Dict:
    """
    Apply genetic algorithm feature selection to ensemble embeddings
    
    Args:
        models: List of trained models
        train_ds: Training dataset
        val_ds: Validation dataset
        test_ds: Test dataset
        output_dir: Directory to save results
        population_size: GA population size
        generations: Number of GA generations
        
    Returns:
        Dictionary with selected features and results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("GENETIC ALGORITHM FEATURE SELECTION")
    print("="*80)
    
    # Extract embeddings from training data
    print("\n📊 Extracting embeddings from models...")
    train_embeddings, train_labels = extract_embeddings(models, train_ds)
    val_embeddings, val_labels = extract_embeddings(models, val_ds)
    test_embeddings, test_labels = extract_embeddings(models, test_ds)
    
    print(f"   Training embeddings shape: {train_embeddings.shape}")
    print(f"   Validation embeddings shape: {val_embeddings.shape}")
    print(f"   Test embeddings shape: {test_embeddings.shape}")
    
    # Run genetic algorithm
    selected_mask, best_fitness = run_genetic_algorithm(
        train_embeddings,
        train_labels,
        population_size=population_size,
        generations=generations
    )
    
    # Apply selected features
    train_selected = train_embeddings[:, selected_mask]
    val_selected = val_embeddings[:, selected_mask]
    test_selected = test_embeddings[:, selected_mask]
    
    # Train final classifier on selected features
    print("\n🎯 Training classifier on selected features...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(train_selected, train_labels)
    
    # Evaluate
    val_pred = clf.predict(val_selected)
    test_pred = clf.predict(test_selected)
    
    val_f1 = f1_score(val_labels, val_pred, average="macro")
    test_f1 = f1_score(test_labels, test_pred, average="macro")
    test_acc = np.mean(test_pred == test_labels)
    
    print(f"\n✅ Results with GA-selected features:")
    print(f"   Validation F1: {val_f1:.4f}")
    print(f"   Test F1: {test_f1:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Save results
    results = {
        "selected_mask": selected_mask.tolist(),
        "num_features_selected": int(selected_mask.sum()),
        "total_features": int(selected_mask.shape[0]),
        "best_fitness": float(best_fitness),
        "val_f1": float(val_f1),
        "test_f1": float(test_f1),
        "test_accuracy": float(test_acc)
    }
    
    with open(os.path.join(output_dir, "ga_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Save mask
    np.save(os.path.join(output_dir, "selected_features.npy"), selected_mask)
    joblib.dump(clf, os.path.join(output_dir, "ga_classifier.joblib"))
    
    print(f"\n✅ Results saved to: {output_dir}")
    
    return results


























