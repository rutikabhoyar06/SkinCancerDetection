"""Genetic algorithm-based feature selection for ensemble embeddings.

Extract embeddings from trained CNNs, concatenate them, and run this module to
select a subset of features that maximizes macro F1 with an XGBoost head.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from deap import algorithms, base, creator, tools
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


def load_embeddings(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["features"], data["labels"]


def make_classifier() -> XGBClassifier:
    return XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        tree_method="gpu_hist" if XGBClassifier().get_params().get("tree_method") else "hist",
        eval_metric="mlogloss",
        use_label_encoder=False,
    )


def build_toolbox(num_features: int, features: np.ndarray, labels: np.ndarray) -> base.Toolbox:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.randint, 0, 2)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual, features=features, labels=labels)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox


def evaluate_individual(individual, *, features: np.ndarray, labels: np.ndarray) -> Tuple[float]:
    mask = np.array(individual, dtype=bool)
    if mask.sum() == 0:
        return 0.0,

    X = features[:, mask]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, labels):
        model = make_classifier()
        model.fit(X[train_idx], labels[train_idx])
        preds = model.predict(X[val_idx])
        scores.append(f1_score(labels[val_idx], preds, average="macro"))
    return (float(np.mean(scores)),)


def save_results(output_dir: Path, mask: np.ndarray, score: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "ga_mask.npy", mask.astype(np.uint8))
    with (output_dir / "ga_score.txt").open("w", encoding="utf-8") as f:
        f.write(f"Best macro F1: {score:.4f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GA feature selection for CNN ensemble embeddings")
    parser.add_argument("--embeddings", type=Path, required=True, help="Path to .npz containing 'features' and 'labels'")
    parser.add_argument("--output", type=Path, default=Path("analysis_results/ga_selection"))
    parser.add_argument("--population", type=int, default=60)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--cxpb", type=float, default=0.5)
    parser.add_argument("--mutpb", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features, labels = load_embeddings(args.embeddings)

    toolbox = build_toolbox(features.shape[1], features, labels)
    population = toolbox.population(n=args.population)

    final_pop, _ = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=args.cxpb,
        mutpb=args.mutpb,
        ngen=args.generations,
        verbose=False,
    )

    best = tools.selBest(final_pop, k=1)[0]
    score = evaluate_individual(best, features=features, labels=labels)[0]
    mask = np.array(best, dtype=bool)
    save_results(args.output, mask, score)
    joblib.dump(mask, args.output / "ga_mask.joblib")
    print(f"Best GA F1: {score:.4f} with {mask.sum()} features")


if __name__ == "__main__":
    main()




























