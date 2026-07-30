#!/usr/bin/env python3
"""Advanced clean-feature classifier sweep.

Clean features are restricted to:

- pairwise docking/rank values from GoldStandardAffinities.zip
- numeric ligand descriptors from pubchem_properties_xlogp.csv
- numeric protein target features from features.tsv

Yamanishi labels are used only as the binary target. No Yamanishi graph-degree,
category, or target-count features are included.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import randint, uniform
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import DatasetBuilder, LIGAND_FEATURE_COLUMNS, TARGET_FEATURE_COLUMNS  # noqa: E402

PAIRWISE_FEATURES = ["affinity", "rank", "inverted_rank", "proportion"]
LIGAND_FEATURES = [f"ligand_{column}" for column in LIGAND_FEATURE_COLUMNS]
TARGET_FEATURES = [f"target_{column}" for column in TARGET_FEATURE_COLUMNS]
FEATURE_SETS = {
    "clean_pairwise_plus_pubchem": PAIRWISE_FEATURES + LIGAND_FEATURES,
    "pairwise_plus_target": PAIRWISE_FEATURES + TARGET_FEATURES,
    "pairwise_plus_pubchem_plus_target": PAIRWISE_FEATURES + LIGAND_FEATURES + TARGET_FEATURES,
}


def numeric_matrix(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    matrix = []
    for row in rows:
        values = []
        for column in columns:
            try:
                values.append(float(row.get(column, "")))
            except (TypeError, ValueError):
                values.append(np.nan)
        matrix.append(values)
    return np.asarray(matrix, dtype=float)


def category_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["category"]] += 1
    return dict(counts)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, y_score),
        "PR AUC": average_precision_score(y_true, y_score),
        "False Positive Rate": fp / (fp + tn),
        "False Negative Rate": fn / (fn + tp),
    }


def scores(model: Pipeline, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    final = model[-1]
    if hasattr(final, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def candidate_models(seed: int, include_slow: bool = False) -> dict[str, tuple[Pipeline, dict[str, object] | None]]:
    standard = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    robust = [("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]
    impute = [("imputer", SimpleImputer(strategy="median"))]

    models = {
        "Logistic L2 tuned": (
            Pipeline(standard + [("model", LogisticRegression(max_iter=3000, solver="liblinear"))]),
            {"model__C": uniform(0.01, 20.0), "model__class_weight": [None, "balanced"]},
        ),
        "Linear SGD tuned": (
            Pipeline(standard + [("model", SGDClassifier(loss="log_loss", random_state=seed, max_iter=5000))]),
            {
                "model__alpha": uniform(1e-5, 1e-2),
                "model__penalty": ["l2", "l1", "elasticnet"],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest tuned": (
            Pipeline(impute + [("model", RandomForestClassifier(random_state=seed, n_jobs=1))]),
            {
                "model__n_estimators": randint(100, 450),
                "model__max_depth": [None, 3, 5, 8, 12, 20],
                "model__min_samples_leaf": randint(1, 12),
                "model__max_features": ["sqrt", "log2", 0.4, 0.7, None],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Extra Trees tuned": (
            Pipeline(impute + [("model", ExtraTreesClassifier(random_state=seed, n_jobs=1))]),
            {
                "model__n_estimators": randint(100, 500),
                "model__max_depth": [None, 3, 5, 8, 12, 20],
                "model__min_samples_leaf": randint(1, 12),
                "model__max_features": ["sqrt", "log2", 0.4, 0.7, None],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Gradient Boosting tuned": (
            Pipeline(impute + [("model", GradientBoostingClassifier(random_state=seed))]),
            {
                "model__n_estimators": randint(50, 250),
                "model__learning_rate": uniform(0.01, 0.25),
                "model__max_depth": randint(1, 5),
                "model__min_samples_leaf": randint(1, 20),
                "model__subsample": uniform(0.55, 0.45),
            },
        ),
        "Hist Gradient Boosting tuned": (
            Pipeline(impute + [("model", HistGradientBoostingClassifier(random_state=seed))]),
            {
                "model__learning_rate": uniform(0.01, 0.25),
                "model__max_iter": randint(75, 250),
                "model__max_leaf_nodes": randint(7, 63),
                "model__min_samples_leaf": randint(5, 60),
                "model__l2_regularization": uniform(0.0, 2.0),
            },
        ),
        "AdaBoost tuned": (
            Pipeline(impute + [("model", AdaBoostClassifier(random_state=seed))]),
            {
                "model__n_estimators": randint(50, 500),
                "model__learning_rate": uniform(0.01, 1.5),
            },
        ),
    }
    if include_slow:
        models.update(
            {
                "RBF SVM tuned": (
                    Pipeline(standard + [("model", SVC(kernel="rbf", probability=True, random_state=seed))]),
                    {
                        "model__C": uniform(0.1, 20.0),
                        "model__gamma": ["scale", "auto", 0.001, 0.01, 0.03, 0.1],
                    },
                ),
                "Bagging SVM": (
                    Pipeline(
                        robust
                        + [
                            (
                                "model",
                                BaggingClassifier(
                                    estimator=SVC(kernel="rbf", C=5.0, gamma="scale", probability=True),
                                    n_estimators=15,
                                    max_samples=0.75,
                                    max_features=0.85,
                                    random_state=seed,
                                    n_jobs=1,
                                ),
                            )
                        ]
                    ),
                    None,
                ),
                "MLP tuned": (
                    Pipeline(standard + [("model", MLPClassifier(random_state=seed, max_iter=600, early_stopping=True))]),
                    {
                        "model__hidden_layer_sizes": [(64,), (128,), (64, 32)],
                        "model__alpha": uniform(1e-5, 5e-3),
                        "model__learning_rate_init": uniform(1e-4, 5e-3),
                    },
                ),
            }
        )
    return models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--include-slow", action="store_true")
    args = parser.parse_args()

    builder = DatasetBuilder(args.data_dir, seed=args.seed)
    positives = builder.build_positive_rows()
    negatives = builder.build_negative_rows(category_counts(positives))
    rows = positives + negatives
    y = np.asarray([int(row["label"]) for row in rows], dtype=int)

    train_idx, test_idx = train_test_split(
        np.arange(len(rows)),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    y_train, y_test = y[train_idx], y[test_idx]

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    results = []
    best_params = {}
    for feature_set_name, columns in FEATURE_SETS.items():
        X = numeric_matrix(rows, columns)
        X_train, X_test = X[train_idx], X[test_idx]
        for name, (pipeline, param_dist) in candidate_models(args.seed, include_slow=args.include_slow).items():
            result_key = f"{feature_set_name}::{name}"
            if param_dist:
                search = RandomizedSearchCV(
                    pipeline,
                    param_distributions=param_dist,
                    n_iter=args.n_iter,
                    scoring="average_precision",
                    cv=cv,
                    random_state=args.seed,
                    n_jobs=1,
                    refit=True,
                )
                search.fit(X_train, y_train)
                model = search.best_estimator_
                best_params[result_key] = search.best_params_
                cv_score = float(search.best_score_)
            else:
                model = pipeline.fit(X_train, y_train)
                best_params[result_key] = {}
                cv_score = float("nan")

            y_pred = model.predict(X_test)
            y_score = scores(model, X_test)
            results.append(
                {
                    "Feature Set": feature_set_name,
                    "Classifier": name,
                    "Train Rows": len(train_idx),
                    "Test Rows": len(test_idx),
                    "Features": len(columns),
                    "CV PR AUC": cv_score,
                    **evaluate(y_test, y_pred, y_score),
                }
            )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.results_dir / "clean_advanced_metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    manifest = {
        "seed": args.seed,
        "test_size": args.test_size,
        "positive_rows": len(positives),
        "negative_rows": len(negatives),
        "total_rows": len(rows),
        "feature_sets": FEATURE_SETS,
        "n_iter": args.n_iter,
        "cv": args.cv,
        "include_slow": args.include_slow,
        "best_params": best_params,
    }
    manifest_path = args.results_dir / "clean_advanced_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote manifest: {manifest_path}")
    for row in sorted(results, key=lambda item: item["PR AUC"], reverse=True):
        print(
            f"{row['Feature Set']:35s} {row['Classifier']:<30} acc={row['Accuracy']:.3f} "
            f"f1={row['F1 Score']:.3f} roc_auc={row['ROC AUC']:.3f} "
            f"pr_auc={row['PR AUC']:.3f} cv_pr_auc={row['CV PR AUC']:.3f}"
        )


if __name__ == "__main__":
    main()
