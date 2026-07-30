#!/usr/bin/env python3
"""Train baseline classifiers on the Yamanishi PSB 2026 joined dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import DatasetBuilder, LIGAND_FEATURE_COLUMNS  # noqa: E402

PAIRWISE_FEATURES = ["affinity", "rank", "inverted_rank", "proportion"]
LIGAND_MODEL_FEATURES = [f"ligand_{column}" for column in LIGAND_FEATURE_COLUMNS]
TARGET_FEATURES = [
    "target_uniprot_count",
    "target_pdb_candidate_count",
]
GRAPH_FEATURES = [
    "target_yamanishi_degree",
    "ligand_yamanishi_degree",
]
CATEGORY_FEATURES = [
    "category_enzyme",
    "category_gpcr",
    "category_ion_channel",
    "category_nuclear_receptor",
]


def feature_sets() -> dict[str, list[str]]:
    return {
        "affinity": ["affinity"],
        "rank": ["rank"],
        "affinity + rank": ["affinity", "rank"],
        "affinity + inverted_rank": ["affinity", "inverted_rank"],
        "pairwise": PAIRWISE_FEATURES,
        "pairwise + ligand_all": PAIRWISE_FEATURES + LIGAND_MODEL_FEATURES,
        "pairwise + target": PAIRWISE_FEATURES + TARGET_FEATURES,
        "pairwise + ligand_all + target": PAIRWISE_FEATURES + LIGAND_MODEL_FEATURES + TARGET_FEATURES,
        "all_available": (
            PAIRWISE_FEATURES
            + LIGAND_MODEL_FEATURES
            + TARGET_FEATURES
            + GRAPH_FEATURES
            + CATEGORY_FEATURES
        ),
    }


def numeric_matrix(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    matrix = []
    for row in rows:
        values = []
        for column in columns:
            if column.startswith("category_"):
                raw = column.removeprefix("category_")
                values.append(1.0 if row.get("category") == raw else 0.0)
            else:
                raw = row.get(column, "")
                try:
                    values.append(float(raw))
                except (TypeError, ValueError):
                    values.append(np.nan)
        matrix.append(values)
    return np.asarray(matrix, dtype=float)


def make_models(seed: int) -> dict[str, Pipeline]:
    scaled = lambda model: Pipeline(  # noqa: E731
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )
    unscaled = lambda model: Pipeline(  # noqa: E731
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )
    return {
        "Logistic Regression": scaled(LogisticRegression(max_iter=1000)),
        "Random Forest": unscaled(
            RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        ),
        "Gradient Boosting": unscaled(GradientBoostingClassifier(random_state=seed)),
        "SVM": scaled(SVC(kernel="rbf", probability=True, random_state=seed)),
        "KNN": scaled(KNeighborsClassifier(n_neighbors=5)),
    }


def predict_scores(model: Pipeline, X: np.ndarray) -> np.ndarray:
    if hasattr(model[-1], "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


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


def category_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["category"]] += 1
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    builder = DatasetBuilder(args.data_dir, seed=args.seed)
    positives = builder.build_positive_rows()
    negatives = builder.build_negative_rows(category_counts(positives))
    rows = positives + negatives

    y = np.asarray([int(row["label"]) for row in rows], dtype=int)
    categories = np.asarray([row["category"] for row in rows], dtype=object)
    train_idx, test_idx = train_test_split(
        np.arange(len(rows)),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    result_rows = []
    models = make_models(args.seed)
    selected_feature_sets = feature_sets()
    for feature_set_name, columns in selected_feature_sets.items():
        X = numeric_matrix(rows, columns)
        for model_name, model in models.items():
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            y_score = predict_scores(model, X[test_idx])
            result_rows.append(
                {
                    "Feature Set": feature_set_name,
                    "Classifier": model_name,
                    "Train Rows": len(train_idx),
                    "Test Rows": len(test_idx),
                    "Features": len(columns),
                    **evaluate(y[test_idx], y_pred, y_score),
                }
            )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.results_dir / "baseline_metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    manifest = {
        "seed": args.seed,
        "test_size": args.test_size,
        "positive_rows": len(positives),
        "negative_rows": len(negatives),
        "total_rows": len(rows),
        "label_counts": {
            "0": int(np.sum(y == 0)),
            "1": int(np.sum(y == 1)),
        },
        "category_counts": {
            category: int(np.sum(categories == category))
            for category in sorted(set(categories))
        },
        "feature_sets": selected_feature_sets,
        "ligand_feature_columns_available": LIGAND_FEATURE_COLUMNS,
    }
    manifest_path = args.results_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote manifest: {manifest_path}")
    for row in sorted(result_rows, key=lambda item: item["F1 Score"], reverse=True)[:10]:
        print(
            f"{row['Feature Set']:24s} {row['Classifier']:20s} "
            f"acc={row['Accuracy']:.3f} f1={row['F1 Score']:.3f} "
            f"roc_auc={row['ROC AUC']:.3f} pr_auc={row['PR AUC']:.3f}"
        )


if __name__ == "__main__":
    main()
