#!/usr/bin/env python3
"""Train transductive Yamanishi models with ligand/target identity features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train_no_affinity_models import FEATURE_SETS, numeric_matrix

TOPOLOGY_COLUMNS = [
    "ligand_yamanishi_degree",
    "target_yamanishi_degree",
    "target_uniprot_count",
]
IDENTITY_COLUMNS = ["category", "kegg_drug", "kegg_target", "uniprot_id"]


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


def dense_numeric(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    return numeric_matrix(rows, columns)


def identity_matrix(rows: list[dict[str, str]], columns: list[str], train_idx: np.ndarray, test_idx: np.ndarray):
    values = np.asarray([[row[column] for column in columns] for row in rows], dtype=object)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    encoder.fit(values[train_idx])
    return encoder.transform(values), len(encoder.get_feature_names_out())


def make_feature_matrix(
    rows: list[dict[str, str]],
    feature_set: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[object, int]:
    parts = []
    feature_count = 0
    if feature_set in {"topology", "identity_plus_topology"}:
        numeric = dense_numeric(rows, TOPOLOGY_COLUMNS)
        parts.append(csr_matrix(numeric))
        feature_count += len(TOPOLOGY_COLUMNS)
    elif feature_set in {"molecular_plus_topology", "molecular_plus_identity_plus_topology"}:
        columns = FEATURE_SETS["pubchem_plus_maccs_plus_target_rich"] + TOPOLOGY_COLUMNS
        numeric = dense_numeric(rows, columns)
        parts.append(csr_matrix(numeric))
        feature_count += len(columns)
    elif feature_set in {"molecular", "molecular_plus_identity"}:
        columns = FEATURE_SETS["pubchem_plus_maccs_plus_target_rich"]
        numeric = dense_numeric(rows, columns)
        parts.append(csr_matrix(numeric))
        feature_count += len(columns)
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

    if "identity" in feature_set:
        identity, identity_count = identity_matrix(rows, IDENTITY_COLUMNS, train_idx, test_idx)
        parts.append(identity)
        feature_count += identity_count

    return hstack(parts, format="csr"), feature_count


def make_models(seed: int) -> dict[str, object]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
                ("model", LogisticRegression(max_iter=3000, C=2.0, solver="liblinear")),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=1)),
            ]
        ),
        "Extra Trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesClassifier(n_estimators=500, random_state=seed, n_jobs=1)),
            ]
        ),
    }


def scores(model: object, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/yamanishi_no_affinity_classifier_dataset.csv"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    with args.dataset.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    y = np.asarray([int(row["label"]) for row in rows], dtype=int)
    train_idx, test_idx = train_test_split(
        np.arange(len(rows)),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    feature_sets = [
        "topology",
        "identity_plus_topology",
        "molecular",
        "molecular_plus_topology",
        "molecular_plus_identity",
        "molecular_plus_identity_plus_topology",
    ]
    results = []
    for feature_set in feature_sets:
        X, feature_count = make_feature_matrix(rows, feature_set, train_idx, test_idx)
        for model_name, model in make_models(args.seed).items():
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            y_score = scores(model, X[test_idx])
            results.append(
                {
                    "Feature Set": feature_set,
                    "Classifier": model_name,
                    "Train Rows": len(train_idx),
                    "Test Rows": len(test_idx),
                    "Features": feature_count,
                    **evaluate(y[test_idx], y_pred, y_score),
                }
            )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.results_dir / "transductive_identity_metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    manifest = {
        "dataset": str(args.dataset),
        "seed": args.seed,
        "test_size": args.test_size,
        "positive_rows": int(np.sum(y == 1)),
        "negative_rows": int(np.sum(y == 0)),
        "total_rows": len(rows),
        "identity_columns": IDENTITY_COLUMNS,
        "topology_columns": TOPOLOGY_COLUMNS,
        "feature_sets": feature_sets,
    }
    manifest_path = args.results_dir / "transductive_identity_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote manifest: {manifest_path}")
    for row in sorted(results, key=lambda item: item["Accuracy"], reverse=True)[:12]:
        print(
            f"{row['Feature Set']:40s} {row['Classifier']:<20s} "
            f"acc={row['Accuracy']:.3f} f1={row['F1 Score']:.3f} "
            f"roc_auc={row['ROC AUC']:.3f} pr_auc={row['PR AUC']:.3f}"
        )


if __name__ == "__main__":
    main()
