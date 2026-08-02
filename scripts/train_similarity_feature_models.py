#!/usr/bin/env python3
"""Train models with literature-style drug and protein similarity features."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from train_no_affinity_models import (
    FEATURE_SETS,
    MACCS_FEATURES,
    MORGAN_FEATURES,
    TARGET_RICH_FEATURES,
    numeric_matrix,
)


def finite_matrix(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    matrix = numeric_matrix(rows, columns)
    medians = np.nanmedian(matrix, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    missing = np.where(~np.isfinite(matrix))
    matrix[missing] = medians[missing[1]]
    return matrix


def tanimoto_summary(binary: np.ndarray, index: int, candidates: list[int]) -> tuple[float, float]:
    if not candidates:
        return 0.0, 0.0
    candidate_matrix = binary[candidates]
    query = binary[index]
    intersection = candidate_matrix @ query
    union = candidate_matrix.sum(axis=1) + query.sum() - intersection
    similarities = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    return float(similarities.max()), float(similarities.mean())


def cosine_summary(normalized: np.ndarray, index: int, candidates: list[int]) -> tuple[float, float]:
    if not candidates:
        return 0.0, 0.0
    similarities = normalized[candidates] @ normalized[index]
    return float(similarities.max()), float(similarities.mean())


def build_similarity_features(
    rows: list[dict[str, str]],
    y: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    maccs = finite_matrix(rows, MACCS_FEATURES)
    morgan = finite_matrix(rows, MORGAN_FEATURES)
    ligand_bits = np.hstack([(maccs > 0).astype(np.float32), (morgan > 0).astype(np.float32)])
    targets = finite_matrix(rows, TARGET_RICH_FEATURES)
    target_norm = np.linalg.norm(targets, axis=1)
    target_norm[target_norm == 0] = 1.0
    target_vectors = targets / target_norm[:, None]

    positive_train_idx = [int(index) for index in train_idx if y[index] == 1]
    positives_by_target: dict[tuple[str, str], list[int]] = defaultdict(list)
    positives_by_drug: dict[tuple[str, str], list[int]] = defaultdict(list)
    positives_by_category: dict[str, list[int]] = defaultdict(list)
    for index in positive_train_idx:
        row = rows[index]
        positives_by_target[(row["category"], row["kegg_target"])].append(index)
        positives_by_drug[(row["category"], row["kegg_drug"])].append(index)
        positives_by_category[row["category"]].append(index)

    feature_names = [
        "drug_sim_to_training_ligands_for_target_max",
        "drug_sim_to_training_ligands_for_target_mean",
        "protein_sim_to_training_targets_for_drug_max",
        "protein_sim_to_training_targets_for_drug_mean",
        "drug_sim_to_category_training_positives_max",
        "drug_sim_to_category_training_positives_mean",
        "protein_sim_to_category_training_positives_max",
        "protein_sim_to_category_training_positives_mean",
        "training_positive_ligands_for_target",
        "training_positive_targets_for_drug",
    ]
    features = []
    for index, row in enumerate(rows):
        same_target = positives_by_target[(row["category"], row["kegg_target"])]
        same_drug = positives_by_drug[(row["category"], row["kegg_drug"])]
        same_category = positives_by_category[row["category"]]
        drug_target_max, drug_target_mean = tanimoto_summary(ligand_bits, index, same_target)
        protein_drug_max, protein_drug_mean = cosine_summary(target_vectors, index, same_drug)
        drug_category_max, drug_category_mean = tanimoto_summary(ligand_bits, index, same_category)
        protein_category_max, protein_category_mean = cosine_summary(target_vectors, index, same_category)
        features.append(
            [
                drug_target_max,
                drug_target_mean,
                protein_drug_max,
                protein_drug_mean,
                drug_category_max,
                drug_category_mean,
                protein_category_max,
                protein_category_mean,
                len(same_target),
                len(same_drug),
            ]
        )
    return np.asarray(features, dtype=float), feature_names


def make_models(seed: int) -> dict[str, Pipeline]:
    return {
        "ExtraTrees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesClassifier(n_estimators=500, random_state=seed, n_jobs=1)),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=1)),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(random_state=seed)),
            ]
        ),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, y_score),
        "PR AUC": average_precision_score(y_true, y_score),
        "FP": int(fp),
        "FN": int(fn),
    }


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

    similarity, similarity_names = build_similarity_features(rows, y, train_idx)
    molecular_columns = FEATURE_SETS["pubchem_plus_maccs_plus_morgan_plus_target_rich"]
    molecular = numeric_matrix(rows, molecular_columns)
    feature_sets = {
        "similarity_only": (similarity, similarity_names),
        "molecular_plus_similarity": (
            np.hstack([molecular, similarity]),
            molecular_columns + similarity_names,
        ),
    }

    results = []
    for feature_set_name, (X, columns) in feature_sets.items():
        for model_name, model in make_models(args.seed).items():
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            y_score = model.predict_proba(X[test_idx])[:, 1]
            results.append(
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
    metrics_path = args.results_dir / "similarity_feature_model_metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    manifest_path = args.results_dir / "similarity_feature_model_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "seed": args.seed,
                "test_size": args.test_size,
                "positive_rows": int(np.sum(y == 1)),
                "negative_rows": int(np.sum(y == 0)),
                "similarity_features": similarity_names,
                "molecular_feature_count": len(molecular_columns),
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote manifest: {manifest_path}")
    for row in sorted(results, key=lambda item: item["Accuracy"], reverse=True):
        print(
            f"{row['Feature Set']:28s} {row['Classifier']:<20s} "
            f"acc={row['Accuracy']:.3f} f1={row['F1 Score']:.3f} "
            f"roc_auc={row['ROC AUC']:.3f} pr_auc={row['PR AUC']:.3f}"
        )


if __name__ == "__main__":
    main()
