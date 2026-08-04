#!/usr/bin/env python3
"""Train balanced 0/1 classifiers using the clean affinity-reranking features."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_affinity_hit_value_model import (  # noqa: E402
    FEATURE_SETS,
    load_feature_maps,
    numeric_matrix,
    scores,
)


FEATURE_SET_NAME = "clean_rank_plus_maccs_morgan_target"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def balanced_rows(
    rows: list[dict[str, str]],
    label_mode: str,
    seed: int,
) -> list[dict[str, str]]:
    rng = np.random.default_rng(seed)
    if label_mode == "yamanishi":
        positives = [dict(row, classifier_label="1") for row in rows if row["label_yamanishi"] == "1"]
        negatives = [dict(row, classifier_label="0") for row in rows if row["label_supported"] == "0"]
    elif label_mode == "any_supported":
        positives = [dict(row, classifier_label="1") for row in rows if row["label_supported"] == "1"]
        negatives = [dict(row, classifier_label="0") for row in rows if row["label_supported"] == "0"]
    else:
        raise ValueError(f"Unknown label mode: {label_mode}")

    if len(negatives) < len(positives):
        raise ValueError("Not enough negatives to balance positives")
    sampled_negative_idx = rng.choice(len(negatives), size=len(positives), replace=False)
    sampled_negatives = [negatives[int(idx)] for idx in sampled_negative_idx]
    out = positives + sampled_negatives
    order = rng.permutation(len(out))
    return [out[int(idx)] for idx in order]


def split_indices(rows: list[dict[str, str]], y: np.ndarray, split_mode: str, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if split_mode == "row_stratified":
        return train_test_split(np.arange(len(rows)), test_size=test_size, random_state=seed, stratify=y)
    if split_mode != "ligand_held_out":
        raise ValueError(f"Unknown split mode: {split_mode}")

    ligand_ids = np.asarray([row["pubchem_cid"] for row in rows])
    unique_ligands = np.asarray(sorted(set(ligand_ids)))
    ligand_has_positive = np.asarray([
        int(any(row["pubchem_cid"] == ligand and row["classifier_label"] == "1" for row in rows))
        for ligand in unique_ligands
    ])
    stratify = ligand_has_positive if len(set(ligand_has_positive)) > 1 else None
    train_ligands, test_ligands = train_test_split(
        unique_ligands,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    train_ligands = set(train_ligands)
    test_ligands = set(test_ligands)
    train_idx = np.asarray([idx for idx, ligand in enumerate(ligand_ids) if ligand in train_ligands])
    test_idx = np.asarray([idx for idx, ligand in enumerate(ligand_ids) if ligand in test_ligands])
    return train_idx, test_idx


def models(seed: int) -> dict[str, object]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        random_state=seed,
                        class_weight="balanced_subsample",
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "Extra Trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        max_features="sqrt",
                        min_samples_leaf=2,
                        random_state=seed,
                        class_weight="balanced",
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "Hist Gradient Boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        random_state=seed,
                        max_iter=220,
                        learning_rate=0.06,
                        max_leaf_nodes=63,
                        min_samples_leaf=15,
                        l2_regularization=0.01,
                    ),
                ),
            ]
        ),
    }


def evaluate(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float | int]:
    y_pred = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, y_score),
        "PR AUC": average_precision_score(y_true, y_score),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "True Positives": int(tp),
        "True Negatives": int(tn),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/affinity_hit_value_dataset_compact.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-prefix", default="balanced_clean_feature_classifier")
    args = parser.parse_args()

    source_rows = read_rows(args.dataset)
    feature_maps = load_feature_maps(args.data_dir, args.seed)
    columns = FEATURE_SETS[FEATURE_SET_NAME]
    out_rows = []
    manifests = {}
    for label_mode in ["yamanishi", "any_supported"]:
        rows = balanced_rows(source_rows, label_mode, args.seed)
        y = np.asarray([int(row["classifier_label"]) for row in rows], dtype=int)
        X = numeric_matrix(rows, columns, feature_maps)
        manifests[label_mode] = {
            "rows": len(rows),
            "positive_rows": int(np.sum(y == 1)),
            "negative_rows": int(np.sum(y == 0)),
            "unique_ligands": len({row["pubchem_cid"] for row in rows}),
        }
        for split_mode in ["row_stratified", "ligand_held_out"]:
            train_idx, test_idx = split_indices(rows, y, split_mode, args.test_size, args.seed)
            for model_name, model in models(args.seed).items():
                model.fit(X[train_idx], y[train_idx])
                y_score = scores(model, X[test_idx])
                out_rows.append(
                    {
                        "Label Mode": label_mode,
                        "Feature Set": FEATURE_SET_NAME,
                        "Split Mode": split_mode,
                        "Classifier": model_name,
                        "Train Rows": len(train_idx),
                        "Test Rows": len(test_idx),
                        "Train Positives": int(np.sum(y[train_idx] == 1)),
                        "Test Positives": int(np.sum(y[test_idx] == 1)),
                        "Features": len(columns),
                        **evaluate(y[test_idx], y_score),
                    }
                )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.results_dir / f"{args.output_prefix}_metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    manifest = {
        "dataset": str(args.dataset),
        "feature_set": FEATURE_SET_NAME,
        "feature_count": len(columns),
        "excluded_label_prior_features": [
            "ligand_yamanishi_degree_any",
            "target_yamanishi_degree_any",
            "target_in_yamanishi_universe",
            "target_in_bindingdb_universe",
        ],
        "seed": args.seed,
        "test_size": args.test_size,
        "label_modes": manifests,
    }
    manifest_path = args.results_dir / f"{args.output_prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote {metrics_path}")
    print(f"Wrote {manifest_path}")
    for row in sorted(out_rows, key=lambda item: item["Accuracy"], reverse=True)[:8]:
        print(
            f"{row['Label Mode']:13s} {row['Split Mode']:16s} {row['Classifier']:22s} "
            f"acc={row['Accuracy']:.3f} bal_acc={row['Balanced Accuracy']:.3f} "
            f"f1={row['F1 Score']:.3f} roc_auc={row['ROC AUC']:.3f}"
        )


if __name__ == "__main__":
    main()
