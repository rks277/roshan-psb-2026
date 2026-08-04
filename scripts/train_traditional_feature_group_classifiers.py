#!/usr/bin/env python3
"""Traditional balanced classifiers for affinity-only, no-affinity, and all features."""

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_affinity_hit_value_dataset import (  # noqa: E402
    LIGAND_OUTPUT_FEATURES,
    MACCS_OUTPUT_FEATURES,
    MORGAN_OUTPUT_FEATURES,
    TARGET_BASIC_OUTPUT_FEATURES,
    TARGET_SEQUENCE_OUTPUT_FEATURES,
)
from train_affinity_hit_value_model import CLEAN_RANK_FEATURES, load_feature_maps, numeric_matrix, scores  # noqa: E402


FEATURE_GROUPS = {
    "affinity_group_only": CLEAN_RANK_FEATURES,
    "all_features_minus_affinity_group": (
        LIGAND_OUTPUT_FEATURES
        + MACCS_OUTPUT_FEATURES
        + MORGAN_OUTPUT_FEATURES
        + TARGET_BASIC_OUTPUT_FEATURES
        + TARGET_SEQUENCE_OUTPUT_FEATURES
    ),
    "all_features": (
        CLEAN_RANK_FEATURES
        + LIGAND_OUTPUT_FEATURES
        + MACCS_OUTPUT_FEATURES
        + MORGAN_OUTPUT_FEATURES
        + TARGET_BASIC_OUTPUT_FEATURES
        + TARGET_SEQUENCE_OUTPUT_FEATURES
    ),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def make_balanced_yamanishi_rows(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    rng = np.random.default_rng(seed)
    positives = [dict(row, classifier_label="1") for row in rows if row["label_yamanishi"] == "1"]
    negatives = [dict(row, classifier_label="0") for row in rows if row["label_supported"] == "0"]
    if len(negatives) < len(positives):
        raise ValueError("Not enough unsupported rows to sample a balanced negative set")
    sampled = rng.choice(len(negatives), size=len(positives), replace=False)
    balanced = positives + [negatives[int(index)] for index in sampled]
    order = rng.permutation(len(balanced))
    return [balanced[int(index)] for index in order]


def split_train_validation_test(y: np.ndarray, seed: int, test_size: float, validation_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_idx = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        all_idx,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    relative_validation = validation_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_validation,
        random_state=seed + 1,
        stratify=y[train_val_idx],
    )
    return train_idx, val_idx, test_idx


def make_models(seed: int) -> dict[str, object]:
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
        "Logistic Regression": scaled(LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
        "Random Forest": unscaled(
            RandomForestClassifier(
                n_estimators=500,
                random_state=seed,
                class_weight="balanced_subsample",
                n_jobs=1,
            )
        ),
        "Extra Trees": unscaled(
            ExtraTreesClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=2,
                random_state=seed,
                class_weight="balanced",
                n_jobs=1,
            )
        ),
        "Hist Gradient Boosting": unscaled(
            HistGradientBoostingClassifier(
                random_state=seed,
                max_iter=220,
                learning_rate=0.06,
                max_leaf_nodes=63,
                min_samples_leaf=15,
                l2_regularization=0.01,
            )
        ),
        "SVM": scaled(SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
        "KNN": scaled(KNeighborsClassifier(n_neighbors=5)),
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
        "False Positive Rate": fp / (fp + tn) if fp + tn else 0.0,
        "False Negative Rate": fn / (fn + tp) if fn + tp else 0.0,
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
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--output-prefix", default="traditional_clean_feature_group_classifier")
    args = parser.parse_args()

    source_rows = read_rows(args.dataset)
    rows = make_balanced_yamanishi_rows(source_rows, args.seed)
    y = np.asarray([int(row["classifier_label"]) for row in rows], dtype=int)
    train_idx, val_idx, test_idx = split_train_validation_test(y, args.seed, args.test_size, args.validation_size)
    feature_maps = load_feature_maps(args.data_dir, args.seed)
    models = make_models(args.seed)

    result_rows = []
    for feature_set_name, columns in FEATURE_GROUPS.items():
        X = numeric_matrix(rows, columns, feature_maps)
        for model_name, model in models.items():
            model.fit(X[train_idx], y[train_idx])
            for split_name, split_idx in [
                ("train", train_idx),
                ("validation", val_idx),
                ("test", test_idx),
            ]:
                y_score = scores(model, X[split_idx])
                result_rows.append(
                    {
                        "Feature Set": feature_set_name,
                        "Classifier": model_name,
                        "Split": split_name,
                        "Rows": len(split_idx),
                        "Positives": int(np.sum(y[split_idx] == 1)),
                        "Negatives": int(np.sum(y[split_idx] == 0)),
                        "Features": len(columns),
                        **evaluate(y[split_idx], y_score),
                    }
                )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.results_dir / f"{args.output_prefix}_metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    manifest = {
        "dataset": str(args.dataset),
        "label_mode": "yamanishi positives vs sampled unsupported/unlabeled negatives",
        "rows": len(rows),
        "positive_rows": int(np.sum(y == 1)),
        "negative_rows": int(np.sum(y == 0)),
        "seed": args.seed,
        "split": {
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(val_idx)),
            "test_rows": int(len(test_idx)),
            "test_size": args.test_size,
            "validation_size": args.validation_size,
            "split_mode": "row-stratified train/validation/test",
        },
        "feature_groups": {name: len(columns) for name, columns in FEATURE_GROUPS.items()},
        "excluded_label_prior_features": [
            "ligand_yamanishi_degree_any",
            "target_yamanishi_degree_any",
            "target_in_yamanishi_universe",
            "target_in_bindingdb_universe",
        ],
    }
    manifest_path = args.results_dir / f"{args.output_prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote {metrics_path}")
    print(f"Wrote {manifest_path}")
    test_rows = [row for row in result_rows if row["Split"] == "test"]
    for row in sorted(test_rows, key=lambda item: item["Accuracy"], reverse=True)[:10]:
        print(
            f"{row['Feature Set']:34s} {row['Classifier']:22s} "
            f"acc={row['Accuracy']:.3f} f1={row['F1 Score']:.3f} "
            f"roc_auc={row['ROC AUC']:.3f} pr_auc={row['PR AUC']:.3f}"
        )


if __name__ == "__main__":
    main()
