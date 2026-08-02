#!/usr/bin/env python3
"""Train no-affinity Yamanishi models under different negative sampling ratios."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import DatasetBuilder  # noqa: E402
from build_no_affinity_dataset import (  # noqa: E402
    build_positive_rows,
    load_maccs_features,
    load_target_sequence_features,
    make_no_affinity_row,
)
from train_no_affinity_models import FEATURE_SETS, numeric_matrix  # noqa: E402


def all_valid_negative_rows(builder, maccs_features, target_sequence_features) -> dict[str, list[dict[str, str]]]:
    labels_by_category = defaultdict(set)
    drugs_by_category = defaultdict(set)
    targets_by_category = defaultdict(set)
    for key in builder.labels:
        labels_by_category[key.category].add((key.kegg_drug, key.kegg_target))
        drugs_by_category[key.category].add(key.kegg_drug)
        targets_by_category[key.category].add(key.kegg_target)

    out = defaultdict(list)
    for category in sorted(drugs_by_category):
        for drug in sorted(drugs_by_category[category]):
            for target in sorted(targets_by_category[category]):
                if (drug, target) in labels_by_category[category]:
                    continue
                row = make_no_affinity_row(
                    builder,
                    maccs_features,
                    target_sequence_features,
                    category,
                    drug,
                    target,
                    0,
                )
                if row is not None:
                    out[category].append(row)
    return out


def category_groups(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out = defaultdict(list)
    for row in rows:
        out[row["category"]].append(row)
    return out


def make_models(seed: int) -> dict[str, Pipeline]:
    return {
        "Hist Gradient Boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(random_state=seed)),
            ]
        ),
        "Extra Trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=1)),
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
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--ratios", type=int, nargs="+", default=[1, 3, 5, 10])
    args = parser.parse_args()

    builder = DatasetBuilder(args.data_dir, seed=args.seed)
    maccs_features = load_maccs_features(args.data_dir)
    target_sequence_features = load_target_sequence_features(args.data_dir)
    positives = build_positive_rows(builder, maccs_features, target_sequence_features)
    positives_by_category = category_groups(positives)
    negatives_by_category = all_valid_negative_rows(builder, maccs_features, target_sequence_features)

    rng = random.Random(args.seed)
    columns = FEATURE_SETS["pubchem_plus_maccs_plus_target_rich"]
    results = []
    for ratio in args.ratios:
        negatives = []
        for category, category_positives in positives_by_category.items():
            pool = negatives_by_category[category][:]
            rng.shuffle(pool)
            negatives.extend(pool[: min(len(pool), len(category_positives) * ratio)])

        rows = positives + negatives
        y = np.asarray([int(row["label"]) for row in rows], dtype=int)
        X = numeric_matrix(rows, columns)
        train_idx, test_idx = train_test_split(
            np.arange(len(rows)),
            test_size=args.test_size,
            random_state=args.seed,
            stratify=y,
        )
        always_negative_accuracy = float(np.mean(y[test_idx] == 0))
        positive_test_fraction = float(np.mean(y[test_idx] == 1))
        for classifier, model in make_models(args.seed).items():
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            y_score = model.predict_proba(X[test_idx])[:, 1]
            results.append(
                {
                    "Negative Ratio": ratio,
                    "Classifier": classifier,
                    "Train Rows": len(train_idx),
                    "Test Rows": len(test_idx),
                    "Positive Test Fraction": positive_test_fraction,
                    "Always Negative Accuracy": always_negative_accuracy,
                    **evaluate(y[test_idx], y_pred, y_score),
                }
            )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.results_dir / "negative_ratio_accuracy_metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote metrics: {metrics_path}")
    for row in sorted(results, key=lambda item: item["Accuracy"], reverse=True):
        print(
            f"{row['Negative Ratio']}:1 {row['Classifier']:<22s} "
            f"acc={row['Accuracy']:.3f} balanced_acc={row['Balanced Accuracy']:.3f} "
            f"always_neg={row['Always Negative Accuracy']:.3f} "
            f"roc_auc={row['ROC AUC']:.3f} pr_auc={row['PR AUC']:.3f}"
        )


if __name__ == "__main__":
    main()
