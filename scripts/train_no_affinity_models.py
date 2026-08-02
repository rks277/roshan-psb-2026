#!/usr/bin/env python3
"""Train Yamanishi models using ligand and target features, without affinities."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import LIGAND_FEATURE_COLUMNS, TARGET_FEATURE_COLUMNS  # noqa: E402
from build_no_affinity_dataset import MACCS_FEATURE_COLUMNS, TARGET_SEQUENCE_FEATURE_COLUMNS  # noqa: E402
from train_clean_advanced import candidate_models, evaluate, scores  # noqa: E402

LIGAND_FEATURES = [f"ligand_{column}" for column in LIGAND_FEATURE_COLUMNS]
MACCS_FEATURES = [f"ligand_{column}" for column in MACCS_FEATURE_COLUMNS]
TARGET_BASIC_FEATURES = [f"target_{column}" for column in TARGET_FEATURE_COLUMNS]
TARGET_SEQUENCE_FEATURES = [f"target_{column}" for column in TARGET_SEQUENCE_FEATURE_COLUMNS]
TARGET_RICH_FEATURES = TARGET_BASIC_FEATURES + TARGET_SEQUENCE_FEATURES
FEATURE_SETS = {
    "pubchem_only": LIGAND_FEATURES,
    "maccs_only": MACCS_FEATURES,
    "pubchem_plus_maccs": LIGAND_FEATURES + MACCS_FEATURES,
    "target_basic_only": TARGET_BASIC_FEATURES,
    "target_rich_only": TARGET_RICH_FEATURES,
    "pubchem_plus_target": LIGAND_FEATURES + TARGET_BASIC_FEATURES,
    "maccs_plus_target": MACCS_FEATURES + TARGET_BASIC_FEATURES,
    "pubchem_plus_maccs_plus_target": LIGAND_FEATURES + MACCS_FEATURES + TARGET_BASIC_FEATURES,
    "pubchem_plus_target_rich": LIGAND_FEATURES + TARGET_RICH_FEATURES,
    "maccs_plus_target_rich": MACCS_FEATURES + TARGET_RICH_FEATURES,
    "pubchem_plus_maccs_plus_target_rich": LIGAND_FEATURES + MACCS_FEATURES + TARGET_RICH_FEATURES,
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
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--include-slow", action="store_true")
    parser.add_argument("--classifiers", nargs="*")
    parser.add_argument("--feature-sets", nargs="*")
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
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
    y_train, y_test = y[train_idx], y[test_idx]
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

    results = []
    best_params = {}
    selected_feature_sets = FEATURE_SETS
    if args.feature_sets:
        selected_feature_sets = {
            name: columns
            for name, columns in FEATURE_SETS.items()
            if name in set(args.feature_sets)
        }
        if not selected_feature_sets:
            raise ValueError(f"No feature sets matched: {args.feature_sets}")

    for feature_set_name, columns in selected_feature_sets.items():
        X = numeric_matrix(rows, columns)
        X_train, X_test = X[train_idx], X[test_idx]
        models = candidate_models(
            args.seed,
            include_slow=args.include_slow,
        )
        if args.classifiers:
            models = {
                name: value
                for name, value in models.items()
                if name in set(args.classifiers)
            }
            if not models:
                raise ValueError(f"No classifiers matched: {args.classifiers}")
        for model_name, (pipeline, param_dist) in models.items():
            result_key = f"{feature_set_name}::{model_name}"
            if param_dist and args.n_iter > 0:
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
                    "Classifier": model_name,
                    "Train Rows": len(train_idx),
                    "Test Rows": len(test_idx),
                    "Features": len(columns),
                    "CV PR AUC": cv_score,
                    **evaluate(y_test, y_pred, y_score),
                }
            )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.metrics_output or args.results_dir / "no_affinity_model_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
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
        "feature_sets": selected_feature_sets,
        "n_iter": args.n_iter,
        "cv": args.cv,
        "include_slow": args.include_slow,
        "best_params": best_params,
    }
    manifest_path = args.manifest_output or args.results_dir / "no_affinity_model_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote manifest: {manifest_path}")
    for row in sorted(results, key=lambda item: item["PR AUC"], reverse=True)[:12]:
        print(
            f"{row['Feature Set']:20s} {row['Classifier']:<30} "
            f"acc={row['Accuracy']:.3f} f1={row['F1 Score']:.3f} "
            f"roc_auc={row['ROC AUC']:.3f} pr_auc={row['PR AUC']:.3f} "
            f"cv_pr_auc={row['CV PR AUC']:.3f}"
        )


if __name__ == "__main__":
    main()
