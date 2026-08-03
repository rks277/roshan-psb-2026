#!/usr/bin/env python3
"""Train/calibrate models that score the value of individual affinity hits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from build_affinity_hit_value_dataset import (
    LIGAND_OUTPUT_FEATURES,
    MACCS_OUTPUT_FEATURES,
    MORGAN_OUTPUT_FEATURES,
    RANK_FEATURES,
    TARGET_BASIC_OUTPUT_FEATURES,
    TARGET_SEQUENCE_OUTPUT_FEATURES,
)

FEATURE_SETS = {
    "rank_only": RANK_FEATURES,
    "rank_plus_basic_context": RANK_FEATURES + LIGAND_OUTPUT_FEATURES + TARGET_BASIC_OUTPUT_FEATURES,
    "rank_plus_maccs_morgan_target": (
        RANK_FEATURES
        + LIGAND_OUTPUT_FEATURES
        + MACCS_OUTPUT_FEATURES
        + MORGAN_OUTPUT_FEATURES
        + TARGET_BASIC_OUTPUT_FEATURES
        + TARGET_SEQUENCE_OUTPUT_FEATURES
    ),
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


def make_models(seed: int) -> dict[str, object]:
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
                        n_estimators=400,
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
                ("model", HistGradientBoostingClassifier(random_state=seed)),
            ]
        ),
    }


def scores(model: object, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def enrichment_rows(y_true: np.ndarray, y_score: np.ndarray) -> list[dict[str, float | int | str]]:
    order = np.argsort(-y_score)
    rows = []
    baseline = float(np.mean(y_true))
    for fraction in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        count = max(1, int(round(len(y_true) * fraction)))
        selected = order[:count]
        observed = float(np.mean(y_true[selected]))
        rows.append(
            {
                "selection": f"top_{fraction:g}",
                "selected_rows": count,
                "supported_fraction": observed,
                "baseline_supported_fraction": baseline,
                "enrichment": observed / baseline if baseline else 0.0,
                "mean_score": float(np.mean(y_score[selected])),
            }
        )
    quantiles = np.quantile(y_score, np.linspace(0, 1, 11))
    for decile in range(10):
        low, high = quantiles[decile], quantiles[decile + 1]
        if decile == 9:
            mask = (y_score >= low) & (y_score <= high)
        else:
            mask = (y_score >= low) & (y_score < high)
        if not np.any(mask):
            continue
        observed = float(np.mean(y_true[mask]))
        rows.append(
            {
                "selection": f"score_decile_{decile + 1}",
                "selected_rows": int(np.sum(mask)),
                "supported_fraction": observed,
                "baseline_supported_fraction": baseline,
                "enrichment": observed / baseline if baseline else 0.0,
                "mean_score": float(np.mean(y_score[mask])),
            }
        )
    return rows


def average_precision_at_recall(y_true: np.ndarray, y_score: np.ndarray, min_recall: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    valid = precision[recall >= min_recall]
    return float(np.max(valid)) if len(valid) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/affinity_hit_value_dataset_compact.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["row", "ligand"], default="row")
    parser.add_argument("--output-prefix", default="affinity_hit_value")
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    with args.dataset.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    y = np.asarray([int(row["label_supported"]) for row in rows], dtype=int)
    if args.split_mode == "ligand":
        ligand_ids = np.asarray([row["pubchem_cid"] for row in rows])
        unique_ligands = np.asarray(sorted(set(ligand_ids)))
        train_ligands, test_ligands = train_test_split(
            unique_ligands,
            test_size=args.test_size,
            random_state=args.seed,
        )
        train_ligands = set(train_ligands)
        test_ligands = set(test_ligands)
        train_idx = np.asarray([idx for idx, ligand_id in enumerate(ligand_ids) if ligand_id in train_ligands])
        test_idx = np.asarray([idx for idx, ligand_id in enumerate(ligand_ids) if ligand_id in test_ligands])
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(rows)),
            test_size=args.test_size,
            random_state=args.seed,
            stratify=y,
        )

    available_columns = set(rows[0])
    metrics = []
    enrichment = []
    best = None
    best_score = -1.0
    for feature_set_name, columns in FEATURE_SETS.items():
        missing_columns = [column for column in columns if column not in available_columns]
        if missing_columns:
            print(
                f"Skipping {feature_set_name}: {len(missing_columns)} feature columns are absent from {args.dataset}"
            )
            continue
        X = numeric_matrix(rows, columns)
        X_train, X_test = X[train_idx], X[test_idx]
        for model_name, model in make_models(args.seed).items():
            fitted = model
            if args.calibrate:
                fitted = CalibratedClassifierCV(model, method="isotonic", cv=3)
            fitted.fit(X_train, y[train_idx])
            y_score = scores(fitted, X_test)
            pr_auc = average_precision_score(y[test_idx], y_score)
            row = {
                "Feature Set": feature_set_name,
                "Classifier": model_name,
                "Train Rows": len(train_idx),
                "Test Rows": len(test_idx),
                "Split Mode": args.split_mode,
                "Features": len(columns),
                "Positive Train Fraction": float(np.mean(y[train_idx])),
                "Positive Test Fraction": float(np.mean(y[test_idx])),
                "ROC AUC": roc_auc_score(y[test_idx], y_score),
                "PR AUC": pr_auc,
                "Brier Score": brier_score_loss(y[test_idx], y_score),
                "Max Precision At Recall >= 0.10": average_precision_at_recall(y[test_idx], y_score, 0.10),
                "Max Precision At Recall >= 0.25": average_precision_at_recall(y[test_idx], y_score, 0.25),
                "Max Precision At Recall >= 0.50": average_precision_at_recall(y[test_idx], y_score, 0.50),
            }
            metrics.append(row)
            for enrich in enrichment_rows(y[test_idx], y_score):
                enrichment.append(
                    {
                        "Feature Set": feature_set_name,
                        "Classifier": model_name,
                        **enrich,
                    }
                )
            if pr_auc > best_score:
                best_score = pr_auc
                best = (feature_set_name, model_name, fitted, columns, X, y_score)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    if not metrics:
        raise SystemExit("No feature sets could be trained from the available dataset columns.")
    metrics_path = args.results_dir / f"{args.output_prefix}_model_metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    enrichment_path = args.results_dir / f"{args.output_prefix}_enrichment.csv"
    with enrichment_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(enrichment[0].keys()))
        writer.writeheader()
        writer.writerows(enrichment)

    assert best is not None
    best_feature_set, best_model_name, best_model, best_columns, X_best, _ = best
    all_scores = scores(best_model, X_best)
    scored_path = args.results_dir / f"{args.output_prefix}_scored_sample.csv"
    scored_rows = []
    for row, score in zip(rows, all_scores):
        scored_rows.append(
            {
                "pubchem_cid": row["pubchem_cid"],
                "uniprot_id": row["uniprot_id"],
                "label_supported": row["label_supported"],
                "label_yamanishi": row["label_yamanishi"],
                "label_bindingdb": row["label_bindingdb"],
                "affinity": row["affinity"],
                "rank_1_based": row["rank_1_based"],
                "rank_percentile": row["rank_percentile"],
                "hit_value_score": score,
            }
        )
    scored_rows.sort(key=lambda row: float(row["hit_value_score"]), reverse=True)
    with scored_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scored_rows)

    manifest_path = args.results_dir / f"{args.output_prefix}_model_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "rows": len(rows),
                "positive_rows": int(np.sum(y == 1)),
                "negative_or_unlabeled_rows": int(np.sum(y == 0)),
                "seed": args.seed,
                "test_size": args.test_size,
                "split_mode": args.split_mode,
                "calibrate": args.calibrate,
                "best_feature_set": best_feature_set,
                "best_classifier": best_model_name,
                "best_pr_auc": best_score,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Wrote {metrics_path}")
    print(f"Wrote {enrichment_path}")
    print(f"Wrote {scored_path}")
    print(f"Wrote {manifest_path}")
    for row in sorted(metrics, key=lambda item: item["PR AUC"], reverse=True)[:8]:
        print(
            f"{row['Feature Set']:30s} {row['Classifier']:<22s} "
            f"roc_auc={row['ROC AUC']:.3f} pr_auc={row['PR AUC']:.3f} "
            f"brier={row['Brier Score']:.4f}"
        )


if __name__ == "__main__":
    main()
