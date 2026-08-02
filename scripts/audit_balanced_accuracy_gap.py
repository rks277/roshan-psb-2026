#!/usr/bin/env python3
"""Audit why balanced Yamanishi accuracy trails stronger literature results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
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
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import DatasetBuilder  # noqa: E402
from build_no_affinity_dataset import (  # noqa: E402
    load_maccs_features,
    load_morgan_features,
    load_target_sequence_features,
    choose_target_uniprot,
)
from train_no_affinity_models import FEATURE_SETS, numeric_matrix  # noqa: E402


MODEL_FEATURE_SET = "pubchem_plus_maccs_plus_morgan_plus_target_rich"


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


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def coverage_status(builder: DatasetBuilder, data_dir: Path) -> list[dict[str, object]]:
    maccs_features = load_maccs_features(data_dir)
    morgan_features = load_morgan_features(data_dir)
    target_sequence_features = load_target_sequence_features(data_dir)
    status_by_category: dict[str, Counter[str]] = defaultdict(Counter)

    for key in builder.labels:
        cid = builder.drug_to_cid.get(key.kegg_drug)
        if cid is None:
            status = "missing_kegg_drug_to_pubchem_cid"
        elif cid not in builder.ligands:
            status = "missing_pubchem_descriptor_row"
        elif key.kegg_target not in builder.hsa_to_uniprot:
            status = "missing_kegg_target_to_uniprot"
        elif choose_target_uniprot(builder, key.kegg_target) is None:
            status = "missing_target_features"
        elif choose_target_uniprot(builder, key.kegg_target) not in target_sequence_features:
            status = "missing_target_sequence_features"
        elif cid not in maccs_features:
            status = "missing_maccs_fingerprint"
        elif morgan_features and cid not in morgan_features:
            status = "missing_morgan_fingerprint"
        else:
            status = "joined"
        status_by_category[key.category][status] += 1

    rows = []
    for category, counts in sorted(status_by_category.items()):
        total = sum(counts.values())
        for status, count in sorted(counts.items()):
            rows.append(
                {
                    "category": category,
                    "status": status,
                    "positive_pairs": count,
                    "category_total_positive_pairs": total,
                    "fraction": count / total if total else 0.0,
                }
            )
    return rows


def feature_conflicts(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    metadata = {
        "category",
        "kegg_drug",
        "pubchem_cid",
        "ligand_title",
        "kegg_target",
        "uniprot_id",
        "target_uniprot_count",
        "target_yamanishi_degree",
        "ligand_yamanishi_degree",
        "label",
    }
    feature_columns = [column for column in rows[0] if column not in metadata]
    buckets: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[column] for column in feature_columns)].append(row)

    conflicts = []
    for same_features in buckets.values():
        if len({row["label"] for row in same_features}) < 2:
            continue
        for row in same_features:
            conflicts.append(
                {
                    "label": row["label"],
                    "category": row["category"],
                    "kegg_drug": row["kegg_drug"],
                    "pubchem_cid": row["pubchem_cid"],
                    "ligand_title": row["ligand_title"],
                    "kegg_target": row["kegg_target"],
                    "uniprot_id": row["uniprot_id"],
                }
            )
    return conflicts


def holdout_metrics(rows: list[dict[str, str]], seed: int, test_size: float) -> list[dict[str, object]]:
    out = []
    feature_columns = FEATURE_SETS[MODEL_FEATURE_SET]
    for group_name, subset in [("combined", rows)] + [
        (category, [row for row in rows if row["category"] == category])
        for category in sorted({row["category"] for row in rows})
    ]:
        y = np.asarray([int(row["label"]) for row in subset], dtype=int)
        X = numeric_matrix(subset, feature_columns)
        train_idx, test_idx = train_test_split(
            np.arange(len(y)),
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )
        for model_name, model in make_models(seed).items():
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            y_score = model.predict_proba(X[test_idx])[:, 1]
            out.append(
                {
                    "evaluation": "holdout",
                    "group": group_name,
                    "model": model_name,
                    "rows": len(subset),
                    "train_rows": len(train_idx),
                    "test_rows": len(test_idx),
                    "features": len(feature_columns),
                    **evaluate(y[test_idx], y_pred, y_score),
                }
            )
    return out


def cv_metrics(rows: list[dict[str, str]], seed: int) -> list[dict[str, object]]:
    out = []
    y = np.asarray([int(row["label"]) for row in rows], dtype=int)
    X = numeric_matrix(rows, FEATURE_SETS[MODEL_FEATURE_SET])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for model_name in ["ExtraTrees", "HistGradientBoosting"]:
        fold_metrics = []
        for train_idx, test_idx in cv.split(X, y):
            model = make_models(seed)[model_name]
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            y_score = model.predict_proba(X[test_idx])[:, 1]
            fold_metrics.append(evaluate(y[test_idx], y_pred, y_score))
        metric_names = fold_metrics[0].keys()
        row: dict[str, object] = {
            "evaluation": "5fold_cv",
            "group": "combined",
            "model": model_name,
            "rows": len(rows),
            "train_rows": "",
            "test_rows": "",
            "features": len(FEATURE_SETS[MODEL_FEATURE_SET]),
        }
        for metric in metric_names:
            values = np.asarray([fold[metric] for fold in fold_metrics], dtype=float)
            row[metric] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
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

    builder = DatasetBuilder(args.data_dir, seed=args.seed)
    coverage_rows = coverage_status(builder, args.data_dir)
    conflict_rows = feature_conflicts(rows)
    metric_rows = holdout_metrics(rows, args.seed, args.test_size) + cv_metrics(rows, args.seed)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.results_dir / "balanced_accuracy_gap_coverage_by_category.csv", coverage_rows)
    write_csv(args.results_dir / "balanced_accuracy_gap_conflicts.csv", conflict_rows)
    write_csv(args.results_dir / "balanced_accuracy_gap_model_metrics.csv", metric_rows)

    summary = {
        "dataset": str(args.dataset),
        "feature_set": MODEL_FEATURE_SET,
        "positive_rows": sum(row["label"] == "1" for row in rows),
        "negative_rows": sum(row["label"] == "0" for row in rows),
        "feature_conflict_rows": len(conflict_rows),
        "best_holdout_accuracy": max(
            row["accuracy"] for row in metric_rows if row["evaluation"] == "holdout"
        ),
        "best_combined_holdout_accuracy": max(
            row["accuracy"]
            for row in metric_rows
            if row["evaluation"] == "holdout" and row["group"] == "combined"
        ),
    }
    summary_path = args.results_dir / "balanced_accuracy_gap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote {args.results_dir / 'balanced_accuracy_gap_coverage_by_category.csv'}")
    print(f"Wrote {args.results_dir / 'balanced_accuracy_gap_conflicts.csv'}")
    print(f"Wrote {args.results_dir / 'balanced_accuracy_gap_model_metrics.csv'}")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
