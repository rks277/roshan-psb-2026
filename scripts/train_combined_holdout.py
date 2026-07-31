#!/usr/bin/env python3
"""Train on combined BindingDB + Yamanishi rows with a 200-row holdout."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
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
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import LIGAND_FEATURE_COLUMNS, TARGET_FEATURE_COLUMNS, read_xlsx_sheet  # noqa: E402

PAIRWISE_FEATURES = ["affinity", "rank", "inverted_rank", "proportion"]
LIGAND_FEATURES = [f"ligand_{column}" for column in LIGAND_FEATURE_COLUMNS]
TARGET_FEATURES = [f"target_{column}" for column in TARGET_FEATURE_COLUMNS]
FEATURE_SETS = {
    "pairwise": PAIRWISE_FEATURES,
    "pairwise + ligand_all": PAIRWISE_FEATURES + LIGAND_FEATURES,
    "pairwise + target_all": PAIRWISE_FEATURES + TARGET_FEATURES,
    "pairwise + ligand_all + target_all": PAIRWISE_FEATURES + LIGAND_FEATURES + TARGET_FEATURES,
}


def load_ligands(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        return {row["CID"].strip(): row for row in csv.DictReader(handle)}


def load_target_features(path: Path) -> dict[str, dict[str, str]]:
    out = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            uniprot = row["entry"].strip()
            if not uniprot:
                continue
            out[uniprot] = {
                "length": row.get("length", ""),
                "mass": row.get("mass", ""),
                "degree_up": row.get("degree (UP)", ""),
            }
    return out


def load_bindingdb_rows(
    workbook: Path,
    ligands: dict[str, dict[str, str]],
    targets: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    rows = []
    for row in read_xlsx_sheet(workbook, "original"):
        if row.get("label") not in {"0", "1"}:
            continue
        cid = row["cid"].strip()
        out = {
            "source": "bindingdb",
            "pubchem_cid": cid,
            "uniprot_id": row.get("uniprot_id", "").strip(),
            "label": row["label"].strip(),
        }
        for column in PAIRWISE_FEATURES:
            out[column] = row.get(column, "")
        ligand = ligands.get(cid, {})
        for column in LIGAND_FEATURE_COLUMNS:
            out[f"ligand_{column}"] = ligand.get(column, "")
        target = targets.get(out["uniprot_id"], {})
        for column in TARGET_FEATURE_COLUMNS:
            out[f"target_{column}"] = target.get(column, "")
        rows.append(out)
    return rows


def load_yamanishi_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            row = dict(row)
            row["source"] = "yamanishi"
            rows.append(row)
    return rows


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


def labels(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([int(row["label"]) for row in rows], dtype=int)


def candidate_models(seed: int) -> dict[str, Pipeline]:
    scaled = lambda model: Pipeline(  # noqa: E731
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)]
    )
    imputed = lambda model: Pipeline(  # noqa: E731
        [("imputer", SimpleImputer(strategy="median")), ("model", model)]
    )
    return {
        "Logistic Regression": scaled(LogisticRegression(max_iter=3000, solver="liblinear")),
        "Random Forest": imputed(RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)),
        "Extra Trees": imputed(ExtraTreesClassifier(n_estimators=500, random_state=seed, n_jobs=-1)),
        "Hist Gradient Boosting": imputed(HistGradientBoostingClassifier(random_state=seed)),
    }


def scores(model: Pipeline, X: np.ndarray) -> np.ndarray:
    final = model[-1]
    if hasattr(final, "predict_proba"):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--yamanishi", type=Path, default=Path("data/processed/yamanishi_classifier_dataset.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--holdout-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ligands = load_ligands(args.data_dir / "pubchem_properties_xlogp.csv")
    targets = load_target_features(args.data_dir / "features.tsv")
    bindingdb_rows = load_bindingdb_rows(args.data_dir / "old_PSB_Data.xlsx", ligands, targets)
    yamanishi_rows = load_yamanishi_rows(args.yamanishi)
    rows = bindingdb_rows + yamanishi_rows
    y = labels(rows)
    sources = np.asarray([row["source"] for row in rows], dtype=object)

    train_idx, test_idx = train_test_split(
        np.arange(len(rows)),
        test_size=args.holdout_size,
        random_state=args.seed,
        stratify=y,
    )

    results = []
    for feature_set_name, columns in FEATURE_SETS.items():
        X = numeric_matrix(rows, columns)
        X_train, X_test = X[train_idx], X[test_idx]
        for model_name, model in candidate_models(args.seed).items():
            model.fit(X_train, y[train_idx])
            y_pred = model.predict(X_test)
            y_score = scores(model, X_test)
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
    metrics_path = args.results_dir / "combined_holdout_metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    manifest = {
        "seed": args.seed,
        "holdout_size": args.holdout_size,
        "bindingdb_rows": len(bindingdb_rows),
        "yamanishi_rows": len(yamanishi_rows),
        "combined_rows": len(rows),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "combined_label_counts": dict(Counter(str(value) for value in y)),
        "train_label_counts": dict(Counter(str(value) for value in y[train_idx])),
        "test_label_counts": dict(Counter(str(value) for value in y[test_idx])),
        "test_source_counts": dict(Counter(str(value) for value in sources[test_idx])),
        "feature_sets": FEATURE_SETS,
    }
    manifest_path = args.results_dir / "combined_holdout_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote manifest: {manifest_path}")
    for row in sorted(results, key=lambda item: item["Accuracy"], reverse=True):
        print(
            f"{row['Feature Set']:35s} {row['Classifier']:22s} "
            f"acc={row['Accuracy']:.3f} f1={row['F1 Score']:.3f} "
            f"roc_auc={row['ROC AUC']:.3f} pr_auc={row['PR AUC']:.3f}"
        )


if __name__ == "__main__":
    main()
