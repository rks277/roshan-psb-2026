#!/usr/bin/env python3
"""Two-step PU reranking with reliable negatives.

This keeps the clean feature set and ligand-held-out evaluation. On the training
ligands only, it first trains a seed HGB classifier against unlabeled rows,
selects reliable negatives from the lowest-scoring unlabeled rows, and retrains
HGB on positives versus those reliable negatives.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_affinity_hit_value_model import (  # noqa: E402
    FEATURE_SETS,
    enrichment_rows,
    load_feature_maps,
    numeric_matrix,
    scores,
)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def ligand_split_indices(rows: list[dict[str, str]], test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ligand_ids = np.asarray([row["pubchem_cid"] for row in rows])
    unique_ligands = np.asarray(sorted(set(ligand_ids)))
    train_ligands, test_ligands = train_test_split(unique_ligands, test_size=test_size, random_state=seed)
    train_ligands = set(train_ligands)
    test_ligands = set(test_ligands)
    train_idx = np.asarray([idx for idx, ligand_id in enumerate(ligand_ids) if ligand_id in train_ligands])
    test_idx = np.asarray([idx for idx, ligand_id in enumerate(ligand_ids) if ligand_id in test_ligands])
    return train_idx, test_idx


def make_hgb(seed: int, balanced: bool = False) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    random_state=seed,
                    class_weight="balanced" if balanced else None,
                ),
            ),
        ]
    )


def top_fraction_rate(y_true: np.ndarray, y_score: np.ndarray, fraction: float) -> float:
    order = np.argsort(-y_score)
    count = max(1, int(round(len(y_true) * fraction)))
    return float(np.mean(y_true[order[:count]]))


def metric_row(
    method: str,
    split: str,
    reliable_negative_fraction: float,
    y_true: np.ndarray,
    y_score: np.ndarray,
    train_rows: int,
    eval_rows: int,
    features: int,
    reliable_negative_rows: int,
) -> dict[str, object]:
    return {
        "Method": method,
        "Split": split,
        "Reliable Negative Fraction": reliable_negative_fraction,
        "Reliable Negative Rows": reliable_negative_rows,
        "Train Rows": train_rows,
        "Eval Rows": eval_rows,
        "Features": features,
        "Positive Eval Fraction": float(np.mean(y_true)),
        "ROC AUC": roc_auc_score(y_true, y_score),
        "PR AUC": average_precision_score(y_true, y_score),
        "Brier Score": brier_score_loss(y_true, y_score),
        "Top 0.1% Support": top_fraction_rate(y_true, y_score, 0.001),
        "Top 0.5% Support": top_fraction_rate(y_true, y_score, 0.005),
        "Top 1.0% Support": top_fraction_rate(y_true, y_score, 0.01),
        "Top 2.0% Support": top_fraction_rate(y_true, y_score, 0.02),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/affinity_hit_value_dataset_compact.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="clean_rank_plus_maccs_morgan_target")
    parser.add_argument("--output-prefix", default="affinity_hit_value_twostep_pu_clean")
    parser.add_argument("--reliable-negative-fractions", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.40])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-test-size", type=float, default=0.2)
    parser.add_argument("--inner-validation-size", type=float, default=0.25)
    args = parser.parse_args()

    rows = load_rows(args.dataset)
    y = np.asarray([int(row["label_supported"]) for row in rows], dtype=int)
    outer_train_idx, test_idx = ligand_split_indices(rows, args.outer_test_size, args.seed)
    outer_train_rows = [rows[idx] for idx in outer_train_idx]
    inner_train_rel, val_rel = ligand_split_indices(outer_train_rows, args.inner_validation_size, args.seed + 1)
    train_idx = outer_train_idx[inner_train_rel]
    val_idx = outer_train_idx[val_rel]

    feature_maps = load_feature_maps(args.data_dir, args.seed)
    columns = FEATURE_SETS[args.feature_set]
    X = numeric_matrix(rows, columns, feature_maps)

    seed_model = make_hgb(args.seed)
    seed_model.fit(X[train_idx], y[train_idx])
    seed_scores_train = scores(seed_model, X[train_idx])
    train_unlabeled_idx = train_idx[y[train_idx] == 0]
    train_unlabeled_scores = seed_scores_train[y[train_idx] == 0]
    train_positive_idx = train_idx[y[train_idx] == 1]

    metrics = []
    best = None
    best_score = -1.0
    for fraction in args.reliable_negative_fractions:
        rn_count = max(len(train_positive_idx), int(round(len(train_unlabeled_idx) * fraction)))
        rn_order = np.argsort(train_unlabeled_scores)
        reliable_negative_idx = train_unlabeled_idx[rn_order[:rn_count]]
        pu_train_idx = np.concatenate([train_positive_idx, reliable_negative_idx])
        pu_y = np.concatenate([np.ones(len(train_positive_idx), dtype=int), np.zeros(len(reliable_negative_idx), dtype=int)])

        model = make_hgb(args.seed + int(fraction * 1000))
        model.fit(X[pu_train_idx], pu_y)
        val_score = scores(model, X[val_idx])
        row = metric_row(
            "Two-step PU HGB",
            "inner_validation_ligands",
            fraction,
            y[val_idx],
            val_score,
            len(pu_train_idx),
            len(val_idx),
            len(columns),
            len(reliable_negative_idx),
        )
        metrics.append(row)
        if row["PR AUC"] > best_score:
            best_score = float(row["PR AUC"])
            best = (fraction, model, len(reliable_negative_idx))
        print(
            f"rn_fraction={fraction:.2f}: val_pr_auc={row['PR AUC']:.3f} "
            f"top1={row['Top 1.0% Support']:.3f} rn={len(reliable_negative_idx)}"
        )

    assert best is not None
    best_fraction, _, _ = best

    # Refit seed and final PU model using all outer-train ligands after selecting fraction.
    outer_seed_model = make_hgb(args.seed)
    outer_seed_model.fit(X[outer_train_idx], y[outer_train_idx])
    outer_seed_scores = scores(outer_seed_model, X[outer_train_idx])
    outer_unlabeled_idx = outer_train_idx[y[outer_train_idx] == 0]
    outer_unlabeled_scores = outer_seed_scores[y[outer_train_idx] == 0]
    outer_positive_idx = outer_train_idx[y[outer_train_idx] == 1]
    rn_count = max(len(outer_positive_idx), int(round(len(outer_unlabeled_idx) * best_fraction)))
    reliable_negative_idx = outer_unlabeled_idx[np.argsort(outer_unlabeled_scores)[:rn_count]]
    final_train_idx = np.concatenate([outer_positive_idx, reliable_negative_idx])
    final_y = np.concatenate([np.ones(len(outer_positive_idx), dtype=int), np.zeros(len(reliable_negative_idx), dtype=int)])
    final_model = make_hgb(args.seed + int(best_fraction * 1000))
    final_model.fit(X[final_train_idx], final_y)
    test_score = scores(final_model, X[test_idx])
    test_row = metric_row(
        "Two-step PU HGB",
        "outer_test_ligands",
        best_fraction,
        y[test_idx],
        test_score,
        len(final_train_idx),
        len(test_idx),
        len(columns),
        len(reliable_negative_idx),
    )
    metrics.append(test_row)

    metrics_path = args.results_dir / f"{args.output_prefix}_metrics.csv"
    write_csv(metrics_path, metrics)
    enrichment = [{"Method": "Two-step PU HGB", "Reliable Negative Fraction": best_fraction, **row} for row in enrichment_rows(y[test_idx], test_score)]
    enrichment_path = args.results_dir / f"{args.output_prefix}_enrichment.csv"
    write_csv(enrichment_path, enrichment)
    manifest_path = args.results_dir / f"{args.output_prefix}_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "feature_set": args.feature_set,
                "feature_count": len(columns),
                "method": "two-step PU reliable negatives with HGB",
                "reliable_negative_fractions_tested": args.reliable_negative_fractions,
                "selected_reliable_negative_fraction": best_fraction,
                "seed": args.seed,
                "outer_test_size": args.outer_test_size,
                "inner_validation_size": args.inner_validation_size,
                "outer_test_pr_auc": test_row["PR AUC"],
                "outer_test_roc_auc": test_row["ROC AUC"],
                "outer_test_top_1_percent_support": test_row["Top 1.0% Support"],
                "outer_train_reliable_negative_rows": len(reliable_negative_idx),
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Selected reliable negative fraction {best_fraction:.2f}")
    print(f"Outer test PR AUC: {test_row['PR AUC']:.3f}")
    print(f"Outer test top 1% support: {test_row['Top 1.0% Support']:.3f}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {enrichment_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
