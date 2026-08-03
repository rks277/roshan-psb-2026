#!/usr/bin/env python3
"""Two-step PU reranker using reliable negatives.

Step 1 trains a clean positive-vs-unlabeled model on training ligands.
Step 2 treats the lowest-scoring unlabeled training rows as reliable negatives
and retrains against all labeled positives. Evaluation is still on held-out
ligands.
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


def hgb(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(random_state=seed)),
        ]
    )


def top_fraction_rate(y_true: np.ndarray, y_score: np.ndarray, fraction: float) -> float:
    order = np.argsort(-y_score)
    count = max(1, int(round(len(y_true) * fraction)))
    return float(np.mean(y_true[order[:count]]))


def metric_row(
    method: str,
    split: str,
    reliable_negative_ratio: float | str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    train_rows: int,
    eval_rows: int,
    features: int,
    reliable_negatives: int,
) -> dict[str, object]:
    return {
        "Method": method,
        "Split": split,
        "Reliable Negative Ratio": reliable_negative_ratio,
        "Train Rows": train_rows,
        "Eval Rows": eval_rows,
        "Features": features,
        "Reliable Negatives": reliable_negatives,
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
    parser.add_argument("--output-prefix", default="affinity_hit_value_pu_reliable_negative")
    parser.add_argument("--ratios", nargs="+", type=float, default=[5, 10, 20, 40])
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

    stage1 = hgb(args.seed)
    stage1.fit(X[train_idx], y[train_idx])
    stage1_train_score = scores(stage1, X[train_idx])
    stage1_val_score = scores(stage1, X[val_idx])
    metrics = [
        metric_row(
            "stage1 positive-unlabeled HGB",
            "inner_validation_ligands",
            "all_unlabeled",
            y[val_idx],
            stage1_val_score,
            len(train_idx),
            len(val_idx),
            len(columns),
            int(np.sum(y[train_idx] == 0)),
        )
    ]

    train_positive_idx = train_idx[y[train_idx] == 1]
    train_unlabeled_idx = train_idx[y[train_idx] == 0]
    unlabeled_scores = stage1_train_score[y[train_idx] == 0]
    unlabeled_order = np.argsort(unlabeled_scores)

    best = None
    best_pr_auc = -1.0
    for ratio in args.ratios:
        reliable_count = min(len(train_unlabeled_idx), int(round(len(train_positive_idx) * ratio)))
        reliable_negative_idx = train_unlabeled_idx[unlabeled_order[:reliable_count]]
        pu_train_idx = np.concatenate([train_positive_idx, reliable_negative_idx])
        pu_y = np.concatenate([np.ones(len(train_positive_idx), dtype=int), np.zeros(len(reliable_negative_idx), dtype=int)])
        model = hgb(args.seed + int(ratio * 10))
        model.fit(X[pu_train_idx], pu_y)
        val_score = scores(model, X[val_idx])
        row = metric_row(
            "two-step PU reliable-negative HGB",
            "inner_validation_ligands",
            ratio,
            y[val_idx],
            val_score,
            len(pu_train_idx),
            len(val_idx),
            len(columns),
            reliable_count,
        )
        metrics.append(row)
        if row["PR AUC"] > best_pr_auc:
            best_pr_auc = float(row["PR AUC"])
            best = (ratio, model, reliable_count)
        print(f"ratio={ratio:g}: val_pr_auc={row['PR AUC']:.3f} top1={row['Top 1.0% Support']:.3f}")

    assert best is not None
    best_ratio, _, _ = best

    final_stage1 = hgb(args.seed)
    final_stage1.fit(X[outer_train_idx], y[outer_train_idx])
    final_train_score = scores(final_stage1, X[outer_train_idx])
    outer_positive_idx = outer_train_idx[y[outer_train_idx] == 1]
    outer_unlabeled_idx = outer_train_idx[y[outer_train_idx] == 0]
    outer_unlabeled_scores = final_train_score[y[outer_train_idx] == 0]
    outer_unlabeled_order = np.argsort(outer_unlabeled_scores)
    final_reliable_count = min(len(outer_unlabeled_idx), int(round(len(outer_positive_idx) * best_ratio)))
    final_reliable_negative_idx = outer_unlabeled_idx[outer_unlabeled_order[:final_reliable_count]]
    final_train_idx = np.concatenate([outer_positive_idx, final_reliable_negative_idx])
    final_y = np.concatenate([np.ones(len(outer_positive_idx), dtype=int), np.zeros(len(final_reliable_negative_idx), dtype=int)])
    final_model = hgb(args.seed + int(best_ratio * 10))
    final_model.fit(X[final_train_idx], final_y)
    test_score = scores(final_model, X[test_idx])
    metrics.append(
        metric_row(
            "two-step PU reliable-negative HGB",
            "outer_test_ligands",
            best_ratio,
            y[test_idx],
            test_score,
            len(final_train_idx),
            len(test_idx),
            len(columns),
            final_reliable_count,
        )
    )

    metrics_path = args.results_dir / f"{args.output_prefix}_metrics.csv"
    write_csv(metrics_path, metrics)
    enrichment = [{"Method": "two-step PU reliable-negative HGB", "Reliable Negative Ratio": best_ratio, **row} for row in enrichment_rows(y[test_idx], test_score)]
    enrichment_path = args.results_dir / f"{args.output_prefix}_enrichment.csv"
    write_csv(enrichment_path, enrichment)
    manifest_path = args.results_dir / f"{args.output_prefix}_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "feature_set": args.feature_set,
                "feature_count": len(columns),
                "method": "two-step PU reliable negatives",
                "ratios_tested": args.ratios,
                "selected_reliable_negative_ratio": best_ratio,
                "selected_inner_validation_pr_auc": best_pr_auc,
                "outer_test_pr_auc": metrics[-1]["PR AUC"],
                "outer_test_roc_auc": metrics[-1]["ROC AUC"],
                "outer_test_top_1_percent_support": metrics[-1]["Top 1.0% Support"],
                "outer_train_reliable_negatives": final_reliable_count,
                "outer_train_labeled_positives": int(len(outer_positive_idx)),
                "seed": args.seed,
                "outer_test_size": args.outer_test_size,
                "inner_validation_size": args.inner_validation_size,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Selected reliable-negative ratio {best_ratio:g}")
    print(f"Outer test PR AUC: {metrics[-1]['PR AUC']:.3f}")
    print(f"Outer test top 1% support: {metrics[-1]['Top 1.0% Support']:.3f}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {enrichment_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
