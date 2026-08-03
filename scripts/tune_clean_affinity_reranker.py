#!/usr/bin/env python3
"""Tune the clean affinity-hit reranker without label-prior features.

The outer test split holds out whole ligands. Hyperparameters are selected on a
second ligand-level validation split inside the training ligands.
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


def ligand_split_indices(
    rows: list[dict[str, str]],
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    ligand_ids = np.asarray([row["pubchem_cid"] for row in rows])
    unique_ligands = np.asarray(sorted(set(ligand_ids)))
    train_ligands, test_ligands = train_test_split(unique_ligands, test_size=test_size, random_state=seed)
    train_ligands = set(train_ligands)
    test_ligands = set(test_ligands)
    train_idx = np.asarray([idx for idx, ligand_id in enumerate(ligand_ids) if ligand_id in train_ligands])
    test_idx = np.asarray([idx for idx, ligand_id in enumerate(ligand_ids) if ligand_id in test_ligands])
    return train_idx, test_idx


def hgb_configs(seed: int) -> list[dict[str, object]]:
    return [
        {
            "name": "hgb_default",
            "model": HistGradientBoostingClassifier(random_state=seed),
        },
        {
            "name": "hgb_balanced_default",
            "model": HistGradientBoostingClassifier(random_state=seed, class_weight="balanced"),
        },
        {
            "name": "hgb_low_lr_300",
            "model": HistGradientBoostingClassifier(
                random_state=seed,
                learning_rate=0.04,
                max_iter=300,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.05,
            ),
        },
        {
            "name": "hgb_low_lr_300_balanced",
            "model": HistGradientBoostingClassifier(
                random_state=seed,
                learning_rate=0.04,
                max_iter=300,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.05,
                class_weight="balanced",
            ),
        },
        {
            "name": "hgb_leaf63_180",
            "model": HistGradientBoostingClassifier(
                random_state=seed,
                learning_rate=0.07,
                max_iter=180,
                max_leaf_nodes=63,
                min_samples_leaf=25,
                l2_regularization=0.02,
            ),
        },
        {
            "name": "hgb_leaf63_180_balanced",
            "model": HistGradientBoostingClassifier(
                random_state=seed,
                learning_rate=0.07,
                max_iter=180,
                max_leaf_nodes=63,
                min_samples_leaf=25,
                l2_regularization=0.02,
                class_weight="balanced",
            ),
        },
        {
            "name": "hgb_shallow_regularized",
            "model": HistGradientBoostingClassifier(
                random_state=seed,
                learning_rate=0.05,
                max_iter=260,
                max_leaf_nodes=15,
                min_samples_leaf=40,
                l2_regularization=0.25,
                max_features=0.7,
            ),
        },
        {
            "name": "hgb_shallow_regularized_balanced",
            "model": HistGradientBoostingClassifier(
                random_state=seed,
                learning_rate=0.05,
                max_iter=260,
                max_leaf_nodes=15,
                min_samples_leaf=40,
                l2_regularization=0.25,
                max_features=0.7,
                class_weight="balanced",
            ),
        },
    ]


def select_configs(configs: list[dict[str, object]], names: list[str] | None) -> list[dict[str, object]]:
    if not names:
        return configs
    by_name = {str(config["name"]): config for config in configs}
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise SystemExit(f"Unknown config(s): {', '.join(missing)}")
    return [by_name[name] for name in names]


def model_pipeline(model: HistGradientBoostingClassifier) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def top_fraction_rate(y_true: np.ndarray, y_score: np.ndarray, fraction: float) -> float:
    order = np.argsort(-y_score)
    count = max(1, int(round(len(y_true) * fraction)))
    return float(np.mean(y_true[order[:count]]))


def metric_row(
    feature_set: str,
    config_name: str,
    split: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    train_rows: int,
    eval_rows: int,
    features: int,
) -> dict[str, object]:
    return {
        "Feature Set": feature_set,
        "Config": config_name,
        "Split": split,
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-test-size", type=float, default=0.2)
    parser.add_argument("--inner-validation-size", type=float, default=0.25)
    parser.add_argument("--output-prefix", default="affinity_hit_value_clean_tuned")
    parser.add_argument("--configs", nargs="+", default=None)
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

    tune_rows = []
    best = None
    best_score = -1.0
    for config in select_configs(hgb_configs(args.seed), args.configs):
        pipeline = model_pipeline(config["model"])
        pipeline.fit(X[train_idx], y[train_idx])
        val_score = scores(pipeline, X[val_idx])
        row = metric_row(
            args.feature_set,
            str(config["name"]),
            "inner_validation_ligands",
            y[val_idx],
            val_score,
            len(train_idx),
            len(val_idx),
            len(columns),
        )
        tune_rows.append(row)
        if row["PR AUC"] > best_score:
            best_score = float(row["PR AUC"])
            best = config
        print(f"{config['name']}: val_pr_auc={row['PR AUC']:.3f} top1={row['Top 1.0% Support']:.3f}")

    assert best is not None
    final_model = model_pipeline(best["model"])
    final_model.fit(X[outer_train_idx], y[outer_train_idx])
    test_score = scores(final_model, X[test_idx])
    test_row = metric_row(
        args.feature_set,
        str(best["name"]),
        "outer_test_ligands",
        y[test_idx],
        test_score,
        len(outer_train_idx),
        len(test_idx),
        len(columns),
    )

    metrics_path = args.results_dir / f"{args.output_prefix}_metrics.csv"
    write_csv(metrics_path, tune_rows + [test_row])

    enrichment = []
    for row in enrichment_rows(y[test_idx], test_score):
        enrichment.append({"Feature Set": args.feature_set, "Config": str(best["name"]), **row})
    enrichment_path = args.results_dir / f"{args.output_prefix}_enrichment.csv"
    write_csv(enrichment_path, enrichment)

    scored_rows = []
    full_score = scores(final_model, X)
    for row, score in zip(rows, full_score):
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
    scored_path = args.results_dir / f"{args.output_prefix}_scored_sample.csv"
    write_csv(scored_path, scored_rows)

    manifest_path = args.results_dir / f"{args.output_prefix}_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "feature_set": args.feature_set,
                "feature_count": len(columns),
                "excluded_label_prior_features": [
                    "ligand_yamanishi_degree_any",
                    "target_yamanishi_degree_any",
                    "target_in_yamanishi_universe",
                    "target_in_bindingdb_universe",
                ],
                "rows": len(rows),
                "positive_rows": int(np.sum(y == 1)),
                "negative_or_unlabeled_rows": int(np.sum(y == 0)),
                "outer_test_size": args.outer_test_size,
                "inner_validation_size": args.inner_validation_size,
                "seed": args.seed,
                "selected_config": str(best["name"]),
                "inner_validation_pr_auc": best_score,
                "outer_test_pr_auc": test_row["PR AUC"],
                "outer_test_roc_auc": test_row["ROC AUC"],
                "outer_test_top_1_percent_support": test_row["Top 1.0% Support"],
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Selected {best['name']}")
    print(f"Outer test PR AUC: {test_row['PR AUC']:.3f}")
    print(f"Outer test top 1% support: {test_row['Top 1.0% Support']:.3f}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {enrichment_path}")
    print(f"Wrote {scored_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
