#!/usr/bin/env python3
"""Annotate full affinity lists with known-target flags and hit likelihoods."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_affinity_rank_positions import full_uniprot_affinity_table, load_bindingdb_positive_pairs  # noqa: E402
from build_affinity_hit_value_dataset import (  # noqa: E402
    RANK_FEATURES,
    affinity_stats,
    make_row,
    yamanishi_degrees,
    yamanishi_positive_pairs,
)
from build_dataset import DatasetBuilder  # noqa: E402


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


def make_model(model_name: str, seed: int) -> object:
    if model_name == "logistic":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
            ]
        )
    if model_name == "hist_gradient_boosting":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(random_state=seed)),
            ]
        )
    raise ValueError(f"Unsupported model: {model_name}")


def predict_scores(model: object, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raise TypeError("Model does not expose predict_proba")


def train_likelihood_model(dataset_path: Path, model_name: str, seed: int, calibrate: bool) -> tuple[object, float]:
    with dataset_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = [column for column in RANK_FEATURES if column not in rows[0]]
    if missing:
        raise ValueError(f"Training dataset is missing rank feature columns: {missing}")

    X = numeric_matrix(rows, RANK_FEATURES)
    y = np.asarray([int(row["label_supported"]) for row in rows], dtype=int)
    model = make_model(model_name, seed)
    if calibrate:
        model = CalibratedClassifierCV(model, method="isotonic", cv=3)
    model.fit(X, y)
    return model, float(np.mean(y))


def prior_correct_scores(scores: np.ndarray, training_prior: float, deployment_prior: float) -> np.ndarray:
    eps = 1e-12
    scores = np.clip(scores, eps, 1.0 - eps)
    training_prior = min(max(training_prior, eps), 1.0 - eps)
    deployment_prior = min(max(deployment_prior, eps), 1.0 - eps)
    odds = scores / (1.0 - scores)
    prior_odds_ratio = (deployment_prior / (1.0 - deployment_prior)) / (
        training_prior / (1.0 - training_prior)
    )
    corrected_odds = odds * prior_odds_ratio
    return corrected_odds / (1.0 + corrected_odds)


def ligand_title(builder: DatasetBuilder, cid: str) -> str:
    return builder.ligands.get(cid, {}).get("Title", "")


def scored_rows_for_ligand(
    builder: DatasetBuilder,
    model: object,
    cid: str,
    yamanishi_pairs: set[tuple[str, str]],
    bindingdb_pairs: set[tuple[str, str]],
    ligand_degree: Counter[str],
    target_degree: Counter[str],
    target_universe: set[str],
    bindingdb_target_universe: set[str],
    training_prior: float,
    deployment_prior: float | None,
) -> list[dict[str, str | float | int]]:
    affinity_table = full_uniprot_affinity_table(builder, cid, exclude_frequent_top_hits=False)
    if not affinity_table:
        return []

    ranked = sorted(affinity_table.items(), key=lambda item: (item[1], item[0]))
    stats = affinity_stats([affinity for _, affinity in ranked])
    feature_rows = []
    for rank, (uniprot, affinity) in enumerate(ranked, start=1):
        feature_rows.append(
            make_row(
                builder,
                builder.ligands,
                {},
                {},
                {},
                ligand_degree,
                target_degree,
                target_universe,
                bindingdb_target_universe,
                cid,
                uniprot,
                affinity,
                rank,
                ranked,
                stats,
                (cid, uniprot) in yamanishi_pairs,
                (cid, uniprot) in bindingdb_pairs,
                include_wide_features=False,
            )
        )

    sample_scores = predict_scores(model, numeric_matrix(feature_rows, RANK_FEATURES))
    scores = (
        prior_correct_scores(sample_scores, training_prior, deployment_prior)
        if deployment_prior is not None
        else sample_scores
    )
    rows: list[dict[str, str | float | int]] = []
    title = ligand_title(builder, cid)
    for row, score, sample_score in zip(feature_rows, scores, sample_scores):
        yamanishi_known = int(row["label_yamanishi"])
        bindingdb_known = int(row["label_bindingdb"])
        rows.append(
            {
                "pubchem_cid": cid,
                "ligand_title": title,
                "uniprot_id": row["uniprot_id"],
                "known_target_yamanishi": yamanishi_known,
                "known_target_bindingdb": bindingdb_known,
                "known_target_any_supported": int(yamanishi_known or bindingdb_known),
                "target_likelihood": float(score),
                "target_likelihood_sample_prior": float(sample_score),
                "affinity": row["affinity"],
                "rank_1_based": row["rank_1_based"],
                "rank_percentile": row["rank_percentile"],
                "reverse_rank_percentile": row["reverse_rank_percentile"],
                "affinity_zscore_within_ligand": row["affinity_zscore_within_ligand"],
                "affinity_robust_zscore_within_ligand": row["affinity_robust_zscore_within_ligand"],
                "affinity_gap_to_next_weaker": row["affinity_gap_to_next_weaker"],
                "affinity_gap_to_previous_stronger": row["affinity_gap_to_previous_stronger"],
                "total_ranked_uniprots": row["total_ranked_uniprots"],
                "ligand_yamanishi_degree_any": row["ligand_yamanishi_degree_any"],
                "target_yamanishi_degree_any": row["target_yamanishi_degree_any"],
                "target_in_yamanishi_universe": row["target_in_yamanishi_universe"],
                "target_in_bindingdb_universe": row["target_in_bindingdb_universe"],
            }
        )
    rows.sort(key=lambda item: (-float(item["target_likelihood"]), int(item["rank_1_based"]), str(item["uniprot_id"])))
    for likelihood_rank, row in enumerate(rows, start=1):
        row["likelihood_rank_1_based"] = likelihood_rank
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--training-dataset", type=Path, default=Path("data/processed/affinity_hit_value_dataset_compact.csv"))
    parser.add_argument("--bindingdb-workbook", type=Path, default=Path("data/raw/old_PSB_Data.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("results/annotated_affinity_targets.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("results/annotated_affinity_target_summary.csv"))
    parser.add_argument("--cid", action="append", help="PubChem CID to score. May be passed more than once.")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=["hist_gradient_boosting", "logistic"], default="hist_gradient_boosting")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument(
        "--deployment-prior",
        type=float,
        default=2862 / 4906996,
        help=(
            "Prior probability used to convert sampled-training probabilities to full-candidate-space "
            "probabilities. Default is the observed Yamanishi-or-BindingDB support rate in the scored "
            "Yamanishi affinity universe."
        ),
    )
    parser.add_argument(
        "--no-prior-correction",
        action="store_true",
        help="Leave target_likelihood on the sampled training prior. Useful for ranking diagnostics only.",
    )
    args = parser.parse_args()

    builder = DatasetBuilder(args.data_dir, seed=args.seed)
    model, training_prior = train_likelihood_model(args.training_dataset, args.model, args.seed, args.calibrate)

    yamanishi_pairs = yamanishi_positive_pairs(builder)
    bindingdb_pairs = load_bindingdb_positive_pairs(args.bindingdb_workbook)
    ligand_degree, target_degree, target_universe = yamanishi_degrees(builder)
    bindingdb_target_universe = {uniprot for _, uniprot in bindingdb_pairs}
    cids = sorted(set(args.cid or builder.affinity_files))

    all_rows: list[dict[str, str | float | int]] = []
    summary_rows = []
    for cid in cids:
        rows = scored_rows_for_ligand(
            builder,
            model,
            cid,
            yamanishi_pairs,
            bindingdb_pairs,
            ligand_degree,
            target_degree,
            target_universe,
            bindingdb_target_universe,
            training_prior,
            None if args.no_prior_correction else args.deployment_prior,
        )
        if not rows:
            continue
        selected = [row for row in rows if float(row["target_likelihood"]) >= args.threshold]
        known_in_file = [row for row in rows if int(row["known_target_yamanishi"]) == 1]
        known_selected = [row for row in selected if int(row["known_target_yamanishi"]) == 1]
        all_rows.extend(rows)
        summary_rows.append(
            {
                "pubchem_cid": cid,
                "ligand_title": ligand_title(builder, cid),
                "ranked_uniprots": len(rows),
                "score_threshold": args.threshold,
                "predicted_targets_at_threshold": len(selected),
                "expected_supported_targets_at_threshold": sum(float(row["target_likelihood"]) for row in selected),
                "known_yamanishi_targets_in_affinity_file": len(known_in_file),
                "known_yamanishi_targets_recovered_at_threshold": len(known_selected),
                "known_yamanishi_targets_missed_at_threshold": len(known_in_file) - len(known_selected),
                "top_target_likelihood": max(float(row["target_likelihood"]) for row in rows),
            }
        )

    if not all_rows:
        raise SystemExit("No affinity targets were scored.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    with args.summary_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest = {
        "training_dataset": str(args.training_dataset),
        "output": str(args.output),
        "summary_output": str(args.summary_output),
        "model": args.model,
        "calibrate": args.calibrate,
        "training_prior": training_prior,
        "deployment_prior": None if args.no_prior_correction else args.deployment_prior,
        "feature_columns": RANK_FEATURES,
        "threshold": args.threshold,
        "ligands_scored": len(summary_rows),
        "rows_scored": len(all_rows),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary_output}")
    print(f"Wrote {manifest_path}")
    print(f"Rows scored: {len(all_rows)}")
    print(f"Ligands scored: {len(summary_rows)}")


if __name__ == "__main__":
    main()
