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

from build_dataset import DatasetBuilder  # noqa: E402
from build_affinity_hit_value_dataset import (
    LIGAND_OUTPUT_FEATURES,
    MACCS_OUTPUT_FEATURES,
    MORGAN_OUTPUT_FEATURES,
    RANK_FEATURES,
    TARGET_BASIC_OUTPUT_FEATURES,
    TARGET_SEQUENCE_OUTPUT_FEATURES,
)
from build_no_affinity_dataset import load_maccs_features, load_morgan_features, load_target_sequence_features

LABEL_PRIOR_FEATURES = [
    "ligand_yamanishi_degree_any",
    "target_yamanishi_degree_any",
    "target_in_yamanishi_universe",
    "target_in_bindingdb_universe",
]
CLEAN_RANK_FEATURES = [column for column in RANK_FEATURES if column not in LABEL_PRIOR_FEATURES]
PAIR_INTERACTION_FEATURES = [
    "pair_xlogp_x_target_hydrophobic",
    "pair_tpsa_x_target_polar",
    "pair_hbd_x_target_polar",
    "pair_hba_x_target_polar",
    "pair_charge_x_target_charged",
    "pair_ring_x_target_aromatic",
    "pair_hydrophobe_x_target_hydrophobic",
    "pair_mw_per_target_length",
    "pair_rotatable_per_target_length",
    "pair_affinity_z_x_xlogp",
    "pair_affinity_z_x_target_hydrophobic",
    "pair_affinity_z_x_target_charged",
]

FEATURE_SETS = {
    "rank_only": RANK_FEATURES,
    "clean_rank_only": CLEAN_RANK_FEATURES,
    "clean_rank_plus_basic_context": CLEAN_RANK_FEATURES + LIGAND_OUTPUT_FEATURES + TARGET_BASIC_OUTPUT_FEATURES,
    "clean_rank_plus_maccs_basic_target": (
        CLEAN_RANK_FEATURES
        + LIGAND_OUTPUT_FEATURES
        + MACCS_OUTPUT_FEATURES
        + TARGET_BASIC_OUTPUT_FEATURES
    ),
    "clean_rank_plus_morgan_basic_target": (
        CLEAN_RANK_FEATURES
        + LIGAND_OUTPUT_FEATURES
        + MORGAN_OUTPUT_FEATURES
        + TARGET_BASIC_OUTPUT_FEATURES
    ),
    "clean_rank_plus_maccs_morgan_basic_target": (
        CLEAN_RANK_FEATURES
        + LIGAND_OUTPUT_FEATURES
        + MACCS_OUTPUT_FEATURES
        + MORGAN_OUTPUT_FEATURES
        + TARGET_BASIC_OUTPUT_FEATURES
    ),
    "clean_rank_plus_maccs_morgan_target": (
        CLEAN_RANK_FEATURES
        + LIGAND_OUTPUT_FEATURES
        + MACCS_OUTPUT_FEATURES
        + MORGAN_OUTPUT_FEATURES
        + TARGET_BASIC_OUTPUT_FEATURES
        + TARGET_SEQUENCE_OUTPUT_FEATURES
    ),
    "clean_rank_plus_maccs_morgan_target_interactions": (
        CLEAN_RANK_FEATURES
        + LIGAND_OUTPUT_FEATURES
        + MACCS_OUTPUT_FEATURES
        + MORGAN_OUTPUT_FEATURES
        + TARGET_BASIC_OUTPUT_FEATURES
        + TARGET_SEQUENCE_OUTPUT_FEATURES
        + PAIR_INTERACTION_FEATURES
    ),
    "rank_plus_basic_context": RANK_FEATURES + LIGAND_OUTPUT_FEATURES + TARGET_BASIC_OUTPUT_FEATURES,
    "rank_plus_maccs_basic_target": (
        RANK_FEATURES
        + LIGAND_OUTPUT_FEATURES
        + MACCS_OUTPUT_FEATURES
        + TARGET_BASIC_OUTPUT_FEATURES
    ),
    "rank_plus_maccs_morgan_target": (
        RANK_FEATURES
        + LIGAND_OUTPUT_FEATURES
        + MACCS_OUTPUT_FEATURES
        + MORGAN_OUTPUT_FEATURES
        + TARGET_BASIC_OUTPUT_FEATURES
        + TARGET_SEQUENCE_OUTPUT_FEATURES
    ),
}


def parse_float(value: str | float | int | None) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def load_feature_maps(data_dir: Path, seed: int) -> dict[str, dict[str, dict[str, str]]]:
    builder = DatasetBuilder(data_dir, seed=seed)
    return {
        "ligand": builder.ligands,
        "maccs": load_maccs_features(data_dir),
        "morgan": load_morgan_features(data_dir),
        "target": builder.target_features,
        "target_sequence": load_target_sequence_features(data_dir),
    }


def augmented_value(
    row: dict[str, str],
    column: str,
    feature_maps: dict[str, dict[str, dict[str, str]]] | None,
) -> str:
    value = row.get(column, "")
    if value or feature_maps is None:
        return value
    if column.startswith("ligand_"):
        cid = row["pubchem_cid"]
        raw_column = column.removeprefix("ligand_")
        for group in ("ligand", "maccs", "morgan"):
            value = feature_maps[group].get(cid, {}).get(raw_column, "")
            if value:
                return value
    if column.startswith("target_"):
        uniprot = row["uniprot_id"]
        raw_column = column.removeprefix("target_")
        for group in ("target", "target_sequence"):
            value = feature_maps[group].get(uniprot, {}).get(raw_column, "")
            if value:
                return value
    return ""


def numeric_matrix(
    rows: list[dict[str, str]],
    columns: list[str],
    feature_maps: dict[str, dict[str, dict[str, str]]] | None = None,
) -> np.ndarray:
    compact_columns = set(rows[0])
    blocks = []
    direct_columns = [column for column in columns if column in compact_columns]
    if direct_columns:
        blocks.append(
            np.asarray(
                [[parse_float(row.get(column, "")) for column in direct_columns] for row in rows],
                dtype=np.float32,
            )
        )
    if feature_maps is not None:
        ligand_columns = [column for column in columns if column.startswith("ligand_") and column not in compact_columns]
        target_columns = [column for column in columns if column.startswith("target_") and column not in compact_columns]
        pair_columns = [column for column in columns if column.startswith("pair_")]
        if ligand_columns:
            blocks.append(entity_feature_block(rows, "pubchem_cid", ligand_columns, feature_maps, ("ligand", "maccs", "morgan")))
        if target_columns:
            blocks.append(entity_feature_block(rows, "uniprot_id", target_columns, feature_maps, ("target", "target_sequence")))
        if pair_columns:
            blocks.append(pair_interaction_block(rows, pair_columns, feature_maps))
    if not blocks:
        return np.empty((len(rows), 0), dtype=np.float32)
    return np.concatenate(blocks, axis=1)


def entity_feature_block(
    rows: list[dict[str, str]],
    id_column: str,
    columns: list[str],
    feature_maps: dict[str, dict[str, dict[str, str]]],
    groups: tuple[str, ...],
) -> np.ndarray:
    entity_ids = np.asarray([row[id_column] for row in rows])
    unique_ids, inverse = np.unique(entity_ids, return_inverse=True)
    prefix = "ligand_" if id_column == "pubchem_cid" else "target_"
    unique_matrix = np.empty((len(unique_ids), len(columns)), dtype=np.float32)
    for entity_idx, entity_id in enumerate(unique_ids):
        for column_idx, column in enumerate(columns):
            raw_column = column.removeprefix(prefix)
            value = ""
            for group in groups:
                value = feature_maps[group].get(entity_id, {}).get(raw_column, "")
                if value:
                    break
            unique_matrix[entity_idx, column_idx] = parse_float(value)
    return unique_matrix[inverse]


def lookup_numeric(
    feature_maps: dict[str, dict[str, dict[str, str]]],
    groups: tuple[str, ...],
    entity_id: str,
    column: str,
) -> float:
    for group in groups:
        value = feature_maps[group].get(entity_id, {}).get(column, "")
        if value:
            return parse_float(value)
    return np.nan


def safe_divide(numerator: float, denominator: float) -> float:
    if np.isnan(numerator) or np.isnan(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def pair_interaction_block(
    rows: list[dict[str, str]],
    columns: list[str],
    feature_maps: dict[str, dict[str, dict[str, str]]],
) -> np.ndarray:
    matrix = np.empty((len(rows), len(columns)), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        cid = row["pubchem_cid"]
        uniprot = row["uniprot_id"]
        affinity_z = parse_float(row.get("affinity_zscore_within_ligand", ""))
        ligand = {
            "xlogp": lookup_numeric(feature_maps, ("ligand",), cid, "XLogP"),
            "tpsa": lookup_numeric(feature_maps, ("ligand",), cid, "TPSA"),
            "hbd": lookup_numeric(feature_maps, ("ligand",), cid, "HBondDonorCount"),
            "hba": lookup_numeric(feature_maps, ("ligand",), cid, "HBondAcceptorCount"),
            "charge": lookup_numeric(feature_maps, ("ligand",), cid, "Charge"),
            "rings": lookup_numeric(feature_maps, ("ligand",), cid, "FeatureRingCount3D"),
            "hydrophobes": lookup_numeric(feature_maps, ("ligand",), cid, "FeatureHydrophobeCount3D"),
            "mw": lookup_numeric(feature_maps, ("ligand",), cid, "MolecularWeight"),
            "rotatable": lookup_numeric(feature_maps, ("ligand",), cid, "RotatableBondCount"),
        }
        target = {
            "length": lookup_numeric(feature_maps, ("target",), uniprot, "length"),
            "hydrophobic": lookup_numeric(feature_maps, ("target_sequence",), uniprot, "group_hydrophobic"),
            "polar": lookup_numeric(feature_maps, ("target_sequence",), uniprot, "group_polar"),
            "charged": lookup_numeric(feature_maps, ("target_sequence",), uniprot, "group_charged"),
            "aromatic": lookup_numeric(feature_maps, ("target_sequence",), uniprot, "group_aromatic"),
        }
        values = {
            "pair_xlogp_x_target_hydrophobic": ligand["xlogp"] * target["hydrophobic"],
            "pair_tpsa_x_target_polar": ligand["tpsa"] * target["polar"],
            "pair_hbd_x_target_polar": ligand["hbd"] * target["polar"],
            "pair_hba_x_target_polar": ligand["hba"] * target["polar"],
            "pair_charge_x_target_charged": ligand["charge"] * target["charged"],
            "pair_ring_x_target_aromatic": ligand["rings"] * target["aromatic"],
            "pair_hydrophobe_x_target_hydrophobic": ligand["hydrophobes"] * target["hydrophobic"],
            "pair_mw_per_target_length": safe_divide(ligand["mw"], target["length"]),
            "pair_rotatable_per_target_length": safe_divide(ligand["rotatable"], target["length"]),
            "pair_affinity_z_x_xlogp": affinity_z * ligand["xlogp"],
            "pair_affinity_z_x_target_hydrophobic": affinity_z * target["hydrophobic"],
            "pair_affinity_z_x_target_charged": affinity_z * target["charged"],
        }
        for column_idx, column in enumerate(columns):
            matrix[row_idx, column_idx] = values[column]
    return matrix


def feature_set_available(
    available_columns: set[str],
    columns: list[str],
    feature_maps: dict[str, dict[str, dict[str, str]]] | None,
) -> tuple[bool, int]:
    if feature_maps is not None:
        return True, 0
    missing_columns = [column for column in columns if column not in available_columns]
    return len(missing_columns) == 0, len(missing_columns)


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
        "Extra Trees Fast": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=160,
                        max_features="sqrt",
                        min_samples_leaf=2,
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
        "Hist Gradient Boosting Deep": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        random_state=seed,
                        max_iter=300,
                        learning_rate=0.04,
                        max_leaf_nodes=31,
                        min_samples_leaf=20,
                        l2_regularization=0.05,
                    ),
                ),
            ]
        ),
        "Hist Gradient Boosting Wide": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        random_state=seed,
                        max_iter=220,
                        learning_rate=0.06,
                        max_leaf_nodes=63,
                        min_samples_leaf=15,
                        l2_regularization=0.01,
                    ),
                ),
            ]
        ),
    }


def selected_items(items: dict[str, object], selected: list[str] | None) -> dict[str, object]:
    if not selected:
        return items
    wanted = set(selected)
    missing = sorted(wanted - set(items))
    if missing:
        raise SystemExit(f"Unknown selection(s): {', '.join(missing)}")
    return {name: item for name, item in items.items() if name in wanted}


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
    parser.add_argument("--feature-sets", nargs="+", choices=sorted(FEATURE_SETS), default=None)
    parser.add_argument("--classifiers", nargs="+", choices=sorted(make_models(42)), default=None)
    parser.add_argument("--skip-scored-output", action="store_true")
    parser.add_argument(
        "--augment-feature-maps",
        action="store_true",
        help="Join ligand chemistry and protein biology features from data/raw and data/processed at training time.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
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
    feature_maps = load_feature_maps(args.data_dir, args.seed) if args.augment_feature_maps else None
    metrics = []
    enrichment = []
    best = None
    best_score = -1.0
    feature_sets = selected_items(FEATURE_SETS, args.feature_sets)
    models = selected_items(make_models(args.seed), args.classifiers)
    for feature_set_name, columns in feature_sets.items():
        is_available, missing_count = feature_set_available(available_columns, columns, feature_maps)
        if not is_available:
            print(f"Skipping {feature_set_name}: {missing_count} feature columns are absent from {args.dataset}")
            continue
        X = numeric_matrix(rows, columns, feature_maps)
        X_train, X_test = X[train_idx], X[test_idx]
        for model_name, model in models.items():
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

    scored_path = args.results_dir / f"{args.output_prefix}_scored_sample.csv"
    assert best is not None
    best_feature_set, best_model_name, best_model, best_columns, X_best, _ = best
    if not args.skip_scored_output:
        all_scores = scores(best_model, X_best)
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
                "augment_feature_maps": args.augment_feature_maps,
                "calibrate": args.calibrate,
                "skip_scored_output": args.skip_scored_output,
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
    if not args.skip_scored_output:
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
