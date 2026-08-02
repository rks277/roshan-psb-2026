#!/usr/bin/env python3
"""Predict glyco metabolite-protein pairs with the no-affinity Yamanishi model."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import LIGAND_FEATURE_COLUMNS, read_xlsx_sheet  # noqa: E402
from build_no_affinity_dataset import TARGET_SEQUENCE_FEATURE_COLUMNS, load_target_sequence_features  # noqa: E402
from train_no_affinity_models import numeric_matrix  # noqa: E402

METABOLITE_TO_CID = {
    "R5P": "77982",
    "RSP": "77982",
    "6GP": "69507",
    "G6P": "69507",
    "F6P": "439958",
    "FBP": "3281376",
    "G3P": "729",
    "3PG": "439183",
    "2PG": "439278",
    "PEP": "1005",
    "PYR": "107735",
    "Pyr": "107735",
    "LAC": "91435",
    "Lac": "91435",
}
METABOLITE_TO_SCORE_COLUMN = {
    "R5P": "RSP",
    "RSP": "RSP",
    "6GP": "6GP",
    "G6P": "6GP",
    "F6P": "F6P",
    "FBP": "FBP",
    "G3P": "G3P",
    "3PG": "3PG",
    "2PG": "2PG",
    "PEP": "PEP",
    "PYR": "Pyr",
    "Pyr": "Pyr",
    "LAC": "Lac",
    "Lac": "Lac",
}

TARGET_RICH_COLUMNS = (
    ["target_length", "target_mass", "target_degree_up"]
    + [f"target_{column}" for column in TARGET_SEQUENCE_FEATURE_COLUMNS]
)
FEATURE_COLUMNS = [f"ligand_{column}" for column in LIGAND_FEATURE_COLUMNS] + TARGET_RICH_COLUMNS


def load_ligands(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        return {row["CID"].strip(): row for row in csv.DictReader(handle)}


def load_target_rows(path: Path) -> dict[str, dict[str, str]]:
    sequence_features = load_target_sequence_features(path.parent)
    out = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            uniprot = row.get("entry", "").strip()
            if not uniprot:
                continue
            target_row = {
                "target_length": row.get("length", ""),
                "target_mass": row.get("mass", ""),
                "target_degree_up": row.get("degree (UP)", ""),
            }
            for column in TARGET_SEQUENCE_FEATURE_COLUMNS:
                target_row[f"target_{column}"] = sequence_features.get(uniprot, {}).get(column, "")
            out[uniprot] = target_row
    return out


def train_model(dataset_path: Path, seed: int, params_path: Path | None) -> Pipeline:
    with dataset_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    y = np.asarray([int(row["label"]) for row in rows], dtype=int)
    X = numeric_matrix(rows, FEATURE_COLUMNS)
    model_params = {"random_state": seed}
    if params_path and params_path.exists():
        manifest = json.loads(params_path.read_text())
        best_params = manifest.get("best_params", {})
        key = "pubchem_plus_target_rich::Hist Gradient Boosting tuned"
        for param_name, value in best_params.get(key, {}).items():
            if param_name.startswith("model__"):
                model_params[param_name.removeprefix("model__")] = value

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(**model_params)),
        ]
    )
    model.fit(X, y)
    return model


def make_prediction_row(
    source: dict[str, str],
    ligand: dict[str, str],
    target: dict[str, str],
    metabolite: str,
    cid: str,
) -> dict[str, str]:
    row = {
        "source_metabolite": metabolite,
        "pubchem_cid": cid,
        "protein_accession": source.get("Protein Accession", "").strip(),
        "gene_names": source.get("Gene names", ""),
        "description": source.get("Description", ""),
        "best_ranked_pdb": source.get("BestRanked", ""),
        "dock_score_for_metabolite": source.get(METABOLITE_TO_SCORE_COLUMN.get(metabolite, metabolite), ""),
        "label": "",
    }
    for column in LIGAND_FEATURE_COLUMNS:
        row[f"ligand_{column}"] = ligand.get(column, "")
    for column in TARGET_RICH_COLUMNS:
        row[column] = target.get(column, "")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-xlsx", type=Path, default=Path("data/raw/Analysis_Glyc_data_PDB_aff.xlsx"))
    parser.add_argument(
        "--ligands",
        type=Path,
        default=Path("data/raw/PubChem_properties_Glyco_intermediates.csv"),
    )
    parser.add_argument(
        "--target-source",
        type=Path,
        default=Path("data/raw/features.tsv"),
    )
    parser.add_argument(
        "--training-dataset",
        type=Path,
        default=Path("data/processed/yamanishi_no_affinity_classifier_dataset.csv"),
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=Path("results/no_affinity_pubchem_rich_target_tuned_model_manifest.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/glyco_pair_predictions.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ligands = load_ligands(args.ligands)
    targets = load_target_rows(args.target_source)
    model = train_model(args.training_dataset, args.seed, args.params)

    source_rows = read_xlsx_sheet(args.pairs_xlsx, "Sheet1")
    prediction_rows = []
    skipped = []
    for source in source_rows:
        metabolite = source.get("Metabolite", "").strip()
        if not metabolite:
            continue
        cid = METABOLITE_TO_CID.get(metabolite)
        protein = source.get("Protein Accession", "").strip()
        if not cid or cid not in ligands or protein not in targets:
            skipped.append(
                {
                    "metabolite": metabolite,
                    "mapped_cid": cid or "",
                    "protein_accession": protein,
                    "reason": (
                        "missing_ligand_properties"
                        if cid and cid not in ligands
                        else "missing_metabolite_mapping"
                        if not cid
                        else "missing_target_features"
                    ),
                }
            )
            continue
        row = make_prediction_row(source, ligands[cid], targets[protein], metabolite, cid)
        X = numeric_matrix([row], FEATURE_COLUMNS)
        score = float(model.predict_proba(X)[0, 1])
        row["predicted_interaction_probability"] = score
        row["predicted_label_at_0_5"] = int(score >= 0.5)
        prediction_rows.append(row)

    prediction_rows.sort(key=lambda row: row["predicted_interaction_probability"], reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if prediction_rows:
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0].keys()))
            writer.writeheader()
            writer.writerows(prediction_rows)

    compact_path = args.output.with_name(args.output.stem + "_compact.csv")
    compact_columns = [
        "source_metabolite",
        "pubchem_cid",
        "protein_accession",
        "gene_names",
        "description",
        "best_ranked_pdb",
        "dock_score_for_metabolite",
        "predicted_interaction_probability",
        "predicted_label_at_0_5",
    ]
    with compact_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=compact_columns)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in compact_columns} for row in prediction_rows)

    skipped_path = args.output.with_name(args.output.stem + "_skipped.csv")
    with skipped_path.open("w", newline="") as handle:
        fieldnames = ["metabolite", "mapped_cid", "protein_accession", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(skipped)

    summary_path = args.output.with_name(args.output.stem + "_summary_by_metabolite.csv")
    by_metabolite: dict[str, list[float]] = defaultdict(list)
    for row in prediction_rows:
        by_metabolite[row["source_metabolite"]].append(row["predicted_interaction_probability"])
    summary_rows = [
        {
            "metabolite": metabolite,
            "rows": len(values),
            "mean_probability": statistics.mean(values),
            "median_probability": statistics.median(values),
            "max_probability": max(values),
            "predicted_positive_at_0_5": sum(value >= 0.5 for value in values),
        }
        for metabolite, values in sorted(by_metabolite.items())
    ]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote predictions: {args.output} ({len(prediction_rows)} rows)")
    print(f"Wrote compact predictions: {compact_path}")
    print(f"Wrote skipped rows: {skipped_path} ({len(skipped)} rows)")
    print(f"Wrote summary: {summary_path}")
    for row in prediction_rows[:20]:
        print(
            f"{row['predicted_interaction_probability']:.3f} "
            f"{row['source_metabolite']} {row['protein_accession']} {row['gene_names']}"
        )


if __name__ == "__main__":
    main()
