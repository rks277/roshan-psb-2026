#!/usr/bin/env python3
"""Export a clean-feature reranked affinity list for one PubChem CID."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_affinity_rank_positions import full_uniprot_affinity_table, load_bindingdb_positive_pairs  # noqa: E402
from build_affinity_hit_value_dataset import (  # noqa: E402
    affinity_stats,
    make_row,
    yamanishi_degrees,
    yamanishi_positive_pairs,
)
from build_dataset import DatasetBuilder  # noqa: E402
from train_affinity_hit_value_model import FEATURE_SETS, load_feature_maps, numeric_matrix, scores  # noqa: E402


FEATURE_SET = "clean_rank_plus_maccs_morgan_target"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def ligand_title(builder: DatasetBuilder, cid: str) -> str:
    title = builder.ligands.get(cid, {}).get("Title", "")
    if title:
        return title
    for affinity_file in builder.affinity_files.get(cid, []):
        match = Path(affinity_file).name
        if match.startswith(f"CID_{cid}_"):
            return match.removeprefix(f"CID_{cid}_").removesuffix(".pdb_affinities.txt")
    return ""


def make_ligand_rows(
    builder: DatasetBuilder,
    cid: str,
    bindingdb_workbook: Path,
) -> list[dict[str, str]]:
    affinity_table = full_uniprot_affinity_table(builder, cid, exclude_frequent_top_hits=False)
    if not affinity_table:
        return []

    yamanishi_pairs = yamanishi_positive_pairs(builder)
    bindingdb_pairs = load_bindingdb_positive_pairs(bindingdb_workbook)
    ligand_degree, target_degree, target_universe = yamanishi_degrees(builder)
    bindingdb_target_universe = {uniprot for _, uniprot in bindingdb_pairs}
    ranked = sorted(affinity_table.items(), key=lambda item: (item[1], item[0]))
    stats = affinity_stats([affinity for _, affinity in ranked])

    out = []
    for rank, (uniprot, affinity) in enumerate(ranked, start=1):
        out.append(
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
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/affinity_hit_value_dataset_compact.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--bindingdb-workbook", type=Path, default=Path("data/raw/old_PSB_Data.xlsx"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    builder = DatasetBuilder(args.data_dir, seed=args.seed)
    rows = read_rows(args.dataset)
    y = np.asarray([int(row["label_supported"]) for row in rows], dtype=int)
    feature_maps = load_feature_maps(args.data_dir, args.seed)
    columns = FEATURE_SETS[FEATURE_SET]
    X = numeric_matrix(rows, columns, feature_maps)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(random_state=args.seed)),
        ]
    )
    model.fit(X, y)

    ligand_rows = make_ligand_rows(builder, args.cid, args.bindingdb_workbook)
    if not ligand_rows:
        raise SystemExit(f"No local affinity rows found for PubChem CID {args.cid}")
    ligand_X = numeric_matrix(ligand_rows, columns, feature_maps)
    hit_scores = scores(model, ligand_X)
    title = ligand_title(builder, args.cid)

    exported = []
    for row, score in zip(ligand_rows, hit_scores):
        exported.append(
            {
                "pubchem_cid": args.cid,
                "ligand_title": title,
                "uniprot_id": row["uniprot_id"],
                "known_target_yamanishi": row["label_yamanishi"],
                "known_target_bindingdb": row["label_bindingdb"],
                "known_target_any_supported": row["label_supported"],
                "hit_value_score": float(score),
                "affinity": row["affinity"],
                "raw_affinity_rank_1_based": row["rank_1_based"],
                "rank_percentile": row["rank_percentile"],
                "reverse_rank_percentile": row["reverse_rank_percentile"],
                "affinity_zscore_within_ligand": row["affinity_zscore_within_ligand"],
                "affinity_robust_zscore_within_ligand": row["affinity_robust_zscore_within_ligand"],
                "affinity_gap_to_next_weaker": row["affinity_gap_to_next_weaker"],
                "affinity_gap_to_previous_stronger": row["affinity_gap_to_previous_stronger"],
                "total_ranked_uniprots": row["total_ranked_uniprots"],
            }
        )

    exported.sort(
        key=lambda row: (
            -float(row["hit_value_score"]),
            int(row["raw_affinity_rank_1_based"]),
            row["uniprot_id"],
        )
    )
    for rank, row in enumerate(exported, start=1):
        row["reranked_rank_1_based"] = rank

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "reranked_rank_1_based",
        "pubchem_cid",
        "ligand_title",
        "uniprot_id",
        "known_target_yamanishi",
        "known_target_bindingdb",
        "known_target_any_supported",
        "hit_value_score",
        "affinity",
        "raw_affinity_rank_1_based",
        "rank_percentile",
        "reverse_rank_percentile",
        "affinity_zscore_within_ligand",
        "affinity_robust_zscore_within_ligand",
        "affinity_gap_to_next_weaker",
        "affinity_gap_to_previous_stronger",
        "total_ranked_uniprots",
    ]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(exported)

    manifest = {
        "cid": args.cid,
        "ligand_title": title,
        "training_dataset": str(args.dataset),
        "feature_set": FEATURE_SET,
        "feature_count": len(columns),
        "model": "HistGradientBoostingClassifier(random_state=42)",
        "training_rows": len(rows),
        "training_supported_rows": int(np.sum(y == 1)),
        "ranked_rows_exported": len(exported),
        "output": str(args.output),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote {args.output}")
    print(f"Wrote {manifest_path}")
    print(f"Ligand title: {title}")
    print(f"Rows exported: {len(exported)}")
    print("Top 10:")
    for row in exported[:10]:
        print(
            f"{row['reranked_rank_1_based']:>3} {row['uniprot_id']:>10} "
            f"score={float(row['hit_value_score']):.4f} "
            f"raw_rank={row['raw_affinity_rank_1_based']} "
            f"affinity={row['affinity']} "
            f"supported={row['known_target_any_supported']}"
        )


if __name__ == "__main__":
    main()
