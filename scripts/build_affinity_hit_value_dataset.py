#!/usr/bin/env python3
"""Build a sampled per-affinity-hit dataset for value/confidence modeling."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_affinity_rank_positions import full_uniprot_affinity_table, load_bindingdb_positive_pairs  # noqa: E402
from build_dataset import DatasetBuilder, LIGAND_FEATURE_COLUMNS, TARGET_FEATURE_COLUMNS  # noqa: E402
from build_no_affinity_dataset import (  # noqa: E402
    MACCS_FEATURE_COLUMNS,
    MORGAN_FEATURE_COLUMNS,
    TARGET_SEQUENCE_FEATURE_COLUMNS,
    load_maccs_features,
    load_morgan_features,
    load_target_sequence_features,
)

LIGAND_OUTPUT_FEATURES = [f"ligand_{column}" for column in LIGAND_FEATURE_COLUMNS]
MACCS_OUTPUT_FEATURES = [f"ligand_{column}" for column in MACCS_FEATURE_COLUMNS]
MORGAN_OUTPUT_FEATURES = [f"ligand_{column}" for column in MORGAN_FEATURE_COLUMNS]
TARGET_BASIC_OUTPUT_FEATURES = [f"target_{column}" for column in TARGET_FEATURE_COLUMNS]
TARGET_SEQUENCE_OUTPUT_FEATURES = [f"target_{column}" for column in TARGET_SEQUENCE_FEATURE_COLUMNS]
RANK_FEATURES = [
    "affinity",
    "rank_1_based",
    "rank_percentile",
    "reverse_rank_percentile",
    "affinity_zscore_within_ligand",
    "affinity_robust_zscore_within_ligand",
    "affinity_gap_to_next_weaker",
    "affinity_gap_to_previous_stronger",
    "total_ranked_uniprots",
    "ligand_yamanishi_degree_any",
    "target_yamanishi_degree_any",
    "target_in_yamanishi_universe",
    "target_in_bindingdb_universe",
]


def yamanishi_positive_pairs(builder: DatasetBuilder) -> set[tuple[str, str]]:
    pairs = set()
    for label in builder.labels:
        cid = builder.drug_to_cid.get(label.kegg_drug)
        if cid is None:
            continue
        for uniprot in builder.hsa_to_uniprot.get(label.kegg_target, []):
            pairs.add((cid, uniprot))
    return pairs


def yamanishi_degrees(builder: DatasetBuilder) -> tuple[Counter[str], Counter[str], set[str]]:
    ligand_degree: Counter[str] = Counter()
    target_degree: Counter[str] = Counter()
    target_universe = set()
    ligand_targets: dict[str, set[str]] = defaultdict(set)
    target_ligands: dict[str, set[str]] = defaultdict(set)
    for label in builder.labels:
        cid = builder.drug_to_cid.get(label.kegg_drug)
        if cid is None:
            continue
        for uniprot in builder.hsa_to_uniprot.get(label.kegg_target, []):
            ligand_targets[cid].add(uniprot)
            target_ligands[uniprot].add(cid)
            target_universe.add(uniprot)
    for cid, targets in ligand_targets.items():
        ligand_degree[cid] = len(targets)
    for uniprot, cids in target_ligands.items():
        target_degree[uniprot] = len(cids)
    return ligand_degree, target_degree, target_universe


def affinity_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    std = float(np.std(arr))
    return {
        "mean": float(np.mean(arr)),
        "std": std if std > 0 else 1.0,
        "median": median,
        "mad": mad if mad > 0 else 1.0,
    }


def make_row(
    builder: DatasetBuilder,
    ligand_features: dict[str, dict[str, str]],
    maccs_features: dict[str, dict[str, str]],
    morgan_features: dict[str, dict[str, str]],
    target_sequence_features: dict[str, dict[str, str]],
    ligand_degree: Counter[str],
    target_degree: Counter[str],
    target_universe: set[str],
    bindingdb_target_universe: set[str],
    cid: str,
    uniprot: str,
    affinity: float,
    rank: int,
    ranked: list[tuple[str, float]],
    stats: dict[str, float],
    yamanishi_supported: bool,
    bindingdb_supported: bool,
    include_wide_features: bool,
) -> dict[str, str]:
    total = len(ranked)
    previous_affinity = ranked[rank - 2][1] if rank > 1 else affinity
    next_affinity = ranked[rank][1] if rank < total else affinity
    row = {
        "pubchem_cid": cid,
        "uniprot_id": uniprot,
        "label_supported": str(int(yamanishi_supported or bindingdb_supported)),
        "label_yamanishi": str(int(yamanishi_supported)),
        "label_bindingdb": str(int(bindingdb_supported)),
        "affinity": str(affinity),
        "rank_1_based": str(rank),
        "rank_percentile": str(rank / total),
        "reverse_rank_percentile": str(1.0 - ((rank - 1) / total)),
        "affinity_zscore_within_ligand": str((affinity - stats["mean"]) / stats["std"]),
        "affinity_robust_zscore_within_ligand": str((affinity - stats["median"]) / stats["mad"]),
        "affinity_gap_to_next_weaker": str(next_affinity - affinity),
        "affinity_gap_to_previous_stronger": str(affinity - previous_affinity),
        "total_ranked_uniprots": str(total),
        "ligand_yamanishi_degree_any": str(ligand_degree[cid]),
        "target_yamanishi_degree_any": str(target_degree[uniprot]),
        "target_in_yamanishi_universe": str(int(uniprot in target_universe)),
        "target_in_bindingdb_universe": str(int(uniprot in bindingdb_target_universe)),
    }
    if not include_wide_features:
        return row

    ligand = ligand_features.get(cid, {})
    for column in LIGAND_FEATURE_COLUMNS:
        row[f"ligand_{column}"] = ligand.get(column, "")
    ligand_maccs = maccs_features.get(cid, {})
    for column in MACCS_FEATURE_COLUMNS:
        row[f"ligand_{column}"] = ligand_maccs.get(column, "")
    ligand_morgan = morgan_features.get(cid, {})
    for column in MORGAN_FEATURE_COLUMNS:
        row[f"ligand_{column}"] = ligand_morgan.get(column, "")
    target = builder.target_features.get(uniprot, {})
    for column in TARGET_FEATURE_COLUMNS:
        row[f"target_{column}"] = target.get(column, "")
    target_sequence = target_sequence_features.get(uniprot, {})
    for column in TARGET_SEQUENCE_FEATURE_COLUMNS:
        row[f"target_{column}"] = target_sequence.get(column, "")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/affinity_hit_value_dataset_compact.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("results/affinity_hit_value_dataset_summary.csv"))
    parser.add_argument("--bindingdb-workbook", type=Path, default=Path("data/raw/old_PSB_Data.xlsx"))
    parser.add_argument("--negative-ratio", type=int, default=20)
    parser.add_argument("--max-unsupported-per-ligand", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-wide-features",
        action="store_true",
        help="Append ligand, fingerprint, and protein feature columns. This can create multi-GB outputs.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    builder = DatasetBuilder(args.data_dir, seed=args.seed)
    ligand_features = builder.ligands
    maccs_features = load_maccs_features(args.data_dir)
    morgan_features = load_morgan_features(args.data_dir)
    target_sequence_features = load_target_sequence_features(args.data_dir)

    yamanishi_pairs = yamanishi_positive_pairs(builder)
    bindingdb_pairs = load_bindingdb_positive_pairs(args.bindingdb_workbook)
    supported_pairs = yamanishi_pairs | bindingdb_pairs
    ligand_degree, target_degree, target_universe = yamanishi_degrees(builder)
    bindingdb_target_universe = {uniprot for _, uniprot in bindingdb_pairs}

    rows = []
    summary_rows = []
    for cid in sorted(builder.affinity_files):
        affinity_table = full_uniprot_affinity_table(builder, cid, exclude_frequent_top_hits=False)
        if not affinity_table:
            continue
        ranked = sorted(affinity_table.items(), key=lambda item: (item[1], item[0]))
        stats = affinity_stats([affinity for _, affinity in ranked])
        supported = [
            (rank, uniprot, affinity)
            for rank, (uniprot, affinity) in enumerate(ranked, start=1)
            if (cid, uniprot) in supported_pairs
        ]
        unsupported = [
            (rank, uniprot, affinity)
            for rank, (uniprot, affinity) in enumerate(ranked, start=1)
            if (cid, uniprot) not in supported_pairs
        ]
        sample_size = min(
            len(unsupported),
            max(args.max_unsupported_per_ligand, len(supported) * args.negative_ratio),
        )
        sampled_unsupported = rng.sample(unsupported, sample_size) if sample_size < len(unsupported) else unsupported
        for rank, uniprot, affinity in supported + sampled_unsupported:
            rows.append(
                make_row(
                    builder,
                    ligand_features,
                    maccs_features,
                    morgan_features,
                    target_sequence_features,
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
                    args.include_wide_features,
                )
            )
        summary_rows.append(
            {
                "pubchem_cid": cid,
                "ranked_uniprots": len(ranked),
                "supported_hits": len(supported),
                "sampled_unsupported_hits": len(sampled_unsupported),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    label_counts = Counter(row["label_supported"] for row in rows)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary_output}")
    print(f"Rows: {len(rows)}")
    print(f"Supported labels: {dict(label_counts)}")
    print(f"Ligands with affinity lists: {len(summary_rows)}")


if __name__ == "__main__":
    main()
