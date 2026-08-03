#!/usr/bin/env python3
"""Analyze where known Yamanishi targets fall in full affinity ranked lists."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import DatasetBuilder, read_xlsx_sheet  # noqa: E402


def full_uniprot_affinity_table(
    builder: DatasetBuilder,
    cid: str,
    exclude_frequent_top_hits: bool,
) -> dict[str, float]:
    pdb_affinities = builder._load_affinity_table(cid)
    uniprot_affinities: dict[str, float] = {}
    excluded = builder.excluded_affinity_uniprots if exclude_frequent_top_hits else set()
    for pdb_chain, affinity in pdb_affinities.items():
        pdb_id = pdb_chain.split("_", 1)[0].upper()
        uniprots = builder.pdb_chain_to_uniprot.get(pdb_chain)
        if not uniprots:
            uniprots = builder.pdb_to_uniprots.get(pdb_id, set())
        for uniprot in uniprots:
            if uniprot in excluded:
                continue
            if uniprot not in uniprot_affinities or affinity < uniprot_affinities[uniprot]:
                uniprot_affinities[uniprot] = affinity
    return uniprot_affinities


def known_positive_uniprots_by_ligand_category(builder: DatasetBuilder) -> dict[tuple[str, str], set[str]]:
    out: dict[tuple[str, str], set[str]] = defaultdict(set)
    for label in builder.labels:
        for uniprot in builder.hsa_to_uniprot.get(label.kegg_target, []):
            out[(label.category, label.kegg_drug)].add(uniprot)
    return out


def load_bindingdb_positive_pairs(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    pairs = set()
    for row in read_xlsx_sheet(path, "original"):
        if row.get("label") != "1":
            continue
        cid = row.get("cid", "").strip()
        uniprot = row.get("uniprot_id", "").strip()
        if cid and uniprot:
            pairs.add((cid, uniprot))
    return pairs


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--bindingdb-workbook", type=Path, default=Path("data/raw/old_PSB_Data.xlsx"))
    parser.add_argument("--exclude-frequent-top-hits", action="store_true")
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100, 200])
    args = parser.parse_args()

    builder = DatasetBuilder(args.data_dir, seed=42)
    known_uniprots = known_positive_uniprots_by_ligand_category(builder)
    bindingdb_positive_pairs = load_bindingdb_positive_pairs(args.bindingdb_workbook)
    run_name = "exclude_top_hits" if args.exclude_frequent_top_hits else "raw"

    rows = []
    status_counts: Counter[str] = Counter()
    for label in sorted(builder.labels, key=lambda item: (item.category, item.kegg_drug, item.kegg_target)):
        cid = builder.drug_to_cid.get(label.kegg_drug)
        if cid is None:
            status_counts["missing_kegg_drug_to_pubchem_cid"] += 1
            continue
        if cid not in builder.affinity_files:
            status_counts["missing_ligand_affinity_file"] += 1
            continue
        affinity_table = full_uniprot_affinity_table(builder, cid, args.exclude_frequent_top_hits)
        if not affinity_table:
            status_counts["empty_uniprot_affinity_table"] += 1
            continue
        target_uniprots = set(builder.hsa_to_uniprot.get(label.kegg_target, []))
        present_targets = target_uniprots & set(affinity_table)
        if not present_targets:
            status_counts["known_target_not_in_affinity_table"] += 1
            continue

        ranked = sorted(affinity_table.items(), key=lambda item: (item[1], item[0]))
        positions = {uniprot: index + 1 for index, (uniprot, _) in enumerate(ranked)}
        chosen_uniprot = min(present_targets, key=lambda uniprot: (positions[uniprot], uniprot))
        rank = positions[chosen_uniprot]
        higher_uniprots = [uniprot for uniprot, _ in ranked[: rank - 1]]
        higher_known = [
            uniprot
            for uniprot in higher_uniprots
            if uniprot in known_uniprots[(label.category, label.kegg_drug)]
        ]
        higher_bindingdb_known = [
            uniprot
            for uniprot in higher_uniprots
            if (cid, uniprot) in bindingdb_positive_pairs
        ]
        apparent_false_positives = len(higher_uniprots) - len(higher_known)
        unsupported_by_yamanishi_or_bindingdb = len(
            [
                uniprot
                for uniprot in higher_uniprots
                if uniprot not in known_uniprots[(label.category, label.kegg_drug)]
                and (cid, uniprot) not in bindingdb_positive_pairs
            ]
        )
        rows.append(
            {
                "category": label.category,
                "kegg_drug": label.kegg_drug,
                "pubchem_cid": cid,
                "kegg_target": label.kegg_target,
                "chosen_uniprot": chosen_uniprot,
                "chosen_affinity": affinity_table[chosen_uniprot],
                "rank_1_based": rank,
                "total_ranked_uniprots": len(ranked),
                "rank_percentile": rank / len(ranked),
                "higher_ranked_targets": len(higher_uniprots),
                "higher_ranked_yamanishi_known_positives": len(higher_known),
                "higher_ranked_bindingdb_known_positives": len(higher_bindingdb_known),
                "higher_ranked_apparent_false_positives": apparent_false_positives,
                "higher_ranked_apparent_fp_fraction": (
                    apparent_false_positives / len(higher_uniprots) if higher_uniprots else 0.0
                ),
                "higher_ranked_unsupported_by_yamanishi_or_bindingdb": unsupported_by_yamanishi_or_bindingdb,
                "higher_ranked_unsupported_fraction": (
                    unsupported_by_yamanishi_or_bindingdb / len(higher_uniprots) if higher_uniprots else 0.0
                ),
            }
        )
        status_counts["ranked_known_positive"] += 1

    args.results_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.results_dir / f"affinity_rank_{run_name}_known_positive_positions.csv"
    with detail_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    cutoff_rows = []
    total_ranked = len(rows)
    for cutoff in args.cutoffs:
        captured = [row for row in rows if int(row["rank_1_based"]) <= cutoff]
        above_counts = [
            int(row["higher_ranked_targets"])
            for row in rows
            if int(row["rank_1_based"]) <= cutoff and int(row["higher_ranked_targets"]) > 0
        ]
        above_fp_counts = [
            int(row["higher_ranked_apparent_false_positives"])
            for row in rows
            if int(row["rank_1_based"]) <= cutoff and int(row["higher_ranked_targets"]) > 0
        ]
        above_bindingdb_supported = [
            int(row["higher_ranked_bindingdb_known_positives"])
            for row in rows
            if int(row["rank_1_based"]) <= cutoff and int(row["higher_ranked_targets"]) > 0
        ]
        above_unsupported = [
            int(row["higher_ranked_unsupported_by_yamanishi_or_bindingdb"])
            for row in rows
            if int(row["rank_1_based"]) <= cutoff and int(row["higher_ranked_targets"]) > 0
        ]
        total_above = sum(above_counts)
        total_apparent_fp = sum(above_fp_counts)
        total_bindingdb_supported = sum(above_bindingdb_supported)
        total_unsupported = sum(above_unsupported)
        cutoff_rows.append(
            {
                "rank_cutoff": cutoff,
                "known_positives_captured": len(captured),
                "known_positive_recall": len(captured) / total_ranked if total_ranked else 0.0,
                "total_higher_ranked_calls_before_known_positive": total_above,
                "apparent_false_positive_calls_before_known_positive": total_apparent_fp,
                "apparent_fp_fraction_among_higher_ranked_calls": (
                    total_apparent_fp / total_above if total_above else 0.0
                ),
                "bindingdb_supported_calls_before_known_positive": total_bindingdb_supported,
                "bindingdb_supported_fraction_among_higher_ranked_calls": (
                    total_bindingdb_supported / total_above if total_above else 0.0
                ),
                "unsupported_calls_before_known_positive": total_unsupported,
                "unsupported_fraction_among_higher_ranked_calls": (
                    total_unsupported / total_above if total_above else 0.0
                ),
                "mean_higher_ranked_calls_for_captured_pairs": (
                    float(np.mean(above_counts)) if above_counts else 0.0
                ),
            }
        )
    cutoff_path = args.results_dir / f"affinity_rank_{run_name}_cutoff_summary.csv"
    with cutoff_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cutoff_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cutoff_rows)

    category_rows = []
    for category in sorted({row["category"] for row in rows}):
        subset = [row for row in rows if row["category"] == category]
        ranks = [float(row["rank_1_based"]) for row in subset]
        category_rows.append({"category": category, **summarize(ranks)})
    category_path = args.results_dir / f"affinity_rank_{run_name}_category_summary.csv"
    with category_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(category_rows[0].keys()))
        writer.writeheader()
        writer.writerows(category_rows)

    ranks = [float(row["rank_1_based"]) for row in rows]
    higher_counts = [float(row["higher_ranked_targets"]) for row in rows]
    summary = {
        "exclude_frequent_top_hits": args.exclude_frequent_top_hits,
        "bindingdb_positive_pairs": len(bindingdb_positive_pairs),
        "status_counts": dict(status_counts),
        "rank_summary": summarize(ranks),
        "higher_ranked_targets_before_known_positive_summary": summarize(higher_counts),
        "pairs_with_known_target_rank_1": sum(1 for row in rows if int(row["rank_1_based"]) == 1),
        "pairs_with_higher_ranked_apparent_false_positives": sum(
            1 for row in rows if int(row["higher_ranked_apparent_false_positives"]) > 0
        ),
        "pairs_with_higher_ranked_bindingdb_supported_targets": sum(
            1 for row in rows if int(row["higher_ranked_bindingdb_known_positives"]) > 0
        ),
    }
    summary_path = args.results_dir / f"affinity_rank_{run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote {detail_path}")
    print(f"Wrote {cutoff_path}")
    print(f"Wrote {category_path}")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
