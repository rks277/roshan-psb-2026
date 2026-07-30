#!/usr/bin/env python3
"""Report why Yamanishi positives drop from all labels to joined feature rows."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from build_dataset import DatasetBuilder


def main() -> None:
    data_dir = Path("data/raw")
    builder = DatasetBuilder(data_dir, seed=42)

    rows = []
    counts = Counter()
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    missing_examples: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    for key in sorted(builder.labels, key=lambda item: (item.category, item.kegg_drug, item.kegg_target)):
        counts["total_yamanishi_positive_pairs"] += 1
        category_counts[key.category]["total"] += 1

        cid = builder.drug_to_cid.get(key.kegg_drug)
        if cid is None:
            reason = "missing_kegg_drug_to_pubchem_cid"
        elif cid not in builder.ligands:
            reason = "missing_pubchem_descriptor_row"
        elif cid not in builder.affinity_files:
            reason = "missing_ligand_affinity_file"
        elif key.kegg_target not in builder.hsa_to_uniprot:
            reason = "missing_kegg_target_to_uniprot"
        else:
            affinity_table = builder._load_uniprot_affinity_table(cid)
            target_uniprots = builder.hsa_to_uniprot.get(key.kegg_target, [])
            if not any(uniprot in affinity_table for uniprot in target_uniprots):
                reason = "missing_uniprot_affinity_for_target"
            else:
                reason = "joined"

        counts[reason] += 1
        category_counts[key.category][reason] += 1
        if reason != "joined" and len(missing_examples[reason]) < 10:
            missing_examples[reason].append((key.category, key.kegg_drug, key.kegg_target))

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    summary_path = out_dir / "coverage_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step_or_reason", "positive_pairs"])
        for key, value in counts.most_common():
            writer.writerow([key, value])

    by_category_path = out_dir / "coverage_by_category.csv"
    reasons = sorted({reason for counter in category_counts.values() for reason in counter})
    with by_category_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", *reasons])
        for category in sorted(category_counts):
            writer.writerow([category, *[category_counts[category][reason] for reason in reasons]])

    examples_path = out_dir / "coverage_missing_examples.csv"
    with examples_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reason", "category", "kegg_drug", "kegg_target"])
        for reason, examples in sorted(missing_examples.items()):
            for category, drug, target in examples:
                writer.writerow([reason, category, drug, target])

    print(f"Wrote {summary_path}")
    print(f"Wrote {by_category_path}")
    print(f"Wrote {examples_path}")
    print()
    for key, value in counts.most_common():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
