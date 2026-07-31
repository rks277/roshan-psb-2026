#!/usr/bin/env python3
"""Report UniProt targets that are frequent top hits in ligand affinity files."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import DatasetBuilder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--min-top-hit-count", type=int, default=10)
    args = parser.parse_args()

    builder = DatasetBuilder(args.data_dir, seed=42)
    # This report should measure the raw affinity tables, not any previously
    # generated exclusion list.
    builder.excluded_affinity_uniprots = set()
    builder._uniprot_affinity_cache.clear()
    counts: Counter[str] = Counter()
    example_cids: dict[str, list[str]] = defaultdict(list)
    example_affinities: dict[str, list[float]] = defaultdict(list)

    for cid in sorted(builder.affinity_files):
        affinity_table = builder._load_uniprot_affinity_table(cid)
        if not affinity_table:
            continue
        best_affinity = min(affinity_table.values())
        best_uniprots = [
            uniprot
            for uniprot, affinity in affinity_table.items()
            if affinity == best_affinity
        ]
        for uniprot in best_uniprots:
            counts[uniprot] += 1
            if len(example_cids[uniprot]) < 8:
                example_cids[uniprot].append(cid)
                example_affinities[uniprot].append(best_affinity)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.results_dir / "top_affinity_hit_uniprots.csv"
    with report_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["uniprot_id", "top_hit_count", "example_cids", "example_affinities"])
        for uniprot, count in counts.most_common():
            writer.writerow(
                [
                    uniprot,
                    count,
                    ";".join(example_cids[uniprot]),
                    ";".join(f"{value:.6g}" for value in example_affinities[uniprot]),
                ]
            )

    exclude_path = args.data_dir / "excluded_top_affinity_uniprots.csv"
    with exclude_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["uniprot_id", "top_hit_count", "exclusion_rule"])
        for uniprot, count in counts.most_common():
            if count >= args.min_top_hit_count:
                writer.writerow([uniprot, count, f"top_hit_count>={args.min_top_hit_count}"])

    print(f"Wrote {report_path}")
    print(f"Wrote {exclude_path}")
    print(f"Excluded UniProt IDs: {sum(1 for count in counts.values() if count >= args.min_top_hit_count)}")
    for uniprot, count in counts.most_common(20):
        marker = "EXCLUDE" if count >= args.min_top_hit_count else "keep"
        print(f"{marker:7s} {count:3d} {uniprot}")


if __name__ == "__main__":
    main()
