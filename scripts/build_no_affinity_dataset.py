#!/usr/bin/env python3
"""Build a Yamanishi classifier table without affinity-derived features."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import DatasetBuilder, LIGAND_FEATURE_COLUMNS, TARGET_FEATURE_COLUMNS  # noqa: E402


def choose_target_uniprot(builder: DatasetBuilder, kegg_target: str) -> str | None:
    for uniprot in builder.hsa_to_uniprot.get(kegg_target, []):
        if uniprot in builder.target_features:
            return uniprot
    return None


def make_no_affinity_row(
    builder: DatasetBuilder,
    category: str,
    kegg_drug: str,
    kegg_target: str,
    label: int,
) -> dict[str, str] | None:
    cid = builder.drug_to_cid.get(kegg_drug)
    if cid is None:
        return None
    ligand = builder.ligands.get(cid)
    if ligand is None:
        return None
    uniprot = choose_target_uniprot(builder, kegg_target)
    if uniprot is None:
        return None

    row: dict[str, str] = {
        "category": category,
        "kegg_drug": kegg_drug,
        "pubchem_cid": cid,
        "ligand_title": ligand.get("Title", ""),
        "kegg_target": kegg_target,
        "uniprot_id": uniprot,
        "target_uniprot_count": str(len(builder.hsa_to_uniprot.get(kegg_target, []))),
        "target_yamanishi_degree": str(builder.target_degrees.get((category, kegg_target), 0)),
        "ligand_yamanishi_degree": str(builder.ligand_degrees.get((category, kegg_drug), 0)),
        "label": str(label),
    }
    for column in LIGAND_FEATURE_COLUMNS:
        row[f"ligand_{column}"] = ligand.get(column, "")
    target_features = builder.target_features.get(uniprot, {})
    for column in TARGET_FEATURE_COLUMNS:
        row[f"target_{column}"] = target_features.get(column, "")
    return row


def build_positive_rows(builder: DatasetBuilder) -> list[dict[str, str]]:
    rows = []
    for key in sorted(builder.labels, key=lambda item: (item.category, item.kegg_drug, item.kegg_target)):
        row = make_no_affinity_row(builder, key.category, key.kegg_drug, key.kegg_target, 1)
        if row is not None:
            rows.append(row)
    return rows


def positive_status_counts(builder: DatasetBuilder) -> Counter[str]:
    counts: Counter[str] = Counter()
    for key in builder.labels:
        cid = builder.drug_to_cid.get(key.kegg_drug)
        if cid is None:
            counts["missing_kegg_drug_to_pubchem_cid"] += 1
        elif cid not in builder.ligands:
            counts["missing_pubchem_descriptor_row"] += 1
        elif key.kegg_target not in builder.hsa_to_uniprot:
            counts["missing_kegg_target_to_uniprot"] += 1
        elif choose_target_uniprot(builder, key.kegg_target) is None:
            counts["missing_target_features"] += 1
        else:
            counts["joined"] += 1
    return counts


def category_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["category"]] += 1
    return dict(counts)


def build_negative_rows(
    builder: DatasetBuilder,
    count_by_category: dict[str, int],
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    labels_by_category = defaultdict(set)
    drugs_by_category = defaultdict(set)
    targets_by_category = defaultdict(set)
    for key in builder.labels:
        labels_by_category[key.category].add((key.kegg_drug, key.kegg_target))
        drugs_by_category[key.category].add(key.kegg_drug)
        targets_by_category[key.category].add(key.kegg_target)

    rows = []
    for category, count in sorted(count_by_category.items()):
        candidates = []
        for drug in sorted(drugs_by_category[category]):
            for target in sorted(targets_by_category[category]):
                if (drug, target) in labels_by_category[category]:
                    continue
                row = make_no_affinity_row(builder, category, drug, target, 0)
                if row is not None:
                    candidates.append(row)
        rng.shuffle(candidates)
        rows.extend(candidates[:count])
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--positive-output",
        type=Path,
        default=Path("data/processed/yamanishi_no_affinity_positive_rows.csv"),
    )
    parser.add_argument(
        "--classifier-output",
        type=Path,
        default=Path("data/processed/yamanishi_no_affinity_classifier_dataset.csv"),
    )
    parser.add_argument(
        "--coverage-output",
        type=Path,
        default=Path("results/no_affinity_coverage_summary.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    builder = DatasetBuilder(args.data_dir, seed=args.seed)
    positives = build_positive_rows(builder)
    negatives = build_negative_rows(builder, category_counts(positives), args.seed)
    status_counts = positive_status_counts(builder)

    write_csv(args.positive_output, positives)
    write_csv(args.classifier_output, positives + negatives)
    args.coverage_output.parent.mkdir(parents=True, exist_ok=True)
    with args.coverage_output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step_or_reason", "positive_pairs"])
        writer.writerow(["total_yamanishi_positive_pairs", len(builder.labels)])
        for key, value in status_counts.most_common():
            writer.writerow([key, value])

    print(f"Yamanishi labels: {len(builder.labels)}")
    print(f"No-affinity positive rows: {len(positives)}")
    print(f"No-affinity negative rows: {len(negatives)}")
    print(f"Wrote positives: {args.positive_output}")
    print(f"Wrote classifier dataset: {args.classifier_output}")
    print(f"Wrote coverage: {args.coverage_output}")


if __name__ == "__main__":
    main()
