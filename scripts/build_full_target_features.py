#!/usr/bin/env python3
"""Build sequence-derived target features for all mapped Yamanishi targets."""

from __future__ import annotations

import argparse
import ast
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import DatasetBuilder  # noqa: E402
from build_no_affinity_dataset import (  # noqa: E402
    AMINO_ACIDS,
    TARGET_GROUP_FEATURES,
    dipeptide_features,
    sequence_fraction,
)


def protein_mass(sequence: str) -> float:
    masses = {
        "A": 89.09,
        "C": 121.16,
        "D": 133.10,
        "E": 147.13,
        "F": 165.19,
        "G": 75.07,
        "H": 155.16,
        "I": 131.18,
        "K": 146.19,
        "L": 131.18,
        "M": 149.21,
        "N": 132.12,
        "P": 115.13,
        "Q": 146.15,
        "R": 174.20,
        "S": 105.09,
        "T": 119.12,
        "V": 117.15,
        "W": 204.23,
        "Y": 181.19,
    }
    sequence = "".join(aa for aa in sequence.upper() if aa in masses)
    if not sequence:
        return 0.0
    return sum(masses[aa] for aa in sequence) - 18.015 * (len(sequence) - 1)


def feature_row(uniprot: str, sequence: str, degree_up: str = "") -> dict[str, str]:
    sequence = "".join(aa for aa in sequence.upper() if aa in AMINO_ACIDS)
    length = len(sequence)
    row = {
        "entry": uniprot,
        "sequence": sequence,
        "length": str(length),
        "mass": str(round(protein_mass(sequence), 3)),
        "degree (UP)": degree_up or "0",
    }
    for aa in AMINO_ACIDS:
        row[f"aa_{aa}"] = str((sequence.count(aa) / length * 100) if length else 0.0)
    for name, residues in TARGET_GROUP_FEATURES.items():
        row[f"group_{name}"] = str(sequence_fraction(sequence, residues))
    row.update(dipeptide_features(sequence))
    return row


def load_existing_features(path: Path) -> dict[str, dict[str, str]]:
    rows = {}
    if not path.exists():
        return rows
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle, delimiter="\t"):
            uniprot = raw.get("entry", "").strip()
            sequence = raw.get("sequence", "")
            if not uniprot or not sequence:
                continue
            try:
                composition = ast.literal_eval(raw.get("composition", "{}"))
            except (SyntaxError, ValueError):
                composition = {}
            row = feature_row(uniprot, sequence, raw.get("degree (UP)", ""))
            for aa in AMINO_ACIDS:
                if aa in composition:
                    row[f"aa_{aa}"] = str(float(composition[aa]))
            rows[uniprot] = row
    return rows


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def fetch_uniprot_sequences(
    uniprots: list[str],
    chunk_size: int,
    sleep_seconds: float,
    reviewed_only: bool,
) -> dict[str, str]:
    sequences = {}
    for index, chunk in enumerate(chunks(uniprots, chunk_size), start=1):
        query = " OR ".join(f"accession:{accession}" for accession in chunk)
        query_filter = f"({query})"
        if reviewed_only:
            query_filter = f"{query_filter} AND reviewed:true"
        params = urlencode(
            {
                "query": query_filter,
                "fields": "accession,sequence",
                "format": "tsv",
            }
        )
        url = f"https://rest.uniprot.org/uniprotkb/stream?{params}"
        text = urlopen(url, timeout=90).read().decode("utf-8")
        for line in text.splitlines()[1:]:
            accession, sequence = (line.split("\t") + [""])[:2]
            if accession and sequence:
                sequences[accession] = sequence
        mode = "reviewed" if reviewed_only else "all"
        print(f"Fetched UniProt {mode} chunk {index}; sequences so far: {len(sequences)}", flush=True)
        time.sleep(sleep_seconds)
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/yamanishi_target_features_full.tsv"))
    parser.add_argument("--missing-output", type=Path, default=Path("results/full_target_features_missing_uniprots.csv"))
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--no-fetch", action="store_true")
    args = parser.parse_args()

    builder = DatasetBuilder(args.data_dir, seed=42)
    target_hsas = {key.kegg_target for key in builder.labels}
    target_uniprots = sorted(
        {
            uniprot
            for hsa in target_hsas
            for uniprot in builder.hsa_to_uniprot.get(hsa, [])
        }
    )

    rows = load_existing_features(args.data_dir / "features.tsv")
    missing = [uniprot for uniprot in target_uniprots if uniprot not in rows]
    if missing and not args.no_fetch:
        fetched = fetch_uniprot_sequences(missing, args.chunk_size, args.sleep_seconds, reviewed_only=True)
        still_missing = [uniprot for uniprot in missing if uniprot not in fetched]
        if still_missing:
            fetched.update(
                fetch_uniprot_sequences(
                    still_missing,
                    args.chunk_size,
                    args.sleep_seconds,
                    reviewed_only=False,
                )
            )
        for uniprot, sequence in fetched.items():
            rows[uniprot] = feature_row(uniprot, sequence)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(next(iter(rows.values())).keys())
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for uniprot in sorted(rows):
            writer.writerow(rows[uniprot])

    remaining = [uniprot for uniprot in target_uniprots if uniprot not in rows]
    args.missing_output.parent.mkdir(parents=True, exist_ok=True)
    with args.missing_output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["uniprot_id"])
        writer.writerows([[uniprot] for uniprot in remaining])

    print(f"Yamanishi mapped UniProt IDs: {len(target_uniprots)}")
    print(f"Feature rows written: {len(rows)}")
    print(f"Remaining missing mapped UniProt IDs: {len(remaining)}")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.missing_output}")


if __name__ == "__main__":
    main()
