#!/usr/bin/env python3
"""Export all Yamanishi positive pairs, regardless of feature availability."""

from __future__ import annotations

import csv
from pathlib import Path

from build_dataset import DatasetBuilder


def feature_status(builder: DatasetBuilder, category: str, kegg_drug: str, kegg_target: str) -> str:
    cid = builder.drug_to_cid.get(kegg_drug)
    if cid is None:
        return "missing_kegg_drug_to_pubchem_cid"
    if cid not in builder.ligands:
        return "missing_pubchem_descriptor_row"
    if cid not in builder.affinity_files:
        return "missing_ligand_affinity_file"
    if kegg_target not in builder.hsa_to_uniprot:
        return "missing_kegg_target_to_uniprot"
    affinity_table = builder._load_uniprot_affinity_table(cid)
    target_uniprots = builder.hsa_to_uniprot.get(kegg_target, [])
    if not any(uniprot in affinity_table for uniprot in target_uniprots):
        return "missing_uniprot_affinity_for_target"
    return "feature_complete"


def main() -> None:
    data_dir = Path("data/raw")
    output = Path("data/processed/yamanishi_all_positive_pairs.csv")
    builder = DatasetBuilder(data_dir, seed=42)

    rows = []
    for key in sorted(builder.labels, key=lambda item: (item.category, item.kegg_drug, item.kegg_target)):
        cid = builder.drug_to_cid.get(key.kegg_drug, "")
        ligand = builder.ligands.get(cid, {})
        uniprots = builder.hsa_to_uniprot.get(key.kegg_target, [])
        status = feature_status(builder, key.category, key.kegg_drug, key.kegg_target)
        rows.append(
            {
                "category": key.category,
                "kegg_drug": key.kegg_drug,
                "pubchem_cid": cid,
                "ligand_title": ligand.get("Title", ""),
                "kegg_target": key.kegg_target,
                "uniprot_ids": ";".join(uniprots),
                "label": "1",
                "feature_status": status,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
