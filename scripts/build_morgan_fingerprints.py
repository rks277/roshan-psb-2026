#!/usr/bin/env python3
"""Build Morgan/ECFP ligand fingerprints from local canonical SMILES."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/maccs_fingerprints (2).csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/morgan_fingerprints.csv"))
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=1024)
    args = parser.parse_args()

    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        from rdkit import RDLogger
    except ImportError as exc:
        raise SystemExit(
            "RDKit is required for Morgan fingerprints. Install it with "
            "`python3 -m pip install rdkit` or conda-forge rdkit."
        ) from exc
    RDLogger.DisableLog("rdApp.warning")

    rows = []
    skipped = []
    with args.input.open(newline="") as handle:
        for row in csv.DictReader(handle):
            cid = row.get("CID", "").strip()
            smiles = row.get("CanonicalSMILES", "").strip()
            if not cid or not smiles:
                skipped.append((cid, smiles, "missing_cid_or_smiles"))
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                skipped.append((cid, smiles, "invalid_smiles"))
                continue
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, args.radius, nBits=args.n_bits)
            bits = np.zeros((args.n_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fingerprint, bits)
            out = {"CID": cid}
            out.update({f"Morgan_{index}": int(value) for index, value in enumerate(bits)})
            rows.append(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["CID"] + [f"Morgan_{index}" for index in range(args.n_bits)]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    skipped_path = args.output.with_name(args.output.stem + "_skipped.csv")
    with skipped_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["CID", "CanonicalSMILES", "reason"])
        writer.writerows(skipped)

    print(f"Input ligands: {len(rows) + len(skipped)}")
    print(f"Morgan fingerprint rows: {len(rows)}")
    print(f"Skipped rows: {len(skipped)}")
    print(f"Wrote {args.output}")
    print(f"Wrote {skipped_path}")


if __name__ == "__main__":
    main()
