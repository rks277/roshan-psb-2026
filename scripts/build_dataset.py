#!/usr/bin/env python3
"""Build a classifier table from Yamanishi labels and local PSB feature files.

Each output row is one ligand/drug + target/protein pair:

    [ligand info][target info][pairwise docking/rank features][label]

Positive labels come from the Yamanishi gold-standard relation lists. Negative
labels are sampled from unlisted ligand-target pairs within the same target
category.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import random
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

MAIN_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
REL_NS = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}

LABEL_FILES = {
    "enzyme": "bind_orfhsa_drug_e.txt",
    "gpcr": "bind_orfhsa_drug_gpcr.txt",
    "ion_channel": "bind_orfhsa_drug_ic.txt",
    "nuclear_receptor": "bind_orfhsa_drug_nr.txt",
}

LIGAND_FEATURE_COLUMNS = [
    "MolecularWeight",
    "MonoisotopicMass",
    "TPSA",
    "Complexity",
    "Charge",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "HeavyAtomCount",
    "IsotopeAtomCount",
    "AtomStereoCount",
    "DefinedAtomStereoCount",
    "UndefinedAtomStereoCount",
    "BondStereoCount",
    "DefinedBondStereoCount",
    "UndefinedBondStereoCount",
    "CovalentUnitCount",
    "Volume3D",
    "FeatureCount3D",
    "FeatureAcceptorCount3D",
    "FeatureDonorCount3D",
    "FeatureAnionCount3D",
    "FeatureCationCount3D",
    "FeatureRingCount3D",
    "FeatureHydrophobeCount3D",
    "ConformerModelRMSD3D",
    "EffectiveRotorCount3D",
    "ConformerCount3D",
    "XLogP",
    "MW",
]

TARGET_FEATURE_COLUMNS = [
    "length",
    "mass",
    "degree_up",
]

TOP_AFFINITY_HIT_LIMIT = 20


def column_index(cell_ref: str | None) -> int:
    match = re.match(r"([A-Z]+)", cell_ref or "")
    out = 0
    for char in match.group(1) if match else "":
        out = out * 26 + ord(char) - 64
    return out


def read_xlsx_sheet(path: Path, sheet_name: str) -> list[dict[str, str]]:
    """Read a simple XLSX worksheet without requiring openpyxl."""
    with ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(text.text or "" for text in item.findall(".//a:t", MAIN_NS))
                for item in root.findall("a:si", MAIN_NS)
            ]

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        relmap = {
            rel.get("Id"): rel.get("Target")
            for rel in rels.findall("rel:Relationship", REL_NS)
        }

        chosen_sheet = None
        for sheet in workbook.findall("a:sheets/a:sheet", MAIN_NS):
            if sheet.get("name") == sheet_name:
                chosen_sheet = sheet
                break
        if chosen_sheet is None:
            raise ValueError(f"Could not find sheet {sheet_name!r} in {path}")

        relationship_id = chosen_sheet.get(
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        )
        target = relmap[relationship_id]
        sheet_path = "xl/" + target.lstrip("/") if not target.startswith("xl/") else target
        root = ET.fromstring(zf.read(sheet_path))

        rows: list[list[str]] = []
        for row in root.findall(".//a:sheetData/a:row", MAIN_NS):
            values: dict[int, str] = {}
            for cell in row.findall("a:c", MAIN_NS):
                cell_type = cell.get("t")
                raw_value = cell.find("a:v", MAIN_NS)
                if cell_type == "inlineStr":
                    value = "".join(
                        text.text or "" for text in cell.findall(".//a:t", MAIN_NS)
                    )
                elif raw_value is None:
                    value = ""
                elif cell_type == "s":
                    value = shared_strings[int(raw_value.text)]
                else:
                    value = raw_value.text or ""
                values[column_index(cell.get("r"))] = value
            if values:
                rows.append([values.get(i, "") for i in range(1, max(values) + 1)])

    header = rows[0]
    records = []
    for row in rows[1:]:
        padded = row + [""] * (len(header) - len(row))
        records.append(dict(zip(header, padded[: len(header)])))
    return records


def parse_affinity_line(line: str) -> tuple[str, float] | None:
    match = re.search(r"/([^/]+\.pdb)/log\.log:\s+\d+\s+([-+0-9.eE]+)", line)
    if not match:
        return None
    return match.group(1).removesuffix(".pdb"), float(match.group(2))


@dataclass(frozen=True)
class PairKey:
    category: str
    kegg_target: str
    kegg_drug: str


class DatasetBuilder:
    def __init__(self, data_dir: Path, seed: int = 42):
        self.data_dir = data_dir
        self.rng = random.Random(seed)
        self.labels = self._load_labels()
        self.drug_to_cid = self._load_drug_to_cid()
        self.hsa_to_uniprot = self._load_hsa_to_uniprot()
        self.ligands = self._load_ligands()
        self.target_features = self._load_target_features()
        self.excluded_affinity_uniprots = self._load_excluded_affinity_uniprots()
        self.pdb_chain_to_uniprot, self.pdb_to_uniprots = self._load_sifts_mapping()
        self.affinity_files = self._index_affinity_files()
        self.ligand_degrees, self.target_degrees = self._compute_label_degrees()
        self._affinity_cache: dict[str, dict[str, float]] = {}
        self._uniprot_affinity_cache: dict[str, dict[str, float]] = {}

    def _load_labels(self) -> set[PairKey]:
        labels = set()
        for category, filename in LABEL_FILES.items():
            path = self.data_dir / filename
            for line in path.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    labels.add(PairKey(category, parts[0], parts[1]))
        return labels

    def _load_drug_to_cid(self) -> dict[str, str]:
        path = self.data_dir / "kegg_drug_pubchem_sid_cid.csv"
        with path.open(newline="") as handle:
            return {
                row["kegg_drug"]: row["pubchem_cid"]
                for row in csv.DictReader(handle)
                if row.get("pubchem_cid")
            }

    def _load_hsa_to_uniprot(self) -> dict[str, list[str]]:
        out: dict[str, set[str]] = defaultdict(set)
        path = self.data_dir / "kegg_hsa_to_uniprot.txt"
        for line in path.read_text().splitlines():
            left, right = line.split("\t")
            hsa = "hsa:" + left.split(":", 1)[1]
            uniprot = right.split(":", 1)[1]
            out[hsa].add(uniprot)
        return {key: sorted(value) for key, value in out.items()}

    def _load_ligands(self) -> dict[str, dict[str, str]]:
        path = self.data_dir / "pubchem_properties_xlogp.csv"
        with path.open(newline="") as handle:
            return {row["CID"].strip(): row for row in csv.DictReader(handle)}

    def _load_target_features(self) -> dict[str, dict[str, str]]:
        path = self.data_dir / "features.tsv"
        if not path.exists():
            return {}
        out = {}
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                uniprot = row.get("entry", "").strip()
                if not uniprot:
                    continue
                features = {
                    "length": row.get("length", ""),
                    "mass": row.get("mass", ""),
                    "degree_up": row.get("degree (UP)", ""),
                }
                out[uniprot] = features
        return out

    def _load_excluded_affinity_uniprots(self) -> set[str]:
        path = self.data_dir / "excluded_top_affinity_uniprots.csv"
        if not path.exists():
            return set()
        with path.open(newline="") as handle:
            return {
                row["uniprot_id"].strip()
                for row in csv.DictReader(handle)
                if row.get("uniprot_id")
            }

    def _load_sifts_mapping(self) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        """Load PDB-chain -> UniProt mappings from SIFTS."""
        path = self.data_dir / "pdb_chain_uniprot.tsv.gz"
        if not path.exists():
            return {}, {}
        chain_to_uniprot: dict[str, set[str]] = defaultdict(set)
        pdb_to_uniprots: dict[str, set[str]] = defaultdict(set)
        with gzip.open(path, "rt", newline="") as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3 or parts[0] == "PDB":
                    continue
                pdb_id = parts[0].upper()
                chain = parts[1]
                uniprot = parts[2].strip()
                if not pdb_id or not chain or not uniprot:
                    continue
                chain_to_uniprot[f"{pdb_id}_{chain}"].add(uniprot)
                pdb_to_uniprots[pdb_id].add(uniprot)
        return dict(chain_to_uniprot), dict(pdb_to_uniprots)

    def _index_affinity_files(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        with ZipFile(self.data_dir / "GoldStandardAffinities.zip") as zf:
            for name in zf.namelist():
                if not name.endswith(".txt") or name.endswith("GoldStandardList.txt"):
                    continue
                match = re.search(r"\((\d+)\)(?:_2D|_3D|\.pdb|$)", Path(name).name)
                if match:
                    out[match.group(1)].append(name)
        return {key: sorted(value) for key, value in out.items()}

    def _compute_label_degrees(self) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
        ligand_targets: dict[tuple[str, str], set[str]] = defaultdict(set)
        target_ligands: dict[tuple[str, str], set[str]] = defaultdict(set)
        for key in self.labels:
            ligand_targets[(key.category, key.kegg_drug)].add(key.kegg_target)
            target_ligands[(key.category, key.kegg_target)].add(key.kegg_drug)
        ligand_degrees = {key: len(value) for key, value in ligand_targets.items()}
        target_degrees = {key: len(value) for key, value in target_ligands.items()}
        return ligand_degrees, target_degrees

    def _load_affinity_table(self, cid: str) -> dict[str, float]:
        if cid in self._affinity_cache:
            return self._affinity_cache[cid]
        files = self.affinity_files.get(cid, [])
        if not files:
            self._affinity_cache[cid] = {}
            return {}

        values: dict[str, float] = {}
        # Prefer the first deterministically sorted file for duplicated 2D/3D ligand variants.
        with ZipFile(self.data_dir / "GoldStandardAffinities.zip") as zf:
            with zf.open(files[0]) as handle:
                for raw_line in handle:
                    parsed = parse_affinity_line(raw_line.decode("utf-8", errors="replace"))
                    if parsed is None:
                        continue
                    pdb_id, affinity = parsed
                    values[pdb_id] = affinity
        self._affinity_cache[cid] = values
        return values

    def _load_uniprot_affinity_table(self, cid: str) -> dict[str, float]:
        if cid in self._uniprot_affinity_cache:
            return self._uniprot_affinity_cache[cid]

        pdb_affinities = self._load_affinity_table(cid)
        uniprot_affinities: dict[str, float] = {}
        for pdb_chain, affinity in pdb_affinities.items():
            pdb_id = pdb_chain.split("_", 1)[0].upper()
            uniprots = self.pdb_chain_to_uniprot.get(pdb_chain)
            if not uniprots:
                uniprots = self.pdb_to_uniprots.get(pdb_id, set())
            for uniprot in uniprots:
                if uniprot in self.excluded_affinity_uniprots:
                    continue
                if uniprot not in uniprot_affinities or affinity < uniprot_affinities[uniprot]:
                    uniprot_affinities[uniprot] = affinity

        if len(uniprot_affinities) > TOP_AFFINITY_HIT_LIMIT:
            uniprot_affinities = dict(
                sorted(uniprot_affinities.items(), key=lambda item: (item[1], item[0]))[
                    :TOP_AFFINITY_HIT_LIMIT
                ]
            )

        self._uniprot_affinity_cache[cid] = uniprot_affinities
        return uniprot_affinities

    def pairwise_features(self, cid: str, uniprot_id: str) -> dict[str, float] | None:
        affinity_table = self._load_uniprot_affinity_table(cid)
        if uniprot_id not in affinity_table:
            return None
        ranked = sorted(affinity_table.items(), key=lambda item: (item[1], item[0]))
        total = len(ranked)
        positions = {target_uniprot: index + 1 for index, (target_uniprot, _) in enumerate(ranked)}
        inverted_rank = positions[uniprot_id]
        rank = total - inverted_rank
        return {
            "affinity": affinity_table[uniprot_id],
            "rank": rank,
            "total": total,
            "inverted_rank": inverted_rank,
            "proportion": rank / total if total else 0.0,
        }

    def make_example(self, category: str, kegg_drug: str, kegg_target: str) -> dict[str, str] | None:
        cid = self.drug_to_cid.get(kegg_drug)
        if cid is None:
            return None
        ligand = self.ligands.get(cid)
        if ligand is None:
            return None

        affinity_table = self._load_uniprot_affinity_table(cid)
        if not affinity_table:
            return None

        target_uniprots = self.hsa_to_uniprot.get(kegg_target, [])
        choices = [(affinity_table[uniprot], uniprot) for uniprot in target_uniprots if uniprot in affinity_table]
        if not choices:
            return None
        _, chosen_uniprot = min(choices, key=lambda item: (item[0], item[1]))
        pairwise = self.pairwise_features(cid, chosen_uniprot)
        if pairwise is None:
            return None

        label = int(PairKey(category, kegg_target, kegg_drug) in self.labels)
        row: dict[str, str] = {
            "category": category,
            "kegg_drug": kegg_drug,
            "pubchem_cid": cid,
            "ligand_title": ligand.get("Title", ""),
            "kegg_target": kegg_target,
            "uniprot_id": chosen_uniprot,
            "target_uniprot_count": str(len(self.hsa_to_uniprot.get(kegg_target, []))),
            "target_yamanishi_degree": str(self.target_degrees.get((category, kegg_target), 0)),
            "ligand_yamanishi_degree": str(self.ligand_degrees.get((category, kegg_drug), 0)),
            "label": str(label),
        }
        for column in LIGAND_FEATURE_COLUMNS:
            row[f"ligand_{column}"] = ligand.get(column, "")
        target_features = self.target_features.get(chosen_uniprot, {})
        for column in TARGET_FEATURE_COLUMNS:
            row[f"target_{column}"] = target_features.get(column, "")
        for key, value in pairwise.items():
            row[key] = str(value)
        return row

    def build_positive_rows(self) -> list[dict[str, str]]:
        rows = []
        for key in sorted(self.labels, key=lambda item: (item.category, item.kegg_drug, item.kegg_target)):
            row = self.make_example(key.category, key.kegg_drug, key.kegg_target)
            if row is not None:
                rows.append(row)
        return rows

    def build_negative_rows(self, count_by_category: dict[str, int]) -> list[dict[str, str]]:
        labels_by_category = defaultdict(set)
        drugs_by_category = defaultdict(set)
        targets_by_category = defaultdict(set)
        for key in self.labels:
            labels_by_category[key.category].add((key.kegg_drug, key.kegg_target))
            drugs_by_category[key.category].add(key.kegg_drug)
            targets_by_category[key.category].add(key.kegg_target)

        rows = []
        seen = set()
        for category, count in count_by_category.items():
            drugs = sorted(drugs_by_category[category])
            targets = sorted(targets_by_category[category])
            attempts = 0
            max_attempts = max(10000, count * 200)
            while len([row for row in rows if row["category"] == category]) < count:
                attempts += 1
                if attempts > max_attempts:
                    break
                drug = self.rng.choice(drugs)
                target = self.rng.choice(targets)
                key = (category, drug, target)
                if key in seen or (drug, target) in labels_by_category[category]:
                    continue
                row = self.make_example(category, drug, target)
                if row is None:
                    continue
                row["label"] = "0"
                rows.append(row)
                seen.add(key)
        return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--positive-output", type=Path, default=Path("data/processed/yamanishi_positive_rows.csv"))
    parser.add_argument("--include-negatives", action="store_true")
    parser.add_argument("--classifier-output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    builder = DatasetBuilder(args.data_dir, seed=args.seed)
    positives = builder.build_positive_rows()

    write_csv(args.positive_output, positives)

    print(f"Yamanishi labels: {len(builder.labels)}")
    print(f"Positive rows with complete features: {len(positives)}")
    print(f"Wrote positives: {args.positive_output}")

    if args.include_negatives:
        if args.classifier_output is None:
            raise ValueError("--classifier-output is required with --include-negatives")
        positive_counts = defaultdict(int)
        for row in positives:
            positive_counts[row["category"]] += 1
        negatives = builder.build_negative_rows(dict(positive_counts))
        all_rows = positives + negatives
        write_csv(args.classifier_output, all_rows)
        print(f"Negative rows sampled with complete features: {len(negatives)}")
        print(f"Wrote classifier dataset: {args.classifier_output}")
    else:
        negatives = []

    positive_counts = defaultdict(int)
    for row in positives:
        positive_counts[row["category"]] += 1
    for category in sorted(positive_counts):
        pos = sum(1 for row in positives if row["category"] == category)
        neg = sum(1 for row in negatives if row["category"] == category)
        print(f"{category}: positives={pos} negatives={neg}")


if __name__ == "__main__":
    main()
