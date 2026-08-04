# Chlorogenic Acid Reranked Target List

This note documents `chlorogenic_acid_reranked_targets.csv`, the reranked affinity-target output for chlorogenic acid.

## Files

- Reranked output: `results/chlorogenic_acid_reranked_targets.csv`
- Run manifest: `results/chlorogenic_acid_reranked_targets.manifest.json`
- Raw affinity input: `data/raw/CID_1794427_ChlorogenicAcid.pdb_affinities.txt`
- Export script: `scripts/export_clean_reranked_ligand.py`

## What This File Contains

Each row is one candidate chlorogenic-acid target after mapping the raw PDB-chain docking affinity file to UniProt identifiers. The CSV contains 7,527 UniProt-mapped candidate targets for PubChem CID `1794427`.

The rows are sorted by `reranked_rank_1_based`, where rank 1 is the highest-priority candidate according to the clean reranking model. This reranked order is not the same as the original docking-affinity order.

## How It Was Generated

The raw input was `CID_1794427_ChlorogenicAcid.pdb_affinities.txt`, which contains 7,403 PDB-chain docking affinity rows. These PDB-chain rows were mapped to UniProt targets using the local PDB-to-UniProt mapping logic in `scripts/build_dataset.py`. Some PDB structures map to more than one UniProt identifier, producing 7,527 UniProt-level candidate rows.

The reranking model was trained on `data/processed/affinity_hit_value_dataset_compact.csv`.

Model details:

| Item | Value |
|---|---|
| Model | `HistGradientBoostingClassifier(random_state=42)` |
| Feature set | `clean_rank_plus_maccs_morgan_target` |
| Feature count | 1,663 |
| Training rows | 214,842 |
| Supported training rows | 2,862 |
| Exported chlorogenic acid rows | 7,527 |

The feature set excludes label-prior/context features such as Yamanishi ligand degree, Yamanishi target degree, Yamanishi target-universe membership, and BindingDB target-universe membership.

## Important Caveat

Chlorogenic acid has a local affinity file, but PubChem CID `1794427` is not currently present in the local PubChem descriptor or fingerprint tables used by the model. For this specific output, the reranker therefore used the available affinity-context features and target features, while missing ligand chemistry/fingerprint values were imputed by the model pipeline.

That means this is a valid reranking of the chlorogenic acid affinity list, but it is not as chemically informed as outputs for ligands that have complete PubChem, MACCS, and Morgan/ECFP features.

## Column Definitions

| Column | Meaning |
|---|---|
| `reranked_rank_1_based` | Final model-based rank after reranking; 1 is highest priority. |
| `pubchem_cid` | PubChem compound identifier for chlorogenic acid; here always `1794427`. |
| `ligand_title` | Ligand name recovered from the affinity filename; here `ChlorogenicAcid`. |
| `uniprot_id` | Candidate target UniProt identifier. |
| `known_target_yamanishi` | `1` if this CID-UniProt pair is supported by Yamanishi labels, else `0`. |
| `known_target_bindingdb` | `1` if this CID-UniProt pair is supported by BindingDB labels, else `0`. |
| `known_target_any_supported` | `1` if supported by Yamanishi or BindingDB, else `0`. |
| `hit_value_score` | Model score used for prioritization. Higher means the pair looks more like supported interactions in the training data. It should be interpreted as a ranking score, not a calibrated binding probability. |
| `affinity` | Original docking affinity after mapping to the UniProt candidate. More negative values are stronger raw docking predictions. |
| `raw_affinity_rank_1_based` | Original rank by docking affinity before model reranking. |
| `rank_percentile` | Original affinity rank divided by the number of ranked UniProt targets. Lower is better in the raw docking list. |
| `reverse_rank_percentile` | Reverse raw rank percentile. Higher is better in the raw docking list. |
| `affinity_zscore_within_ligand` | Affinity z-score within the chlorogenic acid affinity distribution. More negative is stronger relative to this ligand's docking distribution. |
| `affinity_robust_zscore_within_ligand` | Robust within-ligand affinity z-score using median and MAD. More negative is stronger. |
| `affinity_gap_to_next_weaker` | Difference between this affinity and the next weaker raw-ranked candidate. |
| `affinity_gap_to_previous_stronger` | Difference between this affinity and the previous stronger raw-ranked candidate. |
| `total_ranked_uniprots` | Total number of UniProt candidate targets in this chlorogenic acid reranking; here always 7,527. |

## Top Reranked Candidates

| Reranked rank | UniProt | Hit-value score | Raw affinity rank | Affinity |
|---:|---|---:|---:|---:|
| 1 | P35354 | 0.9600 | 467 | -8.8413 |
| 2 | P07550 | 0.9448 | 252 | -9.1398 |
| 3 | P33261 | 0.8871 | 1027 | -8.3892 |
| 4 | P23219 | 0.8121 | 776 | -8.5769 |
| 5 | P06276 | 0.7987 | 759 | -8.5883 |
| 6 | P08684 | 0.7816 | 638 | -8.6850 |
| 7 | P10635 | 0.7302 | 126 | -9.4328 |
| 8 | P04798 | 0.7165 | 525 | -8.7895 |
| 9 | P14867 | 0.7089 | 4167 | -6.9362 |
| 10 | P10632 | 0.6942 | 103 | -9.5008 |

## Known Supported Targets In This Output

There are 6 known supported chlorogenic acid targets in this reranked list. All 6 are BindingDB-supported; none are Yamanishi-supported in the current joined label tables.

| Reranked rank | UniProt | Hit-value score | Affinity | Raw affinity rank | Source |
|---:|---|---:|---:|---:|---|
| 69 | P15121 | 0.0920 | -8.8891 | 426 | BindingDB |
| 361 | O60218 | 0.0130 | -10.2275 | 13 | BindingDB |
| 497 | P05067 | 0.0102 | -4.7715 | 7276 | BindingDB |
| 2788 | Q13547 | 0.0018 | -6.5577 | 5271 | BindingDB |
| 2916 | P18031 | 0.0018 | -7.4363 | 2850 | BindingDB |
| 3566 | P17706 | 0.0013 | -7.1624 | 3531 | BindingDB |

## Interpretation

This file should be used as a prioritization list for follow-up inspection or experimental validation. The top-ranked rows are high-priority model hits, not confirmed binders. Conversely, rows labeled `0` are unlabeled or unsupported by the current Yamanishi/BindingDB labels; they are not experimentally confirmed non-binders.

