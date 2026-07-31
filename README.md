# Roshan PSB 2026 Classifier Data Package

This folder is a self-contained working package for rebuilding a PSB 2026
protein-ligand interaction classifier dataset from the new Yamanishi
drug-target gold standard and the existing local feature artifacts.

## Dataset Shape

Each classifier row is one ligand/drug + target/protein pair:

```text
[ligand info][target info][pairwise docking/rank features][label]
```

The label is binary:

- `label = 1`: the pair is listed in the Yamanishi gold-standard relation files.
- `label = 0`: the pair is an unlisted ligand-target combination sampled as a
  negative example within the same target category.

## Inputs

Required raw files are copied into `data/raw/`:

- `bind_orfhsa_drug_e.txt`
- `bind_orfhsa_drug_gpcr.txt`
- `bind_orfhsa_drug_ic.txt`
- `bind_orfhsa_drug_nr.txt`
- `kegg_drug_pubchem_sid_cid.csv`
- `kegg_hsa_to_uniprot.txt`
- `pubchem_properties_xlogp.csv`
- `GoldStandardAffinities.zip`
- `old_PSB_Data.xlsx`
- `pdb_chain_uniprot.tsv.gz`
- `features.tsv`

The Yamanishi gold standard uses KEGG IDs:

- drugs: `Dxxxxx`
- human targets: `hsa:<gene_id>`

Local features use PubChem CIDs, UniProt IDs, and PDB-chain affinity rows. The
build script resolves:

```text
KEGG drug -> PubChem CID -> ligand descriptors -> affinity file
KEGG target -> UniProt
affinity PDB_CHAIN rows -> UniProt by SIFTS -> target-comparable pairwise features
UniProt -> protein target features
```

The classifier target key is UniProt. PDB chains are only used as the raw
coordinate system inside `GoldStandardAffinities.zip`, then aggregated to
UniProt with `pdb_chain_uniprot.tsv.gz`. Protein target features come from
`features.tsv` and are joined by UniProt.

The affinity join excludes frequent raw top-hit proteins listed in
`data/raw/excluded_top_affinity_uniprots.csv`. These are UniProt IDs that most
often appear as the best, most negative affinity hit across ligand affinity
tables.

## Coverage

The full Yamanishi gold standard has 5,127 positive ligand-target pairs. The
all-positive label artifact contains all 5,127. The feature-complete joined
artifact contains only positives that can be joined all the way to local
features:

```text
total Yamanishi positives:                  5,127
joined positives with complete features:    2,720
```

Current attrition:

```text
missing UniProt affinity for target:        1,463
missing ligand affinity file:                 886
missing PubChem descriptor row:                24
missing KEGG drug -> PubChem CID mapping:      21
missing KEGG target -> UniProt mapping:        13
```

Coverage reports can be regenerated with:

```bash
/Users/roshanklein-seetharaman/.pyenv/shims/python3 scripts/report_coverage.py
```

Outputs:

- `results/coverage_summary.csv`
- `results/coverage_by_category.csv`
- `results/coverage_missing_examples.csv`

## Build

From this folder:

```bash
python3 scripts/build_dataset.py
```

Default output:

- `data/processed/yamanishi_positive_rows.csv`

This positive-row file is the feature-complete joined artifact. It contains only
Yamanishi-known interactions with complete local features.

The full positive-label table is:

- `data/processed/yamanishi_all_positive_pairs.csv`

That file contains all 5,127 Yamanishi positives, regardless of feature
availability. Its `feature_status` column explains whether each pair is
feature-complete or which join is missing.

Balanced negative examples should be generated at training time, because they
are a sampling policy rather than ground truth. To materialize one deterministic
balanced sample for inspection or an experiment, use:

```bash
python3 scripts/build_dataset.py \
  --include-negatives \
  --classifier-output data/processed/yamanishi_classifier_dataset.csv
```

Use a different `--seed` to create a different negative sample.

## Train Baselines

The training script generates balanced negatives at runtime and trains the
baseline model family:

```bash
/Users/roshanklein-seetharaman/.pyenv/shims/python3 scripts/train_baselines.py
```

Outputs:

- `results/baseline_metrics.csv`
- `results/training_manifest.json`

The current baseline run uses `seed=42`, a random stratified 80/20 row split,
2,720 positive rows, and 2,720 runtime-sampled negatives. It uses
`data/raw/GoldStandardAffinities.zip` aggregated to UniProt through SIFTS.

The baseline feature sets now compare pairwise-only, pairwise plus PubChem
ligand descriptors, pairwise plus protein target features, and the combined
feature set. Protein target features are restricted to length, mass, and
degree. Current best baseline results:

```text
Random Forest, pairwise + ligand_all + target_all:     accuracy 0.802, F1 0.801, ROC-AUC 0.876, PR-AUC 0.890
Gradient Boosting, pairwise + ligand_all + target_all: accuracy 0.733, F1 0.734, ROC-AUC 0.821, PR-AUC 0.828
KNN, pairwise + ligand_all + target_all:               accuracy 0.720, F1 0.725, ROC-AUC 0.791, PR-AUC 0.774
```

## Baseline Error Analysis

False-positive and false-negative analysis for the Random Forest baseline can
be regenerated with:

```bash
MPLCONFIGDIR=/private/tmp /Users/roshanklein-seetharaman/.pyenv/shims/python3 scripts/analyze_baseline_errors.py
```

Outputs:

- `results/baseline_error_predictions.csv`
- `results/baseline_error_degree_summary.csv`
- `results/baseline_error_rates_by_degree_bin.csv`
- `results/baseline_error_ligands.csv`
- `results/plots/baseline_error_degree_bins.png`
- `results/plots/baseline_error_rates_by_degree_bin.png`

The current held-out split shows a ligand-degree bias. Positives for ligands
with only one known Yamanishi target have a high false-negative rate
(`35/53 = 0.660`), while positives for ligands with more than 30 known targets
are rarely missed (`7/179 = 0.039`). The opposite happens for sampled
negatives: high-degree ligands are much more likely to be false positives
(`23/41 = 0.561` for degree 11-30 and `8/18 = 0.444` for degree >30).

## Advanced Clean Sweep

For a stricter sweep using the agreed feature sources:

```text
GoldStandardAffinities.zip pairwise features
+ pubchem_properties_xlogp.csv numeric ligand descriptors
+ features.tsv target length, mass, and degree
```

run:

```bash
/Users/roshanklein-seetharaman/.pyenv/shims/python3 scripts/train_clean_advanced.py
```

Outputs:

- `results/clean_advanced_metrics.csv`
- `results/clean_advanced_manifest.json`

This excludes Yamanishi graph-degree, category, and target-count features. The
current best tuned results are:

```text
Hist Gradient Boosting tuned, pairwise + PubChem + target: accuracy 0.785, F1 0.784, ROC-AUC 0.874, PR-AUC 0.883
Random Forest tuned, pairwise + PubChem + target:          accuracy 0.781, F1 0.781, ROC-AUC 0.869, PR-AUC 0.883
Gradient Boosting tuned, pairwise + PubChem + target:      accuracy 0.781, F1 0.782, ROC-AUC 0.861, PR-AUC 0.871
Extra Trees tuned, pairwise + PubChem + target:            accuracy 0.735, F1 0.732, ROC-AUC 0.808, PR-AUC 0.827
```

## Deep Learning

PyTorch is available in the pyenv interpreter, so a small MLP sweep is included:

```bash
/Users/roshanklein-seetharaman/.pyenv/shims/python3 scripts/train_clean_deep.py
```

Outputs:

- `results/clean_deep_metrics.csv`
- `results/clean_deep_manifest.json`

These models use the combined pairwise + PubChem + target feature set. Current
best MLP:

```text
MLP 256-128:   accuracy 0.747, F1 0.746, ROC-AUC 0.818, PR-AUC 0.836
MLP 128-64:    accuracy 0.746, F1 0.746, ROC-AUC 0.811, PR-AUC 0.829
```

The MLP is competitive on F1 and ROC-AUC, but the tuned tree ensembles still
have the best PR-AUC on this small tabular dataset.

On this machine, the package directory resolves `python3` to Apple system
Python, which lacks `numpy`/`sklearn`. Use the pyenv shim command above unless
your shell resolves `python3` to the pyenv interpreter.

## Main Script API

The core row builder is:

```python
DatasetBuilder.make_example(category, kegg_drug, kegg_target)
```

It returns a dictionary containing ligand descriptors, target identifiers,
pairwise docking/rank features, and a `0` or `1` label. It returns `None` when a
required mapping or affinity value is unavailable.

For negatives, use:

```python
builder.build_negative_rows(count_by_category)
```

Those negatives are unlisted Yamanishi ligand-target pairs sampled within the
same target category.
