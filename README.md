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

The Yamanishi gold standard uses KEGG IDs:

- drugs: `Dxxxxx`
- human targets: `hsa:<gene_id>`

Local features use PubChem CIDs, UniProt IDs, and PDB-chain IDs. The build
script resolves:

```text
KEGG drug -> PubChem CID -> ligand descriptors -> affinity file
KEGG target -> UniProt -> PDB_CHAIN
```

The current `UniProt -> PDB_CHAIN` bridge comes from `old_PSB_Data.xlsx`, so
coverage is partial.

## Build

From this folder:

```bash
python3 scripts/build_dataset.py
```

Default output:

- `data/processed/yamanishi_positive_rows.csv`

This positive-row file is the canonical joined artifact. It contains only
Yamanishi-known interactions with complete local features.

Balanced negative examples should be generated at training time, because they
are a sampling policy rather than ground truth. To materialize one deterministic
balanced sample for inspection or an experiment, use:

```bash
python3 scripts/build_dataset.py \
  --include-negatives \
  --classifier-output data/processed/yamanishi_classifier_dataset_seed42.csv
```

Use a different `--seed` to create a different negative sample.

## Train Baselines

The training script generates balanced negatives at runtime and trains the old
baseline model family:

```bash
/Users/roshanklein-seetharaman/.pyenv/shims/python3 scripts/train_baselines.py
```

Outputs:

- `results/baseline_metrics.csv`
- `results/training_manifest.json`

The current baseline run uses `seed=42`, a random stratified 80/20 row split,
1,507 positive rows, and 1,507 runtime-sampled negatives. It uses the new
affinity archive, `data/raw/GoldStandardAffinities.zip`; `old_PSB_Data.xlsx` is
only used as a UniProt-to-PDB-chain bridge.

The best clean non-graph feature set is `pairwise + ligand_all + target`:

```text
Random Forest:     accuracy 0.725, F1 0.725, ROC-AUC 0.774, PR-AUC 0.792
Gradient Boosting: accuracy 0.710, F1 0.706, ROC-AUC 0.760, PR-AUC 0.771
SVM:               accuracy 0.701, F1 0.700, ROC-AUC 0.742, PR-AUC 0.708
```

The `all_available` feature set also includes Yamanishi graph-degree features
and category one-hot features. It performs much better:

```text
Random Forest:     accuracy 0.849, F1 0.848, ROC-AUC 0.916, PR-AUC 0.922
Gradient Boosting: accuracy 0.842, F1 0.844, ROC-AUC 0.919, PR-AUC 0.917
Logistic Reg.:     accuracy 0.837, F1 0.837, ROC-AUC 0.890, PR-AUC 0.877
```

Interpret the graph-degree run carefully: those degree features are computed
from the full Yamanishi label graph, so they are useful for old-style
network-feature experiments but are not a strict holdout-generalization setup.
For stricter evaluation, compute graph features on the training split only or
use grouped splits by ligand/target.

## Advanced Clean Sweep

For a stricter sweep using only the agreed feature sources:

```text
GoldStandardAffinities.zip pairwise features
+ pubchem_properties_xlogp.csv numeric ligand descriptors
```

run:

```bash
/Users/roshanklein-seetharaman/.pyenv/shims/python3 scripts/train_clean_advanced.py
```

Outputs:

- `results/clean_advanced_metrics.csv`
- `results/clean_advanced_manifest.json`

This excludes Yamanishi graph-degree, category, and target-count features. The
current best clean results are:

```text
Extra Trees tuned:            accuracy 0.675, F1 0.678, ROC-AUC 0.734, PR-AUC 0.748
Random Forest tuned:          accuracy 0.673, F1 0.677, ROC-AUC 0.730, PR-AUC 0.744
Hist Gradient Boosting tuned: accuracy 0.675, F1 0.684, ROC-AUC 0.725, PR-AUC 0.735
Gradient Boosting tuned:      accuracy 0.668, F1 0.681, ROC-AUC 0.729, PR-AUC 0.728
```

## Deep Learning

PyTorch is available in the pyenv interpreter, so a small MLP sweep is included:

```bash
/Users/roshanklein-seetharaman/.pyenv/shims/python3 scripts/train_clean_deep.py
```

Outputs:

- `results/clean_deep_metrics.csv`
- `results/clean_deep_manifest.json`

These models use the same clean feature set as the advanced sweep. Current best
MLP:

```text
MLP 256-128: accuracy 0.670, F1 0.685, ROC-AUC 0.738, PR-AUC 0.730
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
