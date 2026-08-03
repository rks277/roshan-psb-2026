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

After UniProt aggregation, each ligand affinity table is further restricted to
the 20 most negative remaining UniProt affinities. Weaker affinity rows are
discarded before pairwise features are generated.

## Coverage

The full Yamanishi gold standard has 5,127 positive ligand-target pairs. The
all-positive label artifact contains all 5,127. The feature-complete joined
artifact contains only positives that can be joined all the way to local
features:

```text
total Yamanishi positives:                  5,127
joined positives with complete features:       57
```

Current attrition:

```text
missing UniProt affinity for target:        4,126
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

Negative examples should be generated at training time, because they are a
sampling policy rather than ground truth. To materialize one deterministic
negative sample for inspection or an experiment, use:

```bash
python3 scripts/build_dataset.py \
  --include-negatives \
  --classifier-output data/processed/yamanishi_classifier_dataset.csv
```

Use a different `--seed` to create a different negative sample.

## Train Baselines

The training script generates negatives at runtime and trains the baseline
model family:

```bash
/Users/roshanklein-seetharaman/.pyenv/shims/python3 scripts/train_baselines.py
```

Outputs:

- `results/baseline_metrics.csv`
- `results/training_manifest.json`

The current baseline run uses `seed=42`, a random stratified 80/20 row split,
57 positive rows, and 50 runtime-sampled negatives. It uses
`data/raw/GoldStandardAffinities.zip` aggregated to UniProt through SIFTS.

The baseline feature sets now compare pairwise-only, pairwise plus PubChem
ligand descriptors, pairwise plus protein target features, and the combined
feature set. Protein target features are restricted to length, mass, and
degree. Current best baseline results:

```text
SVM, pairwise + ligand_all + target_all:               accuracy 0.727, F1 0.769, ROC-AUC 0.783, PR-AUC 0.846
SVM, pairwise + ligand_all:                            accuracy 0.727, F1 0.750, ROC-AUC 0.817, PR-AUC 0.875
Random Forest, pairwise + ligand_all + target_all:     accuracy 0.591, F1 0.609, ROC-AUC 0.717, PR-AUC 0.805
```

This top-20 affinity experiment is intentionally aggressive and leaves only 22
held-out test rows. The affinity-only and pairwise-only comparison against the
previous full-affinity baseline is in
`results/top20_affinity_baseline_comparison.csv`; overall, the affinity-derived
features do not improve reliably after this restriction.

## No-Affinity Yamanishi Models

To test whether affinity coverage is the limiting factor, build a Yamanishi
dataset that ignores affinity files completely and uses ligand descriptors,
MACCS fingerprints, and target features:

```bash
python3 scripts/build_no_affinity_dataset.py
python3 scripts/train_no_affinity_models.py --n-iter 0
python3 scripts/train_no_affinity_models.py \
  --n-iter 8 \
  --feature-sets pubchem_plus_target \
  --classifiers "Random Forest tuned" "Extra Trees tuned" "Hist Gradient Boosting tuned" \
  --metrics-output results/no_affinity_tuned_model_metrics.csv \
  --manifest-output results/no_affinity_tuned_model_manifest.json
python3 scripts/train_no_affinity_models.py \
  --n-iter 5 \
  --feature-sets maccs_plus_target pubchem_plus_target pubchem_plus_maccs_plus_target \
  --classifiers "Random Forest tuned" "Hist Gradient Boosting tuned" \
  --metrics-output results/no_affinity_maccs_tuned_model_metrics.csv \
  --manifest-output results/no_affinity_maccs_tuned_model_manifest.json
python3 scripts/train_no_affinity_models.py \
  --n-iter 0 \
  --feature-sets target_basic_only target_rich_only maccs_plus_target maccs_plus_target_rich pubchem_plus_maccs_plus_target pubchem_plus_maccs_plus_target_rich \
  --classifiers "Random Forest tuned" "Hist Gradient Boosting tuned" "Extra Trees tuned" \
  --metrics-output results/no_affinity_rich_target_model_metrics.csv \
  --manifest-output results/no_affinity_rich_target_model_manifest.json
python3 scripts/train_no_affinity_models.py \
  --n-iter 6 \
  --feature-sets maccs_plus_target_rich pubchem_plus_maccs_plus_target_rich \
  --classifiers "Extra Trees tuned" "Hist Gradient Boosting tuned" \
  --metrics-output results/no_affinity_rich_target_tuned_model_metrics.csv \
  --manifest-output results/no_affinity_rich_target_tuned_model_manifest.json
```

Outputs:

- `data/processed/yamanishi_no_affinity_positive_rows.csv`
- `data/processed/yamanishi_no_affinity_classifier_dataset.csv`
- `data/processed/yamanishi_target_features_full.tsv`
- `data/processed/morgan_fingerprints.csv`
- `results/no_affinity_coverage_summary.csv`
- `results/full_target_features_missing_uniprots.csv`
- `results/no_affinity_model_metrics.csv`
- `results/no_affinity_tuned_model_metrics.csv`
- `results/no_affinity_maccs_tuned_model_metrics.csv`
- `results/no_affinity_rich_target_model_metrics.csv`
- `results/no_affinity_rich_target_tuned_model_metrics.csv`
- `results/expanded_no_affinity_model_metrics.csv`
- `results/similarity_feature_model_metrics.csv`

## Affinity Hit Value Scoring

The docking affinity files can also be treated as ranked candidate lists rather
than direct classifier features. In this setup, each row is one
`PubChem CID + UniProt` hit from an affinity file:

```text
[ligand id][target id][rank/affinity context][support label]
```

The support label is:

- `label_supported = 1`: the hit is supported by Yamanishi or BindingDB.
- `label_supported = 0`: the hit is currently unsupported by those sources.

Unsupported does not mean experimentally false. It means "not in our current
known-positive sets", so precision values should be read as known-support
rates, not true biochemical false-positive rates.

Build the compact affinity-hit value dataset:

```bash
/Users/roshanklein-seetharaman/.pyenv/shims/python3 scripts/build_affinity_hit_value_dataset.py
```

Evaluate whether the ranked affinity lists are enriched for known binders:

```bash
/Users/roshanklein-seetharaman/.pyenv/shims/python3 scripts/analyze_affinity_rank_positions.py
/Users/roshanklein-seetharaman/.pyenv/shims/python3 scripts/train_affinity_hit_value_model.py \
  --split-mode row \
  --output-prefix affinity_hit_value_row
/Users/roshanklein-seetharaman/.pyenv/shims/python3 scripts/train_affinity_hit_value_model.py \
  --split-mode ligand \
  --output-prefix affinity_hit_value_ligand_holdout
```

Current compact dataset:

```text
rows:                         214,842
supported rows:                 2,862
unsupported / unlabeled rows: 211,980
ligand affinity lists:             665
```

Raw rank cutoffs are weak: top 20 captures only 55 / 2,721 ranked Yamanishi
known positives (2.0%). But a learned value score is strongly enriched.
In ligand-held-out splits:

```text
rank-only Extra Trees:                    PR-AUC 0.497
rank + PubChem + target length/mass/deg:  PR-AUC 0.534
rank + PubChem + MACCS + target basic:    PR-AUC 0.539
```

The best current chemistry + biology model uses rank features, PubChem
physicochemical descriptors, MACCS fingerprints, and target length/mass/degree.
It reaches a 56.9% known-support rate in the top 1.0% of scored held-out hits,
versus a 1.0% baseline support rate.

To join chemistry/protein features at training time without materializing a
multi-GB affinity-hit CSV:

```bash
/Users/roshanklein-seetharaman/.pyenv/shims/python3 scripts/train_affinity_hit_value_model.py \
  --augment-feature-maps \
  --split-mode ligand \
  --output-prefix affinity_hit_value_maccs_biology_ligand_holdout \
  --feature-sets rank_plus_maccs_basic_target \
  --classifiers "Logistic Regression" "Hist Gradient Boosting"
```

Reranking figures can be regenerated with:

```bash
/Users/roshanklein-seetharaman/.pyenv/shims/python3 scripts/plot_affinity_reranking.py
/Users/roshanklein-seetharaman/.pyenv/shims/python3 scripts/plot_affinity_reranking_story.py
```

Outputs:

- `results/plots/affinity_reranking_rescue_ribbons.png`
- `results/plots/affinity_reranking_priority_funnel.png`
- `results/plots/affinity_reranking_rescue_map.png`
- `results/plots/affinity_reranking_enrichment_curve.png`
- `results/plots/affinity_reranking_top_fraction_bars.png`
- `results/plots/affinity_reranking_supported_scatter.png`
- `results/plots/affinity_reranking_ligand_examples.png`

The expanded no-affinity table can be rebuilt with:

```bash
python3 scripts/build_full_target_features.py
python3 scripts/build_morgan_fingerprints.py
python3 scripts/build_no_affinity_dataset.py
python3 scripts/train_no_affinity_models.py \
  --n-iter 0 \
  --feature-sets morgan_only pubchem_plus_morgan morgan_plus_target_rich pubchem_plus_maccs_plus_morgan_plus_target_rich \
  --classifiers "Extra Trees tuned" "Hist Gradient Boosting tuned" \
  --metrics-output results/expanded_no_affinity_model_metrics.csv \
  --manifest-output results/expanded_no_affinity_model_manifest.json
python3 scripts/train_similarity_feature_models.py
```

Coverage improves substantially without the affinity requirement:

```text
total Yamanishi positives:                  5,127
joined no-affinity positives:               5,065
sampled no-affinity negatives:              5,065
missing PubChem descriptor rows:            24
missing KEGG drug -> PubChem CID mappings:  21
missing KEGG target -> UniProt mappings:    17
missing target features:                    0
```

The full target feature table fills missing mapped Yamanishi UniProt accessions
from UniProt sequence records. Morgan/ECFP fingerprints are generated from the
local canonical SMILES in `maccs_fingerprints (2).csv`.

Best expanded no-affinity results:

```text
Extra Trees, PubChem + MACCS + Morgan + rich target: accuracy 0.875, F1 0.874, ROC-AUC 0.938, PR-AUC 0.946
Extra Trees, Morgan + rich target:                   accuracy 0.858, F1 0.858, ROC-AUC 0.929, PR-AUC 0.936
Hist Gradient Boosting, PubChem + MACCS + Morgan + rich target: accuracy 0.845, F1 0.849, ROC-AUC 0.918, PR-AUC 0.916
```

This is a clear improvement over the earlier `0.838` balanced accuracy baseline.
It still does not fully reach the strongest 0.90+ balanced literature accuracy
claims, but the ranking metrics are now in that range.

The similarity-feature script adds train-fold-only features analogous to
drug-drug and protein-protein similarity evidence:

```text
Extra Trees, molecular + similarity: accuracy 0.813, ROC-AUC 0.917, PR-AUC 0.935
Extra Trees, similarity only:        accuracy 0.506, ROC-AUC 0.841, PR-AUC 0.840
```

These similarity features are informative for ranking but are poorly calibrated
at the default `0.5` decision threshold, so they do not improve raw balanced
accuracy in the current classifier setup.

### Balanced Accuracy Gap Audit

To audit why the balanced 1:1 accuracy is below stronger literature reports,
run:

```bash
python3 scripts/audit_balanced_accuracy_gap.py
```

Outputs:

- `results/balanced_accuracy_gap_coverage_by_category.csv`
- `results/balanced_accuracy_gap_conflicts.csv`
- `results/balanced_accuracy_gap_model_metrics.csv`
- `results/balanced_accuracy_gap_summary.json`

Current findings:

```text
feature-complete positives:           5,065 / 5,127
combined 80/20 best accuracy:         0.873
combined 5-fold Extra Trees accuracy: 0.865 +/- 0.005
best per-category accuracy:           0.888 on enzyme
conflicting exact feature rows:        2
```

The main gap versus 0.90+ balanced literature numbers appears to be protocol and
feature representation, not a simple label-join bug. Older Yamanishi papers
often train/evaluate per target class and use richer drug-drug/protein-protein
similarity or graph features. After restoring target feature coverage and adding
Morgan fingerprints, the remaining gap is much smaller and is probably model
class/protocol rather than missing target data.

### Negative Ratio Accuracy

To compare against literature reporting raw accuracy under larger negative
sets, run:

```bash
python3 scripts/train_negative_ratio_models.py
```

Output:

- `results/negative_ratio_accuracy_metrics.csv`

Using PubChem + MACCS + rich target features, raw accuracy crosses 90% when the
negative sampling ratio is increased:

```text
1:1 negatives, Extra Trees:  accuracy 0.843, balanced accuracy 0.843, always-negative accuracy 0.500
3:1 negatives, Extra Trees:  accuracy 0.893, balanced accuracy 0.810, always-negative accuracy 0.750
5:1 negatives, Extra Trees:  accuracy 0.913, balanced accuracy 0.771, always-negative accuracy 0.833
10:1 negatives, Extra Trees: accuracy 0.942, balanced accuracy 0.714, always-negative accuracy 0.909
```

This reproduces 90%+ raw accuracy, but it also shows why raw accuracy is
sensitive to the negative sampling protocol. At 10:1 negatives, an
always-negative classifier already gets `0.909` accuracy.

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

The current top-20 affinity split has only 22 held-out rows, so its error plots
are mainly sanity-check artifacts. The broader ligand-degree bias analysis was
more informative before the top-20 affinity restriction, when the held-out set
had 1,088 rows.

## Glyco Pair Predictions

External glyco metabolite-protein pairs from
`data/raw/Analysis_Glyc_data_PDB_aff.xlsx` can be scored with:

```bash
python3 scripts/predict_glyco_pairs.py
```

This trains the compatible no-affinity Yamanishi model on all available
Yamanishi no-affinity rows, then predicts the glyco pairs. The glyco ligand
property file does not include MACCS fingerprints or SMILES, so this prediction
path uses PubChem numeric ligand descriptors plus rich target sequence features.

Outputs:

- `results/glyco_pair_predictions.csv`
- `results/glyco_pair_predictions_compact.csv`
- `results/glyco_pair_predictions_summary_by_metabolite.csv`
- `results/glyco_pair_predictions_skipped.csv`

Current prediction coverage is 1,868 scored pairs and 0 skipped pairs.

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
