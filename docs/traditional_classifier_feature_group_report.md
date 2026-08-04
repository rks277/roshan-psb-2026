# Traditional Classifier Feature-Group Report

This note summarizes the traditional balanced classifier experiments using the same clean feature universe used by the current affinity-reranking model.

## Dataset And Split

The classifier dataset was built from `data/processed/affinity_hit_value_dataset_compact.csv`.

Labels:

- Positive class: Yamanishi-supported ligand-target pairs with affinity/rank rows.
- Negative class: sampled unsupported/unlabeled ligand-target pairs from the same affinity-hit table.
- Important caveat: `0` means unsupported/unlabeled, not experimentally confirmed non-binding.

Balanced dataset:

| Quantity | Value |
|---|---:|
| Total rows | 5,438 |
| Positive rows | 2,719 |
| Negative/unlabeled sampled rows | 2,719 |
| Random seed | 42 |
| Split mode | row-stratified train/validation/test |
| Train rows | 3,262 |
| Validation rows | 1,088 |
| Test rows | 1,088 |

This is not the full 5,127-pair Yamanishi benchmark because these experiments require affinity/rank context. The classifier only includes Yamanishi positives that are represented in the affinity-hit table.

## 1. Feature List

The clean feature universe excludes label-prior/context features that encode known-positive label-set membership:

- `ligand_yamanishi_degree_any`
- `target_yamanishi_degree_any`
- `target_in_yamanishi_universe`
- `target_in_bindingdb_universe`

Feature definitions are generated in:

- [`scripts/build_affinity_hit_value_dataset.py`](../scripts/build_affinity_hit_value_dataset.py)
- [`scripts/build_no_affinity_dataset.py`](../scripts/build_no_affinity_dataset.py)
- [`scripts/train_affinity_hit_value_model.py`](../scripts/train_affinity_hit_value_model.py)

| Feature group | Columns / pattern | Count | Included in affinity-only | Included in all-minus-affinity | Included in all features |
|---|---|---:|---:|---:|---:|
| Affinity/rank context | `affinity`, `rank_1_based`, `rank_percentile`, `reverse_rank_percentile`, `affinity_zscore_within_ligand`, `affinity_robust_zscore_within_ligand`, `affinity_gap_to_next_weaker`, `affinity_gap_to_previous_stronger`, `total_ranked_uniprots` | 9 | yes | no | yes |
| PubChem ligand descriptors | PubChem physicochemical and 3D descriptors, e.g. molecular weight, monoisotopic mass, TPSA, complexity, charge, H-bond counts, rotatable bonds, heavy atoms, stereochemistry counts, covalent units, XLogP, 3D conformer descriptors | 30 | no | yes | yes |
| MACCS fingerprint | `ligand_MACCS_0` through `ligand_MACCS_165` | 166 | no | yes | yes |
| Morgan/ECFP fingerprint | `ligand_Morgan_0` through `ligand_Morgan_1023` | 1,024 | no | yes | yes |
| Basic protein descriptors | `target_length`, `target_mass`, `target_degree_up` | 3 | no | yes | yes |
| Amino-acid composition | `target_aa_A` through `target_aa_Y` for the 20 standard amino acids | 20 | no | yes | yes |
| Residue-group composition | hydrophobic, polar, positive, negative, charged, aromatic, tiny, small, proline, glycine, cysteine fractions | 11 | no | yes | yes |
| Dipeptide composition | `target_dipeptide_AA` through `target_dipeptide_YY`, all 400 ordered adjacent amino-acid pairs | 400 | no | yes | yes |
| **Total** |  | **1,663** | **9** | **1,654** | **1,663** |

## 2. Traditional Classifier With Affinity-Group Features Only

Feature set: `affinity_group_only`

Features used: the 9 affinity/rank context features only.

Best test-set model:

| Model | Test accuracy | Test F1 | Test ROC-AUC | Test PR-AUC |
|---|---:|---:|---:|---:|
| Random Forest | 0.716 | 0.728 | 0.777 | 0.765 |

Interpretation: affinity/rank context alone is informative but not strong enough to reproduce the high traditional classifier accuracy.

## 3. Traditional Classifier With All Features Minus Affinity Group

Feature set: `all_features_minus_affinity_group`

Features used: PubChem descriptors, MACCS, Morgan/ECFP, and protein biology features. Affinity/rank context features were excluded.

Best test-set model:

| Model | Test accuracy | Test F1 | Test ROC-AUC | Test PR-AUC |
|---|---:|---:|---:|---:|
| Extra Trees | 0.917 | 0.915 | 0.972 | 0.975 |

Interpretation: ligand chemistry plus protein biology carries most of the traditional classifier signal.

## 4. Traditional Classifier With All Features

Feature set: `all_features`

Features used: affinity/rank context, PubChem descriptors, MACCS, Morgan/ECFP, and protein biology features.

Best test-set model:

| Model | Test accuracy | Test F1 | Test ROC-AUC | Test PR-AUC |
|---|---:|---:|---:|---:|
| Extra Trees | 0.915 | 0.913 | 0.973 | 0.976 |

Interpretation: all features also reach the 90%+ traditional classifier range. In this split, adding affinity features does not improve raw accuracy relative to chemistry/protein features alone, although ROC-AUC and PR-AUC are essentially tied.

## 5. Performance Metric Comparison

Full train/validation/test table:

| Feature set | Model | Features | Train Acc | Train F1 | Val Acc | Val F1 | Test Acc | Test F1 | Test ROC-AUC | Test PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| affinity_group_only | Logistic Regression | 9 | 0.693 | 0.709 | 0.701 | 0.722 | 0.691 | 0.704 | 0.745 | 0.725 |
| affinity_group_only | Random Forest | 9 | 1.000 | 1.000 | 0.690 | 0.707 | 0.716 | 0.728 | 0.777 | 0.765 |
| affinity_group_only | Extra Trees | 9 | 0.983 | 0.983 | 0.694 | 0.712 | 0.694 | 0.706 | 0.760 | 0.750 |
| affinity_group_only | Hist Gradient Boosting | 9 | 0.989 | 0.989 | 0.684 | 0.694 | 0.708 | 0.718 | 0.778 | 0.765 |
| affinity_group_only | SVM | 9 | 0.700 | 0.715 | 0.708 | 0.726 | 0.701 | 0.711 | 0.752 | 0.723 |
| affinity_group_only | KNN | 9 | 0.773 | 0.782 | 0.675 | 0.687 | 0.688 | 0.701 | 0.744 | 0.700 |
| all_features_minus_affinity_group | Logistic Regression | 1,654 | 0.978 | 0.978 | 0.806 | 0.813 | 0.814 | 0.820 | 0.860 | 0.794 |
| all_features_minus_affinity_group | Random Forest | 1,654 | 1.000 | 1.000 | 0.911 | 0.908 | 0.911 | 0.908 | 0.962 | 0.966 |
| all_features_minus_affinity_group | Extra Trees | 1,654 | 1.000 | 1.000 | 0.920 | 0.917 | 0.917 | 0.915 | 0.972 | 0.975 |
| all_features_minus_affinity_group | Hist Gradient Boosting | 1,654 | 1.000 | 1.000 | 0.917 | 0.917 | 0.912 | 0.911 | 0.971 | 0.973 |
| all_features_minus_affinity_group | SVM | 1,654 | 0.944 | 0.944 | 0.852 | 0.848 | 0.858 | 0.856 | 0.924 | 0.916 |
| all_features_minus_affinity_group | KNN | 1,654 | 0.804 | 0.815 | 0.694 | 0.719 | 0.722 | 0.747 | 0.786 | 0.747 |
| all_features | Logistic Regression | 1,663 | 0.982 | 0.982 | 0.823 | 0.828 | 0.812 | 0.818 | 0.860 | 0.792 |
| all_features | Random Forest | 1,663 | 1.000 | 1.000 | 0.915 | 0.912 | 0.911 | 0.908 | 0.965 | 0.967 |
| all_features | Extra Trees | 1,663 | 1.000 | 1.000 | 0.916 | 0.914 | 0.915 | 0.913 | 0.973 | 0.976 |
| all_features | Hist Gradient Boosting | 1,663 | 1.000 | 1.000 | 0.914 | 0.913 | 0.913 | 0.913 | 0.973 | 0.973 |
| all_features | SVM | 1,663 | 0.945 | 0.945 | 0.853 | 0.850 | 0.858 | 0.857 | 0.930 | 0.923 |
| all_features | KNN | 1,663 | 0.805 | 0.816 | 0.694 | 0.719 | 0.726 | 0.749 | 0.790 | 0.748 |

Best model per requested feature condition:

| Condition | Feature set | Best test model | Test Acc | Test F1 | Test ROC-AUC | Test PR-AUC |
|---|---|---|---:|---:|---:|---:|
| Affinities group only | `affinity_group_only` | Random Forest | 0.716 | 0.728 | 0.777 | 0.765 |
| All features minus affinities group | `all_features_minus_affinity_group` | Extra Trees | 0.917 | 0.915 | 0.972 | 0.975 |
| All features | `all_features` | Extra Trees | 0.915 | 0.913 | 0.973 | 0.976 |

Takeaway: traditional 0/1 classifier accuracy now reaches the 90%+ range when ligand chemistry and protein biology features are included. Affinity/rank features alone are weaker. Adding affinity/rank features to the full chemistry/protein feature set does not improve test accuracy in this row-stratified traditional classifier experiment.

### Did Affinity Information Help Each Model?

This comparison asks whether adding the 9 affinity/rank-context features improved performance relative to the same model trained on all non-affinity features.

| Model | No-affinity test Acc | All-features test Acc | Delta Acc | No-affinity test F1 | All-features test F1 | Delta F1 | Did affinity help? |
|---|---:|---:|---:|---:|---:|---:|---|
| Logistic Regression | 0.814 | 0.812 | -0.002 | 0.820 | 0.818 | -0.003 | no |
| Random Forest | 0.911 | 0.911 | +0.000 | 0.908 | 0.908 | +0.001 | tied |
| Extra Trees | 0.917 | 0.915 | -0.003 | 0.915 | 0.913 | -0.002 | no |
| Hist Gradient Boosting | 0.912 | 0.913 | +0.001 | 0.911 | 0.913 | +0.002 | tied |
| SVM | 0.858 | 0.858 | +0.001 | 0.856 | 0.857 | +0.001 | tied |
| KNN | 0.722 | 0.726 | +0.004 | 0.747 | 0.749 | +0.003 | yes, small |

Overall, affinity/rank information did not materially improve the traditional balanced classifier once ligand chemistry and protein biology features were included. The only model with a positive test-accuracy change was KNN, and the gain was small (`+0.004`). Extra Trees, the best model overall, performed slightly better without the affinity group by accuracy (`0.917` without affinity vs `0.915` with all features), while ROC-AUC and PR-AUC were essentially tied.

## Leakage And Overoptimism Check

Strict label leakage: **not present in the clean feature sets used here**. The explicitly suspicious label-prior/context features were excluded:

- `ligand_yamanishi_degree_any`
- `target_yamanishi_degree_any`
- `target_in_yamanishi_universe`
- `target_in_bindingdb_universe`

Those excluded features directly encode how often ligands or targets appear in the known-positive Yamanishi/BindingDB label sets, so they are not appropriate for the clean model.

Clean feature groups:

- PubChem descriptors
- MACCS fingerprints
- Morgan/ECFP fingerprints
- target length, mass, and `degree_up`
- amino-acid composition
- residue-group composition
- dipeptide composition
- affinity/rank context

Important caveat on `target_degree_up`: this is treated as a biological/network feature from the protein feature table. It is not the same as `target_yamanishi_degree_any`, which was excluded because it encodes Yamanishi label-set degree.

There are still two protocol-level overoptimism risks:

1. **Row-stratified splits can be optimistic.** The same ligand can appear in train, validation, and test rows paired with different targets. The same target can also appear across splits. With Morgan fingerprints and protein sequence features, the model may partially learn ligand/target identity patterns. This is not direct feature leakage, but it can inflate traditional classifier performance.
2. **Affinity context assumes a full docking list is available.** Features such as rank, percentile rank, affinity z-score, robust z-score, and local affinity gaps are computed from the full ligand-specific affinity list. This is valid for the intended reranking use case because a new ligand would have a full affinity file, but it should be described as target prioritization/reranking rather than isolated ligand-target prediction with no docking context.

Suggested paper language:

> The reported clean models exclude label-prior features that encode known Yamanishi or BindingDB membership. However, traditional row-stratified classifier results may be optimistic because the same ligands and targets can appear across train and test rows. Ligand-held-out evaluation is therefore used for the reranking analysis to better estimate generalization to new ligand affinity files.

Raw result files:

- [`results/traditional_clean_feature_group_classifier_metrics.csv`](../results/traditional_clean_feature_group_classifier_metrics.csv)
- [`results/traditional_clean_feature_group_classifier_manifest.json`](../results/traditional_clean_feature_group_classifier_manifest.json)
- [`scripts/train_traditional_feature_group_classifiers.py`](../scripts/train_traditional_feature_group_classifiers.py)

## 6. Equation Check In Current Paper Draft

The current draft equations in Section 4.5 are conceptually right but have formatting/syntax problems after Word conversion. These should be replaced before submission.

### Equation 1: Candidate Set

Current draft:

```text
L={(l_i,t_j)}
```

Recommended:

```latex
\mathcal{L}=\{(l_i,t_j)\}
```

Plain-language meaning: the candidate set contains ligand-target pairs.

### Equation 2: Feature Vector

Current draft has the vector split across lines and missing an equals sign:

```text
\mathbf{x}_{ij}
\left[
\mathbf{x}^{\text{affinity}},
\mathbf{x}^{\text{ligand}},
\mathbf{x}^{\text{target}}
\right].
```

Recommended:

```latex
\mathbf{x}_{ij} =
\left[
\mathbf{x}^{\mathrm{affinity}}_{ij},
\mathbf{x}^{\mathrm{ligand}}_{i},
\mathbf{x}^{\mathrm{target}}_{j}
\right]
```

Plain-language meaning: each pair is represented by concatenating affinity/rank features, ligand features, and target features.

### Equation 3: Positive Labels

Current draft:

```text
y_{ij}=1
```

Recommended:

```latex
y_{ij}=1
```

This is fine as written. The surrounding text should specify that `1` means externally supported by Yamanishi or BindingDB.

### Equation 4: Unlabeled Rows

Current draft:

```text
y_{ij}=0
```

Recommended:

```latex
y_{ij}=0
```

This is mathematically acceptable if the text says that `0` means unlabeled/unsupported, not confirmed non-binding. If we want to be extra precise for positive-unlabeled learning, we could write:

```latex
u_{ij}=0
```

where `u` denotes observed support status rather than true biochemical non-binding.

### Equation 5: Scoring Function

Current draft has missing subscripts:

```text
f(\mathbf{x}{ij}) \rightarrow s{ij}
```

Recommended:

```latex
f(\mathbf{x}_{ij}) \rightarrow s_{ij}
```

or:

```latex
s_{ij}=f(\mathbf{x}_{ij})
```

Plain-language meaning: the model maps each pair-level feature vector to a continuous hit-value score.

### Equation 6: Reranking Model

Current draft has the score and function adjacent without an equals sign:

```text
s_{ij}
f\left(
\mathbf{x}^{\text{affinity}},
\mathbf{x}^{\text{ligand}},
\mathbf{x}^{\text{target}}
\right)
```

Recommended:

```latex
s_{ij} =
f\left(
\mathbf{x}^{\mathrm{affinity}}_{ij},
\mathbf{x}^{\mathrm{ligand}}_{i},
\mathbf{x}^{\mathrm{target}}_{j}
\right)
```

Also update the sentence immediately after this equation:

```text
The final model utilized 212 input features.
```

Replace with:

```text
The final clean model utilized 1,663 input features after excluding label-prior/context features.
```

## Suggested Draft Text For Methods

The traditional classifier was evaluated using three feature conditions: affinity/rank features only, all non-affinity ligand and protein features, and the complete clean feature set. The balanced dataset contained 2,719 Yamanishi-supported positive ligand-target pairs and 2,719 sampled unsupported pairs. Rows were split into train, validation, and test sets using a row-stratified 60/20/20 split with random seed 42. Six classifiers were evaluated: logistic regression, random forest, extra trees, histogram gradient boosting, support vector machine, and k-nearest neighbors. Models were compared using accuracy, F1 score, ROC-AUC, PR-AUC, false-positive rate, and false-negative rate.

## Suggested Draft Text For Results

Affinity/rank features alone provided moderate discrimination, with the best affinity-only model reaching test accuracy 0.716 and F1 0.728. In contrast, ligand chemistry and protein biology features without affinity/rank context reached test accuracy 0.917 and F1 0.915 using an Extra Trees classifier. Adding affinity/rank context to all chemistry and protein features produced similar performance, with test accuracy 0.915 and F1 0.913. Thus, the traditional balanced classifier reaches the 90%+ accuracy range when ligand and target feature coverage is strong, but affinity/rank features alone are insufficient and do not improve raw balanced accuracy when added to the full feature set in this experiment.
