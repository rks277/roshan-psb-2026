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
| Split modes | row-stratified, ligand-held-out, target-held-out |

Split sizes:

| Split mode | Train rows | Validation rows | Test rows | Train positives | Validation positives | Test positives |
|---|---:|---:|---:|---:|---:|---:|
| Row-stratified | 3,262 | 1,088 | 1,088 | 1,631 | 544 | 544 |
| Ligand-held-out | 3,331 | 955 | 1,152 | 1,666 | 435 | 618 |
| Target-held-out | 3,237 | 1,066 | 1,135 | 1,603 | 528 | 588 |

This is not the full 5,127-pair Yamanishi benchmark because these experiments require affinity/rank context. The classifier only includes Yamanishi positives that are represented in the affinity-hit table.

## Raw Affinity Ranking Analysis

The affinity files should not be interpreted as simple "rank 1 equals binder" predictions. Known Yamanishi targets are often buried far down the raw docking list.

Among Yamanishi positives that could be mapped into the affinity files:

| Quantity | Value |
|---|---:|
| Ranked known positives | 2,721 |
| Mean known-target rank | 1,896.6 |
| Median known-target rank | 1,325 |
| 75th percentile rank | 2,770 |
| 90th percentile rank | 4,397 |
| 95th percentile rank | 5,727 |
| Maximum known-target rank | 7,528 |
| Known target ranked first | 4 pairs |
| Known positives with at least one higher-ranked target | 2,717 |

Raw-rank recall is low at practical cutoffs:

| Raw rank cutoff | Known positives captured | Known-positive recall |
|---:|---:|---:|
| 1 | 4 | 0.15% |
| 3 | 16 | 0.59% |
| 5 | 22 | 0.81% |
| 10 | 35 | 1.29% |
| 20 | 55 | 2.02% |
| 50 | 100 | 3.68% |
| 100 | 160 | 5.88% |
| 200 | 290 | 10.66% |

Targets ranked above known positives are mostly unsupported by the current Yamanishi/BindingDB labels, but they should not automatically be called true false positives because many may simply be untested.

| Raw cutoff | Higher-ranked calls before known positive | BindingDB-supported fraction among higher-ranked calls | Unsupported fraction among higher-ranked calls |
|---:|---:|---:|---:|
| 10 | 138 | 1.45% | 92.75% |
| 20 | 432 | 0.46% | 96.99% |
| 50 | 1,997 | 0.10% | 98.65% |
| 100 | 6,637 | 0.15% | 98.95% |
| 200 | 25,908 | 0.06% | 99.32% |

Interpretation:

- Raw affinity rank alone is weak for direct target calling.
- The affinity files are still useful as candidate generators.
- The right framing is positive-unlabeled reranking: unsupported high-ranking targets are candidate novel interactions, not confirmed negatives.
- This explains why the affinity-only classifier below performs much worse than chemistry/protein feature models.

Raw ranking files:

- [`results/affinity_rank_raw_summary.json`](../results/affinity_rank_raw_summary.json)
- [`results/affinity_rank_raw_cutoff_summary.csv`](../results/affinity_rank_raw_cutoff_summary.csv)
- [`results/affinity_rank_raw_known_positive_positions.csv`](../results/affinity_rank_raw_known_positive_positions.csv)
- [`scripts/analyze_affinity_rank_positions.py`](../scripts/analyze_affinity_rank_positions.py)

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

Best test-set models:

| Split mode | Model | Test accuracy | Test F1 | Test ROC-AUC | Test PR-AUC |
|---|---|---:|---:|---:|---:|
| Row-stratified | Random Forest | 0.716 | 0.728 | 0.777 | 0.765 |
| Ligand-held-out | SVM | 0.704 | 0.728 | 0.737 | 0.719 |
| Target-held-out | Hist Gradient Boosting | 0.732 | 0.750 | 0.796 | 0.782 |

Interpretation: affinity/rank context alone is informative but not strong enough to reproduce the high traditional classifier accuracy.

## 3. Traditional Classifier With All Features Minus Affinity Group

Feature set: `all_features_minus_affinity_group`

Features used: PubChem descriptors, MACCS, Morgan/ECFP, and protein biology features. Affinity/rank context features were excluded.

Best test-set models:

| Split mode | Model | Test accuracy | Test F1 | Test ROC-AUC | Test PR-AUC |
|---|---|---:|---:|---:|---:|
| Row-stratified | Extra Trees | 0.917 | 0.915 | 0.972 | 0.975 |
| Ligand-held-out | Random Forest | 0.929 | 0.931 | 0.970 | 0.974 |
| Target-held-out | SVM | 0.789 | 0.780 | 0.875 | 0.870 |

Interpretation: ligand chemistry plus protein biology carries most of the traditional classifier signal.

## 4. Traditional Classifier With All Features

Feature set: `all_features`

Features used: affinity/rank context, PubChem descriptors, MACCS, Morgan/ECFP, and protein biology features.

Best test-set models:

| Split mode | Model | Test accuracy | Test F1 | Test ROC-AUC | Test PR-AUC |
|---|---|---:|---:|---:|---:|
| Row-stratified | Extra Trees | 0.915 | 0.913 | 0.973 | 0.976 |
| Ligand-held-out | Random Forest | 0.922 | 0.925 | 0.970 | 0.971 |
| Target-held-out | SVM | 0.799 | 0.791 | 0.886 | 0.878 |

Interpretation: all features also reach the 90%+ traditional classifier range. In this split, adding affinity features does not improve raw accuracy relative to chemistry/protein features alone, although ROC-AUC and PR-AUC are essentially tied.

## 5. Performance Metric Comparison

The full train/validation/test table for every split mode is in [`results/traditional_clean_feature_group_classifier_metrics.csv`](../results/traditional_clean_feature_group_classifier_metrics.csv). The row-stratified table, which is most similar to the traditional 2025-style split, is shown below:

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

Best model per requested feature condition and split:

| Split mode | Condition | Feature set | Best test model | Test Acc | Test F1 | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|---:|---:|---:|---:|
| Row-stratified | Affinities group only | `affinity_group_only` | Random Forest | 0.716 | 0.728 | 0.777 | 0.765 |
| Row-stratified | All features minus affinities group | `all_features_minus_affinity_group` | Extra Trees | 0.917 | 0.915 | 0.972 | 0.975 |
| Row-stratified | All features | `all_features` | Extra Trees | 0.915 | 0.913 | 0.973 | 0.976 |
| Ligand-held-out | Affinities group only | `affinity_group_only` | SVM | 0.704 | 0.728 | 0.737 | 0.719 |
| Ligand-held-out | All features minus affinities group | `all_features_minus_affinity_group` | Random Forest | 0.929 | 0.931 | 0.970 | 0.974 |
| Ligand-held-out | All features | `all_features` | Random Forest | 0.922 | 0.925 | 0.970 | 0.971 |
| Target-held-out | Affinities group only | `affinity_group_only` | Hist Gradient Boosting | 0.732 | 0.750 | 0.796 | 0.782 |
| Target-held-out | All features minus affinities group | `all_features_minus_affinity_group` | SVM | 0.789 | 0.780 | 0.875 | 0.870 |
| Target-held-out | All features | `all_features` | SVM | 0.799 | 0.791 | 0.886 | 0.878 |

Takeaway: traditional 0/1 classifier accuracy reaches the 90%+ range for row-stratified and ligand-held-out splits when ligand chemistry and protein biology features are included. Target-held-out performance is lower, which indicates that generalizing to unseen targets is harder. Affinity/rank features alone are weaker in all split modes.

### Did Affinity Information Help Each Model?

This comparison asks whether adding the 9 affinity/rank-context features improved performance relative to the same model trained on all non-affinity features.

Row-stratified:

| Model | No-affinity test Acc | All-features test Acc | Delta Acc | No-affinity test F1 | All-features test F1 | Delta F1 | Did affinity help? |
|---|---:|---:|---:|---:|---:|---:|---|
| Logistic Regression | 0.814 | 0.812 | -0.002 | 0.820 | 0.818 | -0.003 | no |
| Random Forest | 0.911 | 0.911 | +0.000 | 0.908 | 0.908 | +0.001 | tied |
| Extra Trees | 0.917 | 0.915 | -0.003 | 0.915 | 0.913 | -0.002 | no |
| Hist Gradient Boosting | 0.912 | 0.913 | +0.001 | 0.911 | 0.913 | +0.002 | tied |
| SVM | 0.858 | 0.858 | +0.001 | 0.856 | 0.857 | +0.001 | tied |
| KNN | 0.722 | 0.726 | +0.004 | 0.747 | 0.749 | +0.003 | yes, small |

Ligand-held-out:

| Model | No-affinity test Acc | All-features test Acc | Delta Acc | No-affinity test F1 | All-features test F1 | Delta F1 | Did affinity help? |
|---|---:|---:|---:|---:|---:|---:|---|
| Logistic Regression | 0.753 | 0.773 | +0.019 | 0.747 | 0.771 | +0.023 | yes |
| Random Forest | 0.929 | 0.922 | -0.007 | 0.931 | 0.925 | -0.007 | no |
| Extra Trees | 0.840 | 0.845 | +0.004 | 0.829 | 0.836 | +0.007 | yes |
| Hist Gradient Boosting | 0.898 | 0.905 | +0.007 | 0.901 | 0.908 | +0.007 | yes |
| SVM | 0.684 | 0.696 | +0.012 | 0.609 | 0.634 | +0.024 | yes |
| KNN | 0.680 | 0.688 | +0.008 | 0.703 | 0.710 | +0.007 | yes |

Target-held-out:

| Model | No-affinity test Acc | All-features test Acc | Delta Acc | No-affinity test F1 | All-features test F1 | Delta F1 | Did affinity help? |
|---|---:|---:|---:|---:|---:|---:|---|
| Logistic Regression | 0.717 | 0.722 | +0.005 | 0.705 | 0.711 | +0.006 | yes |
| Random Forest | 0.625 | 0.667 | +0.042 | 0.444 | 0.540 | +0.096 | yes |
| Extra Trees | 0.719 | 0.759 | +0.040 | 0.638 | 0.709 | +0.071 | yes |
| Hist Gradient Boosting | 0.741 | 0.747 | +0.006 | 0.692 | 0.699 | +0.007 | yes |
| SVM | 0.789 | 0.799 | +0.010 | 0.780 | 0.791 | +0.011 | yes |
| KNN | 0.658 | 0.659 | +0.001 | 0.678 | 0.677 | -0.001 | tied |

Overall: affinity/rank information does not materially help in the row-stratified split and does not improve the best ligand-held-out model. It helps more consistently in the target-held-out split, where generalizing to unseen targets is harder, but the absolute target-held-out accuracy remains below the row and ligand splits.

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

1. **Row-stratified splits can be optimistic.** The same ligand can appear in train, validation, and test rows paired with different targets. The same target can also appear across splits. With Morgan fingerprints and protein sequence features, the model may partially learn ligand/target identity patterns. This is not direct feature leakage, but it can inflate traditional classifier performance. We therefore added ligand-held-out and target-held-out splits in this report.
2. **Affinity context assumes a full docking list is available.** Features such as rank, percentile rank, affinity z-score, robust z-score, and local affinity gaps are computed from the full ligand-specific affinity list. This is valid for the intended reranking use case because a new ligand would have a full affinity file, but it should be described as target prioritization/reranking rather than isolated ligand-target prediction with no docking context.

Suggested paper language:

> The reported clean models exclude label-prior features that encode known Yamanishi or BindingDB membership. However, traditional row-stratified classifier results may be optimistic because the same ligands and targets can appear across train and test rows. We therefore also evaluated ligand-held-out and target-held-out splits to estimate generalization to new ligand affinity files and unseen targets.

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

The traditional classifier was evaluated using three feature conditions: affinity/rank features only, all non-affinity ligand and protein features, and the complete clean feature set. The balanced dataset contained 2,719 Yamanishi-supported positive ligand-target pairs and 2,719 sampled unsupported pairs. Three train/validation/test protocols were evaluated with random seed 42: row-stratified, ligand-held-out, and target-held-out. Six classifiers were evaluated: logistic regression, random forest, extra trees, histogram gradient boosting, support vector machine, and k-nearest neighbors. Models were compared using accuracy, F1 score, ROC-AUC, PR-AUC, false-positive rate, and false-negative rate.

## Suggested Draft Text For Results

Affinity/rank features alone provided moderate discrimination across split modes, with best test accuracy ranging from 0.704 to 0.732. In row-stratified evaluation, ligand chemistry and protein biology features without affinity/rank context reached test accuracy 0.917 and F1 0.915, while all features reached test accuracy 0.915 and F1 0.913. In ligand-held-out evaluation, the best no-affinity model reached test accuracy 0.929 and F1 0.931, while all features reached test accuracy 0.922 and F1 0.925. In target-held-out evaluation, performance was lower: the best no-affinity model reached test accuracy 0.789 and F1 0.780, while all features reached test accuracy 0.799 and F1 0.791. Thus, the traditional balanced classifier reaches the 90%+ accuracy range for row and ligand-held-out splits when ligand and target feature coverage is strong, but generalization to unseen targets is harder. Affinity/rank features alone are insufficient; they help most in the target-held-out split but do not improve the best row-stratified or ligand-held-out model.
