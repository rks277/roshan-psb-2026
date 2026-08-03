# Affinity Output Prioritization: Response to Initial Questions

This note summarizes what we did in response to the original questions about prioritizing predicted affinities in the Yamanishi gold-standard affinity outputs.

The short version is: raw docking/affinity rank is not reliable enough by itself. We built a UniProt-level annotation and scoring workflow that adds known-target flags and a model-based likelihood score to every mapped target in the affinity files.

## Main Code and Outputs

Code:

- [scripts/score_affinity_targets.py](../scripts/score_affinity_targets.py): scores full affinity lists and adds known-target and likelihood columns.
- [scripts/build_affinity_hit_value_dataset.py](../scripts/build_affinity_hit_value_dataset.py): builds the sampled training table used to learn which affinity hits look supported.
- [scripts/train_affinity_hit_value_model.py](../scripts/train_affinity_hit_value_model.py): trains/evaluates hit-value models.
- [scripts/analyze_affinity_rank_positions.py](../scripts/analyze_affinity_rank_positions.py): analyzes where known positives fall in affinity-ranked lists.

Primary outputs:

- `results/annotated_affinity_targets.csv`: full annotated target table, 4,906,996 UniProt-level rows across 665 ligands. This file was generated locally but is about 1 GB, so it should be shared with Git LFS or separate storage rather than normal GitHub.
- [results/annotated_affinity_target_summary.csv](../results/annotated_affinity_target_summary.csv): per-ligand summary at the default threshold.
- [results/annotated_affinity_ligand_degree_estimates.csv](../results/annotated_affinity_ligand_degree_estimates.csv): estimated target degree per ligand.
- [results/annotated_affinity_targets_cid10917.csv](../results/annotated_affinity_targets_cid10917.csv): focused output for (-)-Carnitine / PubChem CID 10917.
- [results/affinity_hit_value_model_metrics.csv](../results/affinity_hit_value_model_metrics.csv): held-out model metrics.
- [results/affinity_hit_value_enrichment.csv](../results/affinity_hit_value_enrichment.csv): enrichment of supported hits among top-scored rows.

Note: the code, manifest, and smaller summaries can stay in the repository; the full annotated CSV is too large for normal GitHub storage.

## 1. Should We Prioritize by Raw Rank / Most Negative Affinity?

Original concern: we had been using raw rank, with the most negative affinity treated as the top target, but that may not be a good strategy.

What we found:

- Raw rank alone is not reliable.
- Known Yamanishi positives often appear far below many stronger-scored unknowns.
- A top docking score can be a common/promiscuous structural hit rather than a known target-like hit.

Relevant outputs:

- [results/affinity_rank_raw_known_positive_positions.csv](../results/affinity_rank_raw_known_positive_positions.csv)
- [results/affinity_rank_raw_cutoff_summary.csv](../results/affinity_rank_raw_cutoff_summary.csv)
- [results/affinity_rank_raw_summary.json](../results/affinity_rank_raw_summary.json)

## 2. What Is the Likelihood That `3FV1_A` Is a Real Target for (-)-Carnitine?

The file in question is:

`(-)-Carnitine_(10917)_3D.pdb_affinities.txt`

The bottom/top-affinity entry shown in the prompt is:

`3FV1_A` with affinity `-6.3043`.

What we did:

- Mapped `3FV1_A` to UniProt using SIFTS/PDB-chain mapping.
- `3FV1_A` maps to UniProt `P39086`.
- Checked whether `P39086` is a known Yamanishi target for PubChem CID `10917`.
- It is not a known Yamanishi target for (-)-Carnitine.

Model scores for `3FV1_A` / `P39086` in [results/annotated_affinity_targets_cid10917.csv](../results/annotated_affinity_targets_cid10917.csv):

- `known_target_yamanishi = 0`
- `known_target_bindingdb = 0`
- `target_likelihood = 0.0043`
- `target_likelihood_sample_prior = 0.0909`
- `rank_1_based = 1`

Interpretation:

Although `3FV1_A` is rank 1 by raw affinity, it receives a low conservative likelihood after correcting for the huge number of candidate targets. This is a concrete example of why raw rank is insufficient.

## 3. Do We Need a Precision-Recall Curve for Each Ligand?

Answer: not as the main analysis.

Most ligands have too few known positives to support a stable ligand-specific precision-recall curve. A per-ligand PR curve would often be dominated by one or a few known targets and would be statistically unstable.

What we did instead:

- Built a global hit-value model over Yamanishi/BindingDB-supported versus unsupported affinity hits.
- Evaluated global held-out performance.
- Applied the trained model back to each ligand-specific affinity list.
- Produced per-ligand summaries and degree estimates.

Relevant files:

- [results/affinity_hit_value_model_metrics.csv](../results/affinity_hit_value_model_metrics.csv)
- [results/affinity_hit_value_enrichment.csv](../results/affinity_hit_value_enrichment.csv)
- [results/annotated_affinity_target_summary.csv](../results/annotated_affinity_target_summary.csv)

The best compact rank-derived model was HistGradientBoosting:

- ROC AUC: `0.991`
- PR AUC: `0.663`
- Brier score: `0.007`

These metrics are for a sampled training/evaluation setup, so we corrected scores for the much lower full-candidate-space prior before writing `target_likelihood`.

## 4. How Far Should We Go Before the Probability Drops Below 80%?

After prior correction, no target reaches 80%.

The maximum conservative `target_likelihood` in the full annotated table is about `0.139`. Therefore:

- An 80% cutoff selects `0` rows.
- It recovers `0` known Yamanishi targets.
- It misses all `2,719` known Yamanishi positives present in the mapped affinity tables.

This means an 80% per-target probability threshold is too strict for the current data and assumptions.

More practical threshold tradeoff:

| Threshold | Selected rows | Known Yamanishi recovered | Known Yamanishi missed | Observed known-positive fraction |
|---:|---:|---:|---:|---:|
| 0.10 | 170 | 111 | 2,608 | 0.653 |
| 0.05 | 6,462 | 1,213 | 1,506 | 0.188 |
| 0.01 | 46,112 | 2,529 | 190 | 0.055 |
| 0.005 | 89,855 | 2,676 | 43 | 0.030 |
| 0.001 | 365,475 | 2,718 | 1 | 0.007 |

This table is based on the locally generated `results/annotated_affinity_targets.csv`.

## 5. How Many Targets Does a Ligand Likely Have?

We estimated ligand degree by summing target likelihoods across all scored targets for each ligand:

`estimated ligand degree = sum(target_likelihood for all targets for that ligand)`

Output:

- [results/annotated_affinity_ligand_degree_estimates.csv](../results/annotated_affinity_ligand_degree_estimates.csv)

For (-)-Carnitine / PubChem CID `10917`:

- Rows scored: `7,482`
- Expected supported-target degree: about `3.31`
- Known Yamanishi targets present in the affinity table: `1`

Across all scored ligands:

- Ligands scored: `665`
- Mean expected supported-target degree: about `3.58`
- Median expected supported-target degree: about `2.57`
- Maximum expected supported-target degree: about `15.44`

These are conservative estimates because they use the observed Yamanishi/BindingDB support rate as the deployment prior.

## 6. Can We Use Other Features to Select Likely Targets?

Partly answered.

The current full production scoring uses:

- affinity value,
- rank,
- rank percentile,
- reverse rank percentile,
- within-ligand affinity z-score,
- robust within-ligand affinity z-score,
- local affinity gaps,
- total ranked UniProt count,
- ligand Yamanishi degree,
- target Yamanishi degree,
- whether the target appears in the Yamanishi universe,
- whether the target appears in the BindingDB universe.

These features are listed in the manifest:

- [results/annotated_affinity_targets.manifest.json](../results/annotated_affinity_targets.manifest.json)

The repository also contains ligand chemistry, fingerprint, and protein-feature modeling code. Those wider features can be added, but the compact scoring pass was chosen because it is interpretable, already covers every affinity list, and avoids creating an even larger production table.

Relevant code:

- [scripts/build_no_affinity_dataset.py](../scripts/build_no_affinity_dataset.py)
- [scripts/train_no_affinity_models.py](../scripts/train_no_affinity_models.py)
- [scripts/train_affinity_hit_value_model.py](../scripts/train_affinity_hit_value_model.py)

## 7. How Do We Handle False Negatives Hidden in the Unknowns?

This is a Positive-Unlabeled learning problem.

Definitions in this setting:

- Positive: a ligand-target pair known from Yamanishi or BindingDB.
- Unlabeled: every other ligand-target pair in the affinity files.
- Unlabeled does not mean confirmed negative.

The current scoring workflow handles this conservatively:

- It trains on known supported hits versus sampled unsupported hits.
- It keeps `target_likelihood_sample_prior`, which is the model score under the sampled training distribution.
- It writes `target_likelihood`, which is corrected to the much lower observed support rate in the full candidate universe.

The correction is controlled by `--deployment-prior` in [scripts/score_affinity_targets.py](../scripts/score_affinity_targets.py).

Default prior:

`2862 / 4906996 = 0.000583`

That is the observed rate of Yamanishi-or-BindingDB-supported rows in the full scored affinity universe. If we believe many hidden positives exist, we should rerun with a larger `--deployment-prior` and compare how many candidates become worth testing.

## 8. Can We Estimate How Large the Hidden-False-Negative Problem Is?

Partly.

We cannot know the true hidden-positive count from the current labels alone, but we can estimate sensitivity to assumptions.

The current output gives a conservative lower-bound view:

- Full scored rows: `4,906,996`
- Known Yamanishi positives in those rows: `2,719`
- Known BindingDB positives in those rows: `164`
- Any Yamanishi/BindingDB-supported rows: `2,862`
- Observed supported rate: `0.000583`

If the true hidden-positive rate is 2x, 5x, or 10x larger than observed, the `--deployment-prior` can be increased accordingly and the target likelihoods recomputed. That would give a plausible range for the hidden-positive burden.

Recommended next analysis:

```bash
python3 scripts/score_affinity_targets.py \
  --deployment-prior 0.001166 \
  --output results/annotated_affinity_targets_prior_2x.csv \
  --summary-output results/annotated_affinity_target_summary_prior_2x.csv \
  --calibrate
```

Repeat for 5x and 10x priors.

## 9. How Many Experiments Do We Need?

Partly answered.

The threshold table provides candidate counts and known-positive recovery rates. To estimate experiments, choose:

- a target likelihood threshold,
- a desired number of discoveries,
- and whether to use the conservative prior or a higher hidden-positive prior.

Example using the conservative output:

- At threshold `0.10`, there are `170` candidates and `111` known Yamanishi positives recovered. This is a small, high-confidence experimental set.
- At threshold `0.05`, there are `6,462` candidates and `1,213` known Yamanishi positives recovered.
- At threshold `0.01`, there are `46,112` candidates and `2,529` known Yamanishi positives recovered.

For an experimental campaign, the most practical starting point is probably not a fixed probability like 80%, but a budgeted design:

- test the top `N` candidates per ligand,
- stratify by likelihood bins,
- include known positives and unknowns,
- then use observed hit rates to update the deployment prior.

## 10. BindingDB Has Discrepant Affinities: Can We Say Which Experiment Is Better?

Not answered yet.

The current workflow uses BindingDB only as supporting labels, not as an assay-quality model. To decide which BindingDB experiment is better, we would need assay-level metadata, such as:

- measurement type: Ki, Kd, IC50, EC50,
- target construct and species,
- assay format,
- assay conditions,
- replicate count,
- curation confidence,
- publication/source,
- whether the experiment is direct binding versus functional activity.

That should be a separate assay-quality analysis. It should not be inferred from the docking affinity output alone.

## 11. Add a Column Highlighting Known Yamanishi Targets

Done.

The main annotated output `results/annotated_affinity_targets.csv` includes:

- `known_target_yamanishi`
- `known_target_bindingdb`
- `known_target_any_supported`

For (-)-Carnitine, see:

- [results/annotated_affinity_targets_cid10917.csv](../results/annotated_affinity_targets_cid10917.csv)

## 12. Add a Column With Likelihood of Being a Target

Done.

The main annotated output includes:

- `target_likelihood`: conservative prior-corrected likelihood.
- `target_likelihood_sample_prior`: model score under the sampled training distribution.

Use `target_likelihood` for conservative prioritization. Use `target_likelihood_sample_prior` only for ranking diagnostics or comparison before prior correction.

## Recommended Interpretation

Do not read `target_likelihood` as an absolute biological truth. It is a model-based prioritization score calibrated to the currently observed support rate. It answers:

> Given what known Yamanishi/BindingDB-supported targets look like in these affinity lists, and given the very large number of candidate targets, how target-like is this row?

The most defensible immediate use is:

1. Use the generated `results/annotated_affinity_targets.csv` to rank candidates by `target_likelihood`.
2. Use [results/annotated_affinity_ligand_degree_estimates.csv](../results/annotated_affinity_ligand_degree_estimates.csv) to estimate how many targets each ligand likely has.
3. Do not use raw affinity rank alone.
4. Treat unknown rows as unlabeled, not confirmed false.
5. Choose experimental candidates by budgeted top-N or likelihood-bin sampling, not by an 80% threshold.
