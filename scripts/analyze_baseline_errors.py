#!/usr/bin/env python3
"""Analyze held-out false positives and false negatives by ligand/target degree."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baselines import (  # noqa: E402
    feature_sets,
    make_models,
    numeric_matrix,
    predict_scores,
)


OUTCOME_ORDER = ["TN", "FP", "FN", "TP"]


def as_float(row: dict[str, str], column: str) -> float:
    try:
        return float(row.get(column, ""))
    except (TypeError, ValueError):
        return np.nan


def degree_bin(value: float) -> str:
    if np.isnan(value):
        return "missing"
    if value <= 1:
        return "1"
    if value <= 3:
        return "2-3"
    if value <= 10:
        return "4-10"
    if value <= 30:
        return "11-30"
    return ">30"


def outcome(label: int, prediction: int) -> str:
    if label == 0 and prediction == 0:
        return "TN"
    if label == 0 and prediction == 1:
        return "FP"
    if label == 1 and prediction == 0:
        return "FN"
    return "TP"


def summarize_numeric(rows: list[dict[str, object]], column: str) -> list[dict[str, object]]:
    output = []
    for name in OUTCOME_ORDER:
        values = np.asarray(
            [float(row[column]) for row in rows if row["outcome"] == name and not np.isnan(float(row[column]))],
            dtype=float,
        )
        if len(values) == 0:
            output.append(
                {
                    "outcome": name,
                    "rows": 0,
                    "column": column,
                    "mean": "",
                    "median": "",
                    "p25": "",
                    "p75": "",
                    "min": "",
                    "max": "",
                }
            )
            continue
        output.append(
            {
                "outcome": name,
                "rows": len(values),
                "column": column,
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p25": float(np.percentile(values, 25)),
                "p75": float(np.percentile(values, 75)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    return output


def summarize_bins(rows: list[dict[str, object]], degree_column: str) -> list[dict[str, object]]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    outcome_totals = Counter(row["outcome"] for row in rows)
    for row in rows:
        counts[(str(row["outcome"]), str(row[f"{degree_column}_bin"]))] += 1

    output = []
    for name in OUTCOME_ORDER:
        for bin_name in ["1", "2-3", "4-10", "11-30", ">30", "missing"]:
            count = counts[(name, bin_name)]
            total = outcome_totals[name]
            output.append(
                {
                    "degree_column": degree_column,
                    "outcome": name,
                    "degree_bin": bin_name,
                    "rows": count,
                    "fraction_within_outcome": count / total if total else 0.0,
                }
            )
    return output


def summarize_error_rates(rows: list[dict[str, object]], degree_column: str) -> list[dict[str, object]]:
    output = []
    for bin_name in ["1", "2-3", "4-10", "11-30", ">30", "missing"]:
        subset = [row for row in rows if row[f"{degree_column}_bin"] == bin_name]
        negatives = [row for row in subset if row["label"] == 0]
        positives = [row for row in subset if row["label"] == 1]
        false_positives = [row for row in negatives if row["prediction"] == 1]
        false_negatives = [row for row in positives if row["prediction"] == 0]
        output.append(
            {
                "degree_column": degree_column,
                "degree_bin": bin_name,
                "negative_rows": len(negatives),
                "false_positives": len(false_positives),
                "false_positive_rate": len(false_positives) / len(negatives) if negatives else "",
                "positive_rows": len(positives),
                "false_negatives": len(false_negatives),
                "false_negative_rate": len(false_negatives) / len(positives) if positives else "",
            }
        )
    return output


def summarize_error_ligands(rows: list[dict[str, object]], outcome_name: str) -> list[dict[str, object]]:
    counts: Counter[tuple[str, str, str, float]] = Counter()
    for row in rows:
        if row["outcome"] == outcome_name:
            counts[
                (
                    str(row["kegg_drug"]),
                    str(row["pubchem_cid"]),
                    str(row["ligand_title"]),
                    float(row["ligand_yamanishi_degree"]),
                )
            ] += 1
    return [
        {
            "outcome": outcome_name,
            "kegg_drug": kegg_drug,
            "pubchem_cid": pubchem_cid,
            "ligand_title": ligand_title,
            "ligand_yamanishi_degree": ligand_degree,
            "error_rows": count,
        }
        for (kegg_drug, pubchem_cid, ligand_title, ligand_degree), count in counts.most_common()
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipped error plot")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    bins = ["1", "2-3", "4-10", "11-30", ">30"]
    colors = {"TN": "#5b8ff9", "FP": "#f4664a", "FN": "#faad14", "TP": "#5ad8a6"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, degree_column, title in [
        (axes[0], "ligand_yamanishi_degree", "Ligand target count"),
        (axes[1], "target_yamanishi_degree", "Target ligand count"),
    ]:
        subset = [row for row in rows if row["degree_column"] == degree_column]
        width = 0.18
        x = np.arange(len(bins))
        for offset, name in enumerate(OUTCOME_ORDER):
            values = [
                next(
                    row["fraction_within_outcome"]
                    for row in subset
                    if row["outcome"] == name and row["degree_bin"] == bin_name
                )
                for bin_name in bins
            ]
            axis.bar(x + (offset - 1.5) * width, values, width, label=name, color=colors[name])
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(bins)
        axis.set_xlabel("Known Yamanishi degree bin")
        axis.set_ylabel("Fraction within outcome")
        axis.set_ylim(0, 1)
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_rate_plot(path: Path, rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipped error-rate plot")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    bins = ["1", "2-3", "4-10", "11-30", ">30"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, degree_column, title in [
        (axes[0], "ligand_yamanishi_degree", "Ligand target count"),
        (axes[1], "target_yamanishi_degree", "Target ligand count"),
    ]:
        subset = [row for row in rows if row["degree_column"] == degree_column]
        fp_rates = [
            next(
                row["false_positive_rate"]
                for row in subset
                if row["degree_bin"] == bin_name
            )
            for bin_name in bins
        ]
        fn_rates = [
            next(
                row["false_negative_rate"]
                for row in subset
                if row["degree_bin"] == bin_name
            )
            for bin_name in bins
        ]
        x = np.arange(len(bins))
        axis.plot(x, fp_rates, marker="o", linewidth=2, label="FP rate among negatives", color="#f4664a")
        axis.plot(x, fn_rates, marker="o", linewidth=2, label="FN rate among positives", color="#faad14")
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(bins)
        axis.set_xlabel("Known Yamanishi degree bin")
        axis.set_ylabel("Error rate")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/yamanishi_classifier_dataset.csv"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--feature-set", default="pairwise + ligand_all + target_all")
    parser.add_argument("--classifier", default="Random Forest")
    args = parser.parse_args()

    with args.dataset.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    y = np.asarray([int(row["label"]) for row in rows], dtype=int)

    train_idx, test_idx = train_test_split(
        np.arange(len(rows)),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    selected_feature_sets = feature_sets()
    if args.feature_set not in selected_feature_sets:
        raise ValueError(f"Unknown feature set: {args.feature_set}")
    models = make_models(args.seed)
    if args.classifier not in models:
        raise ValueError(f"Unknown classifier: {args.classifier}")

    columns = selected_feature_sets[args.feature_set]
    X = numeric_matrix(rows, columns)
    model = models[args.classifier]
    if args.classifier == "Random Forest":
        model.set_params(model__n_jobs=1)
    model.fit(X[train_idx], y[train_idx])
    y_pred = model.predict(X[test_idx])
    y_score = predict_scores(model, X[test_idx])

    prediction_rows: list[dict[str, object]] = []
    for idx, prediction, score in zip(test_idx, y_pred, y_score):
        source = rows[int(idx)]
        label = int(source["label"])
        ligand_degree = as_float(source, "ligand_yamanishi_degree")
        target_degree = as_float(source, "target_yamanishi_degree")
        prediction_rows.append(
            {
                "row_index": int(idx),
                "outcome": outcome(label, int(prediction)),
                "label": label,
                "prediction": int(prediction),
                "score": float(score),
                "category": source["category"],
                "kegg_drug": source["kegg_drug"],
                "pubchem_cid": source["pubchem_cid"],
                "ligand_title": source["ligand_title"],
                "kegg_target": source["kegg_target"],
                "uniprot_id": source["uniprot_id"],
                "ligand_yamanishi_degree": ligand_degree,
                "ligand_yamanishi_degree_bin": degree_bin(ligand_degree),
                "target_yamanishi_degree": target_degree,
                "target_yamanishi_degree_bin": degree_bin(target_degree),
                "affinity": as_float(source, "affinity"),
                "rank": as_float(source, "rank"),
                "inverted_rank": as_float(source, "inverted_rank"),
                "proportion": as_float(source, "proportion"),
            }
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.results_dir / "baseline_error_predictions.csv"
    numeric_path = args.results_dir / "baseline_error_degree_summary.csv"
    bin_path = args.results_dir / "baseline_error_degree_bins.csv"
    rate_path = args.results_dir / "baseline_error_rates_by_degree_bin.csv"
    ligand_path = args.results_dir / "baseline_error_ligands.csv"
    plot_path = args.results_dir / "plots" / "baseline_error_degree_bins.png"
    rate_plot_path = args.results_dir / "plots" / "baseline_error_rates_by_degree_bin.png"

    numeric_summary = (
        summarize_numeric(prediction_rows, "ligand_yamanishi_degree")
        + summarize_numeric(prediction_rows, "target_yamanishi_degree")
    )
    bin_summary = (
        summarize_bins(prediction_rows, "ligand_yamanishi_degree")
        + summarize_bins(prediction_rows, "target_yamanishi_degree")
    )
    rate_summary = (
        summarize_error_rates(prediction_rows, "ligand_yamanishi_degree")
        + summarize_error_rates(prediction_rows, "target_yamanishi_degree")
    )
    ligand_summary = summarize_error_ligands(prediction_rows, "FP") + summarize_error_ligands(
        prediction_rows, "FN"
    )

    write_csv(prediction_path, prediction_rows)
    write_csv(numeric_path, numeric_summary)
    write_csv(bin_path, bin_summary)
    write_csv(rate_path, rate_summary)
    write_csv(ligand_path, ligand_summary)
    write_plot(plot_path, bin_summary)
    write_rate_plot(rate_plot_path, rate_summary)

    print(f"Wrote predictions: {prediction_path}")
    print(f"Wrote numeric degree summary: {numeric_path}")
    print(f"Wrote degree bins: {bin_path}")
    print(f"Wrote error rates: {rate_path}")
    print(f"Wrote error ligands: {ligand_path}")
    print(f"Wrote plot: {plot_path}")
    print(f"Wrote error-rate plot: {rate_plot_path}")
    print(f"Model: {args.classifier}, feature set: {args.feature_set}")
    print("Outcome counts:")
    for name in OUTCOME_ORDER:
        print(f"  {name}: {sum(row['outcome'] == name for row in prediction_rows)}")


if __name__ == "__main__":
    main()
