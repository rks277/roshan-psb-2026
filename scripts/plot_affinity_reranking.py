#!/usr/bin/env python3
"""Plot how affinity-hit value scoring reranks docking candidates."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def heldout_ligands(rows: list[dict[str, str]], manifest_path: Path) -> set[str] | None:
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("split_mode") != "ligand":
        return None
    ligands = np.asarray(sorted({row["pubchem_cid"] for row in rows}))
    _, test_ligands = train_test_split(
        ligands,
        test_size=float(manifest.get("test_size", 0.2)),
        random_state=int(manifest.get("seed", 42)),
    )
    return set(test_ligands)


def filter_rows(rows: list[dict[str, str]], ligands: set[str] | None) -> list[dict[str, str]]:
    if ligands is None:
        return rows
    return [row for row in rows if row["pubchem_cid"] in ligands]


def arrays(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray([int(row["label_supported"]) for row in rows], dtype=int)
    scores = np.asarray([float(row["hit_value_score"]) for row in rows], dtype=float)
    raw_rank_percentile = np.asarray([float(row["rank_percentile"]) for row in rows], dtype=float)
    affinity = np.asarray([float(row["affinity"]) for row in rows], dtype=float)
    return labels, scores, raw_rank_percentile, affinity


def top_fraction_rates(labels: np.ndarray, order: np.ndarray, fractions: list[float]) -> list[float]:
    rates = []
    for fraction in fractions:
        count = max(1, int(round(len(labels) * fraction)))
        rates.append(float(np.mean(labels[order[:count]])))
    return rates


def plot_enrichment_curve(rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib.pyplot as plt

    labels, scores, raw_rank_percentile, _ = arrays(rows)
    fractions = np.logspace(np.log10(0.001), np.log10(0.2), 60)
    model_rates = top_fraction_rates(labels, np.argsort(-scores), fractions.tolist())
    raw_rates = top_fraction_rates(labels, np.argsort(raw_rank_percentile), fractions.tolist())
    baseline = float(np.mean(labels))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fractions * 100, model_rates, linewidth=2.5, color="#2864a6", label="Model reranking")
    ax.plot(fractions * 100, raw_rates, linewidth=2.5, color="#d97532", label="Raw affinity rank")
    ax.axhline(baseline, color="#555555", linestyle="--", linewidth=1.2, label=f"Baseline ({baseline:.1%})")
    ax.set_xscale("log")
    ax.set_xlabel("Top scored/ranked candidates selected (%)")
    ax.set_ylabel("Known-support rate")
    ax.set_title("Affinity Candidate Enrichment")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_top_fraction_bars(rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib.pyplot as plt

    labels, scores, raw_rank_percentile, _ = arrays(rows)
    fractions = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    model_rates = top_fraction_rates(labels, np.argsort(-scores), fractions)
    raw_rates = top_fraction_rates(labels, np.argsort(raw_rank_percentile), fractions)
    baseline = float(np.mean(labels))
    x = np.arange(len(fractions))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, raw_rates, width, color="#d97532", label="Raw affinity rank")
    ax.bar(x + width / 2, model_rates, width, color="#2864a6", label="Model reranking")
    ax.axhline(baseline, color="#555555", linestyle="--", linewidth=1.2, label=f"Baseline ({baseline:.1%})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{fraction:g}%" for fraction in np.asarray(fractions) * 100])
    ax.set_xlabel("Top candidates selected")
    ax.set_ylabel("Known-support rate")
    ax.set_title("Known-Support Rate By Operating Point")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_supported_scatter(rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib.pyplot as plt

    labels, scores, raw_rank_percentile, _ = arrays(rows)
    score_order = np.argsort(-scores)
    model_percentile = np.empty(len(rows), dtype=float)
    model_percentile[score_order] = (np.arange(len(rows)) + 1) / len(rows)
    supported = labels == 1

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        raw_rank_percentile[supported] * 100,
        model_percentile[supported] * 100,
        s=18,
        alpha=0.55,
        color="#2864a6",
        edgecolors="none",
    )
    ax.plot([0, 100], [0, 100], color="#777777", linestyle="--", linewidth=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(100, 0)
    ax.set_xlabel("Raw affinity rank percentile within ligand (lower is better)")
    ax.set_ylabel("Model score percentile globally (higher priority at top)")
    ax.set_title("Known-Supported Hits After Reranking")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_ligand_examples(rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib.pyplot as plt

    by_ligand: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_ligand[row["pubchem_cid"]].append(row)
    supported_counts = Counter(
        ligand
        for ligand, ligand_rows in by_ligand.items()
        for row in ligand_rows
        if row["label_supported"] == "1"
    )
    ligands = [ligand for ligand, count in supported_counts.most_common(6) if count >= 2]
    if not ligands:
        return

    fig, axes = plt.subplots(len(ligands), 1, figsize=(8, 1.6 * len(ligands)), sharex=True)
    if len(ligands) == 1:
        axes = [axes]
    for ax, ligand in zip(axes, ligands):
        ligand_rows = by_ligand[ligand]
        ranked_by_model = sorted(ligand_rows, key=lambda row: float(row["hit_value_score"]), reverse=True)
        model_rank = {f"{row['uniprot_id']}|{row['rank_1_based']}": idx + 1 for idx, row in enumerate(ranked_by_model)}
        total = len(ligand_rows)
        supported_rows = [row for row in ligand_rows if row["label_supported"] == "1"]
        supported_rows = sorted(supported_rows, key=lambda row: float(row["rank_percentile"]))[:12]
        for row in supported_rows:
            raw_pct = float(row["rank_percentile"]) * 100
            model_pct = model_rank[f"{row['uniprot_id']}|{row['rank_1_based']}"] / total * 100
            ax.plot([raw_pct, model_pct], [0, 0], color="#999999", linewidth=1.4, alpha=0.7)
            ax.scatter(raw_pct, 0, color="#d97532", s=28, zorder=3)
            ax.scatter(model_pct, 0, color="#2864a6", s=28, zorder=3)
        ax.set_yticks([])
        ax.set_ylabel(ligand, rotation=0, ha="right", va="center", labelpad=44)
        ax.grid(True, axis="x", alpha=0.2)
    axes[-1].set_xlabel("Percentile within that ligand's candidate list (lower is better)")
    fig.suptitle("Examples: Supported Targets Before And After Reranking", y=0.99)
    fig.text(0.72, 0.035, "orange = raw rank, blue = model rank", ha="center", color="#444444")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scored",
        type=Path,
        default=Path("results/affinity_hit_value_clean_full_best_ligand_holdout_scored_sample.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/affinity_hit_value_clean_full_best_ligand_holdout_model_manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    parser.add_argument("--all-rows", action="store_true", help="Plot all scored rows instead of the held-out ligands.")
    args = parser.parse_args()

    rows = load_rows(args.scored)
    ligands = None if args.all_rows else heldout_ligands(rows, args.manifest)
    rows = filter_rows(rows, ligands)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_enrichment_curve(rows, args.output_dir / "affinity_reranking_enrichment_curve.png")
    plot_top_fraction_bars(rows, args.output_dir / "affinity_reranking_top_fraction_bars.png")
    plot_supported_scatter(rows, args.output_dir / "affinity_reranking_supported_scatter.png")
    plot_ligand_examples(rows, args.output_dir / "affinity_reranking_ligand_examples.png")
    print(f"Rows plotted: {len(rows)}")
    print(f"Supported rows plotted: {sum(1 for row in rows if row['label_supported'] == '1')}")
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
