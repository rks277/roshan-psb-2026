#!/usr/bin/env python3
"""Create colorful, presentation-oriented affinity reranking figures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


PALETTE = {
    "ink": "#16213e",
    "muted": "#6b7280",
    "blue": "#2563eb",
    "cyan": "#06b6d4",
    "green": "#10b981",
    "yellow": "#f59e0b",
    "orange": "#f97316",
    "pink": "#ec4899",
    "purple": "#8b5cf6",
    "red": "#ef4444",
}


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


def annotate_model_percentiles(rows: list[dict[str, str]]) -> None:
    score_order = np.argsort([-float(row["hit_value_score"]) for row in rows])
    for rank, idx in enumerate(score_order, start=1):
        rows[idx]["model_rank_global"] = str(rank)
        rows[idx]["model_percentile_global"] = str(rank / len(rows))

    by_ligand: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_ligand[row["pubchem_cid"]].append(row)
    for ligand_rows in by_ligand.values():
        ligand_rows.sort(key=lambda row: float(row["hit_value_score"]), reverse=True)
        total = len(ligand_rows)
        for rank, row in enumerate(ligand_rows, start=1):
            row["model_rank_within_ligand"] = str(rank)
            row["model_percentile_within_ligand"] = str(rank / total)


def supported_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["label_supported"] == "1"]


def priority_band(percentile: float) -> tuple[str, str]:
    if percentile <= 0.01:
        return "elite", PALETTE["pink"]
    if percentile <= 0.05:
        return "high", PALETTE["purple"]
    if percentile <= 0.10:
        return "medium", PALETTE["blue"]
    return "low", PALETTE["cyan"]


def plot_rescue_map(rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    supported = supported_rows(rows)
    raw = np.asarray([float(row["rank_percentile"]) * 100 for row in supported])
    model = np.asarray([float(row["model_percentile_global"]) * 100 for row in supported])
    bins = np.asarray([0, 1, 2, 5, 10, 20, 40, 60, 80, 100], dtype=float)
    heat, _, _ = np.histogram2d(raw, model, bins=[bins, bins])
    heat = heat.T
    heat_masked = np.ma.masked_where(heat == 0, heat)
    cmap = LinearSegmentedColormap.from_list(
        "rerank_glow",
        ["#fff7ed", PALETTE["yellow"], PALETTE["orange"], PALETTE["pink"], PALETTE["purple"]],
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#fffaf4")
    mesh = ax.pcolormesh(bins, bins, heat_masked, cmap=cmap, shading="flat")
    ax.plot([0, 100], [0, 100], color=PALETTE["ink"], linestyle="--", linewidth=1.4, alpha=0.5)
    ax.fill_between([0, 100], [0, 100], [0, 0], color=PALETTE["green"], alpha=0.07)
    ax.text(12, 7, "rescued upward", color=PALETTE["green"], fontsize=13, weight="bold")
    ax.text(30, 55, "model deprioritized", color=PALETTE["muted"], fontsize=12)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(100, 0)
    ax.set_xticks(bins)
    ax.set_yticks(bins)
    ax.set_xticklabels([f"{int(value)}%" for value in bins])
    ax.set_yticklabels([f"{int(value)}%" for value in bins])
    ax.set_xlabel("Raw affinity rank percentile within ligand")
    ax.set_ylabel("Model priority percentile across held-out hits")
    ax.set_title("Reranking Rescue Map", fontsize=22, color=PALETTE["ink"], pad=14)
    colorbar = fig.colorbar(mesh, ax=ax, shrink=0.85)
    colorbar.set_label("Known-supported hits")
    ax.grid(True, color="white", linewidth=1.2, alpha=0.9)
    fig.tight_layout(rect=[0, 0, 0.96, 1])
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_priority_funnel(rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = np.asarray([int(row["label_supported"]) for row in rows])
    scores = np.asarray([float(row["hit_value_score"]) for row in rows])
    raw_percentiles = np.asarray([float(row["rank_percentile"]) for row in rows])
    score_order = np.argsort(-scores)
    raw_order = np.argsort(raw_percentiles)
    fractions = np.asarray([0.001, 0.005, 0.01, 0.02, 0.05, 0.10])
    names = ["0.1%", "0.5%", "1%", "2%", "5%", "10%"]
    colors = [PALETTE["pink"], PALETTE["purple"], PALETTE["blue"], PALETTE["cyan"], PALETTE["green"], PALETTE["yellow"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    fig.patch.set_facecolor("#fbfbff")
    for ax, order, title in [
        (axes[0], raw_order, "Raw affinity rank"),
        (axes[1], score_order, "Model reranking"),
    ]:
        ax.set_facecolor("#ffffff")
        for idx, (fraction, name, color) in enumerate(zip(fractions, names, colors)):
            count = max(1, int(round(len(labels) * fraction)))
            selected = labels[order[:count]]
            supported = int(np.sum(selected))
            unsupported = count - supported
            ax.barh(idx, unsupported, color="#e5e7eb", height=0.68)
            ax.barh(idx, supported, color=color, height=0.68)
            rate = supported / count
            ax.text(count * 1.03, idx, f"{supported}/{count}  ({rate:.0%})", va="center", fontsize=11)
        ax.set_title(title, fontsize=18, color=PALETTE["ink"])
        ax.set_xlabel("Candidates selected")
        ax.grid(True, axis="x", alpha=0.2)
    axes[0].set_yticks(np.arange(len(names)))
    axes[0].set_yticklabels([f"top {name}" for name in names])
    axes[0].invert_yaxis()
    fig.suptitle("How Many Known Binders Are In The Shortlist?", fontsize=24, color=PALETTE["ink"], y=0.98)
    fig.text(0.5, 0.025, "colored = known-supported, gray = unsupported / untested", ha="center", color=PALETTE["muted"])
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_reranking_ribbons(rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    candidates = [
        row
        for row in supported_rows(rows)
        if float(row["rank_percentile"]) >= 0.05 and float(row["model_percentile_within_ligand"]) <= 0.05
    ]
    candidates.sort(
        key=lambda row: (
            float(row["model_percentile_within_ligand"]) - float(row["rank_percentile"]),
            float(row["model_percentile_within_ligand"]),
        )
    )
    selected = candidates[:70]

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-1, len(selected))
    ax.axis("off")
    ax.text(
        0.5,
        len(selected) + 1.6,
        "Known binders rescued from noisy rankings",
        color="#e0f2fe",
        fontsize=22,
        weight="bold",
        ha="center",
    )
    ax.text(0, len(selected) + 0.1, "raw affinity list", color="#fed7aa", fontsize=15, weight="bold", ha="center")
    ax.text(1, len(selected) + 0.1, "model shortlist", color="#bfdbfe", fontsize=15, weight="bold", ha="center")

    for y, row in enumerate(selected):
        raw_y = len(selected) - y - 1
        model_y = len(selected) - (float(row["model_percentile_within_ligand"]) * len(selected) * 4) - 1
        model_y = max(0, min(len(selected) - 1, model_y))
        raw_pct = float(row["rank_percentile"])
        _, color = priority_band(float(row["model_percentile_within_ligand"]))
        verts = [
            (0, raw_y),
            (0.28, raw_y),
            (0.72, model_y),
            (1, model_y),
        ]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        alpha = 0.25 + min(0.55, raw_pct)
        patch = PathPatch(
            MplPath(verts, codes),
            facecolor="none",
            edgecolor=color,
            lw=1.2 + min(4.0, raw_pct * 5),
            alpha=alpha,
            capstyle="round",
        )
        ax.add_patch(patch)
        ax.scatter([0], [raw_y], s=22, color="#fb923c", alpha=0.9)
        ax.scatter([1], [model_y], s=24, color=color, alpha=0.95)

    ax.plot([0, 0], [0, len(selected) - 1], color="#fed7aa", linewidth=4, alpha=0.7)
    ax.plot([1, 1], [0, len(selected) - 1], color="#bfdbfe", linewidth=4, alpha=0.7)
    ax.text(0, -2.2, "buried known targets", color="#fed7aa", ha="center", fontsize=13)
    ax.text(1, -2.2, "promoted candidates", color="#bfdbfe", ha="center", fontsize=13)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
    fig.savefig(output, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scored",
        type=Path,
        default=Path("results/affinity_hit_value_maccs_biology_ligand_holdout_scored_sample.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/affinity_hit_value_maccs_biology_ligand_holdout_model_manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    args = parser.parse_args()

    rows = load_rows(args.scored)
    ligands = heldout_ligands(rows, args.manifest)
    if ligands is not None:
        rows = [row for row in rows if row["pubchem_cid"] in ligands]
    annotate_model_percentiles(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_rescue_map(rows, args.output_dir / "affinity_reranking_rescue_map.png")
    plot_priority_funnel(rows, args.output_dir / "affinity_reranking_priority_funnel.png")
    plot_reranking_ribbons(rows, args.output_dir / "affinity_reranking_rescue_ribbons.png")
    print(f"Rows plotted: {len(rows)}")
    print(f"Supported rows plotted: {len(supported_rows(rows))}")
    print(f"Wrote story plots to {args.output_dir}")


if __name__ == "__main__":
    main()
