#!/usr/bin/env python3
"""Train a formal positive-unlabeled affinity-hit reranker.

Known Yamanishi/BindingDB pairs are positives. All other affinity hits are
treated as unlabeled, not confirmed negatives. This script trains a small neural
network with the non-negative PU risk estimator and sweeps plausible positive
class priors.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_affinity_hit_value_model import (  # noqa: E402
    FEATURE_SETS,
    enrichment_rows,
    load_feature_maps,
    numeric_matrix,
)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def ligand_split_indices(rows: list[dict[str, str]], test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ligand_ids = np.asarray([row["pubchem_cid"] for row in rows])
    unique_ligands = np.asarray(sorted(set(ligand_ids)))
    train_ligands, test_ligands = train_test_split(unique_ligands, test_size=test_size, random_state=seed)
    train_ligands = set(train_ligands)
    test_ligands = set(test_ligands)
    train_idx = np.asarray([idx for idx, ligand_id in enumerate(ligand_ids) if ligand_id in train_ligands])
    test_idx = np.asarray([idx for idx, ligand_id in enumerate(ligand_ids) if ligand_id in test_ligands])
    return train_idx, test_idx


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def predict(model: nn.Module, X: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(X[start : start + batch_size], dtype=torch.float32)
            out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)


def top_fraction_rate(y_true: np.ndarray, y_score: np.ndarray, fraction: float) -> float:
    order = np.argsort(-y_score)
    count = max(1, int(round(len(y_true) * fraction)))
    return float(np.mean(y_true[order[:count]]))


def metric_row(
    method: str,
    split: str,
    class_prior: float,
    y_true: np.ndarray,
    y_score: np.ndarray,
    train_rows: int,
    eval_rows: int,
    features: int,
    best_epoch: int,
    best_val_pr_auc: float | None,
) -> dict[str, object]:
    return {
        "Method": method,
        "Split": split,
        "Class Prior": class_prior,
        "Train Rows": train_rows,
        "Eval Rows": eval_rows,
        "Features": features,
        "Positive Eval Fraction": float(np.mean(y_true)),
        "ROC AUC": roc_auc_score(y_true, y_score),
        "PR AUC": average_precision_score(y_true, y_score),
        "Brier Score": brier_score_loss(y_true, y_score),
        "Top 0.1% Support": top_fraction_rate(y_true, y_score, 0.001),
        "Top 0.5% Support": top_fraction_rate(y_true, y_score, 0.005),
        "Top 1.0% Support": top_fraction_rate(y_true, y_score, 0.01),
        "Top 2.0% Support": top_fraction_rate(y_true, y_score, 0.02),
        "Best Epoch": best_epoch,
        "Best Validation PR AUC": "" if best_val_pr_auc is None else best_val_pr_auc,
    }


def nnpu_loss(
    logits_p: torch.Tensor,
    logits_u: torch.Tensor,
    class_prior: float,
    beta: float,
) -> tuple[torch.Tensor, float, float]:
    positive_targets = torch.ones_like(logits_p)
    negative_targets_p = torch.zeros_like(logits_p)
    negative_targets_u = torch.zeros_like(logits_u)
    loss_p_pos = nn.functional.binary_cross_entropy_with_logits(logits_p, positive_targets)
    loss_p_neg = nn.functional.binary_cross_entropy_with_logits(logits_p, negative_targets_p)
    loss_u_neg = nn.functional.binary_cross_entropy_with_logits(logits_u, negative_targets_u)
    positive_risk = class_prior * loss_p_pos
    negative_risk = loss_u_neg - class_prior * loss_p_neg
    risk = positive_risk + torch.clamp(negative_risk, min=beta)
    return risk, float(positive_risk.detach()), float(negative_risk.detach())


def train_nnpu(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    class_prior: float,
    hidden: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    seed: int,
) -> tuple[MLP, dict[str, float]]:
    set_seeds(seed)
    model = MLP(X.shape[1], hidden, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    positive_idx = train_idx[y[train_idx] == 1]
    unlabeled_idx = train_idx[y[train_idx] == 0]
    rng = np.random.default_rng(seed)
    steps_per_epoch = max(20, int(np.ceil(len(unlabeled_idx) / batch_size)))
    best_state = None
    best_val_pr_auc = -1.0
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for _ in range(steps_per_epoch):
            p_idx = rng.choice(positive_idx, size=min(batch_size, len(positive_idx)), replace=len(positive_idx) < batch_size)
            u_idx = rng.choice(unlabeled_idx, size=batch_size, replace=len(unlabeled_idx) < batch_size)
            xb_p = torch.tensor(X[p_idx], dtype=torch.float32)
            xb_u = torch.tensor(X[u_idx], dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = nnpu_loss(model(xb_p), model(xb_u), class_prior, beta=0.0)
            loss.backward()
            optimizer.step()

        val_score = predict(model, X[val_idx], batch_size=batch_size * 4)
        val_pr_auc = average_precision_score(y[val_idx], val_score)
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val_pr_auc": best_val_pr_auc}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/affinity_hit_value_dataset_compact.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="clean_rank_plus_maccs_morgan_target")
    parser.add_argument("--output-prefix", default="affinity_hit_value_pu_clean")
    parser.add_argument("--priors", nargs="+", type=float, default=[0.02, 0.05, 0.10])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-test-size", type=float, default=0.2)
    parser.add_argument("--inner-validation-size", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    set_seeds(args.seed)
    rows = load_rows(args.dataset)
    y = np.asarray([int(row["label_supported"]) for row in rows], dtype=int)
    outer_train_idx, test_idx = ligand_split_indices(rows, args.outer_test_size, args.seed)
    outer_train_rows = [rows[idx] for idx in outer_train_idx]
    inner_train_rel, val_rel = ligand_split_indices(outer_train_rows, args.inner_validation_size, args.seed + 1)
    train_idx = outer_train_idx[inner_train_rel]
    val_idx = outer_train_idx[val_rel]

    feature_maps = load_feature_maps(args.data_dir, args.seed)
    columns = FEATURE_SETS[args.feature_set]
    X_raw = numeric_matrix(rows, columns, feature_maps)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = scaler.fit_transform(imputer.fit_transform(X_raw[train_idx])).astype(np.float32)
    X_all = scaler.transform(imputer.transform(X_raw)).astype(np.float32)
    del X_raw

    # Re-index the transformed training matrix back into full-row coordinates for simple sampling.
    X_train_only = X
    X = np.empty((len(rows), X_train_only.shape[1]), dtype=np.float32)
    X[train_idx] = X_train_only
    eval_idx = np.setdiff1d(np.arange(len(rows)), train_idx, assume_unique=False)
    X[eval_idx] = X_all[eval_idx]
    del X_all, X_train_only

    metrics = []
    best = None
    best_pr_auc = -1.0
    observed_train_positive_fraction = float(np.mean(y[train_idx]))
    for prior in args.priors:
        class_prior = max(prior, observed_train_positive_fraction + 1e-4)
        model, info = train_nnpu(
            X,
            y,
            train_idx,
            val_idx,
            class_prior=class_prior,
            hidden=args.hidden,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed + int(class_prior * 1000),
        )
        val_score = predict(model, X[val_idx], args.batch_size * 4)
        row = metric_row(
            "nnPU MLP",
            "inner_validation_ligands",
            class_prior,
            y[val_idx],
            val_score,
            len(train_idx),
            len(val_idx),
            len(columns),
            int(info["best_epoch"]),
            float(info["best_val_pr_auc"]),
        )
        metrics.append(row)
        if row["PR AUC"] > best_pr_auc:
            best_pr_auc = float(row["PR AUC"])
            best = (class_prior, model, info)
        print(f"prior={class_prior:.3f}: val_pr_auc={row['PR AUC']:.3f} top1={row['Top 1.0% Support']:.3f}")

    assert best is not None
    best_prior, best_model, best_info = best
    test_score = predict(best_model, X[test_idx], args.batch_size * 4)
    test_row = metric_row(
        "nnPU MLP",
        "outer_test_ligands",
        best_prior,
        y[test_idx],
        test_score,
        len(train_idx),
        len(test_idx),
        len(columns),
        int(best_info["best_epoch"]),
        best_pr_auc,
    )
    metrics.append(test_row)

    metrics_path = args.results_dir / f"{args.output_prefix}_metrics.csv"
    write_csv(metrics_path, metrics)

    enrichment = [{"Method": "nnPU MLP", "Class Prior": best_prior, **row} for row in enrichment_rows(y[test_idx], test_score)]
    enrichment_path = args.results_dir / f"{args.output_prefix}_enrichment.csv"
    write_csv(enrichment_path, enrichment)

    manifest_path = args.results_dir / f"{args.output_prefix}_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "feature_set": args.feature_set,
                "feature_count": len(columns),
                "method": "non-negative PU risk MLP",
                "priors_tested": args.priors,
                "selected_class_prior": best_prior,
                "observed_inner_train_labeled_positive_fraction": observed_train_positive_fraction,
                "seed": args.seed,
                "outer_test_size": args.outer_test_size,
                "inner_validation_size": args.inner_validation_size,
                "epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "hidden": args.hidden,
                "dropout": args.dropout,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "outer_test_pr_auc": test_row["PR AUC"],
                "outer_test_roc_auc": test_row["ROC AUC"],
                "outer_test_top_1_percent_support": test_row["Top 1.0% Support"],
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Selected prior {best_prior:.3f}")
    print(f"Outer test PR AUC: {test_row['PR AUC']:.3f}")
    print(f"Outer test top 1% support: {test_row['Top 1.0% Support']:.3f}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {enrichment_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
