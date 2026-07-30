#!/usr/bin/env python3
"""Train small PyTorch MLPs on the clean PSB 2026 feature set.

Clean features are pairwise docking/rank features, PubChem numeric ligand
descriptors, and numeric protein target features. Yamanishi is used only for
labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import DatasetBuilder, LIGAND_FEATURE_COLUMNS, TARGET_FEATURE_COLUMNS  # noqa: E402

PAIRWISE_FEATURES = ["affinity", "rank", "inverted_rank", "proportion"]
LIGAND_FEATURES = [f"ligand_{column}" for column in LIGAND_FEATURE_COLUMNS]
TARGET_FEATURES = [f"target_{column}" for column in TARGET_FEATURE_COLUMNS]
CLEAN_FEATURES = PAIRWISE_FEATURES + LIGAND_FEATURES + TARGET_FEATURES


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def category_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["category"]] += 1
    return dict(counts)


def numeric_matrix(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    matrix = []
    for row in rows:
        values = []
        for column in columns:
            try:
                values.append(float(row.get(column, "")))
            except (TypeError, ValueError):
                values.append(np.nan)
        matrix.append(values)
    return np.asarray(matrix, dtype=np.float32)


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for width in hidden:
            layers.extend(
                [
                    nn.Linear(prev, width),
                    nn.BatchNorm1d(width),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def evaluate(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, y_score),
        "PR AUC": average_precision_score(y_true, y_score),
        "False Positive Rate": fp / (fp + tn),
        "False Negative Rate": fn / (fn + tp),
    }


def predict(model: nn.Module, X: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    scores = []
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for (xb,) in loader:
            scores.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(scores)


def train_one(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden: tuple[int, ...],
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    seed: int,
) -> tuple[nn.Module, dict[str, float]]:
    set_seeds(seed)
    model = MLP(X_train.shape[1], hidden, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )

    best_state = None
    best_pr_auc = -1.0
    bad_epochs = 0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        val_score = predict(model, X_val, batch_size)
        val_pr_auc = average_precision_score(y_val, val_score)
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"Best Val PR AUC": best_pr_auc, "Best Epoch": best_epoch}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    set_seeds(args.seed)
    builder = DatasetBuilder(args.data_dir, seed=args.seed)
    positives = builder.build_positive_rows()
    negatives = builder.build_negative_rows(category_counts(positives))
    rows = positives + negatives
    X = numeric_matrix(rows, CLEAN_FEATURES)
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)

    train_val_idx, test_idx = train_test_split(
        np.arange(len(rows)), test_size=0.2, random_state=args.seed, stratify=y
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.2, random_state=args.seed, stratify=y[train_val_idx]
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X[train_idx])).astype(np.float32)
    X_val = scaler.transform(imputer.transform(X[val_idx])).astype(np.float32)
    X_test = scaler.transform(imputer.transform(X[test_idx])).astype(np.float32)
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    configs = [
        {"name": "MLP 64", "hidden": (64,), "dropout": 0.10, "lr": 1e-3, "weight_decay": 1e-4},
        {"name": "MLP 128", "hidden": (128,), "dropout": 0.20, "lr": 8e-4, "weight_decay": 1e-4},
        {"name": "MLP 128-64", "hidden": (128, 64), "dropout": 0.20, "lr": 8e-4, "weight_decay": 3e-4},
        {"name": "MLP 256-128", "hidden": (256, 128), "dropout": 0.30, "lr": 5e-4, "weight_decay": 5e-4},
        {"name": "MLP 128-64-32", "hidden": (128, 64, 32), "dropout": 0.25, "lr": 7e-4, "weight_decay": 3e-4},
    ]

    results = []
    for i, config in enumerate(configs):
        model, val_metrics = train_one(
            X_train,
            y_train,
            X_val,
            y_val,
            hidden=config["hidden"],
            dropout=config["dropout"],
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed + i,
        )
        test_score = predict(model, X_test, args.batch_size)
        results.append(
            {
                "Feature Set": "pairwise_plus_pubchem_plus_target",
                "Classifier": config["name"],
                "Train Rows": len(train_idx),
                "Validation Rows": len(val_idx),
                "Test Rows": len(test_idx),
                "Features": len(CLEAN_FEATURES),
                "Hidden Layers": "-".join(str(x) for x in config["hidden"]),
                "Dropout": config["dropout"],
                "Learning Rate": config["lr"],
                "Weight Decay": config["weight_decay"],
                **val_metrics,
                **evaluate(y_test, test_score),
            }
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.results_dir / "clean_deep_metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    manifest = {
        "seed": args.seed,
        "positive_rows": len(positives),
        "negative_rows": len(negatives),
        "total_rows": len(rows),
        "features": CLEAN_FEATURES,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "torch_version": torch.__version__,
    }
    manifest_path = args.results_dir / "clean_deep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote manifest: {manifest_path}")
    for row in sorted(results, key=lambda item: item["PR AUC"], reverse=True):
        print(
            f"{row['Classifier']:<18} acc={row['Accuracy']:.3f} "
            f"f1={row['F1 Score']:.3f} roc_auc={row['ROC AUC']:.3f} "
            f"pr_auc={row['PR AUC']:.3f} val_pr_auc={row['Best Val PR AUC']:.3f} "
            f"epoch={row['Best Epoch']}"
        )


if __name__ == "__main__":
    main()
