"""Lesson 02 homework: PyTorch GPU 2D binary classification experiment.

Run from the repository root:
    .\.venv\Scripts\python.exe assignment\lesson02_homework.py

Outputs are written to:
    assignment\lesson02_outputs\
"""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import random

OUTPUT_DIR = Path(__file__).parent / "lesson02_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.2


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    # [Data generation] features shape = (240, 2), labels shape = (240,).
    generator = torch.Generator().manual_seed(SEED)
    class_0 = torch.randn(120, 2, generator=generator) * 0.70 + torch.tensor([-1.0, -1.0])
    class_1 = torch.randn(120, 2, generator=generator) * 0.70 + torch.tensor([1.0, 1.0])
    features = torch.cat([class_0, class_1], dim=0).float()
    labels = torch.cat([torch.zeros(120), torch.ones(120)]).float()
    return features, labels


def split_dataset(features: torch.Tensor, labels: torch.Tensor) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    # [Dataset split] train/valid/test = 160/40/40 with a fixed random seed.
    indices = torch.randperm(len(features), generator=torch.Generator().manual_seed(SEED))
    train_idx, valid_idx, test_idx = indices[:160], indices[160:200], indices[200:]
    return (
        TensorDataset(features[train_idx], labels[train_idx]),
        TensorDataset(features[valid_idx], labels[valid_idx]),
        TensorDataset(features[test_idx], labels[test_idx]),
    )


def make_loaders(train_set: TensorDataset, valid_set: TensorDataset, test_set: TensorDataset) -> tuple[DataLoader, DataLoader, DataLoader]:
    # [Tensor/Batch construction] one full training batch has shape:
    # batch_features = (16, 2), batch_labels = (16,).
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
    return train_loader, valid_loader, test_loader


class BinaryClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # [Model] Linear maps input shape (B, 2) to raw logit shape (B, 1).
        self.linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)  # (B, 1) -> (B,), aligned with labels.


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    # [Training loop] every batch does forward, scalar loss, backward and update.
    for batch_features, batch_labels in data_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        logits = model(batch_features)  # logits shape = (B,)
        loss = loss_fn(logits, batch_labels)  # loss shape = ()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_labels)
        total_count += len(batch_labels)

    return total_loss / total_count


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> tuple[float, float]:
    # [Validation/Prediction] evaluate without gradient updates.
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch_features, batch_labels in data_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        logits = model(batch_features)
        loss = loss_fn(logits, batch_labels)
        predictions = (torch.sigmoid(logits) >= 0.5).float()

        total_loss += loss.item() * len(batch_labels)
        total_correct += (predictions == batch_labels).sum().item()
        total_count += len(batch_labels)

    return total_loss / total_count, total_correct / total_count


def save_data_distribution(features: torch.Tensor, labels: torch.Tensor) -> None:
    plt.figure(figsize=(7, 5))
    for label, color in [(0, "tab:blue"), (1, "tab:orange")]:
        mask = labels == label
        plt.scatter(features[mask, 0], features[mask, 1], alpha=0.72, label=f"label {label}", color=color)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Synthetic 2D Binary Classification Data")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig(OUTPUT_DIR / "data_distribution.png", dpi=160, bbox_inches="tight")
    plt.close()


def save_loss_curve(history: list[tuple[int, float, float, float]]) -> None:
    epochs = [row[0] for row in history]
    train_losses = [row[1] for row in history]
    valid_losses = [row[2] for row in history]

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, valid_losses, label="validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=160, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def save_decision_boundary(model: nn.Module, features: torch.Tensor, labels: torch.Tensor, device: torch.device) -> None:
    x_min, x_max = features[:, 0].min().item() - 0.5, features[:, 0].max().item() + 0.5
    y_min, y_max = features[:, 1].min().item() - 0.5, features[:, 1].max().item() + 0.5
    xx, yy = torch.meshgrid(
        torch.linspace(x_min, x_max, 220),
        torch.linspace(y_min, y_max, 220),
        indexing="xy",
    )
    grid_features = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device)
    probabilities = torch.sigmoid(model(grid_features)).reshape(xx.shape).cpu()

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, probabilities, levels=20, cmap="RdBu_r", alpha=0.35)
    plt.contour(xx, yy, probabilities, levels=[0.5], colors="black", linewidths=1.5)
    plt.scatter(features[labels == 0, 0], features[labels == 0, 1], label="label 0", alpha=0.72)
    plt.scatter(features[labels == 1, 0], features[labels == 1, 1], label="label 1", alpha=0.72)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Final Prediction Boundary (p=0.5)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig(OUTPUT_DIR / "decision_boundary.png", dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    seed_everything(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, labels = build_dataset()
    train_set, valid_set, test_set = split_dataset(features, labels)
    train_loader, valid_loader, test_loader = make_loaders(train_set, valid_set, test_set)
    first_batch_features, first_batch_labels = next(iter(train_loader))

    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_runtime={torch.version.cuda}")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"shape | features         | {tuple(features.shape)}")
    print(f"shape | labels           | {tuple(labels.shape)}")
    print(f"shape | train/valid/test | {len(train_set)}/{len(valid_set)}/{len(test_set)}")
    print(f"shape | batch_features   | {tuple(first_batch_features.shape)}")
    print(f"shape | batch_labels     | {tuple(first_batch_labels.shape)}")

    model = BinaryClassifier().to(device)
    # [Loss] BCEWithLogitsLoss receives raw logits, not sigmoid probabilities.
    loss_fn = nn.BCEWithLogitsLoss()
    # [Optimizer] SGD updates model.parameters() with the configured learning rate.
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    sample_logits = model(first_batch_features.to(device))
    sample_loss = loss_fn(sample_logits, first_batch_labels.to(device))
    print(f"shape | logits           | {tuple(sample_logits.shape)}")
    print(f"shape | loss             | {tuple(sample_loss.shape)}")

    save_data_distribution(features, labels)

    history: list[tuple[int, float, float, float]] = []
    best_state = deepcopy(model.state_dict())
    best_valid_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_acc = evaluate(model, valid_loader, loss_fn, device)
        history.append((epoch, train_loss, valid_loss, valid_acc))
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch {epoch:02d} | train_loss={train_loss:.4f} | valid_loss={valid_loss:.4f} | valid_acc={valid_acc:.3f}")

    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
    print(
        f"best_epoch={best_epoch} | best_valid_loss={best_valid_loss:.4f} | "
        f"test_loss={test_loss:.4f} | test_acc={test_acc:.3f}"
    )

    save_loss_curve(history)
    save_decision_boundary(model, features, labels, device)
    print(f"outputs={OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
