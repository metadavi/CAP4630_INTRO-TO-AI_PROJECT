"""
Train Plate Detector CNN
========================
Trains the binary PlateCNN on the patch dataset produced by prepare_data.py.

Dataset layout expected (ImageFolder-compatible):
    data/datasets/detector/
        plate/       ← positive samples  (label 1)
        no_plate/    ← negative samples  (label 0)

Output:
    models/plate_detector.pth   ← best validation-loss weights

Usage:
    python train/train_detector.py
    python train/train_detector.py --epochs 30 --lr 0.0005
"""

import os
import sys
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.detector import PlateCNN


# ── Data ────────────────────────────────────────────────────────────────────

DATA_DIR = "data/datasets/detector"

def get_loaders(batch_size: int, train_split: float):
    h, w = config.DETECTOR_IMG_SIZE

    train_tf = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tf)

    n_train = int(len(full_dataset) * train_split)
    n_val   = len(full_dataset) - n_train
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    # Apply validation transforms to val subset
    val_ds.dataset = datasets.ImageFolder(DATA_DIR, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"[train_detector] {n_train} train  |  {n_val} val  |  "
          f"classes: {full_dataset.classes}")
    return train_loader, val_loader


# ── Training loop ───────────────────────────────────────────────────────────

def train(epochs: int, lr: float, batch_size: int) -> None:
    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: dataset not found at {DATA_DIR}")
        print("Run:  python train/prepare_data.py --task detector")
        sys.exit(1)

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_detector] Device: {device}")

    train_loader, val_loader = get_loaders(batch_size, config.TRAIN_SPLIT)

    model     = PlateCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            # PlateCNN outputs sigmoid → BCELoss needs float labels in [0,1]
            targets = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss  = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images  = images.to(device)
                targets = labels.float().unsqueeze(1).to(device)
                preds   = model(images)
                loss    = criterion(preds, targets)
                val_loss += loss.item() * images.size(0)

                predicted = (preds >= 0.5).long().squeeze(1)
                correct  += (predicted == labels.to(device)).sum().item()
                total    += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc   = correct / total * 100

        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:>3}/{epochs}]  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.1f}%  "
              f"({elapsed:.1f}s)")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.DETECTOR_WEIGHTS)
            print(f"  → Saved best model  (val_loss={val_loss:.4f})")

    print(f"\n[train_detector] Training complete. "
          f"Best val_loss={best_val_loss:.4f}")
    print(f"  Weights saved to: {config.DETECTOR_WEIGHTS}")

    _plot_history(history)


def _plot_history(history: dict) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train loss")
    ax1.plot(epochs, history["val_loss"],   label="Val loss")
    ax1.set_title("Plate Detector — Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss")
    ax1.legend()

    ax2.plot(epochs, history["val_acc"])
    ax2.set_title("Plate Detector — Val Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")

    plt.tight_layout()
    out_path = os.path.join(config.MODEL_DIR, "detector_training.png")
    plt.savefig(out_path)
    print(f"  Training plot saved to {out_path}")
    plt.close()


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Plate Detector CNN")
    parser.add_argument("--epochs",     type=int,   default=config.EPOCHS_DETECTOR)
    parser.add_argument("--lr",         type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch-size", type=int,   default=config.BATCH_SIZE)
    args = parser.parse_args()

    train(args.epochs, args.lr, args.batch_size)


if __name__ == "__main__":
    main()
