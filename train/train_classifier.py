"""
Train Character Classifier CNN
===============================
Trains CharCNN (36-class A-Z / 0-9) on the processed character images
produced by prepare_data.py.

Dataset layout expected (ImageFolder-compatible):
    data/datasets/chars_processed/
        A/   B/   ...   Z/
        0/   1/   ...   9/

Output:
    models/char_classifier.pth   ← best validation-accuracy weights

Usage:
    python train/train_classifier.py
    python train/train_classifier.py --epochs 40 --lr 0.001
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
from sklearn.metrics import classification_report

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.char_classifier import CharCNN


# ── Data ────────────────────────────────────────────────────────────────────

DATA_DIR = "data/datasets/chars_processed"

def get_loaders(batch_size: int, train_split: float):
    h, w = config.CHAR_IMG_SIZE

    train_tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((h, w)),
        transforms.RandomRotation(12),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                scale=(0.9, 1.1), shear=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    val_tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tf)

    # Verify the dataset has the expected classes
    loaded_classes = full_dataset.classes
    print(f"[train_classifier] Found {len(loaded_classes)} classes: {loaded_classes}")
    if len(loaded_classes) != config.NUM_CLASSES:
        print(f"  WARNING: expected {config.NUM_CLASSES} classes, "
              f"got {len(loaded_classes)}. Check your dataset.")

    n_train = int(len(full_dataset) * train_split)
    n_val   = len(full_dataset) - n_train
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    # Give the val split the val transforms
    val_ds.dataset = datasets.ImageFolder(DATA_DIR, transform=val_tf)

    pin = torch.cuda.is_available()   # pin_memory only works on CUDA, not MPS
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=pin)

    print(f"[train_classifier] {n_train} train  |  {n_val} val")
    return train_loader, val_loader, loaded_classes


# ── Training loop ────────────────────────────────────────────────────────────

def train(epochs: int, lr: float, batch_size: int) -> None:
    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: dataset not found at {DATA_DIR}")
        print("Run:  python train/prepare_data.py --task classifier")
        sys.exit(1)

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[train_classifier] Device: {device}")

    train_loader, val_loader, loaded_classes = get_loaders(batch_size, config.TRAIN_SPLIT)

    num_classes = len(loaded_classes)
    model       = CharCNN(num_classes=num_classes).to(device)
    criterion   = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
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
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss   = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)

                preds    = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.CLASSIFIER_WEIGHTS)
            print(f"  → Saved best model  (val_acc={val_acc:.1f}%)")

    print(f"\n[train_classifier] Training complete. "
          f"Best val_acc={best_val_acc:.1f}%")
    print(f"  Weights saved to: {config.CLASSIFIER_WEIGHTS}")

    _eval_report(model, val_loader, loaded_classes, device)
    _plot_history(history)


def _eval_report(model: CharCNN,
                 val_loader: DataLoader,
                 classes: list[str],
                 device: torch.device) -> None:
    """Print a per-class classification report on the validation set."""
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            preds  = model(images).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print("\n── Classification Report ──────────────────────────────────")
    print(classification_report(all_labels, all_preds,
                                 target_names=classes, digits=3))


def _plot_history(history: dict) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train loss")
    ax1.plot(epochs, history["val_loss"],   label="Val loss")
    ax1.set_title("Char Classifier — Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()

    ax2.plot(epochs, history["val_acc"])
    ax2.set_title("Char Classifier — Val Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")

    plt.tight_layout()
    out_path = os.path.join(config.MODEL_DIR, "classifier_training.png")
    plt.savefig(out_path)
    print(f"  Training plot saved to {out_path}")
    plt.close()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Character Classifier CNN")
    parser.add_argument("--epochs",     type=int,   default=config.EPOCHS_CLASSIFIER)
    parser.add_argument("--lr",         type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch-size", type=int,   default=config.BATCH_SIZE)
    args = parser.parse_args()

    train(args.epochs, args.lr, args.batch_size)


if __name__ == "__main__":
    main()
