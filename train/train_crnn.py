"""
Train CRNN + CTC Plate Reader
==============================
Trains PlateCRNN on the combined synthetic + real plate dataset produced by:
    python train/generate_florida_plates.py
    python train/autolabel_real_plates.py

Dataset layout expected:
    data/datasets/crnn_plates/
        images/        ← plate crop PNGs
        labels.csv     ← columns: stem, text

Output:
    models/plate_crnn.pth   ← best validation weights

Usage:
    python train/train_crnn.py
    python train/train_crnn.py --epochs 50 --lr 0.001
"""

import os
import sys
import csv
import argparse
import time
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.plate_reader import PlateCRNN, encode, greedy_decode, IMG_H, IMG_W
from src.plate_reader import CHARS, BLANK, N_CLASS

DATA_DIR   = "data/datasets/crnn_plates"
OUT_WEIGHTS = os.path.join(config.MODEL_DIR, "plate_crnn.pth")


# ── Dataset ────────────────────────────────────────────────────────────────

class PlateDataset(Dataset):
    """Loads plate crop images + text labels from labels.csv."""

    def __init__(self, data_dir: str, augment: bool = False):
        self.img_dir  = os.path.join(data_dir, "images")
        self.augment  = augment
        self.samples: list[tuple[str, str]] = []   # (img_path, text)

        csv_path = os.path.join(data_dir, "labels.csv")
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                text = row["text"].upper().strip()
                if not text:
                    continue
                # Verify all chars are in our charset
                if not all(c in CHARS for c in text):
                    continue
                if not (config.MIN_PLATE_CHARS <= len(text) <= config.MAX_PLATE_CHARS):
                    continue
                img_path = os.path.join(self.img_dir, row["stem"] + ".png")
                if os.path.exists(img_path):
                    self.samples.append((img_path, text))

        print(f"[PlateDataset] {len(self.samples)} samples loaded from {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, text = self.samples[idx]

        bgr  = cv2.imread(img_path)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # CLAHE contrast normalisation
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray  = clahe.apply(gray)

        if self.augment:
            gray = self._augment(gray)

        gray = cv2.resize(gray, (IMG_W, IMG_H), interpolation=cv2.INTER_CUBIC)
        arr  = gray.astype(np.float32) / 255.0
        arr  = (arr - 0.5) / 0.5

        img_tensor  = torch.from_numpy(arr).unsqueeze(0)   # (1,32,128)
        label       = torch.tensor(encode(text), dtype=torch.long)
        label_len   = torch.tensor(len(text),    dtype=torch.long)

        return img_tensor, label, label_len

    @staticmethod
    def _augment(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]

        # Random rotation ±6°
        angle = random.uniform(-6, 6)
        M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img   = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # Random blur
        if random.random() < 0.3:
            k   = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)

        # Brightness / contrast jitter
        alpha = random.uniform(0.8, 1.2)
        beta  = random.randint(-20, 20)
        img   = np.clip(img.astype(np.float32) * alpha + beta,
                        0, 255).astype(np.uint8)

        # Gaussian noise
        noise = np.random.normal(0, random.uniform(1, 8), img.shape).astype(np.int16)
        img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img


def collate_fn(batch):
    """
    Custom collate: pad labels to the same length for CTCLoss.
    Returns:
        images      (B, 1, H, W)
        labels      (sum of all label lengths,)  — concatenated flat
        input_lens  (B,)                         — all equal to T=32
        label_lens  (B,)
    """
    images, labels, label_lens = zip(*batch)
    images     = torch.stack(images, 0)
    label_lens = torch.stack(label_lens, 0)
    labels_cat = torch.cat(labels, 0)         # flat concatenation for CTC
    T          = images.shape[-1] // 4        # time steps = W / 4  (two 2×2 pools)
    input_lens = torch.full((len(images),), T, dtype=torch.long)
    return images, labels_cat, input_lens, label_lens


# ── Accuracy helpers ───────────────────────────────────────────────────────

def char_accuracy(preds: list[str], targets: list[str]) -> float:
    correct = total = 0
    for p, t in zip(preds, targets):
        for pc, tc in zip(p, t):
            correct += (pc == tc)
        total += len(t)
    return correct / total if total else 0.0


def exact_accuracy(preds: list[str], targets: list[str]) -> float:
    return sum(p == t for p, t in zip(preds, targets)) / len(targets)


# ── Training loop ──────────────────────────────────────────────────────────

def train(epochs: int, lr: float, batch_size: int) -> None:
    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: dataset not found at {DATA_DIR}")
        print("Run:  python train/generate_florida_plates.py")
        print("      python train/autolabel_real_plates.py")
        sys.exit(1)

    os.makedirs(config.MODEL_DIR, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[train_crnn] Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────
    full_ds = PlateDataset(DATA_DIR, augment=False)
    n_train = int(len(full_ds) * config.TRAIN_SPLIT)
    n_val   = len(full_ds) - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_ds.dataset = PlateDataset(DATA_DIR, augment=True)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=pin,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=pin,
                              collate_fn=collate_fn)
    print(f"[train_crnn] {n_train} train  |  {n_val} val")

    # ── Model ────────────────────────────────────────────────────────────
    model     = PlateCRNN(n_class=N_CLASS).to(device)
    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_char_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "char_acc": [], "exact_acc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for images, labels, input_lens, label_lens in train_loader:
            images     = images.to(device)

            optimizer.zero_grad()
            log_probs = model(images)                      # (T,B,C)
            # CTCLoss not implemented on MPS — compute on CPU, grads flow back
            loss = criterion(log_probs.cpu(),
                             labels, input_lens, label_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= n_train

        # ── Validate ────────────────────────────────────────────────────
        model.eval()
        val_loss  = 0.0
        all_preds  = []
        all_targets = []

        with torch.no_grad():
            for images, labels, input_lens, label_lens in val_loader:
                images = images.to(device)

                log_probs = model(images)
                # CTCLoss on CPU (not supported on MPS)
                loss = criterion(log_probs.cpu(),
                                 labels, input_lens, label_lens)
                val_loss += loss.item() * images.size(0)

                # Decode each sample in the batch
                lp_cpu = log_probs.cpu()         # (T,B,C)
                offset = 0
                for i in range(images.size(0)):
                    pred   = greedy_decode(lp_cpu[:, i, :])
                    tlen   = label_lens[i].item()
                    target = "".join(
                        [list(CHARS)[labels[offset + j].item()]
                         for j in range(tlen)]
                    )
                    all_preds.append(pred)
                    all_targets.append(target)
                    offset += int(tlen)

        val_loss  /= n_val
        c_acc = char_accuracy(all_preds, all_targets)
        e_acc = exact_accuracy(all_preds, all_targets)

        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:>3}/{epochs}]  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"char_acc={c_acc*100:.1f}%  exact_acc={e_acc*100:.1f}%  "
              f"({elapsed:.1f}s)")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["char_acc"].append(c_acc)
        history["exact_acc"].append(e_acc)

        if c_acc > best_char_acc:
            best_char_acc = c_acc
            torch.save(model.state_dict(), OUT_WEIGHTS)
            print(f"  → Saved best model  (char_acc={c_acc*100:.1f}%)")

    print(f"\n[train_crnn] Done. Best char_acc={best_char_acc*100:.1f}%")
    print(f"  Weights: {OUT_WEIGHTS}")

    _show_examples(model, val_loader, device)
    _plot_history(history)


def _show_examples(model, val_loader, device, n=10):
    """Print a few example predictions vs ground truth."""
    model.eval()
    shown = 0
    print("\n── Sample predictions ─────────────────────────────────────")
    print(f"{'Predicted':<12}  {'Target':<12}  Match")
    with torch.no_grad():
        for images, labels, input_lens, label_lens in val_loader:
            lp  = model(images.to(device)).cpu()
            offset = 0
            for i in range(images.size(0)):
                pred   = greedy_decode(lp[:, i, :])
                tlen   = label_lens[i].item()
                target = "".join([list(CHARS)[labels[offset + j].item()]
                                  for j in range(tlen)])
                match  = "✓" if pred == target else "✗"
                print(f"{pred:<12}  {target:<12}  {match}")
                offset += tlen
                shown  += 1
                if shown >= n:
                    return


def _plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train loss")
    ax1.plot(epochs, history["val_loss"],   label="Val loss")
    ax1.set_title("CRNN — Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("CTC Loss")
    ax1.legend()

    ax2.plot(epochs, [x * 100 for x in history["char_acc"]],  label="Char acc")
    ax2.plot(epochs, [x * 100 for x in history["exact_acc"]], label="Exact acc")
    ax2.set_title("CRNN — Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    plt.tight_layout()
    out = os.path.join(config.MODEL_DIR, "crnn_training.png")
    plt.savefig(out)
    print(f"  Plot saved to {out}")
    plt.close()


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train CRNN + CTC plate reader")
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=64)
    args = parser.parse_args()
    train(args.epochs, args.lr, args.batch_size)


if __name__ == "__main__":
    main()
