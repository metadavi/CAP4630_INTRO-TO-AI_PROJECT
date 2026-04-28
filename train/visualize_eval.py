"""
CRNN Visual Evaluation
======================
Generates four charts saved to models/eval_charts.png:
  1. Summary metrics bar chart
  2. Per-character F1 heatmap
  3. Full confusion matrix (36×36)
  4. Top-15 confusion pairs

Usage:
    python train/visualize_eval.py
"""

import os
import sys
import random
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
from torch.utils.data import random_split, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.plate_reader import PlateCRNN, CHARS, BLANK, N_CLASS, greedy_decode
from train.train_crnn  import PlateDataset, collate_fn

DATA_DIR     = "data/datasets/crnn_plates"
WEIGHTS_PATH = os.path.join(config.MODEL_DIR, "plate_crnn.pth")
OUT_PATH     = os.path.join(config.MODEL_DIR, "eval_charts.png")
SEED         = 42


def run():
    # ── Device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ── Dataset ───────────────────────────────────────────────────────────
    full_ds = PlateDataset(DATA_DIR, augment=False)
    n_train = int(len(full_ds) * config.TRAIN_SPLIT)
    n_val   = len(full_ds) - n_train
    _, val_ds = random_split(full_ds, [n_train, n_val],
                             generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=2, collate_fn=collate_fn)

    # ── Model ─────────────────────────────────────────────────────────────
    model = PlateCRNN(n_class=N_CLASS).to(device)
    state = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # ── Inference ─────────────────────────────────────────────────────────
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels_enc, input_lens, label_lens in val_loader:
            images = images.to(device)
            log_probs = model(images)
            offset = 0
            for i, ll in enumerate(label_lens.tolist()):
                idxs = labels_enc[offset: offset + ll].tolist()
                all_targets.append("".join(CHARS[x] for x in idxs))
                offset += ll
            B = images.size(0)
            for i in range(B):
                all_preds.append(greedy_decode(log_probs[:, i, :].cpu()))

    # ── Compute metrics ───────────────────────────────────────────────────
    correct_chars = total_chars = 0
    for p, t in zip(all_preds, all_targets):
        for pc, tc in zip(p, t):
            correct_chars += (pc == tc)
        total_chars += len(t)
    char_acc  = correct_chars / total_chars if total_chars else 0.0
    exact_acc = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    # Full 36×36 confusion matrix (rows=truth, cols=pred)
    char_idx = {c: i for i, c in enumerate(CHARS)}
    conf_mat = np.zeros((36, 36), dtype=int)

    confusions = []
    for p, t in zip(all_preds, all_targets):
        max_len = max(len(p), len(t))
        p_pad = p.ljust(max_len, '\x00')
        t_pad = t.ljust(max_len, '\x00')
        for pc, tc in zip(p_pad, t_pad):
            if tc != '\x00':
                if pc == tc:
                    tp[tc] += 1
                    if tc in char_idx and pc in char_idx:
                        conf_mat[char_idx[tc]][char_idx[pc]] += 1
                else:
                    fn[tc] += 1
                    if pc != '\x00':
                        fp[pc] += 1
                    if pc != '\x00' and tc in char_idx and pc in char_idx:
                        conf_mat[char_idx[tc]][char_idx[pc]] += 1
            elif pc != '\x00':
                fp[pc] += 1
            if pc != tc:
                confusions.append((pc, tc))

    precisions, recalls, f1s = [], [], []
    per_char = {}
    for c in CHARS:
        t_pos = tp[c]; f_pos = fp[c]; f_neg = fn[c]
        prec  = t_pos / (t_pos + f_pos) if (t_pos + f_pos) > 0 else 0.0
        rec   = t_pos / (t_pos + f_neg) if (t_pos + f_neg) > 0 else 0.0
        f1    = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_char[c] = (prec, rec, f1)
        if (tp[c] + fn[c]) > 0:
            precisions.append(prec); recalls.append(rec); f1s.append(f1)

    macro_prec = sum(precisions) / len(precisions)
    macro_rec  = sum(recalls)    / len(recalls)
    macro_f1   = sum(f1s)        / len(f1s)

    pair_counts = defaultdict(int)
    for pair in confusions:
        pair_counts[pair] += 1
    top_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:15]

    # ── Plot ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 18), facecolor="#0f0f0f")
    fig.suptitle("CRNN Plate Reader — Evaluation Report", fontsize=18,
                 color="white", fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.38, wspace=0.32,
                           left=0.06, right=0.97, top=0.94, bottom=0.05)

    dark  = "#1a1a1a"
    green = "#00c853"
    text  = "#eeeeee"

    # ── Chart 1: Summary metrics ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(dark)
    metrics = ["Char\nAccuracy", "Exact-match\nAccuracy",
               "Macro\nPrecision", "Macro\nRecall", "Macro\nF1"]
    values  = [char_acc, exact_acc, macro_prec, macro_rec, macro_f1]
    colors  = [green if v >= 0.85 else "#ffa726" if v >= 0.70 else "#ef5350"
               for v in values]
    bars = ax1.bar(metrics, [v * 100 for v in values],
                   color=colors, edgecolor="#333", linewidth=0.8, width=0.55)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.8,
                 f"{val*100:.1f}%", ha="center", va="bottom",
                 color=text, fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.set_title("Overall Metrics", color=text, fontsize=13, pad=10)
    ax1.tick_params(colors=text, labelsize=9)
    ax1.spines[:].set_color("#333")
    ax1.yaxis.label.set_color(text)
    ax1.set_ylabel("Score (%)", color=text)
    ax1.axhline(85, color="#555", linestyle="--", linewidth=0.8)

    # ── Chart 2: Per-char F1 bar ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(dark)
    sorted_chars = sorted(CHARS, key=lambda c: per_char[c][2])
    f1_vals  = [per_char[c][2] * 100 for c in sorted_chars]
    bar_cols = [green if v >= 85 else "#ffa726" if v >= 70 else "#ef5350"
                for v in f1_vals]
    ax2.barh(sorted_chars, f1_vals, color=bar_cols,
             edgecolor="#333", linewidth=0.5, height=0.75)
    ax2.axvline(85, color="#555", linestyle="--", linewidth=0.8)
    ax2.set_xlim(0, 105)
    ax2.set_title("Per-Character F1-Score", color=text, fontsize=13, pad=10)
    ax2.tick_params(colors=text, labelsize=8)
    ax2.spines[:].set_color("#333")
    ax2.set_xlabel("F1 Score (%)", color=text)

    # ── Chart 3: Confusion matrix ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(dark)
    # Normalise by row (truth) so colours show recall per class
    row_sums = conf_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    conf_norm = conf_mat / row_sums

    sns.heatmap(conf_norm, ax=ax3,
                xticklabels=list(CHARS), yticklabels=list(CHARS),
                cmap="YlOrRd", linewidths=0.3, linecolor="#222",
                cbar_kws={"shrink": 0.8},
                annot=False, fmt=".0%")
    ax3.set_title("Confusion Matrix (row-normalised recall)", color=text,
                  fontsize=13, pad=10)
    ax3.set_xlabel("Predicted", color=text, fontsize=10)
    ax3.set_ylabel("True", color=text, fontsize=10)
    ax3.tick_params(colors=text, labelsize=7)
    ax3.collections[0].colorbar.ax.tick_params(colors=text, labelsize=8)

    # ── Chart 4: Top confusion pairs ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(dark)
    pair_labels = [f"'{pc}' → '{tc}'" for (pc, tc), _ in top_pairs]
    pair_vals   = [cnt for _, cnt in top_pairs]
    y_pos = range(len(pair_labels) - 1, -1, -1)
    bars4 = ax4.barh(list(y_pos), pair_vals, color="#ef5350",
                     edgecolor="#333", linewidth=0.5, height=0.65)
    ax4.set_yticks(list(y_pos))
    ax4.set_yticklabels(pair_labels, fontsize=9)
    for bar, val in zip(bars4, pair_vals):
        ax4.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 str(val), va="center", color=text, fontsize=8)
    ax4.set_title("Top-15 Confusion Pairs  (predicted → truth)",
                  color=text, fontsize=13, pad=10)
    ax4.set_xlabel("Count", color=text)
    ax4.tick_params(colors=text, labelsize=9)
    ax4.spines[:].set_color("#333")

    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nSaved → {OUT_PATH}")
    print(f"\nSummary:")
    print(f"  Char accuracy  : {char_acc*100:.2f}%")
    print(f"  Exact accuracy : {exact_acc*100:.2f}%")
    print(f"  Macro Precision: {macro_prec*100:.2f}%")
    print(f"  Macro Recall   : {macro_rec*100:.2f}%")
    print(f"  Macro F1       : {macro_f1*100:.2f}%")

    # Open automatically
    os.system(f"open '{OUT_PATH}'")


if __name__ == "__main__":
    run()
