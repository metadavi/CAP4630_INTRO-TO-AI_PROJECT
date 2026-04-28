"""
CRNN Evaluation
===============
Runs the saved plate_crnn.pth against the validation split and reports:
  - Character-level accuracy
  - Exact-match (plate-level) accuracy
  - Per-character Precision, Recall, F1  (macro-averaged across all 36 chars)
  - Confusion summary (top misread pairs)

Usage:
    python train/evaluate_crnn.py
"""

import os
import sys
import random
from collections import defaultdict

import torch
from torch.utils.data import random_split, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.plate_reader import PlateCRNN, CHARS, BLANK, N_CLASS, greedy_decode
from train.train_crnn  import PlateDataset, collate_fn

DATA_DIR     = "data/datasets/crnn_plates"
WEIGHTS_PATH = os.path.join(config.MODEL_DIR, "plate_crnn.pth")
SEED         = 42


def evaluate():
    # ── Device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Dataset (same 80/20 split as training) ───────────────────────────
    full_ds = PlateDataset(DATA_DIR, augment=False)
    n_train = int(len(full_ds) * config.TRAIN_SPLIT)
    n_val   = len(full_ds) - n_train
    _, val_ds = random_split(full_ds, [n_train, n_val],
                             generator=torch.Generator().manual_seed(SEED))

    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=2, collate_fn=collate_fn)
    print(f"Validation samples: {n_val}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = PlateCRNN(n_class=N_CLASS).to(device)
    state = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded weights from {WEIGHTS_PATH}\n")

    # ── Inference ─────────────────────────────────────────────────────────
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for images, labels_enc, input_lens, label_lens in val_loader:
            images = images.to(device)
            log_probs = model(images)          # (T, B, C)

            # Decode targets back to strings
            offset = 0
            for i, ll in enumerate(label_lens.tolist()):
                idxs = labels_enc[offset: offset + ll].tolist()
                all_targets.append("".join(CHARS[x] for x in idxs))
                offset += ll

            # Decode predictions
            B = images.size(0)
            for i in range(B):
                lp = log_probs[:, i, :].cpu()
                all_preds.append(greedy_decode(lp))

    # ── Metrics ───────────────────────────────────────────────────────────
    # 1. Character-level accuracy
    correct_chars = total_chars = 0
    for p, t in zip(all_preds, all_targets):
        for pc, tc in zip(p, t):
            correct_chars += (pc == tc)
        total_chars += len(t)
    char_acc = correct_chars / total_chars if total_chars else 0.0

    # 2. Exact-match (plate-level) accuracy
    exact_acc = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)

    # 3. Per-character TP / FP / FN for precision / recall / F1
    #    Strategy: align predicted string to target by position (zero-pad shorter)
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for p, t in zip(all_preds, all_targets):
        max_len = max(len(p), len(t))
        p_pad = p.ljust(max_len, '\x00')
        t_pad = t.ljust(max_len, '\x00')
        for pc, tc in zip(p_pad, t_pad):
            if tc != '\x00':          # ground-truth position exists
                if pc == tc:
                    tp[tc] += 1
                else:
                    fn[tc] += 1       # missed the correct char
                    if pc != '\x00':
                        fp[pc] += 1   # predicted wrong char
            elif pc != '\x00':        # extra predicted char (over-read)
                fp[pc] += 1

    # Macro-average over all 36 chars that appear in ground truth
    precisions, recalls, f1s = [], [], []
    per_char = {}
    for c in CHARS:
        t_pos  = tp[c]
        f_pos  = fp[c]
        f_neg  = fn[c]
        prec   = t_pos / (t_pos + f_pos) if (t_pos + f_pos) > 0 else 0.0
        rec    = t_pos / (t_pos + f_neg) if (t_pos + f_neg) > 0 else 0.0
        f1     = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_char[c] = (prec, rec, f1, t_pos + f_neg)   # support = actual occurrences
        if (t_pos + f_neg) > 0:      # only average over chars that appear
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

    macro_prec = sum(precisions) / len(precisions) if precisions else 0.0
    macro_rec  = sum(recalls)    / len(recalls)    if recalls    else 0.0
    macro_f1   = sum(f1s)        / len(f1s)        if f1s        else 0.0

    # ── Print results ─────────────────────────────────────────────────────
    print("=" * 52)
    print("  CRNN EVALUATION RESULTS")
    print("=" * 52)
    print(f"  Validation samples    : {n_val}")
    print(f"  Character accuracy    : {char_acc*100:.2f}%")
    print(f"  Exact-match accuracy  : {exact_acc*100:.2f}%")
    print(f"  Macro avg Precision   : {macro_prec*100:.2f}%")
    print(f"  Macro avg Recall      : {macro_rec*100:.2f}%")
    print(f"  Macro avg F1-Score    : {macro_f1*100:.2f}%")
    print("=" * 52)

    # Per-character breakdown (sorted by F1 ascending — worst chars first)
    print("\nPer-character breakdown (worst F1 first):")
    print(f"  {'Char':>4}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Support':>8}")
    print("  " + "-" * 42)
    sorted_chars = sorted(CHARS, key=lambda c: per_char[c][2])
    for c in sorted_chars:
        prec, rec, f1, sup = per_char[c]
        print(f"  {c:>4}  {prec*100:>5.1f}%  {rec*100:>5.1f}%  {f1*100:>5.1f}%  {sup:>8}")

    # Top confusion pairs
    print("\nTop confusion pairs  (predicted → truth):")
    confusions = []
    for p, t in zip(all_preds, all_targets):
        max_len = max(len(p), len(t))
        p_pad = p.ljust(max_len, '_')
        t_pad = t.ljust(max_len, '_')
        for pc, tc in zip(p_pad, t_pad):
            if pc != tc:
                confusions.append((pc, tc))
    pair_counts = defaultdict(int)
    for pair in confusions:
        pair_counts[pair] += 1
    top_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:15]
    for (pred_c, true_c), cnt in top_pairs:
        print(f"  predicted '{pred_c}'  instead of '{true_c}'  →  {cnt}×")


if __name__ == "__main__":
    evaluate()
