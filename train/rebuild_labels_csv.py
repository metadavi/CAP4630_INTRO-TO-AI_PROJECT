"""
Rebuild labels.csv from image filenames
========================================
The generate_florida_plates.py script overwrote labels.csv, losing ~14,800
entries. This script recovers them by reading plate text directly from the
synthetic image filenames (format: {index}_{PLATETEXT}.png).

Real plate entries (neighborhood_*, hardrock_*, real_*) that are already in
the current labels.csv are preserved as-is.

Usage:
    python train/rebuild_labels_csv.py
"""

import os
import sys
import csv
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

IMG_DIR  = "data/datasets/crnn_plates/images"
CSV_PATH = "data/datasets/crnn_plates/labels.csv"

# Valid plate: 5-8 alphanumeric characters
VALID = re.compile(r'^[A-Z0-9]{5,8}$')


def main():
    # ── Load existing CSV entries (keep all real plate labels) ─────────────
    existing = {}   # stem → text
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH) as f:
            for row in csv.DictReader(f):
                existing[row["stem"]] = row["text"].upper().strip()
    print(f"[rebuild] {len(existing)} existing entries in CSV")

    # ── Scan all images ─────────────────────────────────────────────────────
    images = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]
    print(f"[rebuild] {len(images)} images found in {IMG_DIR}")

    recovered = 0
    skipped   = 0
    new_rows  = []

    for fname in images:
        stem = os.path.splitext(fname)[0]   # e.g. 000000_NB14QC

        # Already in CSV — keep as-is
        if stem in existing:
            continue

        # Try to extract plate text from filename: {digits}_{PLATE}
        parts = stem.split("_", 1)
        if len(parts) != 2:
            skipped += 1
            continue

        plate_text = parts[1].upper()

        if not VALID.match(plate_text):
            skipped += 1
            continue

        existing[stem] = plate_text
        new_rows.append({"stem": stem, "text": plate_text})
        recovered += 1

    # ── Write rebuilt CSV ───────────────────────────────────────────────────
    all_rows = [{"stem": s, "text": t} for s, t in existing.items()]

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "text"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[rebuild] Done!")
    print(f"  Recovered from filenames : {recovered}")
    print(f"  Already in CSV (kept)    : {len(existing) - recovered}")
    print(f"  Skipped (bad names)      : {skipped}")
    print(f"  Total entries in CSV now : {len(all_rows)}")
    print(f"\n  Now retrain:")
    print(f"  python train/train_crnn.py --epochs 50 --lr 0.001")


if __name__ == "__main__":
    main()
