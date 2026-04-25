"""
Label Real Florida Plate Images
=================================
Runs the trained CRNN on each image in a folder, saves predictions to a
CSV for human review, then adds confirmed images to the CRNN training set.

Step 1 — generate predictions:
    python train/label_florida_plates.py --predict \
        --src ~/Desktop/Florida\ Plates

Step 2 — open data/datasets/florida_review.csv in Excel / Numbers,
          correct any wrong 'predicted' values in the 'text' column,
          delete rows that aren't actual plates, then save.

Step 3 — add confirmed images to the training dataset:
    python train/label_florida_plates.py --add
"""

import os
import sys
import csv
import shutil
import argparse

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.plate_reader import PlateReader, IMG_H, IMG_W, CHARS

REVIEW_CSV  = "data/datasets/florida_review.csv"
OUT_DIR     = "data/datasets/crnn_plates"
LABELS_CSV  = os.path.join(OUT_DIR, "labels.csv")


def predict(src_dir: str) -> None:
    """Auto-label all images in src_dir using the trained CRNN."""
    src_dir = os.path.expanduser(src_dir)
    if not os.path.isdir(src_dir):
        print(f"ERROR: folder not found: {src_dir}")
        sys.exit(1)

    weights = os.path.join(config.MODEL_DIR, "plate_crnn.pth")
    if not os.path.exists(weights):
        print("ERROR: CRNN weights not found. Run train/train_crnn.py first.")
        sys.exit(1)

    reader = PlateReader(weights)

    img_files = [f for f in os.listdir(src_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".avif"))]
    print(f"[label] Found {len(img_files)} images in {src_dir}")

    rows = []
    for fname in tqdm(img_files, desc="Predicting"):
        fpath = os.path.join(src_dir, fname)
        img   = cv2.imread(fpath)
        if img is None:
            continue

        text, conf = reader.read(img)
        rows.append({
            "filename":  fname,
            "src_path":  fpath,
            "text":      text,
            "conf":      round(conf, 3),
            "keep":      "yes" if text else "no",
        })

    os.makedirs("data/datasets", exist_ok=True)
    with open(REVIEW_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "src_path", "text", "conf", "keep"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[label] Predictions saved to {REVIEW_CSV}")
    print("  → Open that CSV, correct any wrong 'text' values,")
    print("    set 'keep' to 'no' for non-plate images, then run:")
    print("    python train/label_florida_plates.py --add")


def add_to_dataset() -> None:
    """Copy confirmed images into crnn_plates and append to labels.csv."""
    if not os.path.exists(REVIEW_CSV):
        print(f"ERROR: {REVIEW_CSV} not found. Run --predict first.")
        sys.exit(1)

    img_dir = os.path.join(OUT_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)

    # Load existing stems so we don't duplicate
    existing = set()
    if os.path.exists(LABELS_CSV):
        with open(LABELS_CSV) as f:
            for row in csv.DictReader(f):
                existing.add(row["stem"])

    new_rows = []
    skipped  = 0

    with open(REVIEW_CSV) as f:
        for row in csv.DictReader(f):
            if row.get("keep", "yes").strip().lower() != "yes":
                skipped += 1
                continue
            text = row["text"].strip().upper()
            if not text:
                skipped += 1
                continue
            if not all(c in CHARS for c in text):
                print(f"  skipping {row['filename']} — invalid chars in '{text}'")
                skipped += 1
                continue
            if not (config.MIN_PLATE_CHARS <= len(text) <= config.MAX_PLATE_CHARS):
                skipped += 1
                continue

            # Copy image → crnn_plates/images/
            base = os.path.splitext(row["filename"])[0]
            stem = f"fl_{base}"
            if stem in existing:
                skipped += 1
                continue

            dst = os.path.join(img_dir, stem + ".png")
            src = row["src_path"]
            img = cv2.imread(src)
            if img is None:
                skipped += 1
                continue
            cv2.imwrite(dst, img)

            new_rows.append({"stem": stem, "text": text})
            existing.add(stem)

    # Append to labels.csv
    mode = "a" if os.path.exists(LABELS_CSV) else "w"
    with open(LABELS_CSV, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "text"])
        if mode == "w":
            writer.writeheader()
        writer.writerows(new_rows)

    print(f"[label] Added {len(new_rows)} Florida plates to training set "
          f"({skipped} skipped)")
    print(f"  Now retrain: python train/train_crnn.py")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predict", action="store_true",
                       help="Run CRNN on images and save predictions to CSV")
    group.add_argument("--add",     action="store_true",
                       help="Add reviewed CSV entries to training set")
    parser.add_argument("--src", type=str,
                        default="~/Desktop/Florida Plates",
                        help="Source folder of Florida plate images (for --predict)")
    args = parser.parse_args()

    if args.predict:
        predict(args.src)
    else:
        add_to_dataset()


if __name__ == "__main__":
    main()
