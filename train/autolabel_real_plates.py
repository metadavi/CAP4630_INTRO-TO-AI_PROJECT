"""
Auto-Label Real Plate Crops
============================
Uses the trained CRNN (PlateReader) to label the real plate images in
data/train — replacing the old EasyOCR pass with a faster, more accurate
domain-specific model.

Steps:
  1. Read each YOLO label file to get the plate bounding box
  2. Crop the plate from the image
  3. Run PlateReader (CRNN+CTC) to get the text + confidence
  4. Keep only high-confidence readings (>= MIN_CONF)
  5. Save new crops + labels to data/datasets/crnn_plates/
     (existing entries are skipped — no duplicates)

Usage:
    python train/autolabel_real_plates.py
    python train/autolabel_real_plates.py --conf 0.70
"""

import os
import sys
import csv
import argparse

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.plate_reader import PlateReader

# ── Constants ──────────────────────────────────────────────────────────────
RAW_IMG_DIR   = "data/train/images"
RAW_LBL_DIR   = "data/train/labels"
OUT_DIR       = "data/datasets/crnn_plates"
MIN_CONF      = 0.60     # CRNN mean-max-prob confidence threshold
PAD           = 0.05     # 5% padding around the bbox crop


def _crop_plate(img: np.ndarray, yolo_box: tuple) -> np.ndarray | None:
    """Crop plate from image using YOLO normalised bbox (cx,cy,w,h)."""
    h_img, w_img = img.shape[:2]
    cx, cy, bw, bh = yolo_box

    x1 = int((cx - bw / 2 - PAD) * w_img)
    y1 = int((cy - bh / 2 - PAD) * h_img)
    x2 = int((cx + bw / 2 + PAD) * w_img)
    y2 = int((cy + bh / 2 + PAD) * h_img)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    crop = img[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def _parse_yolo(label_path: str) -> list[tuple]:
    """Return list of (cx, cy, w, h) normalised boxes."""
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append(tuple(map(float, parts[1:5])))
    return boxes


def run(min_conf: float) -> None:
    weights = os.path.join(config.MODEL_DIR, "plate_crnn.pth")
    if not os.path.exists(weights):
        print("ERROR: CRNN weights not found. Run: python train/train_crnn.py first.")
        sys.exit(1)

    print("[autolabel] Loading CRNN PlateReader…")
    reader = PlateReader(weights)
    print("[autolabel] Ready.")

    img_dir = os.path.join(OUT_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)

    img_files = [f for f in os.listdir(RAW_IMG_DIR)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    rows    = []
    kept    = 0
    skipped = 0

    # Load existing labels.csv so we can append without duplicates
    csv_path = os.path.join(OUT_DIR, "labels.csv")
    existing_stems = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                existing_stems.add(row["stem"])
        print(f"[autolabel] {len(existing_stems)} existing entries — skipping duplicates")

    for img_name in tqdm(img_files, desc="Auto-labelling plates"):
        stem     = os.path.splitext(img_name)[0]
        img_path = os.path.join(RAW_IMG_DIR, img_name)
        lbl_path = os.path.join(RAW_LBL_DIR, stem + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue

        boxes = _parse_yolo(lbl_path)
        if not boxes:
            continue

        for j, box in enumerate(boxes):
            entry_stem = f"real_{stem}_{j}"
            if entry_stem in existing_stems:
                continue

            crop = _crop_plate(img, box)
            if crop is None:
                continue

            # Run CRNN on the crop
            text, conf = reader.read(crop)

            if not text or conf < min_conf:
                skipped += 1
                continue

            # Save the crop
            out_name = entry_stem + ".png"
            cv2.imwrite(os.path.join(img_dir, out_name), crop)
            rows.append({"stem": entry_stem, "text": text})
            existing_stems.add(entry_stem)
            kept += 1

    # Append to labels.csv
    mode = "a" if os.path.exists(csv_path) else "w"
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "text"])
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)

    print(f"\n[autolabel] Done — {kept} new crops saved, {skipped} skipped (low conf / no text)")
    print(f"  Labels appended to {csv_path}")
    print(f"  Now retrain: python train/train_crnn.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=MIN_CONF,
                        help="Minimum CRNN confidence to keep a label (default: 0.60)")
    args = parser.parse_args()
    run(args.conf)


if __name__ == "__main__":
    main()
