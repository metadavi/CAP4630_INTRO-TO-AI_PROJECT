"""
Ingest Raw Photos into CRNN Training Dataset
=============================================
Designed for folders of unlabeled plate photos (e.g. your own neighborhood shots).
Unlike autolabel_real_plates.py, this script does NOT require YOLO label files.

Pipeline per image:
  1. Run PlateDetector (classical CV + CNN) to find plate region
  2. Run PlateReader (CRNN+CTC) to predict plate text + confidence
  3. High-confidence (>= --auto-conf): accept automatically
  4. Low-confidence: show crop in window, display prediction, ask you to confirm/correct
  5. Save crop + label to data/datasets/crnn_plates/

Usage:
    # Interactive (recommended for neighborhood shots):
    python train/ingest_photos.py --folder ~/Desktop/Neighborhood\ Plates

    # Fully automatic (no prompts, skip low-confidence):
    python train/ingest_photos.py --folder ~/Desktop/Neighborhood\ Plates --auto

    # Lower the auto-accept threshold:
    python train/ingest_photos.py --folder ~/Desktop/Neighborhood\ Plates --auto-conf 0.50
"""

import os
import sys
import csv
import re
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.detector import PlateDetector
from src.plate_reader import PlateReader

OUT_DIR       = os.path.join("data", "datasets", "crnn_plates")
AUTO_CONF     = 0.60   # Default auto-accept threshold
VALID_PLATE   = re.compile(r'^[A-Z0-9]{4,8}$')
WINDOW_NAME   = "Plate Crop — Press any key to continue"


def _load_existing_stems(csv_path: str) -> set:
    stems = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                stems.add(row["stem"])
    return stems


def _show_crop(crop: np.ndarray, predicted: str, conf: float) -> None:
    """Display the plate crop with the CRNN prediction overlaid."""
    h, w = crop.shape[:2]
    # Scale up so shortest edge is at least 300px, keep aspect ratio
    scale   = max(300 / h, 700 / w, 1.0)
    new_w   = int(w * scale)
    new_h   = int(h * scale)
    display = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    label   = f"Prediction: {predicted}  ({conf:.2f})" if predicted else "No prediction"
    color   = (0, 200, 0) if conf >= AUTO_CONF else (0, 140, 255)
    # Draw black shadow then colored text for readability on any background
    cv2.putText(display, label, (12, new_h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
    cv2.putText(display, label, (12, new_h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imshow(WINDOW_NAME, display)
    cv2.waitKey(1)   # Force display refresh; user reads terminal prompt


def _prompt_label(predicted: str, conf: float) -> str | None:
    """
    Ask the user to confirm or correct the plate text.
    Returns the final label string, or None to skip this crop.
    """
    if predicted:
        hint = f"  CRNN says: [{predicted}]  (conf={conf:.2f})\n"
    else:
        hint = "  CRNN could not read this plate.\n"

    print(hint, end="")
    print("  Enter plate text (ENTER to accept, 's' to skip, 'q' to quit): ", end="", flush=True)
    raw = input().strip().upper()

    if raw == "Q":
        return "QUIT"
    if raw == "S" or raw == "":
        if raw == "" and predicted:
            return predicted   # Accept CRNN prediction
        return None            # Skip

    # User typed a correction
    clean = re.sub(r'[^A-Z0-9]', '', raw)
    if VALID_PLATE.match(clean):
        return clean
    print(f"  [!] '{clean}' doesn't look like a valid plate (4-8 alphanumeric). Skipping.")
    return None


def _center_crop(img: np.ndarray) -> np.ndarray:
    """
    Return the center-horizontal, bottom-half strip of an image —
    where a license plate typically lives in a close-up phone shot.
    Falls back to full image if it's already small.
    """
    h, w = img.shape[:2]
    if h < 100 or w < 200:
        return img
    # Take middle 90% width, full height — let CRNN handle the noise
    x1 = int(w * 0.05)
    x2 = int(w * 0.95)
    return img[0:h, x1:x2]


def run(folder: str, auto_mode: bool, auto_conf: float, no_detect: bool) -> None:
    folder = os.path.expanduser(folder)
    if not os.path.isdir(folder):
        print(f"ERROR: Folder not found: {folder}")
        sys.exit(1)

    crnn_weights = os.path.join(config.MODEL_DIR, "plate_crnn.pth")
    if not os.path.exists(crnn_weights):
        print("ERROR: CRNN weights not found at", crnn_weights)
        print("  Run: python train/train_crnn.py  first.")
        sys.exit(1)

    print("[ingest] Loading models…")
    if not no_detect:
        detector = PlateDetector()
    else:
        detector = None
        print("[ingest] --no-detect: skipping plate detector, using full images")
    reader = PlateReader(crnn_weights)
    print("[ingest] Ready.\n")

    img_out_dir = os.path.join(OUT_DIR, "images")
    os.makedirs(img_out_dir, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "labels.csv")

    existing_stems = _load_existing_stems(csv_path)
    print(f"[ingest] {len(existing_stems)} existing entries in dataset (duplicates will be skipped)")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted([
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in image_exts
    ])

    if not image_files:
        print(f"[ingest] No image files found in: {folder}")
        sys.exit(0)

    print(f"[ingest] Found {len(image_files)} images in {folder}")
    if not auto_mode:
        print("[ingest] Interactive mode — a window will show each crop.")
        print("  ENTER = accept CRNN prediction | type to correct | 's' = skip | 'q' = quit\n")

    kept = skipped_conf = skipped_no_detect = skipped_user = 0
    new_rows = []

    for idx, fname in enumerate(image_files):
        img_path = os.path.join(folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [{idx+1}/{len(image_files)}] {fname} — could not read, skipping")
            continue

        stem_base = os.path.splitext(fname)[0]
        print(f"  [{idx+1}/{len(image_files)}] {fname}", end=" ... ", flush=True)

        # ── Stage 1: Detect plate region ───────────────────────────────────
        if no_detect:
            crop = _center_crop(img)
            print("(full-image mode)", end=" ")
        else:
            detections = detector.detect(img)
            if not detections:
                crop = _center_crop(img)
                print("(no detection — using center crop)", end=" ")
            else:
                crop = detections[0]["crop"]

        # ── Stage 2: Run CRNN ───────────────────────────────────────────────
        text, conf = reader.read(crop)
        entry_stem = f"neighborhood_{stem_base}"

        if entry_stem in existing_stems:
            print("already in dataset, skipping")
            continue

        # ── Stage 3: Accept / prompt ────────────────────────────────────────
        if auto_mode:
            if text and conf >= auto_conf:
                label = text
                print(f"auto → {label} ({conf:.2f})")
            else:
                print(f"skipped (conf={conf:.2f}, text='{text}')")
                skipped_conf += 1
                continue
        else:
            # Show crop in window regardless of confidence
            _show_crop(crop, text, conf)
            print(f"pred='{text}' conf={conf:.2f}")
            label = _prompt_label(text, conf)

            if label == "QUIT":
                print("\n[ingest] Quit requested.")
                break
            if label is None:
                skipped_user += 1
                continue

        # ── Stage 4: Save ───────────────────────────────────────────────────
        out_path = os.path.join(img_out_dir, entry_stem + ".png")
        cv2.imwrite(out_path, crop)
        new_rows.append({"stem": entry_stem, "text": label})
        existing_stems.add(entry_stem)
        kept += 1

    cv2.destroyAllWindows()

    # ── Write to labels.csv ─────────────────────────────────────────────────
    if new_rows:
        mode = "a" if os.path.exists(csv_path) else "w"
        with open(csv_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["stem", "text"])
            if mode == "w":
                writer.writeheader()
            writer.writerows(new_rows)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"[ingest] Done!")
    print(f"  Added to dataset : {kept}")
    if auto_mode:
        print(f"  Skipped (low conf): {skipped_conf}")
    else:
        print(f"  Skipped by you   : {skipped_user}")
    print(f"  Labels file      : {csv_path}")
    if kept > 0:
        print(f"\n  Next step → retrain CRNN:")
        print(f"  python train/train_crnn.py")


def main():
    parser = argparse.ArgumentParser(description="Ingest raw plate photos into CRNN training dataset")
    parser.add_argument("--folder",    required=True,
                        help="Path to folder containing plate photos")
    parser.add_argument("--auto",      action="store_true",
                        help="Skip interactive prompts — only keep high-confidence auto reads")
    parser.add_argument("--auto-conf", type=float, default=AUTO_CONF,
                        help=f"Auto-accept confidence threshold (default: {AUTO_CONF})")
    parser.add_argument("--no-detect", action="store_true",
                        help="Skip plate detector — feed full image to CRNN (best for close-up phone shots)")
    args = parser.parse_args()
    run(args.folder, args.auto, args.auto_conf, args.no_detect)


if __name__ == "__main__":
    main()
