"""
Dataset Preparation
===================
Prepares raw Kaggle downloads into the folder structure expected by the
training scripts.

Expected raw dataset layouts
-----------------------------

A) Plate Detector dataset  (binary: plate / no-plate patches)
   Download from Kaggle, e.g.:
     https://www.kaggle.com/datasets/andrewmvd/car-plate-detection

   After download, place in:
       data/datasets/plate_detection/
           images/   ← full vehicle images (JPG/PNG)
           annotations/   ← XML annotations (Pascal VOC format) OR
           labels/        ← YOLO txt labels (class x_c y_c w h)

   This script will:
     1. Parse bounding-box annotations
     2. Crop the plate region (positive samples)
     3. Crop random non-plate regions of similar size (negative samples)
     4. Save 64×64 patches to:
           data/datasets/detector/
               plate/     ← positive crops
               no_plate/  ← negative crops

B) Character Classifier dataset
   Download from Kaggle, e.g.:
     https://www.kaggle.com/datasets/prerak23/license-plate-character-recognition
   OR use any A-Z / 0-9 character image dataset.

   Expected layout after extraction:
       data/datasets/chars/
           A/  B/  C/ ... Z/  0/  1/ ... 9/
           (each subfolder contains grayscale character images)

   This script will:
     1. Resize all images to 32×32 grayscale
     2. Apply augmentation (rotation, noise, perspective)
     3. Save processed images to:
           data/datasets/chars_processed/
               A/  B/ ... (same structure, ready for ImageFolder)

Usage
-----
    python train/prepare_data.py --task detector
    python train/prepare_data.py --task classifier
    python train/prepare_data.py --task all
"""

import os
import sys
import argparse
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from tqdm import tqdm

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ── Constants ──────────────────────────────────────────────────────────────

# Plate detector: use the large YOLO-format train split (7 k images)
RAW_PLATE_DIR   = "data/train"
OUT_DETECTOR    = "data/datasets/detector"
RAW_CHARS_DIR   = "data/datasets/chars"
OUT_CHARS       = "data/datasets/chars_processed"

NEGATIVES_PER_POSITIVE = 3     # random background crops per plate crop
AUG_PER_IMAGE          = 4     # augmented copies per character image


# ── Detector dataset prep ──────────────────────────────────────────────────

def prepare_detector() -> None:
    """Create plate / no_plate patch datasets from annotated vehicle images."""
    images_dir = os.path.join(RAW_PLATE_DIR, "images")
    ann_dir    = os.path.join(RAW_PLATE_DIR, "annotations")

    if not os.path.isdir(images_dir):
        print(f"[prepare_detector] ERROR: {images_dir} not found.")
        print("  Download a license plate detection dataset from Kaggle and")
        print(f"  place images in {images_dir}/")
        print(f"  and Pascal-VOC XML annotations in {ann_dir}/")
        return

    out_pos = os.path.join(OUT_DETECTOR, "plate")
    out_neg = os.path.join(OUT_DETECTOR, "no_plate")
    os.makedirs(out_pos, exist_ok=True)
    os.makedirs(out_neg, exist_ok=True)

    img_files = [f for f in os.listdir(images_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    count_pos = count_neg = 0

    for img_name in tqdm(img_files, desc="Detector patches"):
        img_path = os.path.join(images_dir, img_name)
        frame    = cv2.imread(img_path)
        if frame is None:
            continue

        h_img, w_img = frame.shape[:2]
        stem         = os.path.splitext(img_name)[0]
        boxes        = _parse_voc_xml(os.path.join(ann_dir, stem + ".xml"))

        if not boxes:
            # Try YOLO-format labels
            label_path = os.path.join(RAW_PLATE_DIR, "labels", stem + ".txt")
            boxes = _parse_yolo_label(label_path, w_img, h_img)

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Positive sample
            pos_img = cv2.resize(crop, config.DETECTOR_IMG_SIZE[::-1])
            cv2.imwrite(os.path.join(out_pos, f"{stem}_{i}.jpg"), pos_img)
            count_pos += 1

            # Negative samples (random non-plate regions)
            for j in range(NEGATIVES_PER_POSITIVE):
                neg = _random_negative(frame, boxes,
                                       target_w=x2 - x1, target_h=y2 - y1)
                if neg is not None:
                    neg_img = cv2.resize(neg, config.DETECTOR_IMG_SIZE[::-1])
                    cv2.imwrite(os.path.join(out_neg,
                                             f"{stem}_{i}_neg{j}.jpg"), neg_img)
                    count_neg += 1

    print(f"[prepare_detector] Done — {count_pos} positives, {count_neg} negatives")


def _parse_voc_xml(xml_path: str) -> list[tuple]:
    """Parse Pascal VOC XML and return list of (x1,y1,x2,y2) ints."""
    if not os.path.exists(xml_path):
        return []
    boxes = []
    try:
        tree = ET.parse(xml_path)
        for obj in tree.getroot().findall("object"):
            bndbox = obj.find("bndbox")
            x1 = int(float(bndbox.find("xmin").text))
            y1 = int(float(bndbox.find("ymin").text))
            x2 = int(float(bndbox.find("xmax").text))
            y2 = int(float(bndbox.find("ymax").text))
            boxes.append((x1, y1, x2, y2))
    except Exception:
        pass
    return boxes


def _parse_yolo_label(label_path: str,
                       img_w: int, img_h: int) -> list[tuple]:
    """Parse YOLO-format label file and return (x1,y1,x2,y2) ints."""
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, xc, yc, w, h = map(float, parts[:5])
            x1 = int((xc - w / 2) * img_w)
            y1 = int((yc - h / 2) * img_h)
            x2 = int((xc + w / 2) * img_w)
            y2 = int((yc + h / 2) * img_h)
            boxes.append((x1, y1, x2, y2))
    return boxes


def _random_negative(frame: np.ndarray,
                      avoid_boxes: list[tuple],
                      target_w: int, target_h: int,
                      max_tries: int = 20) -> np.ndarray | None:
    """Sample a random crop that does not overlap any plate box."""
    h_img, w_img = frame.shape[:2]
    for _ in range(max_tries):
        x1 = random.randint(0, max(0, w_img - target_w))
        y1 = random.randint(0, max(0, h_img - target_h))
        x2 = x1 + target_w
        y2 = y1 + target_h

        overlap = any(
            _iou((x1, y1, x2, y2), box) > 0.05
            for box in avoid_boxes
        )
        if not overlap:
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                return crop
    return None


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


# ── Character classifier dataset prep ──────────────────────────────────────

def prepare_classifier() -> None:
    """Resize + augment character images into chars_processed/."""
    if not os.path.isdir(RAW_CHARS_DIR):
        print(f"[prepare_classifier] ERROR: {RAW_CHARS_DIR} not found.")
        print("  Download a character recognition dataset from Kaggle.")
        print(f"  Expected layout: {RAW_CHARS_DIR}/A/, {RAW_CHARS_DIR}/0/, ...")
        return

    total = 0
    for cls_name in tqdm(sorted(os.listdir(RAW_CHARS_DIR)), desc="Char classes"):
        cls_in  = os.path.join(RAW_CHARS_DIR,  cls_name)
        cls_out = os.path.join(OUT_CHARS, cls_name.upper())
        if not os.path.isdir(cls_in):
            continue

        os.makedirs(cls_out, exist_ok=True)

        for fname in os.listdir(cls_in):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            img = cv2.imread(os.path.join(cls_in, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            resized = cv2.resize(img, config.CHAR_IMG_SIZE[::-1])
            stem    = os.path.splitext(fname)[0]

            # Original
            cv2.imwrite(os.path.join(cls_out, f"{stem}.png"), resized)
            total += 1

            # Augmented copies
            for k, aug in enumerate(_augment(resized)):
                cv2.imwrite(os.path.join(cls_out, f"{stem}_aug{k}.png"), aug)
                total += 1

    print(f"[prepare_classifier] Done — {total} images written to {OUT_CHARS}")


def _augment(img: np.ndarray) -> list[np.ndarray]:
    """Return AUG_PER_IMAGE augmented variants of a character image."""
    h, w   = img.shape[:2]
    result = []

    for _ in range(AUG_PER_IMAGE):
        aug = img.copy()

        # Random rotation ±15°
        angle = random.uniform(-15, 15)
        M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        aug   = cv2.warpAffine(aug, M, (w, h),
                                borderMode=cv2.BORDER_REPLICATE)

        # Random perspective warp
        if random.random() < 0.5:
            aug = _random_perspective(aug)

        # Gaussian noise
        noise = np.random.normal(0, random.uniform(5, 15), aug.shape).astype(np.int16)
        aug   = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Random brightness shift
        aug   = np.clip(aug.astype(np.int16) + random.randint(-30, 30),
                        0, 255).astype(np.uint8)

        result.append(aug)

    return result


def _random_perspective(img: np.ndarray,
                         strength: float = 0.08) -> np.ndarray:
    h, w = img.shape[:2]
    d    = int(min(h, w) * strength)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [random.randint(-d, d), random.randint(-d, d)],
        [w + random.randint(-d, d), random.randint(-d, d)],
        [w + random.randint(-d, d), h + random.randint(-d, d)],
        [random.randint(-d, d), h + random.randint(-d, d)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ALPR training datasets")
    parser.add_argument("--task", choices=["detector", "classifier", "all"],
                        default="all",
                        help="Which dataset to prepare (default: all)")
    args = parser.parse_args()

    if args.task in ("detector", "all"):
        prepare_detector()
    if args.task in ("classifier", "all"):
        prepare_classifier()


if __name__ == "__main__":
    main()
