"""
Extract LP Character Crops
==========================
Reads data/LP-characters/ (Pascal VOC XML annotations) and crops each
labeled character from its plate image, saving to data/datasets/chars/<class>/.

Run this BEFORE prepare_data.py --task classifier so the new crops get
included in the augmented chars_processed/ dataset.

Usage
-----
    python train/extract_lp_chars.py
"""

import os
import sys
import xml.etree.ElementTree as ET

import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

IMAGES_DIR  = "data/LP-characters/images"
ANN_DIR     = "data/LP-characters/annotations"
OUT_CHARS   = "data/datasets/chars"

# Only save classes that CharCNN is trained on
VALID_CLASSES = set(config.CLASSES)


def extract() -> None:
    ann_files = [f for f in os.listdir(ANN_DIR) if f.endswith(".xml")]
    if not ann_files:
        print(f"No XML files found in {ANN_DIR}")
        return

    counts: dict[str, int] = {}

    for xml_name in tqdm(ann_files, desc="Extracting LP chars"):
        xml_path = os.path.join(ANN_DIR, xml_name)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"  skip {xml_name}: {e}")
            continue

        filename = root.find("filename").text
        img_path = os.path.join(IMAGES_DIR, filename)
        if not os.path.exists(img_path):
            # Try same stem with .png extension
            stem     = os.path.splitext(xml_name)[0]
            img_path = os.path.join(IMAGES_DIR, stem + ".png")

        img = cv2.imread(img_path)
        if img is None:
            continue

        for obj in root.findall("object"):
            label = obj.find("name").text.upper().strip()
            if label not in VALID_CLASSES:
                continue

            bndbox = obj.find("bndbox")
            x1 = int(float(bndbox.find("xmin").text))
            y1 = int(float(bndbox.find("ymin").text))
            x2 = int(float(bndbox.find("xmax").text))
            y2 = int(float(bndbox.find("ymax").text))

            crop = img[y1:y2, x1:x2]
            if crop.size == 0 or (x2 - x1) < 4 or (y2 - y1) < 4:
                continue

            out_dir = os.path.join(OUT_CHARS, label)
            os.makedirs(out_dir, exist_ok=True)

            stem      = os.path.splitext(xml_name)[0]
            idx       = counts.get(label, 0)
            out_path  = os.path.join(out_dir, f"lp_{stem}_{idx}.png")
            cv2.imwrite(out_path, crop)
            counts[label] = idx + 1

    total = sum(counts.values())
    print(f"\nExtracted {total} character crops across {len(counts)} classes:")
    for cls in sorted(counts):
        print(f"  {cls}: {counts[cls]}")
    print(f"\nNow run:  python train/prepare_data.py --task classifier")
    print("to regenerate chars_processed/ with the new crops included.")


if __name__ == "__main__":
    extract()
