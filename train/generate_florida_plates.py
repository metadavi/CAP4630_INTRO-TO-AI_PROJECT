"""
Florida Plate Synthetic Generator
===================================
Generates synthetic Florida-style licence plate images for training the CRNN.

Florida plate style:
  - White background
  - Green/teal characters (bold, wide)
  - Orange decorative graphic in the centre (partially obscures char 4)
  - "SUNSHINE STATE" footer text
  - "MYFLORIDA.COM" header text

Output:
    data/datasets/crnn_plates/
        images/   ← plate crop PNGs  (e.g. 00001_ABC123.png)
        labels.csv ← stem, text columns

Usage:
    python train/generate_florida_plates.py
    python train/generate_florida_plates.py --count 15000 --out data/datasets/crnn_plates
"""

import os
import sys
import csv
import random
import argparse
import string

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Constants ──────────────────────────────────────────────────────────────

PLATE_W, PLATE_H = 256, 80          # synthetic plate canvas size
CHAR_COLOUR  = (34, 139, 34)        # forest-green  (R,G,B for PIL)
BG_COLOUR    = (255, 255, 255)      # white
ORANGE_BLOB  = (255, 140, 0)        # Florida orange graphic colour
FOOTER_COLOUR = (100, 100, 100)

# Florida plates: 3 letters + 3 digits  OR  2 letters + 2 digits + 2 letters
# We generate both formats
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"   # no I or O (ambiguous on plates)
DIGITS  = "0123456789"


def _random_plate_text() -> str:
    fmt = random.choice(["LLL999", "LL99LL"])
    text = ""
    for ch in fmt:
        if ch == "L":
            text += random.choice(LETTERS)
        else:
            text += random.choice(DIGITS)
    return text


def _draw_plate(text: str, add_noise: bool = True) -> np.ndarray:
    """Render a single synthetic Florida plate as an RGB numpy array."""
    img  = Image.new("RGB", (PLATE_W, PLATE_H), BG_COLOUR)
    draw = ImageDraw.Draw(img)

    # ── Header text ─────────────────────────────────────────────────────
    try:
        hdr_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 9)
    except Exception:
        hdr_font = ImageFont.load_default()
    draw.text((PLATE_W // 2, 4), "MYFLORIDA.COM",
              fill=(150, 150, 150), font=hdr_font, anchor="mt")

    # ── Main characters ──────────────────────────────────────────────────
    try:
        char_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 46)
    except Exception:
        char_font = ImageFont.load_default()

    n      = len(text)          # always 6
    x_step = PLATE_W / (n + 1)
    y_mid  = PLATE_H // 2 + 4

    for i, ch in enumerate(text):
        x = int(x_step * (i + 1))
        draw.text((x, y_mid), ch, fill=CHAR_COLOUR,
                  font=char_font, anchor="mm")

    # ── Orange graphic blob (position 3–4 area) ──────────────────────────
    cx = int(x_step * 3.5)
    cy = y_mid
    r  = random.randint(14, 18)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ORANGE_BLOB)
    # small flower-like detail
    for angle in range(0, 360, 45):
        rad = np.radians(angle)
        px  = int(cx + (r + 5) * np.cos(rad))
        py  = int(cy + (r + 5) * np.sin(rad))
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=ORANGE_BLOB)

    # Redraw the two characters flanking the blob on top so they show through
    for i in [2, 3]:
        x = int(x_step * (i + 1))
        draw.text((x, y_mid), text[i], fill=CHAR_COLOUR,
                  font=char_font, anchor="mm")

    # ── Footer text ──────────────────────────────────────────────────────
    try:
        ftr_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 8)
    except Exception:
        ftr_font = ImageFont.load_default()
    draw.text((PLATE_W // 2, PLATE_H - 4), "SUNSHINE STATE",
              fill=FOOTER_COLOUR, font=ftr_font, anchor="mb")

    # ── Thin border ──────────────────────────────────────────────────────
    draw.rectangle([0, 0, PLATE_W - 1, PLATE_H - 1],
                   outline=(180, 180, 180), width=2)

    arr = np.array(img)         # RGB uint8

    if add_noise:
        arr = _augment(arr)

    return arr


def _augment(img: np.ndarray) -> np.ndarray:
    """Apply random real-world degradations."""
    h, w = img.shape[:2]

    # Random rotation ±8°
    angle = random.uniform(-8, 8)
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img   = cv2.warpAffine(img, M, (w, h),
                           borderMode=cv2.BORDER_REPLICATE)

    # Random perspective warp
    if random.random() < 0.5:
        d   = int(min(h, w) * 0.05)
        src = np.float32([[0,0],[w,0],[w,h],[0,h]])
        dst = np.float32([
            [random.randint(-d,d), random.randint(-d,d)],
            [w+random.randint(-d,d), random.randint(-d,d)],
            [w+random.randint(-d,d), h+random.randint(-d,d)],
            [random.randint(-d,d), h+random.randint(-d,d)],
        ])
        M2  = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M2, (w, h),
                                  borderMode=cv2.BORDER_REPLICATE)

    # Gaussian blur (motion / focus)
    if random.random() < 0.4:
        k   = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Brightness / contrast jitter
    alpha = random.uniform(0.75, 1.25)   # contrast
    beta  = random.randint(-30, 30)      # brightness
    img   = np.clip(img.astype(np.float32) * alpha + beta,
                    0, 255).astype(np.uint8)

    # Gaussian noise
    noise = np.random.normal(0, random.uniform(2, 10),
                             img.shape).astype(np.int16)
    img   = np.clip(img.astype(np.int16) + noise,
                    0, 255).astype(np.uint8)

    return img


# ── Main ───────────────────────────────────────────────────────────────────

def generate(count: int, out_dir: str) -> None:
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "labels.csv")
    rows     = []

    print(f"[generate_florida_plates] Generating {count} plates → {out_dir}")

    for i in range(count):
        text  = _random_plate_text()
        plate = _draw_plate(text, add_noise=True)

        # Convert RGB → BGR for cv2.imwrite
        bgr   = cv2.cvtColor(plate, cv2.COLOR_RGB2BGR)
        stem  = f"{i:06d}_{text}"
        fname = stem + ".png"
        cv2.imwrite(os.path.join(img_dir, fname), bgr)
        rows.append({"stem": stem, "text": text})

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{count}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[generate_florida_plates] Done — {count} images, labels at {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10000,
                        help="Number of synthetic plates to generate")
    parser.add_argument("--out",   type=str,
                        default="data/datasets/crnn_plates",
                        help="Output directory")
    args = parser.parse_args()
    generate(args.count, args.out)


if __name__ == "__main__":
    main()
