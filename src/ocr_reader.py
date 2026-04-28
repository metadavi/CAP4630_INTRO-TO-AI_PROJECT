"""
OCR Reader
==========
Extracts plate text from a detected plate crop.

Primary path  — custom CRNN + CTC (trained from scratch)
Fallback path — custom CharCNN segmentation pipeline (trained from scratch)
"""

import os
import re
import cv2
import numpy as np

from src.detector        import PlateDetector
from src.segmenter       import CharSegmenter
from src.char_classifier import CharClassifier
from src.plate_reader    import PlateReader
import config

_ALNUM = re.compile(r'[^A-Z0-9]')


class OCRReader:
    """
    Reads the text on a detected plate crop.

    Primary: custom CRNN + CTC model (trained from scratch).
    Fallback: custom CharCNN segmentation pipeline (trained from scratch).
    """

    def __init__(self,
                 detector:   PlateDetector  = None,
                 segmenter:  CharSegmenter  = None,
                 classifier: CharClassifier = None):
        self.detector = detector or PlateDetector()

        crnn_weights = os.path.join(config.MODEL_DIR, "plate_crnn.pth")

        if os.path.exists(crnn_weights):
            # ── Primary: custom trained CRNN ──────────────────────────────
            print("[OCR] Loading custom CRNN plate reader…")
            self._plate_reader = PlateReader(crnn_weights)
            self._use_crnn     = True
            print("[OCR] CRNN ready.")

        else:
            # ── Fallback: custom CharCNN segmentation pipeline ────────────
            print("[OCR] CRNN weights not found — falling back to CharCNN.")
            print("      Train CRNN: python train/train_crnn.py")
            self._use_crnn  = False
            self.segmenter  = segmenter  or CharSegmenter()
            self.classifier = classifier or CharClassifier()

    # ── Public API ─────────────────────────────────────────────────────────

    def read(self, plate_crop_bgr: np.ndarray) -> tuple[str, float]:
        """
        Extract the plate string from a colour plate crop.

        Returns (plate_text, confidence) — e.g. ('ABC1234', 0.91)
        Returns ('', 0.0) on failure.
        """
        if plate_crop_bgr is None or plate_crop_bgr.size == 0:
            return "", 0.0

        if self._use_crnn:
            return self._plate_reader.read(self._mask_orange(plate_crop_bgr))
        return self._read_custom(plate_crop_bgr)

    @staticmethod
    def _mask_orange(crop: np.ndarray) -> np.ndarray:
        """
        Replace orange pixels (Florida plate graphic) with white before CRNN.

        The FL orange graphic sits in the center of the plate and causes the
        CRNN to hallucinate 1-2 phantom characters. Masking it to white gives
        the model a neutral gap it can skip over rather than trying to decode.

        Only pixels that are:
          - Orange hue  (10–22 in OpenCV 0–180 scale, i.e. ~20–44° real)
          - High saturation (> 130) — rules out washed-out beige/tan surfaces
          - Reasonable brightness (> 90)  — rules out dark shadows
        are replaced. Everything else (green chars, white bg, black border)
        is untouched.

        No dilation is applied — keeping the mask tight prevents it from
        bleeding into adjacent green characters on large FL graphics.
        """
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           np.array([10, 130,  90], dtype=np.uint8),
                           np.array([22, 255, 255], dtype=np.uint8))

        result = crop.copy()
        result[mask > 0] = (255, 255, 255)
        return result

    # ── Custom CharCNN fallback ─────────────────────────────────────────────

    def _read_custom(self, plate_crop_bgr: np.ndarray) -> tuple[str, float]:
        binary     = PlateDetector.preprocess_crop(plate_crop_bgr)
        char_crops = self.segmenter.segment(binary)
        if not char_crops:
            return "", 0.0

        predictions = self.classifier.predict_batch(char_crops)
        kept_chars, kept_confs = [], []
        for (label, conf) in predictions:
            if conf >= config.CHAR_CONF_THRESHOLD and label:
                kept_chars.append(label)
                kept_confs.append(conf)

        if not (config.MIN_PLATE_CHARS <= len(kept_chars) <= config.MAX_PLATE_CHARS):
            return "", 0.0
        if len(set(kept_chars)) < config.MIN_UNIQUE_CHARS:
            return "", 0.0

        return "".join(kept_chars), sum(kept_confs) / len(kept_confs)
