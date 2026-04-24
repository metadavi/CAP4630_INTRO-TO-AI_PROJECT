"""
OCR Reader
==========
Extracts plate text from a detected plate crop.

Primary path  — EasyOCR (pre-trained, handles glare / angles / phone screens)
Fallback path — custom CharCNN pipeline (requires trained weights)

EasyOCR is initialised once at startup (~3-5 s on first run).
When running in Docker (Cloud Run) the models are pre-baked into the image
at /app/easyocr_models so no network call is needed.
"""

import os
import re
import cv2
import numpy as np

try:
    import easyocr as _easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False

from src.detector        import PlateDetector
from src.segmenter       import CharSegmenter
from src.char_classifier import CharClassifier
import config

_ALNUM    = re.compile(r'[^A-Z0-9]')
_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# When running inside the Docker container the models are pre-downloaded here.
# On a developer machine EasyOCR falls back to ~/.EasyOCR/ automatically.
_CONTAINER_MODEL_DIR = "/app/easyocr_models"
_MODEL_DIR = _CONTAINER_MODEL_DIR if os.path.isdir(_CONTAINER_MODEL_DIR) else None


class OCRReader:
    """
    Reads the text on a detected plate crop.

    Tries EasyOCR first (robust pre-trained model); falls back to the custom
    CharCNN pipeline if EasyOCR is not installed.
    """

    def __init__(self,
                 detector:   PlateDetector  = None,
                 segmenter:  CharSegmenter  = None,
                 classifier: CharClassifier = None):
        self.detector = detector or PlateDetector()

        if _EASYOCR_AVAILABLE:
            kwargs = dict(gpu=False, verbose=False)
            if _MODEL_DIR:
                kwargs["model_storage_directory"] = _MODEL_DIR
            print("[OCR] Initialising EasyOCR…")
            self._reader       = _easyocr.Reader(['en'], **kwargs)
            self._use_easyocr  = True
            print("[OCR] EasyOCR ready.")
        else:
            print("[OCR] EasyOCR not installed — falling back to CharCNN. "
                  "Run: pip install easyocr")
            self._use_easyocr  = False
            self.segmenter     = segmenter  or CharSegmenter()
            self.classifier    = classifier or CharClassifier()

    # ── Public API ─────────────────────────────────────────────────────────

    def read(self, plate_crop_bgr: np.ndarray) -> tuple[str, float]:
        """
        Extract the plate string from a colour plate crop.

        Returns (plate_text, confidence) — e.g. ('ABC1234', 0.91)
        Returns ('', 0.0) on failure.
        """
        if plate_crop_bgr is None or plate_crop_bgr.size == 0:
            return "", 0.0

        if self._use_easyocr:
            return self._read_easyocr(plate_crop_bgr)
        return self._read_custom(plate_crop_bgr)

    # ── EasyOCR path ────────────────────────────────────────────────────────

    def _read_easyocr(self, crop: np.ndarray) -> tuple[str, float]:
        crop = self._preprocess_for_ocr(crop)

        results = self._reader.readtext(
            crop,
            detail=1,
            paragraph=False,
            allowlist=_ALLOWLIST,
        )
        if not results:
            return "", 0.0

        best_text = ""
        best_conf = 0.0
        for (_, text, conf) in results:
            cleaned = _ALNUM.sub('', text.upper())
            if (config.MIN_PLATE_CHARS <= len(cleaned) <= config.MAX_PLATE_CHARS
                    and len(set(cleaned)) >= config.MIN_UNIQUE_CHARS
                    and conf > best_conf):
                best_text = cleaned
                best_conf = conf

        return best_text, best_conf

    @staticmethod
    def _preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
        """Upscale + CLAHE contrast enhancement to handle screen glare."""
        h, w = crop.shape[:2]
        if min(h, w) < 64:
            scale = 64 / min(h, w)
            crop  = cv2.resize(crop, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_CUBIC)
        h, w = crop.shape[:2]
        if w < 200:
            scale = 200 / w
            crop  = cv2.resize(crop, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_CUBIC)

        lab      = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b  = cv2.split(lab)
        clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        lab      = cv2.merge([clahe.apply(l), a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

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
