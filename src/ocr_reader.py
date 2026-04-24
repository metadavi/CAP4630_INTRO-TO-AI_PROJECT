"""
OCR Reader
==========
Extracts plate text from a detected plate crop.

Primary path  — EasyOCR (pre-trained, handles glare / angles / phone screens)
Fallback path — custom CharCNN pipeline (requires trained weights)

EasyOCR is initialised once at startup (~3-5 s) and is fast thereafter.
"""

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

_ALNUM = re.compile(r'[^A-Z0-9]')
# Characters allowed on US license plates
_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


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
            print("[OCR] Initialising EasyOCR (first run downloads ~50 MB)…")
            self._reader = _easyocr.Reader(
                ['en'],
                gpu=False,
                verbose=False,
            )
            self._use_easyocr = True
            print("[OCR] EasyOCR ready.")
        else:
            print("[OCR] EasyOCR not installed — falling back to custom CharCNN. "
                  "Run: pip install easyocr")
            self._use_easyocr = False
            self.segmenter  = segmenter  or CharSegmenter()
            self.classifier = classifier or CharClassifier()

    # ── Public API ─────────────────────────────────────────────────────────

    def read(self, plate_crop_bgr: np.ndarray) -> tuple[str, float]:
        """
        Extract the plate string from a colour plate crop.

        Returns
        -------
        (plate_text, confidence) : (str, float)
            plate_text  — e.g. 'ABC1234', or '' on failure
            confidence  — 0–1 confidence score
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

        # Pick the highest-confidence result that looks like a plate
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
        """
        Upscale + enhance contrast so EasyOCR works on small or washed-out crops.
        Also reduces glare from phone screens via CLAHE.
        """
        h, w = crop.shape[:2]

        # Upscale so the shortest edge is at least 64 px
        min_dim = min(h, w)
        if min_dim < 64:
            scale = 64 / min_dim
            crop = cv2.resize(crop,
                              (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_CUBIC)

        # Upscale width to at least 200 px for better character resolution
        h, w = crop.shape[:2]
        if w < 200:
            scale = 200 / w
            crop = cv2.resize(crop,
                              (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_CUBIC)

        # CLAHE on the L channel to normalise brightness/glare from screens
        lab  = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        l     = clahe.apply(l)
        lab   = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ── Custom CharCNN fallback path ────────────────────────────────────────

    def _read_custom(self, plate_crop_bgr: np.ndarray) -> tuple[str, float]:
        binary     = PlateDetector.preprocess_crop(plate_crop_bgr)
        char_crops = self.segmenter.segment(binary)
        if not char_crops:
            return "", 0.0

        predictions = self.classifier.predict_batch(char_crops)

        kept_chars = []
        kept_confs = []
        for (label, conf) in predictions:
            if conf >= config.CHAR_CONF_THRESHOLD and label:
                kept_chars.append(label)
                kept_confs.append(conf)

        if not (config.MIN_PLATE_CHARS <= len(kept_chars) <= config.MAX_PLATE_CHARS):
            return "", 0.0

        if len(set(kept_chars)) < config.MIN_UNIQUE_CHARS:
            return "", 0.0

        return "".join(kept_chars), sum(kept_confs) / len(kept_confs)
