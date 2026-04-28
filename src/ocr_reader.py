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
from src.plate_reader    import PlateReader
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

        crnn_weights = os.path.join(config.MODEL_DIR, "plate_crnn.pth")

        if os.path.exists(crnn_weights):
            # ── Primary: custom trained CRNN (best option) ────────────────
            print("[OCR] Loading custom CRNN plate reader…")
            self._plate_reader  = PlateReader(crnn_weights)
            self._use_crnn      = True
            self._use_easyocr   = False
            print("[OCR] CRNN ready.")

        elif _EASYOCR_AVAILABLE:
            # ── Fallback 1: EasyOCR (while CRNN is not yet trained) ───────
            self._use_crnn = False
            kwargs = dict(gpu=False, verbose=False)
            if _MODEL_DIR:
                kwargs["model_storage_directory"] = _MODEL_DIR
            print("[OCR] CRNN weights not found — using EasyOCR fallback.")
            print("      Train CRNN: python train/train_crnn.py")
            self._reader       = _easyocr.Reader(['en'], **kwargs)
            self._use_easyocr  = True
            print("[OCR] EasyOCR ready.")

        else:
            # ── Fallback 2: custom CharCNN segmentation pipeline ──────────
            print("[OCR] No CRNN weights and EasyOCR not installed.")
            print("      Train CRNN: python train/train_crnn.py")
            self._use_crnn     = False
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

        if self._use_crnn:
            return self._plate_reader.read(self._mask_orange(plate_crop_bgr))
        if self._use_easyocr:
            return self._read_easyocr(plate_crop_bgr)
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

    # ── EasyOCR path ────────────────────────────────────────────────────────

    def _read_easyocr(self, crop: np.ndarray) -> tuple[str, float]:
        # Strip the top 10% (state name / website) and bottom 10% (tagline /
        # dealer frame) — conservative margin so we don't cut into characters.
        h = crop.shape[0]
        margin = int(h * 0.10)
        if h - 2 * margin >= 10:
            crop = crop[margin: h - margin, :]

        crop = self._preprocess_for_ocr(crop)

        results = self._reader.readtext(
            crop,
            detail=1,
            paragraph=False,
            allowlist=_ALLOWLIST,
            text_threshold=0.5,     # lower = picks up faint/coloured text (FL green plates)
            low_text=0.3,           # more sensitive to low-contrast character edges
            contrast_ths=0.2,       # apply contrast boost if below this level
            adjust_contrast=0.8,    # target contrast after boost
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
        """
        Upscale + contrast enhancement optimised for licence plates.

        Handles both standard dark-on-light plates AND Florida-style
        green/teal characters on white backgrounds by boosting saturation
        before the CLAHE step so coloured text becomes darker than the
        white background in the luminance channel.
        """
        h, w = crop.shape[:2]

        # Upscale so shortest edge >= 64 px
        if min(h, w) < 64:
            scale = 64 / min(h, w)
            crop  = cv2.resize(crop, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_CUBIC)
        h, w = crop.shape[:2]

        # Upscale width to at least 200 px for character resolution
        if w < 200:
            scale = 200 / w
            crop  = cv2.resize(crop, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_CUBIC)

        # Boost saturation so coloured (green/teal) text separates from
        # the white background in the luminance channel
        hsv        = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.8, 0, 255)  # saturation ×1.8
        crop       = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # CLAHE on L channel to normalise brightness / reduce glare
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
