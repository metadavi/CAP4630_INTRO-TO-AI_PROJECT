"""
OCR Reader
==========
Orchestrates the full plate-text extraction pipeline:

    plate_crop (BGR)
        │
        ▼
    PlateDetector.preprocess_crop()   ← adaptive threshold, upscale
        │
        ▼
    CharSegmenter.segment()           ← find individual character blobs
        │
        ▼
    CharClassifier.predict_batch()    ← classify each character
        │
        ▼
    filter by confidence → join → plate string + mean confidence
"""

import numpy as np

from src.detector       import PlateDetector
from src.segmenter      import CharSegmenter
from src.char_classifier import CharClassifier
import config


class OCRReader:
    """
    Reads the text on a detected plate crop.

    Parameters
    ----------
    detector    : PlateDetector   (shared with the main pipeline)
    segmenter   : CharSegmenter
    classifier  : CharClassifier
    """

    def __init__(self,
                 detector:    PlateDetector   = None,
                 segmenter:   CharSegmenter   = None,
                 classifier:  CharClassifier  = None):
        # Allow injection for testing; create defaults otherwise
        self.detector   = detector   or PlateDetector()
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
            confidence  — mean per-character softmax probability (0–1)
        """
        if plate_crop_bgr is None or plate_crop_bgr.size == 0:
            return "", 0.0

        # Stage 1: preprocess
        binary = PlateDetector.preprocess_crop(plate_crop_bgr)

        # Stage 2: segment into character crops
        char_crops = self.segmenter.segment(binary)
        if not char_crops:
            return "", 0.0

        # Stage 3: classify each character
        predictions = self.classifier.predict_batch(char_crops)

        # Stage 4: filter low-confidence predictions and assemble string
        kept_chars = []
        kept_confs = []
        for (label, conf) in predictions:
            if conf >= config.CHAR_CONF_THRESHOLD and label:
                kept_chars.append(label)
                kept_confs.append(conf)

        if len(kept_chars) < config.MIN_PLATE_CHARS:
            return "", 0.0

        if len(kept_chars) > config.MAX_PLATE_CHARS:
            return "", 0.0

        # Reject readings that are nearly all the same character —
        # these come from repeating background patterns (grilles, lattices, etc.)
        if len(set(kept_chars)) < config.MIN_UNIQUE_CHARS:
            return "", 0.0

        plate_text = "".join(kept_chars)
        mean_conf  = sum(kept_confs) / len(kept_confs)

        return plate_text, mean_conf
