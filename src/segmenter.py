"""
Character Segmenter
===================
Given a preprocessed (binary, white-on-black) plate crop, finds and returns
individual character images using classical connected-component analysis.

No machine learning is used here — this is purely OpenCV morphology /
contour analysis.  The character crops are then passed to the CharClassifier.
"""

import cv2
import numpy as np

import config


class CharSegmenter:
    """Splits a plate image into a sorted list of individual character crops."""

    def segment(self, plate_binary: np.ndarray) -> list[np.ndarray]:
        """
        Parameters
        ----------
        plate_binary : np.ndarray
            Grayscale binary image (white characters on black background),
            typically the output of PlateDetector.preprocess_crop().

        Returns
        -------
        list of np.ndarray
            List of square-padded grayscale character crops, sorted left-to-right.
            Returns an empty list if no valid characters are found.
        """
        if plate_binary is None or plate_binary.size == 0:
            return []

        # Ensure binary
        if len(plate_binary.shape) == 3:
            plate_binary = cv2.cvtColor(plate_binary, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(plate_binary, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        plate_h, plate_w = binary.shape[:2]
        char_boxes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < config.MIN_CHAR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            if h < config.MIN_CHAR_HEIGHT:
                continue

            # Reject blobs that span almost the entire plate height
            # (these are usually borders or background blobs)
            if h / plate_h > config.MAX_CHAR_HEIGHT_RATIO:
                continue

            char_boxes.append((x, y, w, h))

        if not char_boxes:
            return []

        # Sort left-to-right by x coordinate
        char_boxes.sort(key=lambda b: b[0])

        crops = []
        for (x, y, w, h) in char_boxes:
            char_crop = binary[y:y + h, x:x + w]
            padded    = self._pad_to_square(char_crop)
            crops.append(padded)

        return crops

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _pad_to_square(img: np.ndarray, size: int = 32) -> np.ndarray:
        """
        Pad a character crop to a square and resize to (size × size).
        Keeps the character centred on a black background.
        """
        h, w = img.shape[:2]
        side  = max(h, w)

        # Create square black canvas
        square = np.zeros((side, side), dtype=np.uint8)
        y_off  = (side - h) // 2
        x_off  = (side - w) // 2
        square[y_off:y_off + h, x_off:x_off + w] = img

        resized = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
        return resized

    @staticmethod
    def visualise(plate_bgr: np.ndarray,
                  char_boxes: list[tuple]) -> np.ndarray:
        """Draw bounding boxes on a colour plate crop (for debugging)."""
        vis = plate_bgr.copy()
        for (x, y, w, h) in char_boxes:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
        return vis
