"""
Plate Detector
==============
Two-stage pipeline:
  1. Classical CV  — Canny edges + contour filtering to generate candidate
                     bounding boxes that look like license plates.
  2. Custom CNN    — A lightweight binary classifier trained on plate / no-plate
                     patches confirms or rejects each candidate.

No pre-trained weights from YOLO, Hugging Face, etc. are used.
The CNN weights are produced by train/train_detector.py.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

import config


# ── CNN Architecture ───────────────────────────────────────────────────────

class PlateCNN(nn.Module):
    """
    Small binary CNN: outputs P(is_license_plate) for a fixed-size patch.

    Architecture:
        Conv(3→16) → ReLU → MaxPool
        Conv(16→32) → ReLU → MaxPool
        Conv(32→64) → ReLU → AdaptiveAvgPool(4×4)
        Flatten → FC(1024→128) → ReLU → Dropout → FC(128→1) → Sigmoid
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                     # 32×32 → 16×16

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                     # 16×16 → 8×8

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),            # → 4×4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ── Detector class ─────────────────────────────────────────────────────────

class PlateDetector:
    """
    Locates license plate regions in a BGR frame.

    Usage:
        detector = PlateDetector()          # loads trained weights
        detections = detector.detect(frame) # list of dicts
    """

    def __init__(self, weights_path: str = config.DETECTOR_WEIGHTS):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = PlateCNN().to(self.device)

        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[Detector] Loaded weights from {weights_path}")
        else:
            print(f"[Detector] WARNING: no weights found at {weights_path}. "
                  "Run train/train_detector.py first.")

        self.model.eval()

        h, w = config.DETECTOR_IMG_SIZE
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((h, w)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    # ── Public API ─────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Returns a list of plate detections, each a dict:
            {
                "bbox": (x1, y1, x2, y2),
                "conf": float,              # CNN confidence (0-1)
                "crop": np.ndarray          # BGR crop of the plate region
            }
        """
        candidates = self._generate_candidates(frame)
        detections  = []

        for (x1, y1, x2, y2) in candidates:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            conf = self._score_candidate(crop)
            if conf >= config.DETECT_CONF:
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                    "crop": crop.copy(),
                })

        # Keep the highest-confidence detection per frame
        if len(detections) > 1:
            detections = [max(detections, key=lambda d: d["conf"])]

        return detections

    # ── Private helpers ────────────────────────────────────────────────────

    def _generate_candidates(self, frame: np.ndarray) -> list[tuple]:
        """
        Classical-CV stage: return a list of (x1,y1,x2,y2) bounding boxes
        that have the shape of a license plate.
        """
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, config.CANNY_LOW, config.CANNY_HIGH)

        # Dilate to connect broken edges
        kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated  = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        h_frame, w_frame = frame.shape[:2]
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (config.MIN_PLATE_AREA <= area <= config.MAX_PLATE_AREA):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0:
                continue

            aspect = w / h
            if not (config.ASPECT_RATIO_MIN <= aspect <= config.ASPECT_RATIO_MAX):
                continue

            # Reject non-rectangular blobs (fingers, cables, irregular shapes).
            # Solidity = contour_area / convex_hull_area — plates are convex.
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            if hull_area > 0 and (area / hull_area) < config.MIN_PLATE_SOLIDITY:
                continue

            # Reject dark regions (brick walls, shadows) — license plates are bright
            region = gray[y:y + h, x:x + w]
            if region.size > 0 and region.mean() < config.MIN_PLATE_BRIGHTNESS:
                continue

            # Add a small padding so the CNN sees context around the plate
            pad = 4
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_frame, x + w + pad)
            y2 = min(h_frame, y + h + pad)
            candidates.append((x1, y1, x2, y2))

        return candidates

    def _score_candidate(self, crop: np.ndarray) -> float:
        """Run the CNN on a single BGR crop and return P(is_plate)."""
        tensor = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = self.model(tensor).item()
        return prob

    # ── Preprocessing helper (used by training & OCR pipeline) ────────────

    @staticmethod
    def preprocess_crop(crop: np.ndarray) -> np.ndarray:
        """
        Preprocess a detected plate crop for better character segmentation:
        upscale → grayscale → adaptive threshold → slight dilation.
        """
        if crop is None or crop.size == 0:
            return crop

        h, w = crop.shape[:2]
        if w < 200:
            scale = 200 / w
            crop = cv2.resize(crop,
                              (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.dilate(thresh, kernel, iterations=1)
        return processed
