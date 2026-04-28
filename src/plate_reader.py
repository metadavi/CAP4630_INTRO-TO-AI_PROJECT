"""
Plate Reader — CRNN + CTC
==========================
Custom sequence recognition model that reads text directly from a plate crop.
Replaces EasyOCR entirely — this is a fully trained-from-scratch model.

Architecture
------------
  Plate crop (grayscale 32×128)
    → CNN backbone      (4 conv blocks, extracts visual features)
    → Adaptive pool     (collapses height to 1, keeps 32 time-steps)
    → 2-layer BiLSTM    (reads sequence left→right and right→left)
    → FC + LogSoftmax   (per-timestep class probabilities)
    → CTC greedy decode (collapses to final plate string)

Charset: 0-9 A-Z  (36 chars) + blank token = 37 classes total
"""

import os
import re

import cv2
import numpy as np
import torch
import torch.nn as nn

import config

# ── Charset ────────────────────────────────────────────────────────────────
CHARS   = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # 36 chars
BLANK   = len(CHARS)                                # index 36 = CTC blank
N_CLASS = len(CHARS) + 1                           # 37

CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
IDX2CHAR = {i: c for i, c in enumerate(CHARS)}

# Input size fed to the CNN
IMG_H = 32
IMG_W = 128

_ALNUM = re.compile(r'[^A-Z0-9]')


# ── Model ──────────────────────────────────────────────────────────────────

class PlateCRNN(nn.Module):
    """
    CNN + BiLSTM sequence model for licence-plate text recognition.

    Input : (B, 1, 32, 128)  — normalised grayscale plate crop
    Output: (32, B, N_CLASS) — log-probabilities per timestep
    """

    def __init__(self, n_class: int = N_CLASS):
        super().__init__()

        # ── CNN backbone ──────────────────────────────────────────────────
        # Each block: Conv → BN → ReLU → (optional) MaxPool
        # Height is halved at blocks 1-4 until AdaptivePool collapses it.
        # Width stays at 128 → 64 → 32, giving 32 time-steps for the RNN.
        self.cnn = nn.Sequential(
            # Block 1: 1×32×128 → 32×16×64
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # Block 2: 32×16×64 → 64×8×32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # Block 3: 64×8×32 → 128×4×32  (pool height only)
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d((2, 1)),

            # Block 4: 128×4×32 → 256×2×32  (pool height only)
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1)),

            # Collapse height to 1: 256×2×32 → 256×1×32
            nn.AdaptiveAvgPool2d((1, None)),
        )

        # ── Recurrent layers ──────────────────────────────────────────────
        # Input per timestep = 256 features (from CNN)
        # BiLSTM → 128 hidden × 2 directions = 256 output features
        self.rnn = nn.LSTM(
            input_size  = 256,
            hidden_size = 128,
            num_layers  = 2,
            batch_first = False,
            bidirectional = True,
            dropout = 0.3,
        )

        # ── Output projection ─────────────────────────────────────────────
        self.fc = nn.Linear(256, n_class)   # 256 = 128*2 (bidirectional)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, H, W)
        returns: (T, B, C) log-softmax scores  where T = W/4 = 32
        """
        # CNN: (B,1,32,128) → (B,256,1,32)
        feat = self.cnn(x)

        # Squeeze height, permute to (T, B, C): (B,256,1,32) → (32,B,256)
        feat = feat.squeeze(2)          # (B, 256, 32)
        feat = feat.permute(2, 0, 1)    # (32, B, 256)

        # RNN: (32,B,256) → (32,B,256)
        out, _ = self.rnn(feat)

        # FC + log-softmax: (32,B,256) → (32,B,N_CLASS)
        out = self.fc(out)
        return torch.log_softmax(out, dim=2)


# ── Codec helpers ──────────────────────────────────────────────────────────

def encode(text: str) -> list[int]:
    """Convert plate string to list of class indices."""
    return [CHAR2IDX[c] for c in text.upper() if c in CHAR2IDX]


def greedy_decode(log_probs: torch.Tensor) -> str:
    """
    CTC greedy decode: argmax → collapse repeats → remove blanks.

    log_probs: (T, N_CLASS) for a single sample
    """
    indices = log_probs.argmax(dim=1).tolist()   # (T,)

    # Collapse consecutive duplicates
    collapsed = [indices[0]]
    for idx in indices[1:]:
        if idx != collapsed[-1]:
            collapsed.append(idx)

    # Remove blank tokens
    chars = [IDX2CHAR[i] for i in collapsed if i != BLANK]
    return "".join(chars)


# ── Inference wrapper ──────────────────────────────────────────────────────

class PlateReader:
    """
    Drop-in replacement for the EasyOCR path in OCRReader.

    Usage
    -----
        reader = PlateReader()
        text, conf = reader.read(plate_crop_bgr)
    """

    def __init__(self, weights_path: str = None):
        if weights_path is None:
            weights_path = os.path.join(config.MODEL_DIR, "plate_crnn.pth")
        self.weights_path = weights_path

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        self._model = PlateCRNN().to(self._device)

        if os.path.exists(weights_path):
            state = torch.load(weights_path,
                               map_location=self._device,
                               weights_only=True)
            self._model.load_state_dict(state)
            print(f"[PlateReader] Loaded weights from {weights_path}")
        else:
            print(f"[PlateReader] WARNING: no weights at {weights_path} — "
                  "run train/train_crnn.py first")

        self._model.eval()

    # ── Public API ──────────────────────────────────────────────────────

    def read(self, plate_crop_bgr: np.ndarray) -> tuple[str, float]:
        """
        Read plate text from a BGR plate crop.

        Returns (text, confidence) — e.g. ('ABC123', 0.91)
        Returns ('', 0.0) on failure.
        """
        if plate_crop_bgr is None or plate_crop_bgr.size == 0:
            return "", 0.0

        tensor = self._preprocess(plate_crop_bgr)   # (1,1,32,128)

        with torch.no_grad():
            log_probs = self._model(tensor)          # (32,1,37)
            log_probs_cpu = log_probs.squeeze(1).cpu()  # (32,37)

        text = greedy_decode(log_probs_cpu)
        text = _ALNUM.sub('', text.upper())

        if not (config.MIN_PLATE_CHARS <= len(text) <= config.MAX_PLATE_CHARS):
            return "", 0.0
        if len(set(text)) < config.MIN_UNIQUE_CHARS:
            return "", 0.0

        # Confidence = mean max-probability across timesteps
        probs = log_probs_cpu.exp()
        conf  = float(probs.max(dim=1).values.mean())

        return text, conf

    # ── Private ─────────────────────────────────────────────────────────

    def _preprocess(self, bgr: np.ndarray) -> torch.Tensor:
        """BGR crop → normalised (1,1,32,128) tensor on device."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Upscale small crops before CLAHE so character edges are sharp.
        # Distant/small plates arrive as tiny crops (e.g. 30×90px) — resizing
        # directly to 32×128 loses edge detail. Upscaling to at least 64×200
        # first preserves stroke sharpness before the final resize.
        h, w = gray.shape[:2]
        if w < 200 or h < 64:
            scale = max(200 / w, 64 / h)
            gray  = cv2.resize(gray,
                               (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_CUBIC)

        # CLAHE for contrast normalisation (handles green FL text)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray  = clahe.apply(gray)

        gray  = cv2.resize(gray, (IMG_W, IMG_H),
                           interpolation=cv2.INTER_CUBIC)
        arr   = gray.astype(np.float32) / 255.0
        arr   = (arr - 0.5) / 0.5          # normalise to [-1, 1]

        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,32,128)
        return tensor.to(self._device)
