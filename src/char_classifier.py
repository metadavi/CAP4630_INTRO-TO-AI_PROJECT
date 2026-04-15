"""
Character Classifier
====================
Custom CNN that maps a 32×32 grayscale character image to one of 36 classes
(digits 0-9, letters A-Z).

Architecture
------------
    Block 1: Conv(1→32, 3×3) → BN → ReLU → MaxPool(2)   → 16×16
    Block 2: Conv(32→64, 3×3) → BN → ReLU → MaxPool(2)  → 8×8
    Block 3: Conv(64→128, 3×3) → BN → ReLU → AvgPool(4) → 2×2
    Classifier: Flatten → FC(512→256) → ReLU → Dropout(0.5) → FC(256→36)

No pre-trained weights used — the model is trained from scratch on a
Kaggle character-recognition dataset via train/train_classifier.py.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

import config


# ── Model definition ───────────────────────────────────────────────────────

class CharCNN(nn.Module):
    """
    Lightweight CNN for 36-class alphanumeric character recognition.
    Input:  (N, 1, 32, 32)  — single-channel (grayscale)
    Output: (N, 36)          — raw logits
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 32 → 16
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 16 → 8
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),  # → 2×2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


# ── Inference wrapper ──────────────────────────────────────────────────────

class CharClassifier:
    """
    Wraps CharCNN with:
      - weight loading
      - pre-processing transforms
      - per-character inference returning (label, confidence)
    """

    def __init__(self, weights_path: str = config.CLASSIFIER_WEIGHTS):
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model   = CharCNN().to(self.device)
        self.classes = config.CLASSES

        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[CharClassifier] Loaded weights from {weights_path}")
        else:
            print(f"[CharClassifier] WARNING: no weights at {weights_path}. "
                  "Run train/train_classifier.py first.")

        self.model.eval()

        h, w = config.CHAR_IMG_SIZE
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((h, w)),
            T.ToTensor(),                        # [0,1] single channel
            T.Normalize(mean=[0.5], std=[0.5]),  # → [-1, 1]
        ])

    def predict(self, char_crop: np.ndarray) -> tuple[str, float]:
        """
        Classify a single character crop.

        Parameters
        ----------
        char_crop : np.ndarray
            Grayscale (H×W) or BGR (H×W×3) crop of one character.

        Returns
        -------
        (label, confidence) : (str, float)
            label      — one of CLASSES (e.g. 'A', '3')
            confidence — softmax probability of the top prediction
        """
        if char_crop is None or char_crop.size == 0:
            return "", 0.0

        tensor = self.transform(char_crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)             # (1, 36)
            probs  = F.softmax(logits, dim=1)       # (1, 36)
            top_p, top_idx = probs.max(dim=1)

        label = self.classes[top_idx.item()]
        conf  = top_p.item()
        return label, conf

    def predict_batch(self,
                      crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Classify a list of character crops in one forward pass."""
        if not crops:
            return []

        tensors = torch.stack(
            [self.transform(c) for c in crops]
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(tensors)
            probs  = F.softmax(logits, dim=1)
            top_p, top_idx = probs.max(dim=1)

        return [
            (self.classes[idx.item()], p.item())
            for idx, p in zip(top_idx, top_p)
        ]
