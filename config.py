"""
ALPR System Configuration
All tunable parameters live here so nothing is hard-coded elsewhere.
"""

# ── Paths ──────────────────────────────────────────────────────────────────
WHITELIST_PATH   = "data/whitelist.csv"
LOG_DIR          = "data/logs"
MODEL_DIR        = "models"

# Saved model weights produced by the training scripts
DETECTOR_WEIGHTS    = "models/plate_detector.pth"
CLASSIFIER_WEIGHTS  = "models/char_classifier.pth"

# ── Plate Detector CNN ─────────────────────────────────────────────────────
# Input image size fed to the detector CNN (H, W)
DETECTOR_IMG_SIZE   = (64, 64)
DETECT_CONF         = 0.75      # Minimum detector confidence to keep a candidate

# Classical-CV candidate generation
CANNY_LOW           = 50
CANNY_HIGH          = 150
MIN_PLATE_AREA      = 800       # px² — lower to catch plates on phone screens
MAX_PLATE_AREA      = 100000   # higher to catch large close-up plates
ASPECT_RATIO_MIN    = 1.5       # looser lower bound (angled/phone-screen plates)
ASPECT_RATIO_MAX    = 6.0
MIN_PLATE_BRIGHTNESS = 60       # mean grayscale brightness — filters dark brick/wall contours
MIN_PLATE_SOLIDITY  = 0.70      # contour area / convex hull area — rejects fingers/cables

# ── Character Classifier CNN ───────────────────────────────────────────────
CHAR_IMG_SIZE       = (32, 32)  # Input size for the character CNN (H, W)
# 35 classes: digits 0-9 then letters A-Z (excluding O — mapped to 0 on plates)
CLASSES             = list("0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ")
NUM_CLASSES         = len(CLASSES)  # 35

# OCR / character segmentation
MIN_CHAR_AREA       = 80        # px² — blobs smaller than this are noise
MIN_CHAR_HEIGHT     = 10        # px
MAX_CHAR_HEIGHT_RATIO = 0.95    # char height / plate height — full-height blobs are borders
CHAR_CONF_THRESHOLD = 0.40      # Per-character confidence to include in reading
MIN_PLATE_CHARS     = 5         # Minimum characters for a valid plate reading (US plates are 5-7)
MAX_PLATE_CHARS     = 8         # Maximum — more than this is likely a false positive
MIN_UNIQUE_CHARS    = 3         # Reject readings whose characters are all the same (e.g. IIIIII)

# ── Multi-frame Voting ─────────────────────────────────────────────────────
VOTE_WINDOW         = 7         # Accumulate this many frames before deciding
VOTE_MIN_HITS       = 3         # Plate string must appear ≥ this many times (was 2)
VOTE_CONF_THRESHOLD = 0.55      # Aggregated confidence → ALLOWED (EasyOCR scores differ)

# ── Access Control ─────────────────────────────────────────────────────────
UNCERTAIN_THRESHOLD = 0.40      # Below this → UNCERTAIN (not DENIED)

# ── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE          = 32
EPOCHS_DETECTOR     = 20
EPOCHS_CLASSIFIER   = 30
LEARNING_RATE       = 1e-3
TRAIN_SPLIT         = 0.80      # 80 % train, 20 % validation

# ── Display ────────────────────────────────────────────────────────────────
SHOW_WINDOW         = True
FRAME_SKIP          = 2         # Process every Nth frame (1 = every frame)
BBOX_SMOOTH_ALPHA   = 0.5       # EMA weight for bbox smoothing (0=frozen, 1=raw)

# ── Gatekeeper Mode ───────────────────────────────────────────────────────
GATEKEEPER_MODE       = True        # Enable interactive registration prompts
DECISION_DISPLAY_SECS = 3.0         # How long to show ALLOWED/DENIED before resuming
CONFIRM_DISPLAY_SECS  = 3.0         # How long to show registration confirmation
DEFAULT_OWNER         = "Gate Registration"
DEFAULT_NOTES         = "Registered at gate"
