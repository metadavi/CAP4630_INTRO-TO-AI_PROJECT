# AI Vehicle Access Control — ALPR
**CAP4630 · Introduction to Artificial Intelligence**

An AI-powered gate-keeping system that reads Florida license plates in real time via a phone camera, decides whether to grant or deny access, and lets an operator register new plates on the spot.

---

## How It Works

1. Phone camera streams video to the server over WebSocket
2. A classical CV + CNN detector finds the plate bounding box
3. The plate crop is preprocessed (CLAHE, orange graphic masking) and fed to a custom **CRNN + CTC** model
4. Readings across multiple frames are fuzzy-clustered (Levenshtein) into a confident vote
5. The voted plate is checked against a whitelist CSV → **ALLOWED / DENIED / UNCERTAIN**
6. The operator can register or deny unknown plates live from the UI

---

## Tech Stack

| Component | Technology |
|---|---|
| AI Models | PyTorch (custom CNNs + CRNN + CTC) |
| Computer Vision | OpenCV + CLAHE contrast normalization |
| Web Server | Flask + Flask-SocketIO |
| Real-time Comms | WebSocket (Socket.IO) |
| Frontend | HTML5 + JavaScript (getUserMedia API) |
| Database | CSV (whitelist) + CSV (event logs) |
| Training Acceleration | Apple MPS (Metal) — M1/M2/M3 Macs |
| Primary OCR | Custom CRNN + CTC (trained from scratch) |
| OCR Fallback | EasyOCR → CharCNN segmentation pipeline |
| Multi-frame Voting | Levenshtein fuzzy clustering |
| Deployment | Google Cloud Run |
| Async Worker | Eventlet |

---

## Dataset Sources

| Dataset | Source | Used For |
|---|---|---|
| Car Plate Detection | [Kaggle — andrewmvd](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection) | Plate detector training |
| LP Character Recognition | [Kaggle — prerak23](https://www.kaggle.com/datasets/prerak23/license-plate-character-recognition) | Char classifier training |
| License Plate Recognition v11 | [Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e) — CC BY 4.0 | Plate detector training |
| Synthetic FL plates (~15,000) | Self-generated (`train/generate_florida_plates.py`) | CRNN training |
| Self-collected real photos (~502) | Neighborhood + parking garage (manually labeled) | CRNN training |

---

## Model Evaluation (Validation Set — 3,102 samples)

| Metric | Score |
|---|---|
| Character Accuracy | 88.04% |
| Exact-match Accuracy | 84.82% |
| Macro Precision | 86.39% |
| Macro Recall | 83.53% |
| Macro F1-Score | 84.88% |

![Evaluation Charts](models/eval_charts.png)

---

## Prerequisites

- **Python 3.10 or 3.11** (3.12+ not yet supported by all dependencies)
- **pip** (comes with Python)
- **Git**
- A Mac with Apple Silicon **or** a machine with a CUDA GPU (CPU works but training will be slow)
- A phone and computer on the **same Wi-Fi network** to use the live camera UI

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/metadavi/CAP4630_INTRO-TO-AI_PROJECT.git
cd CAP4630_INTRO-TO-AI_PROJECT
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3):** PyTorch will automatically use the Metal (MPS) backend — no extra steps needed.  
> **NVIDIA GPU:** Make sure your CUDA version matches the PyTorch wheel. See [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Verify the trained weights are present
```
models/
  plate_crnn.pth       ← CRNN OCR model  (required)
  plate_detector.pth   ← plate detector  (required)
  char_classifier.pth  ← char fallback   (optional)
```
These are committed to the repo. If they are missing, retrain — see **Training** below.

---

## Running the Server

```bash
python server.py
```

The terminal will print a local URL like:
```
https://192.168.x.x:8080
```

Open that URL on your phone (same Wi-Fi). Accept the self-signed certificate warning, then tap **Enable Camera** and point it at a license plate.

### Flags
| Flag | Default | Description |
|---|---|---|
| `--port` | `8080` | Port to listen on |
| `--no-ssl` | off | Disable HTTPS (use if behind a reverse proxy) |

---

## Project Structure

```
├── server.py                  # Flask + SocketIO entry point
├── config.py                  # All tunable parameters
├── src/
│   ├── detector.py            # Plate detector (CV + CNN)
│   ├── ocr_reader.py          # OCR orchestrator (CRNN → EasyOCR → CharCNN)
│   ├── plate_reader.py        # CRNN model definition + inference
│   ├── voter.py               # Multi-frame Levenshtein voter
│   ├── access_control.py      # Whitelist + decision logic
│   └── logger.py              # CSV event logger
├── train/
│   ├── train_crnn.py          # Train the CRNN OCR model
│   ├── train_detector.py      # Train the plate detector
│   ├── train_classifier.py    # Train the char classifier fallback
│   ├── generate_florida_plates.py  # Synthetic FL plate generator
│   ├── ingest_photos.py       # Label real photos into the dataset
│   ├── evaluate_crnn.py       # Print evaluation metrics
│   └── visualize_eval.py      # Generate eval_charts.png
├── models/
│   ├── plate_crnn.pth
│   ├── plate_detector.pth
│   ├── char_classifier.pth
│   └── eval_charts.png
├── data/
│   ├── whitelist.csv          # Authorized plates
│   └── logs/                  # Per-day event logs
└── templates/
    └── index.html             # Phone camera UI
```

---

## Training

> Only needed if you want to retrain from scratch or add more data.

### CRNN (primary OCR model)
```bash
python train/train_crnn.py --epochs 50 --lr 0.001
```

### Plate Detector
```bash
python train/train_detector.py
```

### Add real photos to the training set
```bash
python train/ingest_photos.py --photos /path/to/your/photos
```

---

## Configuration

All parameters are in `config.py`. Key ones:

| Parameter | Default | Description |
|---|---|---|
| `VOTE_WINDOW` | `6` | Frames to accumulate before deciding |
| `VOTE_FUZZY_DIST` | `3` | Max Levenshtein distance to cluster reads |
| `VOTE_COOLDOWN_SECS` | `6.0` | Pause after a decision before re-scanning |
| `VOTE_CONF_THRESHOLD` | `0.45` | Confidence floor for ALLOWED decision |
| `MIN_PLATE_CHARS` | `5` | Minimum characters for a valid plate |
| `MAX_PLATE_CHARS` | `7` | Maximum characters for a valid plate |
