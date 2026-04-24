"""
ALPR — AI Vehicle Access Control System
========================================
Entry point supporting three input modes:

    webcam  — live feed from a connected camera (default)
    video   — process a recorded video file
    image   — process a single image file

Usage examples:
    python main.py                          # webcam mode (gatekeeper on)
    python main.py --no-gatekeeper          # webcam without registration prompts
    python main.py --mode video  --input path/to/video.mp4
    python main.py --mode image  --input path/to/plate.jpg
    python main.py --mode webcam --camera 1   # use camera index 1

Press 'q' to quit the preview window.
Press 'r' to manually reset the multi-frame voter buffer.

Gatekeeper mode (default):
    When an unrecognised plate is scanned, the operator is prompted:
        Y = register the plate    N = deny access

Pipeline (per frame):
    1. Detect plate region  → PlateDetector  (classical CV + custom CNN)
    2. Read plate text      → OCRReader      (CharSegmenter + CharCNN)
    3. Accumulate readings  → FrameVoter     (multi-frame voting)
    4. Decide access        → AccessController (whitelist + confidence)
    5. Log event            → EventLogger    (CSV)
    6. Overlay result       → OpenCV display
"""

import argparse
import os
import sys
import time
from enum import Enum

import cv2
import numpy as np

import config
from src.detector       import PlateDetector
from src.ocr_reader     import OCRReader
from src.voter          import FrameVoter
from src.access_control import AccessController, Decision
from src.logger         import EventLogger


# ── Colour palette ─────────────────────────────────────────────────────────

COLOURS = {
    Decision.ALLOWED:   (0,   200,  0),     # green
    Decision.DENIED:    (0,    0,  220),    # red
    Decision.UNCERTAIN: (0,  165,  255),    # orange
}

LABELS = {
    Decision.ALLOWED:   "ALLOWED",
    Decision.DENIED:    "DENIED",
    Decision.UNCERTAIN: "UNCERTAIN",
}


# ── Gatekeeper state machine ──────────────────────────────────────────────

class GatekeeperState(Enum):
    SCANNING           = "SCANNING"
    PROMPT_REGISTER    = "PROMPT_REGISTER"
    CONFIRM_REGISTERED = "CONFIRM_REGISTERED"
    DECISION_SHOWN     = "DECISION_SHOWN"


class GateState:
    """Tracks gatekeeper interaction state within the frame loop."""

    def __init__(self):
        self.reset_to_scanning()

    def enter(self, state: GatekeeperState, **kwargs):
        self.state = state
        self.state_entered_at = time.time()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def elapsed(self) -> float:
        return time.time() - self.state_entered_at

    def reset_to_scanning(self):
        self.state            = GatekeeperState.SCANNING
        self.pending_plate    = ""
        self.pending_conf     = 0.0
        self.pending_decision = None
        self.pending_info     = {}
        self.state_entered_at = 0.0


# ── Overlay helpers ─────────────────────────────────────────────────────────

def _draw_detection(frame: np.ndarray,
                    bbox:  tuple,
                    plate: str,
                    conf:  float,
                    decision: Decision | None) -> None:
    """Draw bounding box + plate text + decision badge on the frame."""
    x1, y1, x2, y2 = bbox
    colour = COLOURS.get(decision, (200, 200, 200))

    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    label = f"{plate}  {conf:.0%}"
    if decision:
        label += f"  [{LABELS[decision]}]"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), colour, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def _draw_hud(frame: np.ndarray,
              fps: float,
              frame_idx: int,
              voter_buf: int,
              gate_state: GatekeeperState = GatekeeperState.SCANNING) -> None:
    """Draw a small heads-up display in the top-left corner."""
    if gate_state == GatekeeperState.PROMPT_REGISTER:
        keys_line = "Y=register  N=deny  Q=quit"
    else:
        keys_line = "Q=quit  R=reset voter"

    lines = [
        f"FPS: {fps:.1f}",
        f"Frame: {frame_idx}",
        f"Voter buffer: {voter_buf}/{config.VOTE_WINDOW}",
        keys_line,
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (8, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
                    cv2.LINE_AA)


def _draw_gatekeeper_overlay(frame: np.ndarray, gate: GateState) -> None:
    """Draw gatekeeper prompt / confirmation overlays on the frame."""
    h, w = frame.shape[:2]

    if gate.state == GatekeeperState.PROMPT_REGISTER:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 120), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        plate = gate.pending_plate
        cv2.putText(frame, f"Unrecognized plate: {plate}",
                    (20, h - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Register this plate?  Y = Yes   N = No",
                    (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)

    elif gate.state == GatekeeperState.CONFIRM_REGISTERED:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 90), (w, h), (0, 120, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, f"REGISTERED: {gate.pending_plate}",
                    (20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        remaining = config.CONFIRM_DISPLAY_SECS - gate.elapsed()
        cv2.putText(frame, f"Resuming in {max(0, remaining):.0f}s...",
                    (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)

    elif gate.state == GatekeeperState.DECISION_SHOWN:
        decision = gate.pending_decision
        colour = COLOURS.get(decision, (200, 200, 200))
        label  = LABELS.get(decision, "")

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 90), (w, h), colour, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, f"{label}: {gate.pending_plate}",
                    (20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        remaining = config.DECISION_DISPLAY_SECS - gate.elapsed()
        cv2.putText(frame, f"Resuming in {max(0, remaining):.0f}s...",
                    (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)


def _show_frame(frame, bbox, gate, fps, frame_idx, buf,
                live_plate: str = "", live_conf: float = 0.0):
    """Render the display frame with detection, HUD, and gatekeeper overlays."""
    if not config.SHOW_WINDOW:
        return
    display = frame.copy()

    if bbox:
        if gate.state == GatekeeperState.SCANNING:
            # Show whatever the OCR is currently reading in real-time
            _draw_detection(display, bbox, live_plate, live_conf, None)
        elif gate.state == GatekeeperState.PROMPT_REGISTER:
            _draw_detection(display, bbox, gate.pending_plate,
                            gate.pending_conf, Decision.DENIED)
        else:
            _draw_detection(display, bbox, gate.pending_plate,
                            gate.pending_conf, gate.pending_decision)

    _draw_hud(display, fps, frame_idx, buf, gate.state)

    if gate.state != GatekeeperState.SCANNING:
        _draw_gatekeeper_overlay(display, gate)

    cv2.imshow("ALPR — Vehicle Access Control", display)


# ── Core pipeline ───────────────────────────────────────────────────────────

def run_pipeline(source,
                 detector:    PlateDetector,
                 ocr:         OCRReader,
                 voter:       FrameVoter,
                 controller:  AccessController,
                 logger:      EventLogger) -> None:
    """
    Main loop: read frames from `source` (cv2.VideoCapture or a single image
    wrapped as an iterator), run the ALPR pipeline, display results.
    """
    frame_idx       = 0
    last_bbox       = None
    _smooth_bbox    = None    # exponential-moving-average box (reduces jitter)
    last_plate_txt  = ""      # most recent OCR reading (shown live on bbox)
    last_plate_conf = 0.0
    fps_t           = time.time()
    fps             = 0.0
    gate            = GateState()

    while True:
        ret, frame = source.read()
        if not ret:
            break

        frame_idx += 1

        # FPS counter
        now = time.time()
        if now - fps_t >= 1.0:
            fps   = frame_idx / (now - fps_t + 1e-6)
            fps_t = now

        # ── Auto-transitions for timed states ─────────────────────────
        if gate.state == GatekeeperState.CONFIRM_REGISTERED:
            if gate.elapsed() > config.CONFIRM_DISPLAY_SECS:
                gate.reset_to_scanning()

        if gate.state == GatekeeperState.DECISION_SHOWN:
            if gate.elapsed() > config.DECISION_DISPLAY_SECS:
                gate.reset_to_scanning()

        # ── Pipeline runs ONLY in SCANNING state ──────────────────────
        if gate.state == GatekeeperState.SCANNING:

            # Frame skip logic
            if frame_idx % config.FRAME_SKIP != 0:
                _show_frame(frame, last_bbox, gate, fps, frame_idx,
                            voter.buffer_size, last_plate_txt, last_plate_conf)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    voter.reset()
                continue

            # Stage 1: Detect plate
            detections = detector.detect(frame)

            if not detections:
                _smooth_bbox = None   # reset smoothing when plate leaves frame
                _show_frame(frame, None, gate, fps, frame_idx,
                            voter.buffer_size, last_plate_txt, last_plate_conf)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    voter.reset()
                continue

            best = detections[0]
            raw_bbox = best["bbox"]

            # Smooth bbox position across frames to remove jitter.
            # alpha=0.5: new pos pulls box halfway toward detected pos each frame.
            alpha = config.BBOX_SMOOTH_ALPHA
            if _smooth_bbox is None:
                _smooth_bbox = raw_bbox
            else:
                _smooth_bbox = tuple(
                    int(alpha * n + (1 - alpha) * p)
                    for n, p in zip(raw_bbox, _smooth_bbox)
                )
            bbox = _smooth_bbox
            crop = best["crop"]

            # Stage 2: Read plate text
            plate_text, ocr_conf = ocr.read(crop)

            # Update live display reading
            if plate_text:
                last_plate_txt  = plate_text
                last_plate_conf = ocr_conf

            # Stage 3: Multi-frame vote
            voted = voter.update(plate_text, ocr_conf)

            if voted:
                voted_plate, voted_conf = voted

                # Stage 4: Access decision
                decision, info = controller.check(voted_plate, voted_conf)

                # Stage 5: Log
                logger.log(
                    plate      = voted_plate,
                    confidence = voted_conf,
                    decision   = decision.value,
                    owner      = info.get("owner", ""),
                    notes      = info.get("notes", ""),
                )

                voter.reset()

                # Gatekeeper state transitions
                if config.GATEKEEPER_MODE and decision == Decision.DENIED:
                    gate.enter(GatekeeperState.PROMPT_REGISTER,
                               pending_plate=voted_plate,
                               pending_conf=voted_conf,
                               pending_decision=decision,
                               pending_info=info)
                else:
                    gate.enter(GatekeeperState.DECISION_SHOWN,
                               pending_plate=voted_plate,
                               pending_conf=voted_conf,
                               pending_decision=decision,
                               pending_info=info)

            last_bbox = bbox

        # ── Display (all states) ──────────────────────────────────────
        _show_frame(frame, last_bbox, gate, fps, frame_idx,
                    voter.buffer_size, last_plate_txt, last_plate_conf)

        # ── Key handling (state-dependent) ────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if gate.state == GatekeeperState.SCANNING:
            if key == ord("r"):
                voter.reset()
                print("[main] Voter buffer reset manually.")

        elif gate.state == GatekeeperState.PROMPT_REGISTER:
            if key in (ord("y"), ord("Y")):
                controller.add_plate(
                    gate.pending_plate,
                    owner=config.DEFAULT_OWNER,
                    notes=config.DEFAULT_NOTES,
                )
                print(f"[main] Registered plate: {gate.pending_plate}")
                logger.log(
                    plate      = gate.pending_plate,
                    confidence = gate.pending_conf,
                    decision   = "REGISTERED",
                    owner      = config.DEFAULT_OWNER,
                    notes      = "Registered at gate by operator",
                )
                gate.enter(GatekeeperState.CONFIRM_REGISTERED,
                           pending_plate=gate.pending_plate,
                           pending_conf=gate.pending_conf,
                           pending_decision=Decision.ALLOWED,
                           pending_info=gate.pending_info)

            elif key in (ord("n"), ord("N")):
                print(f"[main] Access denied for: {gate.pending_plate}")
                gate.enter(GatekeeperState.DECISION_SHOWN,
                           pending_plate=gate.pending_plate,
                           pending_conf=gate.pending_conf,
                           pending_decision=Decision.DENIED,
                           pending_info=gate.pending_info)

    cv2.destroyAllWindows()


# ── Image-as-video-source adapter ──────────────────────────────────────────

class SingleImageSource:
    """
    Wraps a single image so it works with the same read() interface.
    The image is returned VOTE_WINDOW * 3 times so the multi-frame voter has
    enough readings to reach a decision (voter needs VOTE_WINDOW full frames).
    """
    def __init__(self, path: str):
        self._img   = cv2.imread(path)
        self._count = 0
        self._max   = config.VOTE_WINDOW * 3
        if self._img is None:
            print(f"ERROR: cannot read image at {path}")
            sys.exit(1)

    def read(self):
        if self._count >= self._max:
            return False, None
        self._count += 1
        return True, self._img.copy()

    def release(self):
        pass


# ── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ALPR — AI Vehicle Access Control System"
    )
    parser.add_argument("--mode",   choices=["webcam", "video", "image"],
                        default="webcam")
    parser.add_argument("--input",  type=str, default=None,
                        help="Path to video or image file (video/image modes)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index for webcam mode (default: 0)")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable the OpenCV preview window")
    parser.add_argument("--gatekeeper", action="store_true", default=True,
                        help="Enable interactive gatekeeper mode (default: on)")
    parser.add_argument("--no-gatekeeper", dest="gatekeeper",
                        action="store_false",
                        help="Disable gatekeeper mode (original behavior)")
    args = parser.parse_args()

    if args.no_display:
        config.SHOW_WINDOW = False
    config.GATEKEEPER_MODE = args.gatekeeper

    # ── Validate inputs ────────────────────────────────────────────────────
    if args.mode in ("video", "image") and not args.input:
        parser.error(f"--input is required for mode '{args.mode}'")

    if args.mode in ("video", "image") and not os.path.exists(args.input):
        print(f"ERROR: file not found: {args.input}")
        sys.exit(1)

    # ── Initialise components ──────────────────────────────────────────────
    print("Initialising ALPR pipeline …")
    detector   = PlateDetector()
    ocr        = OCRReader(detector=detector)
    voter      = FrameVoter()
    controller = AccessController()
    logger     = EventLogger()
    mode_label = "gatekeeper" if config.GATEKEEPER_MODE else "standard"
    print(f"Ready. Mode: {mode_label}\n")

    # ── Open source ────────────────────────────────────────────────────────
    if args.mode == "webcam":
        source = cv2.VideoCapture(args.camera)
        if not source.isOpened():
            print(f"ERROR: cannot open camera {args.camera}")
            sys.exit(1)
        print(f"Streaming from camera {args.camera}. Press Q to quit.")
    elif args.mode == "video":
        source = cv2.VideoCapture(args.input)
        if not source.isOpened():
            print(f"ERROR: cannot open video {args.input}")
            sys.exit(1)
        print(f"Processing video: {args.input}")
    else:  # image
        source = SingleImageSource(args.input)
        print(f"Processing image: {args.input}")
        config.SHOW_WINDOW = True  # always show for a single image

    # ── Run ────────────────────────────────────────────────────────────────
    try:
        run_pipeline(source, detector, ocr, voter, controller, logger)
    finally:
        if hasattr(source, "release"):
            source.release()


if __name__ == "__main__":
    main()
