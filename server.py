"""
ALPR Web Server
===============
Serves a phone-facing camera page and runs the ALPR pipeline on each
frame received via WebSocket.

Usage
-----
    python server.py

Then open the printed URL on your phone (same WiFi network required).
Press Y / N on screen to register or deny unrecognised plates.
"""

import base64
import socket as _socket

import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

import config
from src.detector       import PlateDetector
from src.ocr_reader     import OCRReader
from src.voter          import FrameVoter
from src.access_control import AccessController, Decision
from src.logger         import EventLogger


# ── App setup ──────────────────────────────────────────────────────────────

app    = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


# ── Pipeline (initialised once at startup) ─────────────────────────────────

print("Initialising ALPR pipeline…")
detector   = PlateDetector()
ocr        = OCRReader(detector=detector)
controller = AccessController()
logger     = EventLogger()
print("Ready.\n")

# Per-connection state: voter keyed by socket session id
_voters: dict[str, FrameVoter] = {}
_busy:   dict[str, bool]       = {}   # drop frame if previous is still processing


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Socket events ──────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    from flask import request
    sid = request.sid
    _voters[sid] = FrameVoter()
    _busy[sid]   = False


@socketio.on("disconnect")
def on_disconnect():
    from flask import request
    sid = request.sid
    _voters.pop(sid, None)
    _busy.pop(sid, None)


@socketio.on("frame")
def on_frame(data):
    from flask import request
    sid = request.sid

    if _busy.get(sid):
        return                          # still processing previous frame

    _busy[sid] = True
    try:
        _process_frame(sid, data)
    finally:
        _busy[sid] = False


def _process_frame(sid: str, data: dict) -> None:
    voter = _voters.get(sid)
    if voter is None:
        return

    # Decode base64 JPEG from browser
    raw    = data.get("image", "")
    if "," in raw:
        raw = raw.split(",", 1)[1]
    img_bytes = base64.b64decode(raw)
    nparr     = np.frombuffer(img_bytes, np.uint8)
    frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return

    # Stage 1 — detect plate
    detections = detector.detect(frame)
    if not detections:
        emit("result", {"status": "scanning"})
        return

    best = detections[0]
    bbox = best["bbox"]
    crop = best["crop"]

    # Stage 2 — read plate text
    plate_text, ocr_conf = ocr.read(crop)

    result: dict = {
        "status": "detected",
        "bbox":   list(bbox),
        "plate":  plate_text,
        "conf":   round(ocr_conf, 3),
    }

    # Stage 3 — multi-frame vote
    voted = voter.update(plate_text, ocr_conf)
    if voted:
        voted_plate, voted_conf = voted
        decision, info = controller.check(voted_plate, voted_conf)

        logger.log(
            plate      = voted_plate,
            confidence = voted_conf,
            decision   = decision.value,
            owner      = info.get("owner", ""),
            notes      = info.get("notes", ""),
        )

        voter.reset()

        result.update({
            "voted_plate": voted_plate,
            "voted_conf":  round(voted_conf, 3),
            "decision":    decision.value,
            "owner":       info.get("owner", ""),
        })

    emit("result", result)


@socketio.on("register")
def on_register(data):
    plate = data.get("plate", "").strip().upper()
    conf  = data.get("conf", 0.0)
    if not plate:
        return

    controller.add_plate(
        plate,
        owner = config.DEFAULT_OWNER,
        notes = config.DEFAULT_NOTES,
    )
    logger.log(
        plate      = plate,
        confidence = conf,
        decision   = "REGISTERED",
        owner      = config.DEFAULT_OWNER,
        notes      = "Registered via web app",
    )
    emit("registered", {"plate": plate})


# ── Entry point ─────────────────────────────────────────────────────────────

def _local_ip() -> str:
    try:
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    ip   = _local_ip()
    port = args.port
    print(f"╔══════════════════════════════════════╗")
    print(f"║  ALPR Web Server                     ║")
    print(f"║  Open on your phone (same WiFi):     ║")
    print(f"║  https://{ip}:{port:<18}║")
    print(f"║  (accept the certificate warning)    ║")
    print(f"╚══════════════════════════════════════╝\n")
    socketio.run(app, host="0.0.0.0", port=port, debug=False,
                 ssl_context="adhoc")
