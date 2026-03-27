"""
dashboard.py
────────────
Flask web dashboard for real-time PPE monitoring.

Endpoints
---------
GET  /                → main dashboard page
GET  /video_feed      → MJPEG live stream
GET  /api/stats       → JSON stats snapshot
GET  /api/violations  → recent violation log (JSON)
GET  /api/report      → session report (JSON)
POST /api/reset       → reset session stats
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, request

import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    MODEL_CONFIG, ENABLE_ZONES, HAZARD_ZONES,
    DASHBOARD_CONFIG, LOG_CONFIG,
)
from detection.model     import PPEModel
from detection.inference import InferencePipeline
from utils.tracker       import WorkerTracker
from utils.alerts        import AlertManager
from utils.report        import ReportManager

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared state accessed by background thread + Flask handlers
# ─────────────────────────────────────────────────────────────────────────────
_state = {
    "frame":           None,       # latest annotated frame (bytes)
    "stats":           {},         # latest stats snapshot
    "violations":      [],         # recent alerts list
    "running":         False,
    "lock":            threading.Lock(),
}


def _inference_thread(source, conf):
    """Background thread: read frames, run pipeline, update _state."""
    model    = PPEModel()
    pipeline = InferencePipeline(model=model, conf=conf)
    tracker  = WorkerTracker()
    alerts   = AlertManager()
    reporter = ReportManager()

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        logger.error("Cannot open source: %s", source)
        return

    zones      = HAZARD_ZONES if ENABLE_ZONES else None
    frame_no   = 0
    fps_buf    = []
    _state["running"] = True

    try:
        while _state["running"]:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
                continue

            frame_no += 1
            workers, _, _ = pipeline.process_frame(frame, zones=zones)
            tracked = tracker.update(workers)
            reporter.ingest_workers(tracked, frame_no)

            fps_buf.append(time.perf_counter() - t0)
            if len(fps_buf) > 30:
                fps_buf.pop(0)
            fps = len(fps_buf) / sum(fps_buf)

            annotated = alerts.draw_frame(frame, tracked, fps, frame_no)

            # Encode JPEG
            _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])

            # Build stats
            total      = len(tracked)
            viols      = sum(1 for tw in tracked if not tw.is_compliant)
            comp_pct   = ((total - viols) / total * 100) if total else 100.0

            recent_viols = []
            for tw in tracked:
                for vt in tw.current_violations:
                    recent_viols.append({
                        "worker_id": tw.track_id,
                        "type":      vt,
                        "severity":  tw.highest_severity,
                        "zone":      tw.zone_name or "—",
                        "duration":  round(
                            tw.active_violations[vt].duration, 1
                        ) if vt in tw.active_violations else 0,
                    })

            with _state["lock"]:
                _state["frame"]      = jpeg.tobytes()
                _state["stats"]      = {
                    "fps":           round(fps, 1),
                    "frame_no":      frame_no,
                    "total_workers": total,
                    "violations":    viols,
                    "compliance_pct": round(comp_pct, 1),
                }
                _state["violations"] = recent_viols

    finally:
        cap.release()
        _state["running"] = False


# ─────────────────────────────────────────────────────────────────────────────
def create_app(source="0", conf=MODEL_CONFIG["confidence_threshold"]) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(ROOT / "templates"),
        static_folder=str(ROOT / "static"),
    )

    # Start inference thread
    t = threading.Thread(
        target=_inference_thread,
        args=(source, conf),
        daemon=True,
    )
    t.start()

    # ── Routes ────────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template("dashboard.html")

    @app.route("/video_feed")
    def video_feed():
        def generate():
            while True:
                with _state["lock"]:
                    frame = _state.get("frame")
                if frame:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                time.sleep(1 / DASHBOARD_CONFIG["stream_fps"])

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/stats")
    def api_stats():
        with _state["lock"]:
            return jsonify(_state["stats"])

    @app.route("/api/violations")
    def api_violations():
        with _state["lock"]:
            return jsonify(_state["violations"])

    @app.route("/api/report")
    def api_report():
        path = Path(LOG_CONFIG["session_report_json"])
        if path.exists():
            return app.response_class(
                response=path.read_text(),
                mimetype="application/json",
            )
        return jsonify([])

    @app.route("/api/violation_log")
    def api_violation_log():
        path = Path(LOG_CONFIG["violation_log_json"])
        if path.exists():
            return app.response_class(
                response=path.read_text(),
                mimetype="application/json",
            )
        return jsonify([])

    @app.route("/api/reset", methods=["POST"])
    def api_reset():
        # Can't easily reset background thread here;
        # restart the app for a clean session.
        return jsonify({"status": "restart app to reset session"})

    return app


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0")
    p.add_argument("--conf",   type=float, default=MODEL_CONFIG["confidence_threshold"])
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = create_app(source=a.source, conf=a.conf)
    app.run(
        host=DASHBOARD_CONFIG["host"],
        port=DASHBOARD_CONFIG["port"],
        debug=False,
        threaded=True,
    )
app = Flask(__name__)

@app.route("/")
def home():
    return "Smart Construction Safety System 🚧 Running"

import os
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
@app.route('/')
def home():
    return "Hello! The Smart Construction Safety System is running."
@app.route('/')
def home():
    return "<h1>Smart Construction Safety System</h1><p>The server is running successfully!</p><a href='/dashboard'>Go to Dashboard</a>"
