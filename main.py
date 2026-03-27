"""
main.py
───────
Smart Construction Site Safety System — Entry Point

Usage
-----
# Webcam (default)
python main.py

# Video file
python main.py --source path/to/video.mp4

# RTSP / HTTP stream
python main.py --source rtsp://192.168.1.100/stream

# Custom weights & confidence
python main.py --weights models/ppe_yolov8.pt --conf 0.45

# Save annotated output
python main.py --source video.mp4 --save-output

# Headless (no window) — for server / CI
python main.py --source video.mp4 --headless

# Launch Flask dashboard instead of raw OpenCV window
python main.py --dashboard

# Train / fine-tune model
python main.py --train --data data/ppe_dataset.yaml --epochs 50

# Evaluate model
python main.py --evaluate
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

# ── ensure project root is importable ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    MODEL_CONFIG, ALERT_CONFIG, ENABLE_ZONES, HAZARD_ZONES,
    DASHBOARD_CONFIG, LOG_CONFIG,
)
from detection.model     import PPEModel
from detection.inference import InferencePipeline
from utils.tracker       import WorkerTracker
from utils.alerts        import AlertManager
from utils.report        import ReportManager

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smart Construction Site PPE Safety System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source",     default="0",
                   help="Video source: 0=webcam, file path, RTSP URL")
    p.add_argument("--weights",    default=None,
                   help="Path to YOLO weights (.pt). Overrides config.")
    p.add_argument("--conf",       type=float, default=MODEL_CONFIG["confidence_threshold"],
                   help="Confidence threshold")
    p.add_argument("--device",     default=MODEL_CONFIG["device"],
                   help="Device: cpu | cuda | mps")
    p.add_argument("--save-output", action="store_true",
                   help="Save annotated video to reports/output.mp4")
    p.add_argument("--headless",   action="store_true",
                   help="Run without display window (server mode)")
    p.add_argument("--dashboard",  action="store_true",
                   help="Launch Flask web dashboard")
    p.add_argument("--train",      action="store_true",
                   help="Train / fine-tune the model")
    p.add_argument("--data",       default=None,
                   help="Dataset YAML path (for --train / --evaluate)")
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--evaluate",   action="store_true",
                   help="Evaluate model on validation set")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after N frames (0 = unlimited)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def run_inference_loop(args: argparse.Namespace):
    """
    Main video processing loop.
    Reads frames, runs inference, tracks workers, overlays results,
    logs violations, and optionally saves output.
    """
    # ── Model & pipeline ────────────────────────────────────────────────────
    MODEL_CONFIG["device"] = args.device
    model    = PPEModel(weights_path=args.weights)
    pipeline = InferencePipeline(model=model, conf=args.conf)
    tracker  = WorkerTracker()
    alerts   = AlertManager()
    reporter = ReportManager()

    # ── Open video source ───────────────────────────────────────────────────
    source = args.source
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Cannot open video source: %s", args.source)
        sys.exit(1)

    orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    logger.info("Source: %s  |  %dx%d @ %.1f fps", args.source, orig_w, orig_h, src_fps)

    # ── Optional video writer ───────────────────────────────────────────────
    writer = None
    if args.save_output:
        out_path = str(ROOT / "reports" / "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, src_fps, (orig_w, orig_h))
        logger.info("Saving output → %s", out_path)

    # ── Zones ───────────────────────────────────────────────────────────────
    zones = HAZARD_ZONES if ENABLE_ZONES else None

    # ── FPS tracking ────────────────────────────────────────────────────────
    fps_window = 30
    frame_times: list = []
    smoothed_fps = 0.0

    frame_no   = 0
    prev_violation_ids: set = set()

    logger.info("Starting inference loop  (press Q to quit)")

    try:
        while True:
            t_frame_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                logger.info("End of stream / video.")
                break

            frame_no += 1

            # ── Inference ─────────────────────────────────────────────────
            worker_states, raw_dets, infer_ms = pipeline.process_frame(frame, zones=zones)
            logger.debug("Frame %d: %d raw detections, %d worker states, %.1fms inference", 
                        frame_no, len(raw_dets), len(worker_states), infer_ms)

            # ── Tracking ──────────────────────────────────────────────────
            tracked_workers = tracker.update(worker_states)
            logger.debug("Frame %d: %d tracked workers", frame_no, len(tracked_workers))

            # ── Reporter ──────────────────────────────────────────────────
            reporter.ingest_workers(tracked_workers, frame_no)

            # ── Push new violation alerts ──────────────────────────────────
            current_viol_ids = set()
            for tw in tracked_workers:
                for vtype in tw.current_violations:
                    vid = (tw.track_id, vtype)
                    current_viol_ids.add(vid)
                    if vid not in prev_violation_ids:
                        sev = tw.active_violations[vtype].severity \
                              if vtype in tw.active_violations else "MEDIUM"
                        alerts.push_alert(f"W{tw.track_id}: {vtype}", sev)
                        logger.warning("VIOLATION  worker=%d  type=%s  sev=%s  zone=%s",
                                       tw.track_id, vtype, sev, tw.zone_name)
            prev_violation_ids = current_viol_ids

            # ── FPS ────────────────────────────────────────────────────────
            t_elapsed = time.perf_counter() - t_frame_start
            frame_times.append(t_elapsed)
            if len(frame_times) > fps_window:
                frame_times.pop(0)
            smoothed_fps = len(frame_times) / sum(frame_times)

            # ── Annotate frame ─────────────────────────────────────────────
            annotated = alerts.draw_frame(frame, tracked_workers, smoothed_fps, frame_no)

            # ── Display ────────────────────────────────────────────────────
            if not args.headless:
                cv2.imshow("PPE Safety Monitor", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    logger.info("User quit.")
                    break

            # ── Write ──────────────────────────────────────────────────────
            if writer:
                writer.write(annotated)

            # ── Frame limit ────────────────────────────────────────────────
            if args.max_frames and frame_no >= args.max_frames:
                logger.info("Reached max frames (%d).", args.max_frames)
                break

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

        # ── Final report ──────────────────────────────────────────────────
        report = reporter.finalize()
        reporter.print_summary(report)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = build_args()

    if args.train:
        logger.info("=== Training mode ===")
        model = PPEModel()
        model.train(data_yaml=args.data, epochs=args.epochs)
        return

    if args.evaluate:
        logger.info("=== Evaluation mode ===")
        model = PPEModel()
        metrics = model.evaluate(data_yaml=args.data)
        print("\nEvaluation Results:")
        for k, v in metrics.items():
            print(f"  {k:<20} {v:.4f}")
        return

    if args.dashboard:
        logger.info("=== Dashboard mode ===")
        from dashboard import create_app
        app = create_app()
        app.run(
            host=DASHBOARD_CONFIG["host"],
            port=DASHBOARD_CONFIG["port"],
            debug=DASHBOARD_CONFIG["debug"],
        )
        return

    # Default: inference loop
    run_inference_loop(args)


if __name__ == "__main__":
    main()
from flask import Flask

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
