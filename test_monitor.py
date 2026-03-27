#!/usr/bin/env python3
"""
test_monitor.py
---------------
Simple test script that cycles through test images to demonstrate
the PPE safety monitoring system with violation detection and alarms.
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from detection.model import PPEModel
from detection.inference import InferencePipeline
from utils.tracker import WorkerTracker
from utils.alerts import AlertManager
from utils.report import ReportManager
from config import HAZARD_ZONES, ENABLE_ZONES
import cv2
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Test PPE monitoring with image cycling")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between images (seconds)")
    parser.add_argument("--save", action="store_true", help="Save annotated frames")
    args = parser.parse_args()

    # Initialize components
    logger.info("Initializing PPE Safety Monitor...")
    model = PPEModel()
    pipeline = InferencePipeline(model=model, conf=args.conf)
    tracker = WorkerTracker()
    alerts = AlertManager()
    reporter = ReportManager()

    # Find test images
    test_images = list(ROOT.glob("data/images/train/*.jpg"))[:5]  # Use first 5 images
    if not test_images:
        logger.error("No test images found in data/images/train/")
        return

    logger.info(f"Found {len(test_images)} test images")
    logger.info("Starting monitoring loop (press Ctrl+C to stop)")

    frame_no = 0
    try:
        while True:
            # Cycle through images
            img_path = test_images[frame_no % len(test_images)]
            logger.info(f"Processing: {img_path.name}")

            # Read image
            frame = cv2.imread(str(img_path))
            if frame is None:
                logger.error(f"Failed to read image: {img_path}")
                continue

            frame_no += 1

            # Run inference
            worker_states, raw_dets, infer_ms = pipeline.process_frame(frame, zones=HAZARD_ZONES if ENABLE_ZONES else None)
            tracked_workers = tracker.update(worker_states)
            reporter.ingest_workers(tracked_workers, frame_no)

            # Generate alerts for violations
            for tw in tracked_workers:
                for vtype in tw.current_violations:
                    alerts.push_alert(f"W{tw.track_id}: {vtype}", "MEDIUM")
                    logger.warning(f"VIOLATION - Worker {tw.track_id}: {vtype} in zone {tw.zone_name or 'Unknown'}")

            # Annotate frame
            annotated = alerts.draw_frame(frame, tracked_workers, 30.0, frame_no)

            # Display results
            total = len(tracked_workers)
            violations = sum(1 for tw in tracked_workers if not tw.is_compliant)
            compliant = total - violations
            compliance_pct = (compliant / total * 100) if total else 100.0

            print(f"Frame {frame_no}: {total} workers, {violations} violations, {compliance_pct:.1f}% compliant")

            # Show frame (if display available)
            try:
                cv2.imshow("PPE Safety Monitor - Test", annotated)
                key = cv2.waitKey(int(args.delay * 1000))
                if key in (ord('q'), ord('Q'), 27):
                    break
            except:
                logger.info("No display available - continuing without visual output")

            # Save annotated frame if requested
            if args.save:
                output_path = ROOT / "reports" / f"test_frame_{frame_no:04d}.jpg"
                cv2.imwrite(str(output_path), annotated)
                logger.info(f"Saved annotated frame: {output_path}")

            time.sleep(args.delay)

    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")

    # Final report
    logger.info("Generating final report...")
    report = reporter.finalize()
    reporter.print_summary(report)

    cv2.destroyAllWindows()
    logger.info("Test monitoring completed")

if __name__ == "__main__":
    main()
