#!/usr/bin/env python3
"""
camera_monitor.py
--------------
Simple camera monitor to test live detection
"""

import cv2
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from detection.model import PPEModel
from detection.inference import InferencePipeline
from utils.tracker import WorkerTracker
from utils.alerts import AlertManager

def main():
    print("🎥 Starting Camera Monitor...")
    print("Position yourself in front of the camera for detection")
    print("Press 'q' to quit")
    
    # Initialize components
    model = PPEModel()
    pipeline = InferencePipeline(model=model, conf=0.1)  # Very low confidence
    tracker = WorkerTracker()
    alerts = AlertManager()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("✅ Camera opened - Starting detection...")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read frame")
            break
        
        frame_count += 1
        
        # Run detection every few frames to improve performance
        if frame_count % 3 == 0:
            worker_states, raw_dets, infer_ms = pipeline.process_frame(frame)
            tracked_workers = tracker.update(worker_states)
            
            # Print detection info
            total = len(tracked_workers)
            violations = sum(1 for tw in tracked_workers if not tw.is_compliant)
            
            if total > 0:
                print(f"👥 Detected {total} worker(s), {violations} violations")
                for tw in tracked_workers:
                    status = "COMPLIANT" if tw.is_compliant else "VIOLATIONS"
                    print(f"   Worker {tw.track_id}: {status}")
                    for v in tw.current_violations:
                        print(f"     - {v}")
            elif frame_count % 30 == 0:  # Print every 30 frames if no detection
                print(f"🔍 Scanning... (frame {frame_count})")
            
            # Annotate frame
            annotated = alerts.draw_frame(frame, tracked_workers, 30.0, frame_count)
        else:
            annotated = frame
        
        # Show camera feed
        cv2.imshow("PPE Safety Camera Monitor", annotated)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Camera monitor stopped")

if __name__ == "__main__":
    main()
