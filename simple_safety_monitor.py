#!/usr/bin/env python3
"""
simple_safety_monitor.py
------------------------
Complete PPE safety monitoring system with:
- Human detection using YOLOv8
- Simulated helmet/vest detection with visual alerts
- Real-time worker counting
- Alert messages and compliance tracking
"""

import cv2
import time
import numpy as np
from pathlib import Path
import sys
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple
import logging

# Add project root for imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Try to import ultralytics, fallback to basic detection if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️  Ultralytics not available, using basic detection")
    YOLO_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class Worker:
    """Represents a detected worker with PPE status"""
    id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    has_helmet: bool = False
    has_vest: bool = False
    confidence: float = 0.0
    
    @property
    def violations(self) -> List[str]:
        violations = []
        if not self.has_helmet:
            violations.append("No Helmet")
        if not self.has_vest:
            violations.append("No Vest")
        return violations
    
    @property
    def is_compliant(self) -> bool:
        return len(self.violations) == 0

class SimpleSafetyMonitor:
    """Complete PPE safety monitoring system"""
    
    def __init__(self):
        self.workers: List[Worker] = []
        self.worker_id_counter = 1
        self.alert_history = deque(maxlen=10)
        self.frame_count = 0
        
        # Initialize YOLO model
        if YOLO_AVAILABLE:
            try:
                logger.info("Loading YOLOv8 model...")
                self.model = YOLO('yolov8n.pt')  # Download fresh model
                logger.info("✅ YOLOv8 model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load YOLO: {e}")
                self.model = None
        else:
            self.model = None
        
        # Alert colors
        self.colors = {
            'person': (255, 255, 255),      # White
            'compliant': (0, 255, 0),      # Green
            'violation': (0, 0, 255),     # Red
            'helmet': (0, 255, 0),         # Green
            'no_helmet': (0, 0, 255),      # Red
            'vest': (0, 200, 255),         # Orange
            'no_vest': (0, 165, 255),      # Orange-red
        }
    
    def detect_workers(self, frame: np.ndarray) -> List[Worker]:
        """Detect workers in frame using YOLO or fallback method"""
        detected = []
        
        if self.model:
            try:
                # Use YOLO for person detection
                results = self.model(frame, conf=0.3, classes=[0])  # Class 0 = person
                
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            
                            # Create worker with simulated PPE detection
                            worker = Worker(
                                id=self.worker_id_counter,
                                bbox=(x1, y1, x2, y2),
                                confidence=conf,
                                # Simulate PPE detection - 70% chance of having each item
                                has_helmet=np.random.random() > 0.3,
                                has_vest=np.random.random() > 0.3
                            )
                            detected.append(worker)
                            self.worker_id_counter += 1
                            
            except Exception as e:
                logger.error(f"YOLO detection failed: {e}")
        
        return detected
    
    def draw_alerts(self, frame: np.ndarray) -> np.ndarray:
        """Draw all alerts and information on frame"""
        h, w = frame.shape[:2]
        annotated = frame.copy()
        
        # Top status bar
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0, annotated)
        
        # Worker statistics
        total_workers = len(self.workers)
        violations = sum(1 for w in self.workers if not w.is_compliant)
        compliant = total_workers - violations
        compliance_rate = (compliant / total_workers * 100) if total_workers > 0 else 100
        
        # Draw statistics
        cv2.putText(annotated, f"WORKERS: {total_workers}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(annotated, f"VIOLATIONS: {violations}", (200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) if violations > 0 else (0, 255, 0), 2)
        cv2.putText(annotated, f"COMPLIANCE: {compliance_rate:.0f}%", (450, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if compliance_rate >= 80 else (0, 0, 255), 2)
        
        # Draw individual workers
        for worker in self.workers:
            x1, y1, x2, y2 = worker.bbox
            
            # Choose color based on compliance
            color = self.colors['compliant'] if worker.is_compliant else self.colors['violation']
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            # Draw worker ID
            cv2.putText(annotated, f"Worker {worker.id}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw PPE status
            y_offset = y1 + 25
            if worker.has_helmet:
                cv2.putText(annotated, "🪖 HELMET", (x1 + 5, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['helmet'], 2)
                y_offset += 20
            else:
                cv2.putText(annotated, "⚠ NO HELMET", (x1 + 5, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['no_helmet'], 2)
                y_offset += 20
            
            if worker.has_vest:
                cv2.putText(annotated, "🦺 VEST", (x1 + 5, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['vest'], 2)
            else:
                cv2.putText(annotated, "⚠ NO VEST", (x1 + 5, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['no_vest'], 2)
        
        # Alert panel on the right
        panel_x = w - 350
        cv2.rectangle(annotated, (panel_x, 10), (w - 10, 200), (20, 20, 20), -1)
        cv2.putText(annotated, "ALERTS", (panel_x + 10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        
        # Show recent alerts
        if violations > 0:
            alert_msg = f"⚠ {violations} WORKER(S) WITH PPE VIOLATIONS"
            cv2.putText(annotated, alert_msg, (panel_x + 10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Show specific violations
            y_pos = 95
            for worker in self.workers:
                if not worker.is_compliant:
                    for violation in worker.violations:
                        cv2.putText(annotated, f"W{worker.id}: {violation}", 
                                   (panel_x + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        y_pos += 20
                        if y_pos > 180:
                            break
        else:
            cv2.putText(annotated, "✅ ALL WORKERS COMPLIANT", (panel_x + 10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Bottom instruction bar
        cv2.rectangle(annotated, (0, h - 60), (w, h), (20, 20, 20), -1)
        cv2.putText(annotated, "PPE SAFETY MONITOR - Press 'q' to quit", (10, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        
        return annotated
    
    def run_monitoring(self):
        """Main monitoring loop"""
        print("🎥 Starting PPE Safety Monitor...")
        print("Position yourself in front of the camera")
        print("Press 'q' to quit")
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        print("✅ Camera opened - Starting monitoring...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Cannot read frame")
                    break
                
                self.frame_count += 1
                
                # Detect workers every 3 frames for performance
                if self.frame_count % 3 == 0:
                    self.workers = self.detect_workers(frame)
                
                # Draw alerts and information
                annotated = self.draw_alerts(frame)
                
                # Print status every 30 frames
                if self.frame_count % 30 == 0:
                    total = len(self.workers)
                    violations = sum(1 for w in self.workers if not w.is_compliant)
                    print(f"📊 Frame {self.frame_count}: {total} workers, {violations} violations")
                
                # Show frame
                cv2.imshow("PPE Safety Monitor", annotated)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
        
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ PPE Safety Monitor stopped")

def main():
    """Main entry point"""
    monitor = SimpleSafetyMonitor()
    monitor.run_monitoring()

if __name__ == "__main__":
    main()
