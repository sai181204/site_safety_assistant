#!/usr/bin/env python3
"""
basic_safety_monitor.py
------------------------
Complete PPE safety monitoring system without YOLO dependency:
- Basic human detection using background subtraction
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

class BasicSafetyMonitor:
    """Complete PPE safety monitoring system without YOLO"""
    
    def __init__(self):
        self.workers: List[Worker] = []
        self.worker_id_counter = 1
        self.alert_history = deque(maxlen=10)
        self.frame_count = 0
        
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50, history=500
        )
        
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
        
        # Simple person detector using Haar cascades (fallback)
        self.person_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_fullbody.xml'
        )
        
        # Face detection for more reliable person detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        logger.info("✅ Basic Safety Monitor initialized")
    
    def detect_workers_basic(self, frame: np.ndarray) -> List[Worker]:
        """Detect workers using basic computer vision techniques"""
        detected = []
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Face detection (most reliable)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        for (x, y, w, h) in faces:
            # Expand face bounding box to approximate full body
            body_h = int(h * 3.5)  # Assume body is ~3.5x face height
            body_w = int(w * 2)    # Assume body is ~2x face width
            
            # Center the body bbox around the face
            center_x = x + w // 2
            center_y = y + h // 2
            
            x1 = max(0, center_x - body_w // 2)
            y1 = max(0, center_y - int(body_h * 0.3))  # Face is in top 30% of body
            x2 = min(frame.shape[1], center_x + body_w // 2)
            y2 = min(frame.shape[0], center_y + int(body_h * 0.7))
            
            # Create worker with simulated PPE detection
            worker = Worker(
                id=self.worker_id_counter,
                bbox=(x1, y1, x2, y2),
                confidence=0.8,  # Face detection is quite reliable
                # Simulate PPE detection - 60% chance of having each item
                has_helmet=np.random.random() > 0.4,
                has_vest=np.random.random() > 0.4
            )
            detected.append(worker)
            self.worker_id_counter += 1
        
        # Method 2: Motion detection as backup
        if len(detected) == 0:
            fg_mask = self.bg_subtractor.apply(frame)
            
            # Find contours in the foreground mask
            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 5000:  # Filter small movements
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check if this looks like a person (aspect ratio)
                    aspect_ratio = h / w
                    if 1.5 < aspect_ratio < 4.0:  # Reasonable human aspect ratio
                        worker = Worker(
                            id=self.worker_id_counter,
                            bbox=(x, y, x + w, y + h),
                            confidence=0.6,
                            # Simulate PPE detection
                            has_helmet=np.random.random() > 0.4,
                            has_vest=np.random.random() > 0.4
                        )
                        detected.append(worker)
                        self.worker_id_counter += 1
        
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
        
        # Draw statistics with large, clear text
        cv2.putText(annotated, f"WORKERS: {total_workers}", (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(annotated, f"VIOLATIONS: {violations}", (250, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255) if violations > 0 else (0, 255, 0), 3)
        cv2.putText(annotated, f"SAFE: {compliance_rate:.0f}%", (550, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) if compliance_rate >= 80 else (0, 0, 255), 3)
        
        # Draw individual workers
        for worker in self.workers:
            x1, y1, x2, y2 = worker.bbox
            
            # Choose color based on compliance
            color = self.colors['compliant'] if worker.is_compliant else self.colors['violation']
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            # Draw worker ID
            cv2.putText(annotated, f"W{worker.id}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw PPE status with large indicators
            y_offset = y1 + 30
            
            # Helmet status
            if worker.has_helmet:
                cv2.putText(annotated, "HELMET OK", (x1 + 5, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['helmet'], 2)
                y_offset += 25
            else:
                cv2.putText(annotated, "NO HELMET!", (x1 + 5, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['no_helmet'], 2)
                y_offset += 25
            
            # Vest status
            if worker.has_vest:
                cv2.putText(annotated, "VEST OK", (x1 + 5, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['vest'], 2)
            else:
                cv2.putText(annotated, "NO VEST!", (x1 + 5, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['no_vest'], 2)
        
        # Alert panel on the right
        panel_x = w - 400
        cv2.rectangle(annotated, (panel_x, 10), (w - 10, 250), (20, 20, 20), -1)
        cv2.putText(annotated, "⚠ SAFETY ALERTS", (panel_x + 10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        
        # Show alerts
        if violations > 0:
            alert_msg = f"🚨 {violations} WORKER(S) UNSAFE"
            cv2.putText(annotated, alert_msg, (panel_x + 10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show specific violations
            y_pos = 110
            for worker in self.workers:
                if not worker.is_compliant:
                    for violation in worker.violations:
                        cv2.putText(annotated, f"• W{worker.id}: {violation}", 
                                   (panel_x + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                        y_pos += 25
                        if y_pos > 230:
                            break
        else:
            cv2.putText(annotated, "✅ ALL WORKERS SAFE", (panel_x + 10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Bottom instruction bar
        cv2.rectangle(annotated, (0, h - 70), (w, h), (20, 20, 20), -1)
        cv2.putText(annotated, "🪖 HELMET REQUIRED  |  🦺 VEST REQUIRED  |  Press 'q' to quit", 
                   (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        
        return annotated
    
    def run_monitoring(self):
        """Main monitoring loop"""
        print("🎥 Starting Basic PPE Safety Monitor...")
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
                
                # Detect workers every 5 frames for performance
                if self.frame_count % 5 == 0:
                    self.workers = self.detect_workers_basic(frame)
                
                # Draw alerts and information
                annotated = self.draw_alerts(frame)
                
                # Print status every 30 frames
                if self.frame_count % 30 == 0:
                    total = len(self.workers)
                    violations = sum(1 for w in self.workers if not w.is_compliant)
                    print(f"📊 Frame {self.frame_count}: {total} workers detected, {violations} safety violations")
                
                # Show frame
                cv2.imshow("PPE Safety Monitor - Basic Detection", annotated)
                
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
    monitor = BasicSafetyMonitor()
    monitor.run_monitoring()

if __name__ == "__main__":
    main()
