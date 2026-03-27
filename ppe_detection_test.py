#!/usr/bin/env python3
"""
ppe_detection_test.py
--------------
Detailed PPE detection test to see what the model actually detects
"""

import cv2
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from detection.model import PPEModel
from detection.inference import InferencePipeline
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s %(message)s")

def main():
    print("🔍 PPE Detection Test - What does the model actually see?")
    print("Position yourself in front of camera")
    print("Press 'q' to quit")
    
    # Initialize model with very low confidence
    model = PPEModel()
    pipeline = InferencePipeline(model=model, conf=0.05)  # Very low confidence
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("✅ Camera opened - Analyzing detections...")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run inference every 5 frames
        if frame_count % 5 == 0:
            print(f"\n=== Frame {frame_count} ===")
            
            # Get raw results from model
            raw_results = model.predict(frame, conf=0.05)
            
            # Parse detections manually to see everything
            all_detections = []
            for r in raw_results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    class_names = {0: "person", 1: "helmet", 2: "vest", 3: "no_helmet", 4: "no_vest"}
                    cls_name = class_names.get(cls_id, f"class_{cls_id}")
                    
                    all_detections.append({
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'confidence': conf,
                        'bbox': (int(x1), int(y1), int(x2), int(y2))
                    })
                    
                    print(f"  📦 {cls_name} (ID:{cls_id}) - {conf:.2f} confidence")
            
            # Run full pipeline
            worker_states, raw_dets, infer_ms = pipeline.process_frame(frame)
            
            print(f"  👥 Workers detected: {len(worker_states)}")
            for i, ws in enumerate(worker_states):
                print(f"    Worker {i+1}: helmet={ws.has_helmet}, vest={ws.has_vest}, violations={ws.violations}")
            
            # Draw all detections on frame
            annotated = frame.copy()
            for det in all_detections:
                x1, y1, x2, y2 = det['bbox']
                color = {
                    0: (255, 255, 255),  # person - white
                    1: (0, 255, 0),      # helmet - green  
                    2: (0, 200, 255),    # vest - orange
                    3: (0, 0, 255),      # no_helmet - red
                    4: (0, 165, 255)     # no_vest - orange-red
                }.get(det['class_id'], (128, 128, 128))
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{det['class_name']} {det['confidence']:.2f}"
                cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            annotated = frame
        
        # Show camera feed
        cv2.imshow("PPE Detection Analysis", annotated)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Detection test completed")

if __name__ == "__main__":
    main()
