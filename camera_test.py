#!/usr/bin/env python3
"""
camera_test.py
--------------
Test different camera indices to find working cameras
"""

import cv2

def test_cameras():
    print("Testing camera indices 0-4...")
    
    for i in range(5):
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            print(f"✅ Camera {i}: Available")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"   Resolution: {w}x{h}")
                
                # Show test window
                cv2.imshow(f"Camera {i} Test", frame)
                print(f"   Test window opened (press any key to continue)")
                cv2.waitKey(2000)  # Show for 2 seconds
                cv2.destroyWindow(f"Camera {i} Test")
            else:
                print(f"   Error: Cannot read frame")
            
            cap.release()
        else:
            print(f"❌ Camera {i}: Not available")
    
    cv2.destroyAllWindows()
    print("\nCamera test completed")

if __name__ == "__main__":
    test_cameras()
