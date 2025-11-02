"""
Test script for sunglasses detection
Tests the face detector with sample images or webcam
"""

import cv2
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import FaceDetector

def test_webcam():
    """Test sunglasses detection with webcam"""
    print("Testing sunglasses detection with webcam...")
    print("Press 'q' to quit")
    
    # Initialize detector
    detector = FaceDetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break
        
        # Detect faces and sunglasses
        faces = detector.detect(frame)
        
        # Draw results
        for face in faces:
            x, y, w, h = face['x'], face['y'], face['width'], face['height']
            has_sunglasses = face.get('has_sunglasses', False)
            confidence = face.get('sunglasses_confidence', 0)
            eyes_detected = face.get('eyes_detected', 0)
            
            # Choose color based on sunglasses
            color = (0, 215, 255) if has_sunglasses else (0, 255, 0)  # Gold or Green
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw label
            label = f"{'Sunglasses' if has_sunglasses else 'No Sunglasses'}"
            conf_text = f"{confidence*100:.1f}%"
            eyes_text = f"Eyes: {eyes_detected}"
            
            # Background for text
            cv2.rectangle(frame, (x, y-70), (x+250, y), color, -1)
            
            # Text
            cv2.putText(frame, label, (x+5, y-45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, conf_text, (x+5, y-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, eyes_text, (x+5, y-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Display info
        info_text = f"Faces: {len(faces)}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Sunglasses Detection Test', frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def test_image(image_path):
    """Test sunglasses detection with an image file"""
    print(f"Testing sunglasses detection with image: {image_path}")
    
    # Initialize detector
    detector = FaceDetector()
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return
    
    # Detect faces and sunglasses
    faces = detector.detect(image)
    
    print(f"\nDetected {len(faces)} face(s):")
    
    # Draw results
    for i, face in enumerate(faces):
        x, y, w, h = face['x'], face['y'], face['width'], face['height']
        has_sunglasses = face.get('has_sunglasses', False)
        confidence = face.get('sunglasses_confidence', 0)
        eyes_detected = face.get('eyes_detected', 0)
        method = face.get('sunglasses_detection_method', 'unknown')
        
        print(f"\nFace {i+1}:")
        print(f"  Position: ({x}, {y})")
        print(f"  Size: {w}x{h}")
        print(f"  Has Sunglasses: {has_sunglasses}")
        print(f"  Confidence: {confidence*100:.2f}%")
        print(f"  Eyes Detected: {eyes_detected}")
        print(f"  Detection Method: {method}")
        
        # Choose color based on sunglasses
        color = (0, 215, 255) if has_sunglasses else (0, 255, 0)
        
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        
        # Draw label
        label = f"{'Sunglasses' if has_sunglasses else 'No Sunglasses'}"
        conf_text = f"{confidence*100:.1f}%"
        
        cv2.rectangle(image, (x, y-50), (x+200, y), color, -1)
        cv2.putText(image, label, (x+5, y-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(image, conf_text, (x+5, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
    
    # Show result
    cv2.imshow('Sunglasses Detection Result', image)
    print("\nPress any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """Main test function"""
    print("=" * 60)
    print("Sunglasses Detection Test Script")
    print("=" * 60)
    print()
    print("Options:")
    print("1. Test with webcam")
    print("2. Test with image file")
    print()
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        test_webcam()
    elif choice == "2":
        image_path = input("Enter image path: ").strip()
        test_image(image_path)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
