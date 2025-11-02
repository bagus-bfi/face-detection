"""
Test script for mask detection
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
    """Test mask detection with webcam"""
    print("Testing mask detection with webcam...")
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
        
        # Detect faces, sunglasses, and masks
        faces = detector.detect(frame)
        
        # Draw results
        for face in faces:
            x, y, w, h = face['x'], face['y'], face['width'], face['height']
            has_sunglasses = face.get('has_sunglasses', False)
            has_mask = face.get('has_mask', False)
            mask_confidence = face.get('mask_confidence', 0)
            sunglasses_confidence = face.get('sunglasses_confidence', 0)
            eyes_detected = face.get('eyes_detected', 0)
            mouth_detected = face.get('mouth_detected', 0)
            
            # Choose color based on mask and sunglasses
            if has_mask and has_sunglasses:
                color = (107, 107, 255)  # Red
            elif has_mask:
                color = (196, 205, 78)   # Teal
            elif has_sunglasses:
                color = (0, 215, 255)    # Gold
            else:
                color = (0, 255, 0)      # Green
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw label background
            cv2.rectangle(frame, (x, y-95), (x+280, y), color, -1)
            
            # Face info
            cv2.putText(frame, f"Face Detection", (x+5, y-75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Sunglasses status
            sunglasses_text = f"{'Sunglasses' if has_sunglasses else 'No Sunglasses'}"
            cv2.putText(frame, sunglasses_text, (x+5, y-58), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, f"S.Conf: {sunglasses_confidence*100:.1f}%", (x+5, y-44), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Mask status
            mask_text = f"{'Mask Detected' if has_mask else 'No Mask'}"
            cv2.putText(frame, mask_text, (x+5, y-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, f"M.Conf: {mask_confidence*100:.1f}%", (x+5, y-16), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Additional info
            cv2.putText(frame, f"Eyes:{eyes_detected} Mouth:{mouth_detected}", 
                       (x+5, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        
        # Display info
        info_text = f"Faces: {len(faces)}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Legend
        cv2.putText(frame, "Legend: Green=None | Gold=Sunglasses | Teal=Mask | Red=Both", 
                   (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('Mask & Sunglasses Detection Test', frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def test_image(image_path):
    """Test mask detection with an image file"""
    print(f"Testing mask detection with image: {image_path}")
    
    # Initialize detector
    detector = FaceDetector()
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return
    
    # Detect faces, sunglasses, and masks
    faces = detector.detect(image)
    
    print(f"\nDetected {len(faces)} face(s):")
    
    # Draw results
    for i, face in enumerate(faces):
        x, y, w, h = face['x'], face['y'], face['width'], face['height']
        has_sunglasses = face.get('has_sunglasses', False)
        has_mask = face.get('has_mask', False)
        sunglasses_confidence = face.get('sunglasses_confidence', 0)
        mask_confidence = face.get('mask_confidence', 0)
        eyes_detected = face.get('eyes_detected', 0)
        mouth_detected = face.get('mouth_detected', 0)
        sunglasses_method = face.get('sunglasses_detection_method', 'unknown')
        mask_method = face.get('mask_detection_method', 'unknown')
        
        print(f"\nFace {i+1}:")
        print(f"  Position: ({x}, {y})")
        print(f"  Size: {w}x{h}")
        print(f"  Has Sunglasses: {has_sunglasses}")
        print(f"  Sunglasses Confidence: {sunglasses_confidence*100:.2f}%")
        print(f"  Sunglasses Method: {sunglasses_method}")
        print(f"  Eyes Detected: {eyes_detected}")
        print(f"  Has Mask: {has_mask}")
        print(f"  Mask Confidence: {mask_confidence*100:.2f}%")
        print(f"  Mask Method: {mask_method}")
        print(f"  Mouth Detected: {mouth_detected}")
        
        # Choose color
        if has_mask and has_sunglasses:
            color = (107, 107, 255)  # Red
        elif has_mask:
            color = (196, 205, 78)   # Teal
        elif has_sunglasses:
            color = (0, 215, 255)    # Gold
        else:
            color = (0, 255, 0)      # Green
        
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        
        # Draw label background
        cv2.rectangle(image, (x, y-80), (x+250, y), color, -1)
        
        # Labels
        cv2.putText(image, f"Face {i+1}", (x+5, y-62), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        sunglasses_text = f"{'Sunglasses' if has_sunglasses else 'No Sunglasses'} ({sunglasses_confidence*100:.0f}%)"
        cv2.putText(image, sunglasses_text, (x+5, y-45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        mask_text = f"{'Mask' if has_mask else 'No Mask'} ({mask_confidence*100:.0f}%)"
        cv2.putText(image, mask_text, (x+5, y-28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        cv2.putText(image, f"Eyes:{eyes_detected} Mouth:{mouth_detected}", 
                   (x+5, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    
    # Show result
    cv2.imshow('Mask & Sunglasses Detection Result', image)
    print("\nPress any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """Main test function"""
    print("=" * 60)
    print("Mask & Sunglasses Detection Test Script")
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
