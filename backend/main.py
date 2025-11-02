# backend/main.py
"""
FastAPI WebSocket Backend for Real-time Face Detection
Receives base64 encoded images from React frontend
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import json
import asyncio
from datetime import datetime
from typing import List, Dict
import logging
import mediapipe as mp

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Face Detection API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FaceDetector:
    """Face detection using OpenCV's Haar Cascade with sunglasses and mask detection (DL + CV hybrid)"""
    
    def __init__(self):
        # Using Haar Cascade (fast, CPU-friendly)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        # Eye cascade for sunglasses detection
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        # Mouth cascade for mask detection
        self.mouth_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
        
        # Initialize MediaPipe Face Mesh for facial landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        logger.info("FaceDetector initialized")

    
    def detect_sunglasses(self, image: np.ndarray, face_bbox: tuple) -> Dict:
        """
        Detect if a person is wearing sunglasses using multiple methods
        Returns: dict with has_sunglasses (bool) and confidence (float)
        """
        x, y, w, h = face_bbox
        
        # Extract face ROI
        face_roi = image[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Eye detection using Haar Cascade
        # If eyes are not detected, likely wearing sunglasses
        eyes = self.eye_cascade.detectMultiScale(
            gray_roi,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(int(w * 0.1), int(h * 0.1))
        )
        
        # Method 2: Use MediaPipe Face Mesh to detect eye landmarks
        rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_roi)
        
        has_sunglasses = False
        confidence = 0.0
        method_used = "none"
        
        # Analyze eye detection results
        if len(eyes) == 0:
            # No eyes detected by Haar Cascade - possible sunglasses
            has_sunglasses = True
            confidence = 0.6
            method_used = "haar_no_eyes"
        elif len(eyes) == 1:
            # Only one eye detected - possible sunglasses or side view
            has_sunglasses = True
            confidence = 0.5
            method_used = "haar_partial"
        else:
            # Both eyes detected
            has_sunglasses = False
            confidence = 0.8
            method_used = "haar_eyes_visible"
        
        # Refine using MediaPipe facial landmarks
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Eye region landmarks indices
                # Left eye: 33, 133, 159, 145
                # Right eye: 362, 263, 386, 374
                left_eye_indices = [33, 133, 159, 145]
                right_eye_indices = [362, 263, 386, 374]
                
                # Get eye region visibility score
                h_roi, w_roi = face_roi.shape[:2]
                
                # Calculate eye region contrast (darker regions suggest sunglasses)
                left_eye_region = self._get_eye_region_roi(
                    gray_roi, face_landmarks, left_eye_indices, w_roi, h_roi
                )
                right_eye_region = self._get_eye_region_roi(
                    gray_roi, face_landmarks, right_eye_indices, w_roi, h_roi
                )
                
                # Calculate average brightness in eye regions
                left_brightness = np.mean(left_eye_region) if left_eye_region.size > 0 else 128
                right_brightness = np.mean(right_eye_region) if right_eye_region.size > 0 else 128
                avg_brightness = (left_brightness + right_brightness) / 2
                
                # If eye region is very dark, likely sunglasses
                if avg_brightness < 60:
                    has_sunglasses = True
                    confidence = min(0.9, confidence + 0.3)
                    method_used = "mediapipe_dark_eyes"
                elif avg_brightness < 100 and len(eyes) == 0:
                    has_sunglasses = True
                    confidence = 0.75
                    method_used = "combined_analysis"
                else:
                    # Bright eye regions and eyes detected - no sunglasses
                    has_sunglasses = False
                    confidence = 0.85
                    method_used = "mediapipe_bright_eyes"
        
        return {
            "has_sunglasses": has_sunglasses,
            "confidence": confidence,
            "method": method_used,
            "eyes_detected": len(eyes),
            "eye_brightness": avg_brightness if 'avg_brightness' in locals() else None
        }
    
    def _get_eye_region_roi(self, gray_img: np.ndarray, landmarks, eye_indices: List[int], 
                           width: int, height: int) -> np.ndarray:
        """Extract eye region from face using landmarks"""
        try:
            points = []
            for idx in eye_indices:
                landmark = landmarks.landmark[idx]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                points.append([x, y])
            
            points = np.array(points)
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            
            # Add padding
            padding = 5
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(width, x_max + padding)
            y_max = min(height, y_max + padding)
            
            return gray_img[y_min:y_max, x_min:x_max]
        except Exception as e:
            logger.error(f"Error extracting eye region: {e}")
            return np.array([])
    
    def detect_mask(self, image: np.ndarray, face_bbox: tuple) -> Dict:
        """
        Improved full face analysis with multiple methods
        Returns: dict with has_mask (bool) and confidence (float)
        """
        x, y, w, h = face_bbox
        
        # Extract face ROI
        face_roi = image[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        h_roi, w_roi = face_roi.shape[:2]
        
        # Initialize scoring system
        mask_score = 0.0  # Positive = mask, Negative = no mask
        evidence = []
        
        # === METHOD 1: Mouth Detection (Haar Cascade) ===
        # Try multiple scales for better detection
        lower_face_y = int(h * 0.4)  # Start from 40% down (include more area)
        lower_face_roi = gray_roi[lower_face_y:, :]
        
        mouth_detections = self.mouth_cascade.detectMultiScale(
            lower_face_roi,
            scaleFactor=1.05,  # More sensitive
            minNeighbors=8,     # Lower threshold
            minSize=(int(w * 0.12), int(h * 0.08))
        )
        
        if len(mouth_detections) == 0:
            mask_score += 2.0
            evidence.append("no_mouth_detected")
        elif len(mouth_detections) >= 1:
            mask_score -= 2.0
            evidence.append("mouth_visible")
        
        # === METHOD 2: MediaPipe Face Mesh Analysis ===
        rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_roi)
        
        mouth_region_brightness = None
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Define key facial regions with MORE comprehensive landmark sets
                
                # Mouth region (lips and around mouth)
                mouth_indices = [0, 11, 12, 13, 14, 15, 16, 17, 37, 39, 40, 61, 84, 91, 
                                146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 312, 
                                314, 317, 321, 375, 405, 409]
                
                # Nose region
                nose_indices = [1, 2, 5, 6, 19, 94, 98, 168, 188, 195, 197, 240, 241, 
                               242, 326, 327, 328, 370, 419, 420, 421, 429, 437, 460, 462]
                
                # Cheek region (for texture comparison)
                left_cheek_indices = [116, 117, 118, 119, 120, 121, 123, 147, 187, 207, 
                                     216, 234, 227, 216]
                right_cheek_indices = [345, 346, 347, 348, 349, 350, 352, 376, 411, 427, 
                                      436, 454, 447, 436]
                
                # Eye region (for baseline comparison)
                left_eye_indices = [33, 133, 159, 145, 246, 161, 160, 144, 163]
                right_eye_indices = [362, 263, 386, 374, 466, 388, 387, 373, 390]
                
                # Extract regions
                mouth_region = self._get_region_roi(gray_roi, face_landmarks, mouth_indices, w_roi, h_roi)
                nose_region = self._get_region_roi(gray_roi, face_landmarks, nose_indices, w_roi, h_roi)
                left_cheek = self._get_region_roi(gray_roi, face_landmarks, left_cheek_indices, w_roi, h_roi)
                right_cheek = self._get_region_roi(gray_roi, face_landmarks, right_cheek_indices, w_roi, h_roi)
                left_eye = self._get_region_roi(gray_roi, face_landmarks, left_eye_indices, w_roi, h_roi)
                right_eye = self._get_region_roi(gray_roi, face_landmarks, right_eye_indices, w_roi, h_roi)
                
                # === ANALYSIS 1: Texture Uniformity ===
                # Masks have more uniform texture than skin
                if mouth_region.size > 50:
                    mouth_std = np.std(mouth_region)
                    mouth_region_brightness = np.mean(mouth_region)
                    
                    # Compare with cheeks and eyes for baseline
                    cheek_std = (np.std(left_cheek) + np.std(right_cheek)) / 2 if left_cheek.size > 0 and right_cheek.size > 0 else 30
                    eye_std = (np.std(left_eye) + np.std(right_eye)) / 2 if left_eye.size > 0 and right_eye.size > 0 else 30
                    
                    # If mouth region is significantly more uniform than cheeks/eyes
                    if mouth_std < 18 and (cheek_std > 25 or eye_std > 25):
                        mask_score += 1.5
                        evidence.append(f"uniform_mouth_texture(std={mouth_std:.1f})")
                    elif mouth_std > 25:
                        mask_score -= 1.5
                        evidence.append(f"varied_mouth_texture(std={mouth_std:.1f})")
                
                # === ANALYSIS 2: Nose Region Analysis ===
                if nose_region.size > 50:
                    nose_std = np.std(nose_region)
                    nose_brightness = np.mean(nose_region)
                    
                    # Masks covering nose show uniform texture
                    if nose_std < 16:
                        mask_score += 1.2
                        evidence.append(f"uniform_nose(std={nose_std:.1f})")
                    elif nose_std > 28:
                        mask_score -= 1.0
                        evidence.append(f"visible_nose_detail(std={nose_std:.1f})")
                
                # === ANALYSIS 3: Brightness Pattern Analysis ===
                # Split face into thirds: upper, middle, lower
                third_h = h_roi // 3
                upper_third = gray_roi[:third_h, :]
                middle_third = gray_roi[third_h:2*third_h, :]
                lower_third = gray_roi[2*third_h:, :]
                
                upper_brightness = np.mean(upper_third)
                middle_brightness = np.mean(middle_third)
                lower_brightness = np.mean(lower_third)
                
                # Calculate brightness gradients
                upper_to_middle_diff = abs(upper_brightness - middle_brightness)
                middle_to_lower_diff = abs(middle_brightness - lower_brightness)
                upper_to_lower_diff = abs(upper_brightness - lower_brightness)
                
                # Masks create distinct brightness boundaries
                if middle_to_lower_diff > 15 and upper_to_middle_diff < 10:
                    mask_score += 1.3
                    evidence.append(f"brightness_boundary(diff={middle_to_lower_diff:.1f})")
                elif upper_to_lower_diff > 25:
                    mask_score += 1.0
                    evidence.append(f"strong_brightness_diff({upper_to_lower_diff:.1f})")
                elif upper_to_lower_diff < 10:
                    mask_score -= 0.8
                    evidence.append(f"uniform_face_brightness({upper_to_lower_diff:.1f})")
                
                # === ANALYSIS 4: Edge Detection (Full Face) ===
                # Masks create distinct horizontal edges
                edges = cv2.Canny(gray_roi, 50, 150)
                
                # Count horizontal edges in lower 60% of face
                lower_60_y = int(h_roi * 0.4)
                lower_edges = edges[lower_60_y:, :]
                
                # Sum edges along rows to find strong horizontal patterns
                horizontal_edge_profile = np.sum(lower_edges, axis=1)
                
                # Look for strong horizontal edge (mask boundary)
                max_horizontal_edge = np.max(horizontal_edge_profile) if len(horizontal_edge_profile) > 0 else 0
                mean_horizontal_edge = np.mean(horizontal_edge_profile) if len(horizontal_edge_profile) > 0 else 0
                
                if max_horizontal_edge > mean_horizontal_edge * 3 and max_horizontal_edge > 100:
                    mask_score += 1.5
                    evidence.append(f"horizontal_edge_detected({max_horizontal_edge:.0f})")
                
                # === ANALYSIS 5: Color Analysis (if available) ===
                # Masks often have different color than skin
                if face_roi.shape[2] == 3:
                    # Convert to HSV for better color analysis
                    hsv_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
                    
                    # Split into upper and lower
                    lower_60_y_start = int(h_roi * 0.4)
                    upper_face_hsv = hsv_roi[:lower_60_y_start, :]
                    lower_face_hsv = hsv_roi[lower_60_y_start:, :]
                    
                    # Compare saturation (masks often less saturated than skin)
                    upper_saturation = np.mean(upper_face_hsv[:, :, 1])
                    lower_saturation = np.mean(lower_face_hsv[:, :, 1])
                    saturation_diff = abs(upper_saturation - lower_saturation)
                    
                    if saturation_diff > 20:
                        mask_score += 0.8
                        evidence.append(f"color_saturation_diff({saturation_diff:.1f})")
        
        # === FINAL DECISION ===
        # Convert score to probability
        # Score > 3.0 = very likely mask
        # Score 1.5-3.0 = likely mask
        # Score -1.5 to 1.5 = uncertain
        # Score < -1.5 = likely no mask
        
        has_mask = mask_score > 1.5
        
        # Calculate confidence based on score magnitude
        if mask_score > 4.0:
            confidence = 0.95
            method_used = "high_confidence_mask"
        elif mask_score > 3.0:
            confidence = 0.90
            method_used = "strong_mask_indicators"
        elif mask_score > 1.5:
            confidence = 0.80
            method_used = "moderate_mask_indicators"
        elif mask_score > 0.5:
            confidence = 0.65
            method_used = "weak_mask_indicators"
        elif mask_score > -0.5:
            confidence = 0.50
            method_used = "uncertain"
        elif mask_score > -1.5:
            confidence = 0.70
            method_used = "weak_no_mask"
        elif mask_score > -3.0:
            confidence = 0.85
            method_used = "moderate_no_mask"
        else:
            confidence = 0.90
            method_used = "strong_no_mask"
        
        # Adjust has_mask for uncertain cases
        if abs(mask_score) < 1.0:
            # Very uncertain - default to no mask with low confidence
            has_mask = False
            confidence = 0.55
            method_used = "uncertain_default_no_mask"
        
        return {
            "has_mask": has_mask,
            "confidence": confidence,
            "method": method_used,
            "mouth_detected": len(mouth_detections),
            "mouth_brightness": mouth_region_brightness,
            "mask_score": round(mask_score, 2),
            "evidence": evidence[:5]  # Top 5 evidence points
        }
    
    def _get_region_roi(self, gray_img: np.ndarray, landmarks, indices: List[int], 
                       width: int, height: int) -> np.ndarray:
        """Extract region from face using landmarks"""
        try:
            points = []
            for idx in indices:
                landmark = landmarks.landmark[idx]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                points.append([x, y])
            
            points = np.array(points)
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            
            # Add padding
            padding = 5
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(width, x_max + padding)
            y_max = min(height, y_max + padding)
            
            return gray_img[y_min:y_max, x_min:x_max]
        except Exception as e:
            logger.error(f"Error extracting region: {e}")
            return np.array([])
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using Haar Cascade with sunglasses and mask detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        results = []
        for (x, y, w, h) in faces:
            # Detect sunglasses for each face
            sunglasses_info = self.detect_sunglasses(image, (x, y, w, h))
            
            # Detect mask for each face
            mask_info = self.detect_mask(image, (x, y, w, h))
            
            results.append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "confidence": 0.95,  # Haar doesn't provide confidence
                "has_sunglasses": sunglasses_info["has_sunglasses"],
                "sunglasses_confidence": sunglasses_info["confidence"],
                "sunglasses_detection_method": sunglasses_info["method"],
                "eyes_detected": sunglasses_info["eyes_detected"],
                "has_mask": mask_info["has_mask"],
                "mask_confidence": mask_info["confidence"],
                "mask_detection_method": mask_info["method"],
                "mouth_detected": mask_info["mouth_detected"]
            })
        
        return results


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_result(self, websocket: WebSocket, data: dict):
        await websocket.send_text(json.dumps(data))


# Initialize
detector = FaceDetector()
manager = ConnectionManager()


def base64_to_image(base64_string: str) -> np.ndarray:
    """Convert base64 string to OpenCV image"""
    try:
        # Decode base64
        img_data = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return img
    except Exception as e:
        logger.error(f"Error converting base64 to image: {e}")
        return None


async def process_frame(image_data: str, frame_info: dict) -> dict:
    """Process a single frame for face detection"""
    try:
        # Convert base64 to image
        image = base64_to_image(image_data)
        
        if image is None:
            return {
                "type": "error",
                "message": "Failed to decode image",
                "timestamp": datetime.now().timestamp() * 1000
            }
        
        # Detect faces
        faces = detector.detect(image)
        
        # Prepare response
        result = {
            "type": "detection_result",
            "faces": faces,
            "frame_number": frame_info.get("frame_number", 0),
            "timestamp": frame_info.get("timestamp", datetime.now().timestamp() * 1000),
            "processing_time": datetime.now().timestamp() * 1000 - frame_info.get("timestamp", 0),
            "image_size": {
                "width": image.shape[1],
                "height": image.shape[0]
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return {
            "type": "error",
            "message": str(e),
            "timestamp": datetime.now().timestamp() * 1000
        }


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Face Detection API",
        "version": "1.0.0",
        "detector": "haar",
        "active_connections": len(manager.active_connections)
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "connections": len(manager.active_connections),
        "detector_ready": True
    }


@app.websocket("/ws/face-detection")
async def websocket_face_detection(websocket: WebSocket):
    """WebSocket endpoint for real-time face detection"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "frame":
                # Extract image data
                image_base64 = message.get("image")
                
                if not image_base64:
                    await manager.send_result(websocket, {
                        "type": "error",
                        "message": "No image data provided"
                    })
                    continue
                
                # Process frame asynchronously
                result = await process_frame(image_base64, message)
                
                # Send result back to client
                await manager.send_result(websocket, result)
            
            elif message.get("type") == "ping":
                # Respond to ping
                await manager.send_result(websocket, {
                    "type": "pong",
                    "timestamp": datetime.now().timestamp() * 1000
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected normally")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# Optional: REST API endpoint for single image detection
@app.post("/api/detect")
async def detect_face_rest(request: dict):
    """REST API endpoint for single image detection"""
    try:
        image_base64 = request.get("image")
        
        if not image_base64:
            return {"error": "No image data provided"}
        
        result = await process_frame(image_base64, {
            "timestamp": datetime.now().timestamp() * 1000,
            "frame_number": 0
        })
        
        return result
    
    except Exception as e:
        logger.error(f"REST API error: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    print("Starting Face Detection API...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )