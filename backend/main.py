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
    """Face detection using OpenCV's Haar Cascade with sunglasses detection"""
    
    def __init__(self):
        # Using Haar Cascade (fast, CPU-friendly)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        # Eye cascade for sunglasses detection
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Initialize MediaPipe Face Mesh for facial landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        logger.info("FaceDetector initialized with Haar Cascade method")
    
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
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using Haar Cascade with sunglasses detection"""
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
            
            results.append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "confidence": 0.95,  # Haar doesn't provide confidence
                "has_sunglasses": sunglasses_info["has_sunglasses"],
                "sunglasses_confidence": sunglasses_info["confidence"],
                "sunglasses_detection_method": sunglasses_info["method"],
                "eyes_detected": sunglasses_info["eyes_detected"]
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