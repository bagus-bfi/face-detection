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
    """Face detection using OpenCV's Haar Cascade or DNN"""
    
    def __init__(self, method="haar"):
        self.method = method
        
        if method == "haar":
            # Using Haar Cascade (fast, CPU-friendly)
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        elif method == "dnn":
            # Using DNN model (more accurate)
            model_file = "models/res10_300x300_ssd_iter_140000.caffemodel"
            config_file = "models/deploy.prototxt"
            self.net = cv2.dnn.readNetFromCaffe(config_file, model_file)
        
        logger.info(f"FaceDetector initialized with method: {method}")
    
    def detect_faces_haar(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using Haar Cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        results = []
        for (x, y, w, h) in faces:
            results.append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "confidence": 0.95  # Haar doesn't provide confidence
            })
        
        return results
    
    def detect_faces_dnn(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using DNN model"""
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                results.append({
                    "x": int(x1),
                    "y": int(y1),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                    "confidence": float(confidence)
                })
        
        return results
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Main detection method"""
        if self.method == "haar":
            return self.detect_faces_haar(image)
        elif self.method == "dnn":
            return self.detect_faces_dnn(image)
        else:
            return []


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
detector = FaceDetector(method="haar")  # Change to "dnn" for better accuracy
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
        "detector": detector.method,
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