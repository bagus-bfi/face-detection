import React, { useRef, useState, useEffect } from "react";
import { Camera, Wifi, WifiOff, AlertCircle } from "lucide-react";
import "./App.css";

const FaceDetectionClient = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState("disconnected");
  const [detectionResults, setDetectionResults] = useState(null);
  const [latency, setLatency] = useState(0);
  const [frameCount, setFrameCount] = useState(0);

  // WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      // Determine WebSocket URL based on current location
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const host = window.location.hostname;
      const port = window.location.port;

      // If accessing through nginx (port 80/443), use /ws/ path
      // If accessing directly on port 3000, connect to backend on port 8000
      let wsUrl;
      if (port === "80" || port === "443" || port === "") {
        // Through nginx
        wsUrl = `${protocol}//${host}/ws/face-detection`;
      } else {
        // Direct access, connect to backend
        wsUrl = `ws://${host}:8000/ws/face-detection`;
      }

      console.log("Connecting to WebSocket:", wsUrl);
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log("WebSocket connected");
        setConnectionStatus("connected");
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        const receiveTime = Date.now();

        if (data.type === "detection_result") {
          setDetectionResults(data);
          setLatency(receiveTime - data.timestamp);

          // Draw bounding boxes
          if (data.faces && data.faces.length > 0) {
            drawBoundingBoxes(data.faces);
          }
        }
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        setConnectionStatus("error");
      };

      ws.onclose = () => {
        console.log("WebSocket disconnected");
        setConnectionStatus("disconnected");

        // Reconnect after 3 seconds
        setTimeout(() => {
          if (isStreaming) {
            connectWebSocket();
          }
        }, 3000);
      };

      wsRef.current = ws;
    };

    if (isStreaming) {
      connectWebSocket();
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [isStreaming]);

  // Start webcam
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setIsStreaming(true);
      }
    } catch (err) {
      console.error("Error accessing webcam:", err);
      alert("Unable to access webcam. Please grant permission.");
    }
  };

  // Stop webcam
  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
    }

    setIsStreaming(false);
    setDetectionResults(null);
  };

  // Capture and send frame
  const captureAndSend = () => {
    if (!videoRef.current || !canvasRef.current || !wsRef.current) return;
    if (wsRef.current.readyState !== WebSocket.OPEN) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to base64 (with compression)
    const base64Image = canvas.toDataURL("image/jpeg", 0.7);

    // Remove data:image/jpeg;base64, prefix
    const base64Data = base64Image.split(",")[1];

    // Send to backend
    const payload = {
      type: "frame",
      image: base64Data,
      timestamp: Date.now(),
      frame_number: frameCount,
    };

    wsRef.current.send(JSON.stringify(payload));
    setFrameCount((prev) => prev + 1);
  };

  // Send frames continuously
  useEffect(() => {
    if (!isStreaming) return;

    // Send frame every 100ms (10 FPS)
    const interval = setInterval(() => {
      captureAndSend();
    }, 100);

    return () => clearInterval(interval);
  }, [isStreaming, frameCount]);

  // Draw bounding boxes on overlay canvas
  const drawBoundingBoxes = (faces) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    faces.forEach((face, index) => {
      const {
        x,
        y,
        width,
        height,
        confidence,
        has_sunglasses,
        sunglasses_confidence,
      } = face;

      // Choose color based on sunglasses detection
      const boxColor = has_sunglasses ? "#FFD700" : "#00ff00"; // Gold for sunglasses, green for no sunglasses
      const labelBgColor = has_sunglasses
        ? "rgba(255, 215, 0, 0.8)"
        : "rgba(0, 255, 0, 0.8)";

      // Draw rectangle
      ctx.strokeStyle = boxColor;
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, width, height);

      // Draw label background
      ctx.fillStyle = labelBgColor;
      ctx.fillRect(x, y - 60, 200, 55);

      // Draw label text
      ctx.fillStyle = "#000";
      ctx.font = "14px Arial";
      ctx.fillText(
        `Face ${index + 1}: ${(confidence * 100).toFixed(1)}%`,
        x + 5,
        y - 40
      );

      // Draw sunglasses status
      ctx.font = "bold 13px Arial";
      const sunglassesText = has_sunglasses
        ? "😎 Sunglasses"
        : "👀 No Sunglasses";
      ctx.fillText(sunglassesText, x + 5, y - 22);

      // Draw sunglasses confidence
      ctx.font = "11px Arial";
      ctx.fillText(
        `Confidence: ${(sunglasses_confidence * 100).toFixed(1)}%`,
        x + 5,
        y - 7
      );
    });
  };

  const getStatusColor = () => {
    switch (connectionStatus) {
      case "connected":
        return "bg-green-500";
      case "disconnected":
        return "bg-red-500";
      case "error":
        return "bg-yellow-500";
      default:
        return "bg-gray-500";
    }
  };

  const getStatusIcon = () => {
    switch (connectionStatus) {
      case "connected":
        return <Wifi className="w-5 h-5" />;
      case "disconnected":
        return <WifiOff className="w-5 h-5" />;
      case "error":
        return <AlertCircle className="w-5 h-5" />;
      default:
        return <WifiOff className="w-5 h-5" />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="bg-gray-800 rounded-lg shadow-2xl overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-6">
            <h1 className="text-3xl font-bold text-white flex items-center gap-3">
              <Camera className="w-8 h-8" />
              Real-time Face & Sunglasses Detection
            </h1>
            <p className="text-blue-100 mt-2">
              WebSocket-based face detection with sunglasses recognition
            </p>
          </div>

          {/* Status Bar */}
          <div className="bg-gray-700 px-6 py-3 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div
                  className={`w-3 h-3 rounded-full ${getStatusColor()} animate-pulse`}
                ></div>
                <span className="text-white text-sm">
                  {connectionStatus.toUpperCase()}
                </span>
              </div>
              {getStatusIcon()}
            </div>

            <div className="flex items-center gap-6 text-sm text-gray-300">
              <div>
                Frames:{" "}
                <span className="text-white font-mono">{frameCount}</span>
              </div>
              <div>
                Latency:{" "}
                <span className="text-white font-mono">{latency}ms</span>
              </div>
              <div>
                Faces:{" "}
                <span className="text-white font-mono">
                  {detectionResults?.faces?.length || 0}
                </span>
              </div>
            </div>
          </div>

          {/* Video Container */}
          <div className="relative bg-black" style={{ aspectRatio: "16/9" }}>
            <video
              ref={videoRef}
              className="w-full h-full object-contain"
              playsInline
              muted
            />
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 w-full h-full object-contain pointer-events-none"
            />

            {!isStreaming && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <Camera className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                  <p className="text-gray-400 text-lg">Camera not started</p>
                </div>
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="p-6 bg-gray-750">
            <div className="flex gap-4">
              {!isStreaming ? (
                <button
                  onClick={startCamera}
                  className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
                >
                  <Camera className="w-5 h-5" />
                  Start Detection
                </button>
              ) : (
                <button
                  onClick={stopCamera}
                  className="flex-1 bg-red-600 hover:bg-red-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
                >
                  Stop Detection
                </button>
              )}
            </div>
          </div>

          {/* Detection Results */}
          {detectionResults &&
            detectionResults.faces &&
            detectionResults.faces.length > 0 && (
              <div className="p-6 bg-gray-750 border-t border-gray-700">
                <h3 className="text-white font-semibold mb-3">
                  Detection Results
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {detectionResults.faces.map((face, index) => (
                    <div key={index} className="bg-gray-700 rounded-lg p-4">
                      <div className="text-sm text-gray-300">
                        <div className="font-semibold text-white mb-2 flex items-center gap-2">
                          <span>Face {index + 1}</span>
                          {face.has_sunglasses ? (
                            <span className="text-yellow-400">😎</span>
                          ) : (
                            <span className="text-green-400">👀</span>
                          )}
                        </div>
                        <div>
                          Confidence:{" "}
                          <span className="text-green-400">
                            {(face.confidence * 100).toFixed(2)}%
                          </span>
                        </div>
                        <div>
                          Position: ({face.x}, {face.y})
                        </div>
                        <div>
                          Size: {face.width}x{face.height}
                        </div>
                        <div className="mt-2 pt-2 border-t border-gray-600">
                          <div className="font-semibold text-white">
                            {face.has_sunglasses
                              ? "Wearing Sunglasses"
                              : "No Sunglasses"}
                          </div>
                          <div>
                            S. Confidence:{" "}
                            <span
                              className={
                                face.has_sunglasses
                                  ? "text-yellow-400"
                                  : "text-blue-400"
                              }
                            >
                              {(face.sunglasses_confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          {face.eyes_detected !== undefined && (
                            <div>Eyes detected: {face.eyes_detected}</div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
        </div>

        {/* Info Panel */}
        <div className="mt-6 bg-gray-800 rounded-lg p-6">
          <h3 className="text-white font-semibold mb-3">System Information</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="text-gray-400">
              <div className="font-semibold text-white">Protocol</div>
              <div>WebSocket</div>
            </div>
            <div className="text-gray-400">
              <div className="font-semibold text-white">Format</div>
              <div>Base64 JPEG</div>
            </div>
            <div className="text-gray-400">
              <div className="font-semibold text-white">Frame Rate</div>
              <div>10 FPS</div>
            </div>
            <div className="text-gray-400">
              <div className="font-semibold text-white">Quality</div>
              <div>70%</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FaceDetectionClient;
