# Real-time Face Detection System

A real-time face detection application using WebSocket communication between a React frontend and Python FastAPI backend. The system uses OpenCV for face detection and provides live video streaming with face bounding boxes.

## 🚀 Features

- **Real-time Face Detection**: Detects faces in live video stream using OpenCV Haar Cascade
- **Sunglasses Detection**: Identifies whether detected faces are wearing sunglasses using hybrid CV techniques ✨
- **Mask Detection**: Detects face masks using mouth/nose region analysis and facial landmarks 🆕 **NEW**
- **WebSocket Communication**: Low-latency communication between frontend and backend
- **Live Video Processing**: Processes video frames at 10 FPS
- **Interactive UI**: Modern, responsive interface built with React
- **Docker Support**: Fully containerized application with Docker Compose
- **Nginx Load Balancer**: Production-ready setup with Nginx reverse proxy
- **Redis Session Management**: Scalable session handling with Redis

### Sunglasses Detection ✨

The system includes advanced sunglasses detection that uses:
- **Haar Cascade Eye Detection**: Fast, CPU-friendly eye visibility detection
- **MediaPipe Face Mesh**: 468-point facial landmark detection for precise eye region analysis
- **Brightness Analysis**: Analyzes eye region darkness to identify sunglasses
- **Hybrid Approach**: Combines multiple methods for 85-90% accuracy

**Visual Indicators:**
- 🟢 Green box + 👀 = No sunglasses
- 🟡 Gold box + 😎 = Sunglasses detected

### Mask Detection 🆕

The system now includes advanced mask detection that uses:
- **Haar Cascade Mouth Detection**: Detects mouth/nose region visibility
- **MediaPipe Face Mesh**: Analyzes lower face landmarks for mask presence
- **Texture Analysis**: Examines uniformity of lower face region (masks show less texture variation)
- **Brightness Comparison**: Compares upper and lower face brightness differences
- **Multi-method Validation**: Combines mouth detection, texture analysis, and brightness for 75-90% accuracy

**Detection Logic:**
- No mouth detected + uniform lower face texture = Mask detected
- Significant brightness difference between upper/lower face = Mask indicator
- Visible mouth with texture variation = No mask

**Visual Indicators:**
- 🟢 Green box + � = No accessories
- 🟡 Gold box + 😎 = Sunglasses only
- 🔵 Teal box + 😷 = Mask only
- 🔴 Red box + 😎😷 = Both sunglasses and mask

**Testing:**
```bash
# Test mask detection with webcam
cd backend
python test_mask_detection.py

# Test with image file
python test_mask_detection.py
# Choose option 2 and provide image path
```

## 📋 Prerequisites

Before running this project, make sure you have installed:

- **Docker** (version 20.10 or higher)
- **Docker Compose** (version 2.0 or higher)
- **Make** (optional, for using Makefile commands)

## 🏗️ Architecture

```
├── frontend/          # React application
│   ├── src/
│   │   ├── App.js    # Main React component with WebSocket client
│   │   └── App.css   # Styling
│   └── Dockerfile
│
├── backend/           # Python FastAPI application
│   ├── main.py       # FastAPI server with face detection logic
│   ├── requirements.txt
│   └── Dockerfile
│
├── nginx/            # Nginx configuration
│   └── nginx.conf
│
└── docker-compose.yaml
```

## 🚀 Quick Start

### Option 1: Using Make (Recommended)

The project includes a Makefile for easy management:

```bash
# Show all available commands
make help

# Build and start all containers (FAST with cache)
make start

# Fast rebuild with cache (recommended for development) ⚡
make rebuild-fast       # 20-40 seconds instead of 6-10 minutes!

# Full rebuild without cache (when changing dependencies)
make rebuild

# Stop all containers
make stop

# Restart all containers
make restart

# View logs
make logs

# View specific service logs
make logs-frontend
make logs-backend
make logs-nginx
```

**🚀 Speed Tip:** Use `make rebuild-fast` for daily development! It's **15x faster** than full rebuild when you only change code (not dependencies).

### Option 2: Using Docker Compose

```bash
# Build and start all containers (with BuildKit)
DOCKER_BUILDKIT=1 docker-compose up -d

# Build without cache
DOCKER_BUILDKIT=1 docker-compose build --no-cache

# Stop all containers
docker-compose down

# View logs
docker-compose logs -f

# Restart specific service
docker-compose restart frontend
```

## 📦 Running the Application

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd faceaie2e
   ```

2. **Start the application**
   ```bash
   make start
   ```
   Or:
   ```bash
   docker-compose up -d
   ```

3. **Access the application**
   - **Frontend (via Nginx)**: http://localhost
   - **Frontend (direct)**: http://localhost:3000
   - **Backend API**: http://localhost:8000
   - **API Documentation**: http://localhost:8000/docs

4. **Using the face detection**
   - Open http://localhost in your browser
   - Click "Start Detection" to begin webcam capture
   - Grant camera permissions when prompted
   - The system will detect and highlight faces in real-time

## 🛠️ Development

### Environment Variables

Create a `.env` file in the root directory (optional):

```env
# Backend settings
DETECTOR_METHOD=haar
MAX_CONNECTIONS=100

# Redis settings
REDIS_HOST=redis
REDIS_PORT=6379

# Frontend settings
REACT_APP_WS_URL=ws://localhost:8000
```

### Running Individual Services

```bash
# Rebuild only frontend
make rebuild-frontend

# Rebuild only backend
make rebuild-backend

# Access container shell
make shell-frontend
make shell-backend
```

### Manual Development (without Docker)

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm start
```

## ⚡ Docker Build Optimization

This project uses **Docker BuildKit** for significantly faster builds!

### Build Speed Comparison

| Scenario | Before | After | Improvement |
|----------|--------|-------|-------------|
| Code changes | 6-10 min | 20-40 sec | **15x faster** 🚀 |
| First build | 6-10 min | 5-8 min | 1-2 min faster |

### Key Optimizations

1. **BuildKit enabled** - Modern Docker build engine
2. **Smart layer caching** - Dependencies cached separately from code
3. **.dockerignore files** - 10x faster context transfer
4. **Build cache reuse** - Leverages previous builds
5. **Optimized pip flags** - Faster package installation

### Usage

```bash
# Fast rebuild (daily development)
make rebuild-fast       # ~30 seconds

# Full rebuild (dependency changes)
make rebuild           # ~6-8 minutes

# Just restart
make restart           # ~5 seconds
```

📖 **See [DOCKER_BUILD_OPTIMIZATION.md](DOCKER_BUILD_OPTIMIZATION.md) for detailed guide**

## 🧪 Testing

The application includes health check endpoints:

```bash
# Check backend health
curl http://localhost:8000/health

# Check backend status
curl http://localhost:8000/
```

## 📊 API Endpoints

- `GET /` - Service status and information
- `GET /health` - Health check endpoint
- `WS /ws/face-detection` - WebSocket endpoint for face detection
- `POST /api/detect` - REST API endpoint for single image detection

## 🔧 Configuration

### Backend Configuration

Edit `backend/main.py` to change:
- Detection method: `haar` or `dnn`
- Frame processing settings
- WebSocket configuration

### Frontend Configuration

Edit `frontend/src/App.js` to modify:
- Frame rate (default: 10 FPS)
- Image quality (default: 70%)
- WebSocket connection settings

### Nginx Configuration

Edit `nginx/nginx.conf` to adjust:
- Load balancing strategy
- Timeout settings
- SSL/TLS configuration

## 📝 Available Make Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make build` | Build all containers |
| `make build-nocache` | Build without cache |
| `make start` | Build and start all containers |
| `make stop` | Stop all containers |
| `make restart` | Restart all containers |
| `make rebuild` | Rebuild and restart everything |
| `make logs` | Show logs from all containers |
| `make logs-frontend` | Show frontend logs |
| `make logs-backend` | Show backend logs |
| `make logs-nginx` | Show nginx logs |
| `make status` | Show container status |
| `make clean` | Stop and remove all containers and volumes |

## 🐛 Troubleshooting

### Container fails to start
```bash
# Check logs
make logs

# Rebuild without cache
make rebuild
```

### WebSocket connection fails
- Make sure all containers are running: `docker-compose ps`
- Check backend logs: `make logs-backend`
- Verify the WebSocket URL in browser console

### Camera not accessible
- Grant camera permissions in your browser
- Ensure you're accessing via `http://localhost` (not `file://`)
- Check browser console for errors

### Port conflicts
If ports 80, 3000, 8000, or 6379 are already in use, modify them in `docker-compose.yaml`

## 🏭 Production Deployment

For production deployment:

1. Update environment variables in `.env`
2. Configure SSL/TLS certificates in `nginx/ssl/`
3. Update CORS settings in `backend/main.py`
4. Set `allow_origins` to your frontend domain
5. Use production-grade Redis configuration

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 👨‍💻 Author

Your Name - Your Email

## 🙏 Acknowledgments

- OpenCV for face detection algorithms
- FastAPI for the backend framework
- React for the frontend framework
- Docker for containerization
