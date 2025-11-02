# Real-time Face Detection System

A real-time face detection application using WebSocket communication between a React frontend and Python FastAPI backend. The system uses OpenCV for face detection and provides live video streaming with face bounding boxes.

## 🚀 Features

- **Real-time Face Detection**: Detects faces in live video stream using OpenCV Haar Cascade
- **WebSocket Communication**: Low-latency communication between frontend and backend
- **Live Video Processing**: Processes video frames at 10 FPS
- **Interactive UI**: Modern, responsive interface built with React
- **Docker Support**: Fully containerized application with Docker Compose
- **Nginx Load Balancer**: Production-ready setup with Nginx reverse proxy
- **Redis Session Management**: Scalable session handling with Redis

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

# Build and start all containers
make start

# Stop all containers
make stop

# Restart all containers
make restart

# Rebuild everything without cache
make rebuild

# View logs
make logs

# View specific service logs
make logs-frontend
make logs-backend
make logs-nginx
```

### Option 2: Using Docker Compose

```bash
# Build and start all containers
docker-compose up -d

# Build without cache
docker-compose build --no-cache

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
