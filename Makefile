.PHONY: help build up down restart logs logs-frontend logs-backend logs-nginx clean rebuild start stop status

# Default target
help:
	@echo "Face Detection Docker Commands"
	@echo "=============================="
	@echo "make build          - Build all containers"
	@echo "make build-nocache  - Build all containers without cache"
	@echo "make up             - Start all containers"
	@echo "make start          - Build and start all containers"
	@echo "make down           - Stop and remove all containers"
	@echo "make stop           - Stop all containers"
	@echo "make restart        - Restart all containers"
	@echo "make rebuild        - Rebuild and restart all containers"
	@echo "make logs           - Show logs from all containers"
	@echo "make logs-frontend  - Show frontend logs"
	@echo "make logs-backend   - Show backend logs"
	@echo "make logs-nginx     - Show nginx logs"
	@echo "make status         - Show container status"
	@echo "make clean          - Stop containers and remove volumes"
	@echo "make shell-frontend - Open shell in frontend container"
	@echo "make shell-backend  - Open shell in backend container"

# Build containers
build:
	docker-compose build

# Build without cache
build-nocache:
	docker-compose build --no-cache

# Start containers
up:
	docker-compose up -d

# Build and start
start: build up
	@echo "✅ Containers started successfully"
	@echo "Frontend: http://localhost:3000"
	@echo "Backend: http://localhost:8000"
	@echo "Nginx: http://localhost"

# Stop containers
down:
	docker-compose down

# Stop without removing
stop:
	docker-compose stop

# Restart containers
restart:
	docker-compose restart
	@echo "✅ Containers restarted"

# Rebuild everything
rebuild:
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d
	@echo "✅ Containers rebuilt and started"

# Show logs
logs:
	docker-compose logs -f

logs-frontend:
	docker-compose logs -f frontend

logs-backend:
	docker-compose logs -f backend

logs-nginx:
	docker-compose logs -f nginx

logs-redis:
	docker-compose logs -f redis

# Show status
status:
	docker-compose ps

# Clean everything
clean:
	docker-compose down -v
	@echo "✅ Containers and volumes removed"

# Shell access
shell-frontend:
	docker exec -it face-detection-frontend sh

shell-backend:
	docker exec -it face-detection-backend sh

shell-nginx:
	docker exec -it face-detection-nginx sh

# Quick rebuild frontend only
rebuild-frontend:
	docker-compose build --no-cache frontend
	docker-compose up -d frontend
	@echo "✅ Frontend rebuilt and restarted"

# Quick rebuild backend only
rebuild-backend:
	docker-compose build --no-cache backend
	docker-compose up -d backend
	@echo "✅ Backend rebuilt and restarted"

# Development mode - rebuild and show logs
dev: rebuild logs
