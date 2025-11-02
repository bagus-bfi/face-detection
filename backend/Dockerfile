FROM python:3.11-slim

WORKDIR /app

# Install system dependencies in one layer
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies with optimizations
# --no-cache-dir: Don't cache packages (saves space)
# --compile: Pre-compile Python files
# pip install with multiple workers for faster parallel downloads
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --compile -r requirements.txt

# Copy application code (this layer changes most often)
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]