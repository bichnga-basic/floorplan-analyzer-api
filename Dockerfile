FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        poppler-utils \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY floorplan_analyzer.py .

# Create uploads directory
RUN mkdir -p uploads
RUN chmod -R 755 uploads

EXPOSE 5000

# Debug startup
CMD ["/bin/sh", "-c", "ls -la && echo '--- Files in /app ---' && find /app -type f && echo '--- Python Path ---' && python -c \"import sys; print(sys.path)\" && echo '--- Testing import of app.py ---' && python -c \"from app import app; print(\"✅ Imported app successfully\")\" && echo '--- Starting Gunicorn ---' && gunicorn --bind 0.0.0.0:5000 --workers 1 --timeout 120 app:app"]