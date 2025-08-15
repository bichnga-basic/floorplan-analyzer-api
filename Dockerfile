FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV FLASK_APP=app.py

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

# Debug: Verify installed packages
RUN dpkg -l | grep poppler
RUN dpkg -l | grep libglib
RUN dpkg -l | grep libsm
RUN dpkg -l | grep libxext
RUN dpkg -l | grep libxrender
RUN dpkg -l | grep libgomp

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Clean up pip cache
RUN pip cache purge

# Copy application files
COPY floorplan_analyzer.py .
COPY app.py .

# Create uploads directory with proper permissions
RUN mkdir -p uploads
RUN chmod -R 755 uploads

# Health check endpoint
EXPOSE 5000

# Start Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]
CMD ["/bin/sh", "-c", "ls -la && python -c 'from floorplan_analyzer import analyze_floorplan; print(\"✅ Imported analyze_floorplan\")' && gunicorn --bind 0.0.0.0:5000 --workers 1 --timeout 120 app:app"]