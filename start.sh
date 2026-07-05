#!/bin/sh
# One-command start on any machine (Linux/macOS):
#   ./start.sh
# Probes whether Docker can expose an NVIDIA GPU; if yes, starts with GPU
# acceleration (YOLO11m person detection), otherwise starts CPU-only.
set -e
cd "$(dirname "$0")"

if ! docker info >/dev/null 2>&1; then
    echo "[!] Docker is not running (or you lack permission). Start the Docker service first." >&2
    exit 1
fi

if docker run --rm --gpus all alpine true >/dev/null 2>&1; then
    echo "NVIDIA GPU detected - starting with GPU acceleration"
    exec docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
else
    echo "No GPU available to Docker - starting in CPU mode"
    exec docker compose up -d --build
fi
