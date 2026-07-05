ARG BUILD_FROM=python:3.11-slim
FROM ${BUILD_FROM}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# libgl1/libglib2.0-0: required for opencv-python-headless to run on Debian slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU-only build of torch first (many GB lighter than the default
# CUDA build on PyPI) so ultralytics below reuses it instead of downloading the CUDA build
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py run.sh ./
COPY templates ./templates
RUN chmod +x /app/run.sh

EXPOSE 8000

CMD ["/app/run.sh"]
