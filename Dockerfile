ARG BUILD_FROM=python:3.11-slim
FROM ${BUILD_FROM}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# libgl1/libglib2.0-0: required for opencv-python-headless to run on Debian slim
# xz-utils/curl: to unpack the static ffmpeg below
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 curl xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Static ffmpeg with NVDEC/NVENC support (Debian's ffmpeg is built without
# NVIDIA hardware codecs). Installed to /usr/local/bin so it takes PATH
# precedence; the apt ffmpeg above stays as fallback. On machines without an
# NVIDIA GPU this build still works - it just decodes/encodes on CPU.
RUN curl -fsSL https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz -o /tmp/ffmpeg.tar.xz \
    && tar -xf /tmp/ffmpeg.tar.xz -C /tmp \
    && cp /tmp/ffmpeg-*/bin/ffmpeg /tmp/ffmpeg-*/bin/ffprobe /usr/local/bin/ \
    && rm -rf /tmp/ffmpeg*

WORKDIR /app

# Torch flavor - the same image runs everywhere, pick at build time:
#   cu124 (default): CUDA build; uses the NVIDIA GPU when the host exposes one
#     (docker-compose.gpu.yml + nvidia-container-toolkit), falls back to CPU
#     automatically on machines without a GPU.
#   cpu: several GB slimmer image for devices that will never have a GPU:
#     docker compose build --build-arg TORCH_FLAVOR=cpu
ARG TORCH_FLAVOR=cu124
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/${TORCH_FLAVOR}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py run.sh ./
COPY templates ./templates
RUN chmod +x /app/run.sh

EXPOSE 8000

CMD ["/app/run.sh"]
