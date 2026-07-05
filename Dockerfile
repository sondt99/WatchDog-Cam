ARG BUILD_FROM=python:3.11-slim
FROM ${BUILD_FROM}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# libgl1/libglib2.0-0: can cho opencv-python-headless chay duoc tren Debian slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cai ban CPU-only cua torch truoc (nhe hon nhieu GB so voi ban CUDA mac dinh
# tren PyPI) de ultralytics ben duoi tan dung lai, khong tai ban CUDA nua
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py run.sh ./
COPY templates ./templates
RUN chmod +x /app/run.sh

EXPOSE 8000

CMD ["/app/run.sh"]
