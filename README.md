# Camera Hub

A web app to watch RTSP cameras (Hikvision / EZVIZ and similar cameras) directly
in your browser + automatically send periodic photos and **video when a person
is detected** to Telegram.

## Features

- **Adding a camera is super easy**: just enter the **IP address + password**
  (the security pass printed on the camera's label). The app automatically
  builds the RTSP URL using the Hikvision/EZVIZ standard:

  ```
  rtsp://admin:<PASSWORD>@<IP>:554/Streaming/Channels/101   (main stream)
  rtsp://admin:<PASSWORD>@<IP>:554/Streaming/Channels/102   (sub stream)
  ```
- **Live view in the browser**: video is read via RTSP/TCP by ffmpeg and
  converted to MJPEG so it can be viewed in any browser, even for
  H.265/HEVC cameras.
- **Camera grid**: watch multiple cameras at once (using the sub stream to
  keep it lightweight), click one to enlarge and view the high-quality main
  stream.
- **Periodic Telegram photo sending**: on each cycle the bot takes a photo
  from **all cameras and sends them together as a single album in one go**.
  There's also a **📤 Send to Telegram** button in the UI to send
  immediately.
- **Send video when a person is detected**: each camera runs a YOLO model
  (YOLO11m on the NVIDIA GPU when available, YOLOv8s on CPU otherwise) to
  detect people in the sub stream. As
  soon as a person is seen, it records a clip (20s by default) and sends it
  to Telegram; if the person keeps appearing continuously, it keeps
  recording longer follow-up clips (60s by default, roughly once per
  minute) until the person leaves the frame. If there's no person, no video
  is sent.
- **Take a snapshot** to download to your device; **test the connection**
  before saving a camera.
- For cameras with a different standard (Dahua, ONVIF, etc.): adjust the
  account/port/stream path in the **Advanced settings** section.

## Running with Docker (recommended)

```bash
# 1. Create the Telegram configuration file
cp .env.example .env    # then fill in the token + chat ID

# 2. Start - auto-detects whether the machine has an NVIDIA GPU
./start.sh              # Linux/macOS
start.bat               # Windows (Docker Desktop)

# 3. Open http://localhost:8000
```

The start script probes Docker for an NVIDIA GPU and starts in GPU mode
(YOLO11m person detection, ~10ms/check) when available, CPU mode otherwise.
Plain `docker compose up -d --build` also works anywhere in CPU mode.

Per-platform GPU notes:

- **Windows**: nothing to install beyond Docker Desktop + the normal NVIDIA
  driver — GPU support is built into Docker Desktop (WSL2).
- **Linux**: one-time setup on GPU hosts:
  ```bash
  sudo apt install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
  sudo systemctl restart docker
  ```
- **Devices that will never have a GPU**: build a several-GB-slimmer image
  with `docker compose build --build-arg TORCH_FLAVOR=cpu`.

Camera data is stored in `./data/cameras.json`. The `.env` file is mounted
into the container, so **changing the Telegram configuration doesn't require
a rebuild** — it takes effect on the next send cycle.

View logs: `docker logs -f camera-hub` • Stop: `docker compose down`

## Running directly with Python

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# ffmpeg is required: sudo apt install ffmpeg
python3 app.py   # opens http://localhost:8000
```

Install the CPU build of `torch`/`torchvision` first (the line above) to
avoid `pip` automatically pulling the default CUDA build — which is many GB
heavier and unnecessary on a machine without a GPU.
When running directly, the camera list is stored in `cameras.json` next to
the source code.
The YOLO model weights (~20-40MB) are downloaded automatically into `data/` (or
`DATA_DIR`) on the first run — an internet connection is required at that
time; subsequent runs reuse the already downloaded file.

## Telegram configuration (.env)

```
TELEGRAM_BOT_TOKEN=<token from @BotFather>
TELEGRAM_CHAT_ID=<group ID, e.g. -100xxxxxxxxxx>
TELEGRAM_PHOTO_INTERVAL_MINUTES=10   # photo sending interval (minutes)
```

Notes:

- The bot must be **added to the group** first.
- If the group is upgraded by Telegram to a supergroup, the chat ID will
  change to the `-100...` format — get the new ID via
  `https://api.telegram.org/bot<TOKEN>/getUpdates`.
- Videos are converted to H.264 before sending so they can be played
  directly in Telegram.
- More than 10 cameras: the photo album is automatically split into
  multiple batches (Telegram's limit is 10 photos per album).

## Person detection & video sending (YOLO11m on GPU / YOLOv8s on CPU)

When an NVIDIA GPU is available (host has `nvidia-container-toolkit`
installed and the GPU block in `docker-compose.yml` is present), detection
runs **YOLO11m on the GPU at ~10ms per check** — far more accurate than the
CPU models and with zero impact on streaming performance. Without a GPU it
automatically falls back to YOLOv8s on CPU (`PERSON_DETECT_DEVICE` forces
either mode).

Each camera has its own thread reading a downscaled version of the video
(low fps) and only runs the AI when it detects **motion**, while someone is
being tracked in frame, or at most once every `PERSON_FORCE_CHECK_SECONDS`
seconds — this keeps the machine nearly idle when the frame is static,
saving resources for other tasks running in parallel.

The live view in the web UI draws a **bounding box around any person it
sees**, refreshed in near real time, straight from the same AI check - no
extra setup needed.

On the **first sighting** of a human, two things happen with zero delay:
an **instant alert photo** is sent to Telegram (with the bounding box drawn
on it, so you can judge at a glance whether it's real), and a
`PERSON_FIRST_CLIP_SECONDS`-second recording starts (20s by default) so the
footage catches them from the very first second.

The recorded **video** is only sent if the human stayed in frame for at
least `PERSON_MIN_DWELL_SECONDS` (4s by default) — a car or motorbike that
just flickers through the frame gets its clip silently discarded instead of
spamming the group. If the human is still around after the clip, it keeps
recording follow-up `PERSON_CONTINUOUS_CLIP_SECONDS`-second clips (60s by
default, roughly once per minute) until no one has been seen for
`PERSON_GRACE_SECONDS` seconds (8s by default). All of this can be
configured in `.env` — see `.env.example` for the full list of variables
(`PERSON_DETECT_MODEL`, `PERSON_DETECT_CONF`, `PERSON_DETECT_IMGSZ`,
`PERSON_DETECT_FPS`, `PERSON_MOTION_THRESHOLD`, `PERSON_MIN_DWELL_SECONDS`,
etc.). Set `PERSON_DETECT_ENABLED=false` to disable this feature entirely.

Getting false person alerts? Raise `PERSON_MIN_DWELL_SECONDS` (passing
vehicles never stay in frame long) and/or `PERSON_DETECT_CONF`. Real people
getting missed (seated, far away, partially hidden)? Lower
`PERSON_DETECT_CONF` or raise `PERSON_DETECT_IMGSZ` — and make sure you're
on a strong enough model (`yolo11m.pt` on GPU / `yolov8s.pt` on CPU), not the weaker `yolov8n.pt`.

## Testing a camera from the command line

```bash
ffplay -rtsp_transport tcp "rtsp://admin:<PASSWORD>@<IP>:554/Streaming/Channels/101"
```
