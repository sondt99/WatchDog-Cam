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
- **Send video when a person is detected**: each camera runs its own YOLOv8n
  model (CPU only, no GPU required) to detect people in the sub stream. As
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
cp .env.example .env   # then fill in the token + chat ID

# 2. Build and run
docker compose up -d --build

# 3. Open http://localhost:8000
```

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
The `yolov8n.pt` model (~6MB) is downloaded automatically into `data/` (or
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

## Person detection & video sending (YOLOv8n, running on CPU)

No GPU required. Each camera has its own thread reading a downscaled
version of the video (low fps, 2 frames per second by default) and only
runs the AI when it detects **motion** (or at most once every
`PERSON_FORCE_CHECK_SECONDS` seconds even without motion) — this keeps the
machine nearly idle when the frame is static, saving CPU for other tasks
running in parallel.

When a person is detected: it records a `PERSON_FIRST_CLIP_SECONDS`-second
clip (20s by default) then sends it to Telegram immediately. If the person
is still in frame, it keeps recording follow-up
`PERSON_CONTINUOUS_CLIP_SECONDS`-second clips (60s by default, roughly once
per minute) and continues until no person has been seen for
`PERSON_GRACE_SECONDS` seconds (8s by default). All of this can be
configured in `.env` — see `.env.example` for the full list of variables
(`PERSON_DETECT_MODEL`, `PERSON_DETECT_CONF`, `PERSON_DETECT_FPS`,
`PERSON_MOTION_THRESHOLD`, etc.). Set `PERSON_DETECT_ENABLED=false` to
disable this feature entirely.

## Testing a camera from the command line

```bash
ffplay -rtsp_transport tcp "rtsp://admin:<PASSWORD>@<IP>:554/Streaming/Channels/101"
```

## Note

The AI features previously in this repo have been removed — the full
technical summary is in
[AI_FEATURES_NOTES.txt](./AI_FEATURES_NOTES.txt).
