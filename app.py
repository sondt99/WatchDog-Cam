"""Camera Hub - Flask web app for viewing RTSP cameras (Hikvision/EZVIZ).

Adding a camera only requires entering the IP + password; by default it uses the format:
    rtsp://admin:<password>@<ip>:554/Streaming/Channels/101  (main stream)
    rtsp://admin:<password>@<ip>:554/Streaming/Channels/102  (sub stream)

Video is read by ffmpeg over RTSP/TCP and converted to MJPEG for direct viewing
in the browser (no plugin needed, works even with H.265/HEVC cameras).

Sends periodic photos + automatically sends video when a person is detected
(YOLOv8n, CPU) to Telegram.
"""

import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid

import cv2
import numpy as np
import requests
from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration & storage
# ---------------------------------------------------------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))
# When running inside a Home Assistant addon, /data is the persistent storage directory
DATA_DIR = os.environ.get("DATA_DIR") or ("/data" if os.path.isdir("/data") else APP_DIR)
CAMERAS_FILE = os.path.join(DATA_DIR, "cameras.json")

DEFAULT_USERNAME = "admin"
DEFAULT_PORT = 554
DEFAULT_PATH_MAIN = "/Streaming/Channels/101"
DEFAULT_PATH_SUB = "/Streaming/Channels/102"

# Limit the stream to save CPU
STREAM_FPS = 10
MAIN_MAX_WIDTH = 1280      # main stream scaled down to at most 1280px
JPEG_QUALITY = "7"         # 2 (best) .. 31 (worst)
WORKER_IDLE_STOP_SECONDS = 15  # stop ffmpeg after the last client leaves
                               # (> the UI's 6-8s retry cadence, to reuse the still-warm worker)
FRAME_STALL_TIMEOUT = 15       # no new frame within N seconds -> treat as disconnected

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("camera-hub")

# Use only 1 CPU thread for OpenCV: the machine also runs other tasks, don't
# let opencv/YOLO hog the entire CPU core
cv2.setNumThreads(1)


# Variables already present in the real environment (before loading .env) always take priority
_REAL_ENV_KEYS = frozenset(os.environ)


def _env_file_values():
    """Read values from the .env file (APP_DIR first, DATA_DIR overrides on key conflict)."""
    values = {}
    for path in (os.path.join(APP_DIR, ".env"), os.path.join(DATA_DIR, ".env")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key:
                values[key] = value.strip().strip('"').strip("'")
    return values


def _env(name, default=""):
    """Get config: real env > .env file (re-read on every call) > options.json."""
    if name in _REAL_ENV_KEYS:
        return os.environ.get(name, default)
    file_values = _env_file_values()
    if name in file_values:
        return file_values[name]
    return os.environ.get(name, default)


def _load_env_files():
    """Load .env into environ at startup (for variables read only once, like PORT)."""
    for key, value in _env_file_values().items():
        os.environ.setdefault(key, value)


def _load_addon_options():
    """Home Assistant addon: read config from /data/options.json if present."""
    try:
        with open("/data/options.json", "r", encoding="utf-8") as f:
            options = json.load(f)
    except (OSError, ValueError):
        return
    if isinstance(options, dict):
        for key, value in options.items():
            if value not in (None, ""):
                os.environ.setdefault(key.upper(), str(value))


_load_env_files()
_load_addon_options()

app = Flask(__name__)

_store_lock = threading.Lock()


def _load_cameras():
    try:
        with open(CAMERAS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        cams = data.get("cameras", [])
        return cams if isinstance(cams, list) else []
    except (OSError, ValueError):
        return []


def _save_cameras(cams):
    tmp = CAMERAS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"cameras": cams}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CAMERAS_FILE)


def _get_camera(cam_id):
    for cam in _load_cameras():
        if cam.get("id") == cam_id:
            return cam
    return None


def _rtsp_url(cam, src="main"):
    username = cam.get("username") or DEFAULT_USERNAME
    password = cam.get("password") or ""
    host = cam["host"]
    port = cam.get("port") or DEFAULT_PORT
    if src == "sub":
        path = cam.get("path_sub") or DEFAULT_PATH_SUB
    else:
        path = cam.get("path_main") or DEFAULT_PATH_MAIN
    if not path.startswith("/"):
        path = "/" + path
    return f"rtsp://{username}:{password}@{host}:{port}{path}"


_HOST_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_camera_payload(payload, existing=None):
    """Validate add/edit camera data. Returns (camera_dict, error_message)."""
    if not isinstance(payload, dict):
        return None, "Invalid data"

    name = str(payload.get("name") or "").strip()
    host = str(payload.get("host") or "").strip()
    password = str(payload.get("password") or "")
    username = str(payload.get("username") or "").strip() or DEFAULT_USERNAME
    path_main = str(payload.get("path_main") or "").strip() or DEFAULT_PATH_MAIN
    path_sub = str(payload.get("path_sub") or "").strip() or DEFAULT_PATH_SUB

    if not host:
        return None, "Please enter the camera's IP address"
    if not _HOST_RE.match(host):
        return None, "Invalid IP address/hostname"

    try:
        port = int(payload.get("port") or DEFAULT_PORT)
        if not 1 <= port <= 65535:
            raise ValueError
    except (TypeError, ValueError):
        return None, "Invalid RTSP port"

    if existing:
        # Editing camera: leaving password blank = keep the old password
        if not password:
            password = existing.get("password", "")
    if not password:
        return None, "Please enter the camera password (pass security)"

    if not name:
        name = f"Camera {host}"

    cam = {
        "id": existing["id"] if existing else uuid.uuid4().hex[:12],
        "name": name,
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "path_main": path_main,
        "path_sub": path_sub,
    }
    return cam, None


def _public_camera(cam):
    """Camera copy to return to the frontend - doesn't leak the password."""
    out = {k: v for k, v in cam.items() if k != "password"}
    out["has_password"] = bool(cam.get("password"))
    return out


# ---------------------------------------------------------------------------
# ffmpeg: read RTSP -> MJPEG, share 1 process across multiple viewers
# ---------------------------------------------------------------------------


class StreamWorker:
    """One ffmpeg process reads 1 RTSP stream, broadcasts JPEG frames to each client."""

    def __init__(self, url, max_width=None, fps=STREAM_FPS):
        self.url = url
        self.max_width = max_width
        self.fps = fps
        self.cond = threading.Condition()
        self.frame = None
        self.frame_id = 0
        self.last_frame_at = 0.0
        self.clients = 0
        self.running = True
        self.proc = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _build_cmd(self):
        vf = f"fps={self.fps}"
        if self.max_width:
            vf += f",scale='min({self.max_width},iw)':-2"
        # Don't use -fflags nobuffer: with HEVC it makes ffmpeg decode mid-
        # GOP before the keyframe arrives -> a batch of gray frames on every new connection
        return [
            "ffmpeg", "-nostdin", "-loglevel", "error",
            "-rtsp_transport", "tcp",
            # Shorten the probing time on each new connection (codec already known from SDP)
            "-probesize", "1000000",
            "-analyzeduration", "1000000",
            "-i", self.url,
            "-an",
            "-vf", vf,
            "-c:v", "mjpeg", "-q:v", JPEG_QUALITY,
            "-f", "mjpeg", "-",
        ]

    def _run(self):
        try:
            self.proc = subprocess.Popen(
                self._build_cmd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            stdout = self.proc.stdout
            if stdout is None:
                return
            buf = b""
            fd = stdout.fileno()
            while self.running:
                # os.read returns as soon as data is available (read() usually waits until
                # 64KB is filled before returning -> frames get batched, lowering actual fps)
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                buf += chunk
                # Split JPEG images by SOI (FFD8) / EOI (FFD9) markers
                while True:
                    soi = buf.find(b"\xff\xd8")
                    if soi < 0:
                        buf = buf[-1:]
                        break
                    eoi = buf.find(b"\xff\xd9", soi + 2)
                    if eoi < 0:
                        if soi > 0:
                            buf = buf[soi:]
                        break
                    frame = buf[soi:eoi + 2]
                    buf = buf[eoi + 2:]
                    with self.cond:
                        self.frame = frame
                        self.frame_id += 1
                        self.last_frame_at = time.time()
                        self.cond.notify_all()
        finally:
            self.running = False
            with self.cond:
                self.cond.notify_all()
            self._kill_proc()

    def _kill_proc(self):
        proc = self.proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    proc.kill()
                except OSError:
                    pass

    def stop(self):
        self.running = False
        self._kill_proc()
        with self.cond:
            self.cond.notify_all()


_workers = {}
_workers_lock = threading.Lock()


def _acquire_worker(cam, src, fps=STREAM_FPS):
    key = (cam["id"], src, fps)
    url = _rtsp_url(cam, src)
    with _workers_lock:
        worker = _workers.get(key)
        if worker is None or not worker.running or worker.url != url:
            if worker is not None:
                worker.stop()
            max_width = MAIN_MAX_WIDTH if src == "main" else None
            worker = StreamWorker(url, max_width=max_width, fps=fps)
            _workers[key] = worker
        worker.clients += 1
        return key, worker


def _release_worker(key, worker):
    with _workers_lock:
        worker.clients -= 1
        if worker.clients > 0:
            return

    def _stop_if_idle():
        time.sleep(WORKER_IDLE_STOP_SECONDS)
        with _workers_lock:
            current = _workers.get(key)
            if current is worker and worker.clients <= 0:
                worker.stop()
                _workers.pop(key, None)

    threading.Thread(target=_stop_if_idle, daemon=True).start()


def _mjpeg_stream(cam, src, fps=STREAM_FPS):
    key, worker = _acquire_worker(cam, src, fps)
    last_id = 0
    try:
        while True:
            with worker.cond:
                worker.cond.wait_for(
                    lambda: worker.frame_id != last_id or not worker.running,
                    timeout=FRAME_STALL_TIMEOUT,
                )
                if not worker.running:
                    break
                frame = worker.frame
                if worker.frame_id == last_id or frame is None:
                    break  # too long without a new frame -> disconnect so the client reconnects
                last_id = worker.frame_id
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                + frame + b"\r\n"
            )
    finally:
        _release_worker(key, worker)


def _snapshot(cam, src="main", timeout=15):
    """Capture 1 JPEG image from the camera."""
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-rtsp_transport", "tcp",
        "-i", _rtsp_url(cam, src),
        "-frames:v", "1", "-q:v", "3",
        "-f", "image2", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    return result.stdout


def _probe(url, timeout=12):
    """Check the RTSP connection using ffprobe. Returns (info_dict, error_message)."""
    cmd = [
        "ffprobe", "-v", "error",
        "-rtsp_transport", "tcp",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate",
        "-of", "json",
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "Timed out - camera is not responding"
    if result.returncode != 0:
        err = (result.stderr or b"").decode("utf-8", "replace").strip()
        if "401" in err or "Unauthorized" in err:
            return None, "Wrong username or password (401 Unauthorized)"
        if "404" in err or "Not Found" in err:
            return None, "Video stream not found (404) - check the RTSP path"
        if "Connection refused" in err:
            return None, "Camera refused the connection - check the IP and RTSP port"
        return None, err.splitlines()[-1] if err else "Could not connect to the camera"
    try:
        streams = json.loads(result.stdout).get("streams", [])
        if not streams:
            return None, "Video stream not found"
        s = streams[0]
        fps = s.get("avg_frame_rate", "0/1")
        try:
            num, den = fps.split("/")
            fps_val = round(int(num) / int(den)) if int(den) else 0
        except (ValueError, ZeroDivisionError):
            fps_val = 0
        return {
            "codec": s.get("codec_name", "?"),
            "width": s.get("width", 0),
            "height": s.get("height", 0),
            "fps": fps_val,
        }, None
    except ValueError:
        return None, "Could not read the video stream information"


# ---------------------------------------------------------------------------
# Telegram: send periodic photos + videos to the group
# ---------------------------------------------------------------------------

TELEGRAM_API = "https://api.telegram.org"

# A separate topic (forum) for each camera + 1 shared topic for periodic photos.
# Only works if the group has "Topics" enabled and the bot has "Manage
# Topics" permission - if not, these functions return None and messages
# land in the "General" topic as usual (no error).
TELEGRAM_TOPICS_FILE = os.path.join(DATA_DIR, "telegram_topics.json")
PHOTOS_TOPIC_KEY = "photos"
PHOTOS_TOPIC_NAME = "\U0001F4F7 Photos"
TOPIC_RETRY_SECONDS = 300  # topic creation failed -> wait this many seconds before retrying, to avoid spamming the API with errors

_topics_lock = threading.Lock()
_topic_retry_after = {}


def _telegram_settings():
    """Read Telegram config - re-reads .env on every call so editing the file
    takes effect on the next cycle immediately, without restarting."""
    def _num(name, default, cast):
        try:
            return cast(_env(name) or default)
        except (TypeError, ValueError):
            return default

    token = _env("TELEGRAM_BOT_TOKEN").strip()
    chat_id = _env("TELEGRAM_CHAT_ID").strip()
    enabled_raw = _env("TELEGRAM_ENABLED").strip().lower()
    # Video source when a person is detected: "sub" (sub stream - lighter on bandwidth, default) or "main"
    video_source = "main" if _env("TELEGRAM_VIDEO_SOURCE").strip().lower() == "main" else "sub"
    base_interval = max(1.0, _num("TELEGRAM_INTERVAL_MINUTES", 10.0, float))
    return {
        "token": token,
        "chat_id": chat_id,
        "enabled": bool(token and chat_id) and enabled_raw not in ("0", "false", "no", "off"),
        "photo_interval_minutes": max(1.0, _num("TELEGRAM_PHOTO_INTERVAL_MINUTES", base_interval, float)),
        "video_source": video_source,
    }


def _tg_send(settings, method, data, files=None, timeout=120):
    url = f"{TELEGRAM_API}/bot{settings['token']}/{method}"
    resp = requests.post(url, data=data, files=files, timeout=timeout)
    try:
        payload = resp.json()
    except ValueError:
        raise RuntimeError(f"Telegram {method}: HTTP {resp.status_code}")
    if not payload.get("ok"):
        raise RuntimeError(f"Telegram {method}: {payload.get('description') or resp.status_code}")
    return payload.get("result")


def _load_topics():
    try:
        with open(TELEGRAM_TOPICS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_topics(topics):
    tmp = TELEGRAM_TOPICS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)
    os.replace(tmp, TELEGRAM_TOPICS_FILE)


def _camera_topic_key(cam_id):
    return f"camera:{cam_id}"


def _create_forum_topic(settings, name):
    try:
        result = _tg_send(settings, "createForumTopic", {
            "chat_id": settings["chat_id"],
            "name": name[:128],
        })
        return result.get("message_thread_id")
    except Exception as exc:
        log.warning("Telegram: failed to create topic '%s': %s", name, exc)
        return None


def _get_topic_thread_id(settings, key, name):
    """Return the message_thread_id for a topic (from the cache file, or by creating
    a new one if it doesn't exist yet). If the group hasn't enabled Topics or the
    bot doesn't have topic-management permission, topic creation will fail - in
    that case wait TOPIC_RETRY_SECONDS before retrying (instead of calling the
    failing API repeatedly), and return None so the message still gets sent
    normally (without a message_thread_id)."""
    with _topics_lock:
        topics = _load_topics()
        thread_id = topics.get(key)
        if thread_id:
            return thread_id
        now = time.monotonic()
        if _topic_retry_after.get(key, 0) > now:
            return None
        thread_id = _create_forum_topic(settings, name)
        if thread_id is None:
            _topic_retry_after[key] = now + TOPIC_RETRY_SECONDS
            return None
        topics[key] = thread_id
        _save_topics(topics)
        log.info("Telegram: created topic '%s' (thread_id=%s)", name, thread_id)
        return thread_id


def _photos_topic_thread_id(settings):
    return _get_topic_thread_id(settings, PHOTOS_TOPIC_KEY, PHOTOS_TOPIC_NAME)


def _camera_topic_thread_id(settings, cam):
    return _get_topic_thread_id(settings, _camera_topic_key(cam["id"]), cam["name"])


def _ensure_all_topics():
    """Pre-create the topic for photos + all existing cameras (instead of waiting
    until a person is detected/the photo cycle to create them) - runs in the
    background at startup."""
    settings = _telegram_settings()
    if not settings["enabled"]:
        return
    _photos_topic_thread_id(settings)
    for cam in _load_cameras():
        _camera_topic_thread_id(settings, cam)


def _record_clip(cam, seconds, src="sub"):
    """Record a short video clip from the camera, return the mp4 (H.264) file path.

    Records from the sub stream by default: ~6x lighter on bandwidth than the
    main stream, good enough for viewing on a phone. Set TELEGRAM_VIDEO_SOURCE=main
    if you need higher resolution.
    """
    fd, path = tempfile.mkstemp(suffix=".mp4", prefix="camera_hub_")
    os.close(fd)
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-rtsp_transport", "tcp",
        # EZVIZ cameras timestamp video/audio streams hours apart ->
        # without this flag, -t would cut out all the audio
        "-use_wallclock_as_timestamps", "1",
        "-i", _rtsp_url(cam, src),
        "-t", str(seconds),
        "-vf", "scale='min(1280,iw)':-2",
        # Force H.264 because Telegram/browsers can't play HEVC directly
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
        "-c:a", "aac", "-b:a", "64k",
        "-movflags", "+faststart",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=seconds + 60)
        if result.returncode == 0 and os.path.getsize(path) > 0:
            return path
        err = (result.stderr or b"").decode("utf-8", "replace").strip()
        log.warning("Failed to record clip for camera %s: %s", cam["name"], err.splitlines()[-1] if err else "?")
    except subprocess.TimeoutExpired:
        log.warning("Recording clip for camera %s timed out", cam["name"])
    try:
        os.remove(path)
    except OSError:
        pass
    return None


def _send_photo_album(settings, photos, stamp):
    """Send all photos in one go (album). Telegram limits albums to 10 photos,
    so if there are more cameras, split into multiple consecutive albums.
    All photos (per camera) are sent together into 1 "Photos" topic."""
    thread_id = _photos_topic_thread_id(settings)
    for start in range(0, len(photos), 10):
        chunk = photos[start:start + 10]
        if len(chunk) == 1:
            cam, data = chunk[0]
            data_form = {"chat_id": settings["chat_id"], "caption": f"\U0001F4F7 {cam['name']} • {stamp}"}
            if thread_id:
                data_form["message_thread_id"] = thread_id
            _tg_send(
                settings, "sendPhoto", data_form,
                files={"photo": (f"{cam['host']}.jpg", data, "image/jpeg")},
            )
            continue
        media = []
        files = {}
        for idx, (cam, data) in enumerate(chunk):
            key = f"photo{idx}"
            media.append({
                "type": "photo",
                "media": f"attach://{key}",
                "caption": f"\U0001F4F7 {cam['name']} • {stamp}",
            })
            files[key] = (f"{cam['host']}.jpg", data, "image/jpeg")
        data_form = {"chat_id": settings["chat_id"], "media": json.dumps(media)}
        if thread_id:
            data_form["message_thread_id"] = thread_id
        _tg_send(
            settings, "sendMediaGroup", data_form,
            files=files,
        )


_telegram_report_lock = threading.Lock()


def _send_photo_batch(settings):
    """Capture photos from ALL cameras then send them together in one batch (album).
    Returns results per camera (dict cam_id -> row)."""
    cams = _load_cameras()
    stamp = time.strftime("%d/%m/%Y %H:%M:%S")
    rows = {c["id"]: {"camera": c["name"], "ok": True, "sent": []} for c in cams}

    photos = []
    for cam in cams:
        data = _snapshot(cam, "main")
        if data:
            photos.append((cam, data))
        else:
            rows[cam["id"]]["ok"] = False
            rows[cam["id"]]["error"] = "could not capture photo"
            log.warning("Telegram: could not capture photo for camera %s", cam["name"])
    if photos:
        try:
            _send_photo_album(settings, photos, stamp)
            for cam, _ in photos:
                rows[cam["id"]]["sent"].append("photo")
        except Exception as exc:
            for cam, _ in photos:
                rows[cam["id"]]["ok"] = False
                rows[cam["id"]]["error"] = str(exc)
            log.warning("Telegram: error sending photo album: %s", exc)
    failed_snap = [r["camera"] for r in rows.values() if r.get("error") == "could not capture photo"]
    if failed_snap:
        try:
            data_form = {
                "chat_id": settings["chat_id"],
                "text": f"⚠️ Could not capture photo from: {', '.join(failed_snap)} ({stamp})",
            }
            thread_id = _photos_topic_thread_id(settings)
            if thread_id:
                data_form["message_thread_id"] = thread_id
            _tg_send(settings, "sendMessage", data_form)
        except Exception as exc:
            log.warning("Telegram: error sending warning: %s", exc)
    log.info("Telegram [photo]: sent album %d/%d cameras", len(photos), len(cams))
    return rows


def _send_all_reports():
    """Immediately send a photo album of all cameras (used by the 'Send to Telegram'
    button on the web UI). Video is no longer sent on demand - it's only sent
    automatically when a person is detected in the frame, see the "Person
    detection" section below."""
    settings = _telegram_settings()
    if not settings["enabled"]:
        return {"enabled": False, "results": []}
    with _telegram_report_lock:
        photo_rows = _send_photo_batch(settings)
    return {"enabled": True, "results": list(photo_rows.values())}


def _send_person_clip(settings, cam, clip_path, seconds):
    """Send a recorded clip (with a person) to Telegram then delete the temp file.
    Each camera sends to its own topic (if the group has Topics enabled)."""
    stamp = time.strftime("%d/%m/%Y %H:%M:%S")
    try:
        with open(clip_path, "rb") as f:
            data_form = {
                "chat_id": settings["chat_id"],
                "caption": f"\U0001F6A8 Person detected • {cam['name']} • {stamp} ({seconds}s)",
                "supports_streaming": "true",
            }
            thread_id = _camera_topic_thread_id(settings, cam)
            if thread_id:
                data_form["message_thread_id"] = thread_id
            _tg_send(
                settings, "sendVideo", data_form,
                files={"video": (f"{cam['host']}.mp4", f, "video/mp4")},
                timeout=300,
            )
        log.info("Telegram [person]: sent clip for camera %s (%ds)", cam["name"], seconds)
    except Exception as exc:
        log.warning("Telegram: error sending person-detection clip for camera %s: %s", cam["name"], exc)
    finally:
        try:
            os.remove(clip_path)
        except OSError:
            pass


def _run_periodic(kind, batch_fn, interval_key, first_delay):
    """Periodic sending loop. The cycle is timed from the START of each round to keep
    the correct cadence even if a round runs long; wait at least 20s between rounds."""
    time.sleep(first_delay)
    while True:
        started = time.monotonic()
        settings = _telegram_settings()
        if settings["enabled"]:
            try:
                with _telegram_report_lock:
                    batch_fn(settings)
            except Exception as exc:
                log.warning("Telegram [%s]: error during send round: %s", kind, exc)
        elapsed = time.monotonic() - started
        time.sleep(max(20, settings[interval_key] * 60 - elapsed))


def _photo_worker():
    _run_periodic("photo", _send_photo_batch, "photo_interval_minutes", first_delay=15)


def _start_telegram_workers():
    settings = _telegram_settings()
    if settings["enabled"]:
        log.info(
            "Telegram: photo every %.0f minutes (album) | video: automatic on person detection (source %s) | chat %s",
            settings["photo_interval_minutes"], settings["video_source"], settings["chat_id"],
        )
    else:
        log.info("Telegram: TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not configured - periodic sending disabled")
    threading.Thread(target=_ensure_all_topics, daemon=True).start()
    threading.Thread(target=_photo_worker, daemon=True).start()


# ---------------------------------------------------------------------------
# Person detection (YOLOv8n, CPU) -> record clip -> send to Telegram
#
# Each camera has 2 threads:
#  - _detection_loop: reads the sub stream at a low fps, only runs YOLO when
#    there's motion (or it's been too long since the last check) so it
#    doesn't hog the CPU when the frame is static - which is most of the
#    time for a security camera.
#  - _recorder_loop: waits until a person appears, then records a clip
#    (default 20s) and sends it to Telegram; if the person keeps appearing
#    continuously, it chains longer clips (default 60s) until the person
#    leaves the frame.
# ---------------------------------------------------------------------------

_yolo_model = None
_yolo_lock = threading.Lock()


def _person_detect_settings():
    def _num(name, default, cast):
        try:
            return cast(_env(name) or default)
        except (TypeError, ValueError):
            return default

    enabled_raw = _env("PERSON_DETECT_ENABLED", "1").strip().lower()
    return {
        "enabled": enabled_raw not in ("0", "false", "no", "off"),
        "model": _env("PERSON_DETECT_MODEL", "yolov8n.pt").strip() or "yolov8n.pt",
        "conf": max(0.05, min(0.95, _num("PERSON_DETECT_CONF", 0.5, float))),
        "fps": max(1, min(5, _num("PERSON_DETECT_FPS", 2, int))),
        "first_clip_seconds": max(3, _num("PERSON_FIRST_CLIP_SECONDS", 20, int)),
        "continuous_clip_seconds": max(10, _num("PERSON_CONTINUOUS_CLIP_SECONDS", 60, int)),
        "grace_seconds": max(2.0, _num("PERSON_GRACE_SECONDS", 8.0, float)),
        "motion_threshold": max(0.1, _num("PERSON_MOTION_THRESHOLD", 1.5, float)),
        "force_check_seconds": max(1.0, _num("PERSON_FORCE_CHECK_SECONDS", 5.0, float)),
    }


def _get_yolo_model(model_name):
    """Load the YOLO model exactly once, shared across all cameras (the global
    lock also serializes inference runs - only 1 camera runs AI at a time,
    preventing multiple AI processes from competing for CPU simultaneously)."""
    global _yolo_model
    with _yolo_lock:
        if _yolo_model is None:
            weight_path = model_name
            if not os.path.isabs(weight_path) and os.sep not in weight_path:
                # Store it in DATA_DIR (persistent directory) so it isn't re-downloaded
                # from the network every time the container restarts
                weight_path = os.path.join(DATA_DIR, model_name)
            log.info("Loading person-detection model: %s", weight_path)
            _yolo_model = YOLO(weight_path)
        return _yolo_model


def _detect_person(model, frame_bgr, conf):
    with _yolo_lock:
        results = model.predict(
            frame_bgr, imgsz=320, conf=conf, classes=[0],
            verbose=False, device="cpu",
        )
    return bool(results and len(results[0].boxes) > 0)


class _PersonState:
    """The most recent time a person was detected for 1 camera - read/written from 2
    different threads (detection & recorder) so it needs a lock."""

    def __init__(self):
        self.lock = threading.Lock()
        self.last_seen = 0.0

    def mark_seen(self):
        with self.lock:
            self.last_seen = time.time()

    def recently_seen(self, within_seconds):
        with self.lock:
            return (time.time() - self.last_seen) < within_seconds


_person_states = {}
_person_states_lock = threading.Lock()


def _get_person_state(cam_id):
    with _person_states_lock:
        state = _person_states.get(cam_id)
        if state is None:
            state = _PersonState()
            _person_states[cam_id] = state
        return state


def _detection_loop(cam_id, stop_event):
    cam = _get_camera(cam_id)
    if cam is None:
        return
    settings = _person_detect_settings()
    try:
        model = _get_yolo_model(settings["model"])
    except Exception as exc:
        log.warning("Failed to load person-detection model for camera %s: %s", cam["name"], exc)
        return
    state = _get_person_state(cam_id)

    # Outer loop: automatically reconnect if the RTSP stream drops (camera reset,
    # network loss...) instead of exiting entirely and permanently stopping monitoring for this camera.
    while not stop_event.is_set():
        cam = _get_camera(cam_id)
        if cam is None:
            return
        key, worker = _acquire_worker(cam, "sub", fps=settings["fps"])
        prev_small = None
        last_checked = 0.0
        last_id = 0
        got_first_frame = False
        log.info("Person monitoring: started for camera %s", cam["name"])
        try:
            while not stop_event.is_set():
                with worker.cond:
                    worker.cond.wait_for(
                        lambda: worker.frame_id != last_id or not worker.running,
                        timeout=FRAME_STALL_TIMEOUT,
                    )
                    if not worker.running:
                        break
                    frame_bytes = worker.frame
                    if worker.frame_id == last_id or frame_bytes is None:
                        continue
                    last_id = worker.frame_id
                if not got_first_frame:
                    got_first_frame = True
                    log.info("Person monitoring: received the first frame from camera %s", cam["name"])
                arr = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                small = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY)
                now = time.time()
                motion = prev_small is None
                if prev_small is not None:
                    diff = cv2.absdiff(small, prev_small)
                    changed_pct = float(np.count_nonzero(diff > 25)) / diff.size * 100.0
                    motion = changed_pct > settings["motion_threshold"]
                prev_small = small
                forced = (now - last_checked) >= settings["force_check_seconds"]
                if not (motion or forced):
                    continue
                last_checked = now
                try:
                    if _detect_person(model, frame, settings["conf"]):
                        state.mark_seen()
                except Exception as exc:
                    log.warning("Error running person-detection AI for camera %s: %s", cam["name"], exc)
        finally:
            _release_worker(key, worker)

        if stop_event.is_set():
            break
        log.warning("Person monitoring: lost connection to camera %s, retrying in 5s", cam["name"])
        stop_event.wait(5)


def _recorder_loop(cam_id, stop_event):
    state = _get_person_state(cam_id)
    while not stop_event.is_set():
        if not state.recently_seen(1.0):
            time.sleep(0.5)
            continue
        # Person just detected -> start a recording session, first segment 20s
        settings = _person_detect_settings()
        duration = settings["first_clip_seconds"]
        while not stop_event.is_set():
            cam = _get_camera(cam_id)
            tg = _telegram_settings()
            if cam is None or not tg["enabled"]:
                break
            clip = _record_clip(cam, duration, tg["video_source"])
            if clip:
                _send_person_clip(tg, cam, clip, duration)
            settings = _person_detect_settings()
            if state.recently_seen(settings["grace_seconds"]):
                duration = settings["continuous_clip_seconds"]
                continue
            break


_camera_workers = {}
_camera_workers_lock = threading.Lock()


def _stop_camera_worker(cam_id):
    with _camera_workers_lock:
        entry = _camera_workers.pop(cam_id, None)
    if entry:
        entry["stop"].set()


def _start_camera_worker(cam_id):
    stop_event = threading.Event()
    threads = [
        threading.Thread(target=_detection_loop, args=(cam_id, stop_event), daemon=True),
        threading.Thread(target=_recorder_loop, args=(cam_id, stop_event), daemon=True),
    ]
    with _camera_workers_lock:
        _camera_workers[cam_id] = {"stop": stop_event, "threads": threads}
    for t in threads:
        t.start()


def _person_detect_manager():
    """Sync the list of cameras being monitored for persons with cameras.json
    and the Telegram enabled/disabled state - re-checks every 20s."""
    while True:
        settings = _person_detect_settings()
        tg = _telegram_settings()
        active_ids = set()
        if settings["enabled"] and tg["enabled"]:
            active_ids = {c["id"] for c in _load_cameras()}
        with _camera_workers_lock:
            running_ids = set(_camera_workers.keys())
        for cam_id in running_ids - active_ids:
            _stop_camera_worker(cam_id)
        for cam_id in active_ids - running_ids:
            _start_camera_worker(cam_id)
        time.sleep(20)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/cameras", methods=["GET"])
def api_list_cameras():
    with _store_lock:
        cams = _load_cameras()
    return jsonify({"cameras": [_public_camera(c) for c in cams]})


@app.route("/api/cameras", methods=["POST"])
def api_add_camera():
    cam, err = _validate_camera_payload(request.get_json(silent=True))
    if err:
        return jsonify({"error": err}), 400
    with _store_lock:
        cams = _load_cameras()
        cams.append(cam)
        _save_cameras(cams)
    settings = _telegram_settings()
    if settings["enabled"]:
        threading.Thread(target=_camera_topic_thread_id, args=(settings, cam), daemon=True).start()
    return jsonify({"camera": _public_camera(cam)}), 201


@app.route("/api/cameras/<cam_id>", methods=["PUT"])
def api_update_camera(cam_id):
    with _store_lock:
        cams = _load_cameras()
        existing = next((c for c in cams if c.get("id") == cam_id), None)
        if existing is None:
            return jsonify({"error": "Camera not found"}), 404
        cam, err = _validate_camera_payload(request.get_json(silent=True), existing=existing)
        if err:
            return jsonify({"error": err}), 400
        cams = [cam if c.get("id") == cam_id else c for c in cams]
        _save_cameras(cams)
    return jsonify({"camera": _public_camera(cam)})


@app.route("/api/cameras/<cam_id>", methods=["DELETE"])
def api_delete_camera(cam_id):
    with _store_lock:
        cams = _load_cameras()
        remaining = [c for c in cams if c.get("id") != cam_id]
        # Idempotent: deleting a camera that doesn't exist still returns ok (avoids an
        # error when the delete button is clicked twice in a row)
        if len(remaining) != len(cams):
            _save_cameras(remaining)
    with _workers_lock:
        for key in list(_workers):
            if key[0] == cam_id:
                _workers.pop(key).stop()
    _stop_camera_worker(cam_id)
    with _topics_lock:
        topics = _load_topics()
        if topics.pop(_camera_topic_key(cam_id), None) is not None:
            _save_topics(topics)
    return jsonify({"ok": True})


@app.route("/api/test", methods=["POST"])
def api_test_connection():
    """Check the connection before saving the camera."""
    payload = request.get_json(silent=True) or {}
    existing = None
    cam_id = payload.get("id")
    if cam_id and not payload.get("password"):
        existing = _get_camera(cam_id)
    cam, err = _validate_camera_payload(payload, existing=existing)
    if err:
        return jsonify({"error": err}), 400
    info, err = _probe(_rtsp_url(cam, "main"))
    if err:
        return jsonify({"error": err}), 502
    return jsonify({"ok": True, "info": info})


@app.route("/stream/<cam_id>")
def stream(cam_id):
    cam = _get_camera(cam_id)
    if cam is None:
        return jsonify({"error": "Camera not found"}), 404
    src = "sub" if request.args.get("src") == "sub" else "main"
    try:
        fps = max(1, min(15, int(request.args.get("fps", STREAM_FPS))))
    except (TypeError, ValueError):
        fps = STREAM_FPS
    return Response(
        _mjpeg_stream(cam, src, fps),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.route("/snapshot/<cam_id>")
def snapshot(cam_id):
    cam = _get_camera(cam_id)
    if cam is None:
        return jsonify({"error": "Camera not found"}), 404
    data = _snapshot(cam, "main")
    if data is None:
        return jsonify({"error": "Could not capture photo from camera"}), 502
    headers = {"Cache-Control": "no-store"}
    if request.args.get("download"):
        ts = time.strftime("%Y%m%d_%H%M%S")
        headers["Content-Disposition"] = f'attachment; filename="{cam["host"]}_{ts}.jpg"'
    return Response(data, mimetype="image/jpeg", headers=headers)


@app.route("/api/telegram/send-now", methods=["POST"])
def api_telegram_send_now():
    """Immediately send a photo album of all cameras to the Telegram group (video
    is only sent automatically when a person is detected)."""
    settings = _telegram_settings()
    if not settings["enabled"]:
        return jsonify({"error": "Telegram is not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in .env)"}), 400
    report = _send_all_reports()
    ok = bool(report["results"]) and all(r["ok"] for r in report["results"])
    return jsonify(report), (200 if ok else 502)


@app.route("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    _start_telegram_workers()
    threading.Thread(target=_person_detect_manager, daemon=True).start()
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, threaded=True)
