"""Camera Hub - Flask web app xem camera RTSP (Hikvision/EZVIZ).

Them camera chi can nhap IP + mat khau, mac dinh dung format:
    rtsp://admin:<password>@<ip>:554/Streaming/Channels/101  (luong chinh)
    rtsp://admin:<password>@<ip>:554/Streaming/Channels/102  (luong phu)

Video duoc ffmpeg doc qua RTSP/TCP va chuyen thanh MJPEG de xem truc tiep
tren trinh duyet (khong can plugin, hoat dong ca voi camera H.265/HEVC).

Gui anh dinh ky + gui video tu dong khi phat hien nguoi (YOLOv8n, CPU) len
Telegram.
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
# Cau hinh & luu tru
# ---------------------------------------------------------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Khi chay trong Home Assistant addon, /data la thu muc luu tru ben vung
DATA_DIR = os.environ.get("DATA_DIR") or ("/data" if os.path.isdir("/data") else APP_DIR)
CAMERAS_FILE = os.path.join(DATA_DIR, "cameras.json")

DEFAULT_USERNAME = "admin"
DEFAULT_PORT = 554
DEFAULT_PATH_MAIN = "/Streaming/Channels/101"
DEFAULT_PATH_SUB = "/Streaming/Channels/102"

# Gioi han stream de tiet kiem CPU
STREAM_FPS = 10
MAIN_MAX_WIDTH = 1280      # luong chinh scale xuong toi da 1280px
JPEG_QUALITY = "7"         # 2 (tot nhat) .. 31 (te nhat)
WORKER_IDLE_STOP_SECONDS = 15  # dung ffmpeg sau khi client cuoi cung roi di
                               # (> nhip thu lai 6-8s cua UI de tai su dung worker dang am)
FRAME_STALL_TIMEOUT = 15       # khong co frame moi trong N giay -> coi nhu mat ket noi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("camera-hub")

# Chi dung 1 luong CPU cho OpenCV: may chay them nhieu tac vu khac, khong
# de opencv/YOLO chiem het nhan CPU
cv2.setNumThreads(1)


# Cac bien co san trong moi truong that (truoc khi nap .env) luon uu tien nhat
_REAL_ENV_KEYS = frozenset(os.environ)


def _env_file_values():
    """Doc gia tri tu file .env (APP_DIR truoc, DATA_DIR ghi de neu trung key)."""
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
    """Lay cau hinh: env that > file .env (doc lai moi lan goi) > options.json."""
    if name in _REAL_ENV_KEYS:
        return os.environ.get(name, default)
    file_values = _env_file_values()
    if name in file_values:
        return file_values[name]
    return os.environ.get(name, default)


def _load_env_files():
    """Nap .env vao environ luc khoi dong (cho cac bien chi doc 1 lan nhu PORT)."""
    for key, value in _env_file_values().items():
        os.environ.setdefault(key, value)


def _load_addon_options():
    """Home Assistant addon: doc cau hinh tu /data/options.json neu co."""
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
    """Kiem tra du lieu them/sua camera. Tra ve (camera_dict, error_message)."""
    if not isinstance(payload, dict):
        return None, "Du lieu khong hop le"

    name = str(payload.get("name") or "").strip()
    host = str(payload.get("host") or "").strip()
    password = str(payload.get("password") or "")
    username = str(payload.get("username") or "").strip() or DEFAULT_USERNAME
    path_main = str(payload.get("path_main") or "").strip() or DEFAULT_PATH_MAIN
    path_sub = str(payload.get("path_sub") or "").strip() or DEFAULT_PATH_SUB

    if not host:
        return None, "Vui lòng nhập địa chỉ IP của camera"
    if not _HOST_RE.match(host):
        return None, "Địa chỉ IP/hostname không hợp lệ"

    try:
        port = int(payload.get("port") or DEFAULT_PORT)
        if not 1 <= port <= 65535:
            raise ValueError
    except (TypeError, ValueError):
        return None, "Cổng RTSP không hợp lệ"

    if existing:
        # Sua camera: bo trong mat khau = giu mat khau cu
        if not password:
            password = existing.get("password", "")
    if not password:
        return None, "Vui lòng nhập mật khẩu camera (pass security)"

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
    """Ban sao camera de tra ve frontend - khong lo mat khau."""
    out = {k: v for k, v in cam.items() if k != "password"}
    out["has_password"] = bool(cam.get("password"))
    return out


# ---------------------------------------------------------------------------
# ffmpeg: doc RTSP -> MJPEG, chia se 1 tien trinh cho nhieu nguoi xem
# ---------------------------------------------------------------------------


class StreamWorker:
    """Mot tien trinh ffmpeg doc 1 luong RTSP, phat frame JPEG cho moi client."""

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
        # Khong dung -fflags nobuffer: voi HEVC no lam ffmpeg giai ma giua
        # GOP truoc khi keyframe den -> hang loat frame xam khi moi ket noi
        return [
            "ffmpeg", "-nostdin", "-loglevel", "error",
            "-rtsp_transport", "tcp",
            # Rut ngan thoi gian do luong khi moi ket noi (codec da biet tu SDP)
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
                # os.read tra ve ngay khi co du lieu (read() thuong se cho du
                # 64KB moi tra -> frame bi don cum lam giam fps thuc te)
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                buf += chunk
                # Tach cac anh JPEG theo marker SOI (FFD8) / EOI (FFD9)
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
                    break  # qua lau khong co frame moi -> ngat de client tu ket noi lai
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
    """Chup 1 anh JPEG tu camera."""
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
    """Kiem tra ket noi RTSP bang ffprobe. Tra ve (info_dict, error_message)."""
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
        return None, "Hết thời gian chờ - camera không phản hồi"
    if result.returncode != 0:
        err = (result.stderr or b"").decode("utf-8", "replace").strip()
        if "401" in err or "Unauthorized" in err:
            return None, "Sai tài khoản hoặc mật khẩu (401 Unauthorized)"
        if "404" in err or "Not Found" in err:
            return None, "Không tìm thấy luồng video (404) - kiểm tra lại đường dẫn RTSP"
        if "Connection refused" in err:
            return None, "Camera từ chối kết nối - kiểm tra IP và cổng RTSP"
        return None, err.splitlines()[-1] if err else "Không kết nối được camera"
    try:
        streams = json.loads(result.stdout).get("streams", [])
        if not streams:
            return None, "Không tìm thấy luồng video"
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
        return None, "Không đọc được thông tin luồng video"


# ---------------------------------------------------------------------------
# Telegram: gui anh + video dinh ky vao nhom
# ---------------------------------------------------------------------------

TELEGRAM_API = "https://api.telegram.org"

# Topic (forum) rieng cho tung camera + 1 topic chung cho anh dinh ky.
# Chi hoat dong neu nhom da bat "Topics" va bot duoc cap quyen "Manage
# Topics" - neu chua thi cac ham nay tra ve None va tin nhan roi vao topic
# "General" nhu binh thuong (khong loi).
TELEGRAM_TOPICS_FILE = os.path.join(DATA_DIR, "telegram_topics.json")
PHOTOS_TOPIC_KEY = "photos"
PHOTOS_TOPIC_NAME = "\U0001F4F7 Ảnh"
TOPIC_RETRY_SECONDS = 300  # tao topic that bai -> cho tung nay giay moi thu lai, tranh spam API loi

_topics_lock = threading.Lock()
_topic_retry_after = {}


def _telegram_settings():
    """Doc cau hinh Telegram - doc lai .env moi lan goi nen sua file la ap dung
    ngay o chu ky ke tiep, khong can khoi dong lai."""
    def _num(name, default, cast):
        try:
            return cast(_env(name) or default)
        except (TypeError, ValueError):
            return default

    token = _env("TELEGRAM_BOT_TOKEN").strip()
    chat_id = _env("TELEGRAM_CHAT_ID").strip()
    enabled_raw = _env("TELEGRAM_ENABLED").strip().lower()
    # Nguon quay video khi phat hien nguoi: "sub" (luong phu - nhe mang, mac dinh) hoac "main"
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
        log.warning("Telegram: khong tao duoc topic '%s': %s", name, exc)
        return None


def _get_topic_thread_id(settings, key, name):
    """Tra ve message_thread_id cho 1 topic (tu cache file, hoac tu tao moi
    neu chua co). Neu nhom chua bat Topics hoac bot chua co quyen quan ly
    topic, tao topic se that bai - luc do cho TOPIC_RETRY_SECONDS moi thu
    lai (thay vi goi API loi lien tuc), va tra ve None de tin nhan van gui
    binh thuong (khong dinh kem message_thread_id)."""
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
        log.info("Telegram: da tao topic '%s' (thread_id=%s)", name, thread_id)
        return thread_id


def _photos_topic_thread_id(settings):
    return _get_topic_thread_id(settings, PHOTOS_TOPIC_KEY, PHOTOS_TOPIC_NAME)


def _camera_topic_thread_id(settings, cam):
    return _get_topic_thread_id(settings, _camera_topic_key(cam["id"]), cam["name"])


def _ensure_all_topics():
    """Tao truoc topic cho anh + tat ca camera dang co (khong cho toi khi co
    nguoi/den chu ky gui anh moi tao) - chay ngam khi khoi dong."""
    settings = _telegram_settings()
    if not settings["enabled"]:
        return
    _photos_topic_thread_id(settings)
    for cam in _load_cameras():
        _camera_topic_thread_id(settings, cam)


def _record_clip(cam, seconds, src="sub"):
    """Quay 1 doan video ngan tu camera, tra ve duong dan file mp4 (H.264).

    Mac dinh quay tu luong phu: nhe mang hon ~6 lan so voi luong chinh,
    du xem tren dien thoai. Dat TELEGRAM_VIDEO_SOURCE=main neu can net cao.
    """
    fd, path = tempfile.mkstemp(suffix=".mp4", prefix="camera_hub_")
    os.close(fd)
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-rtsp_transport", "tcp",
        # Camera EZVIZ danh timestamp video/audio lech nhau hang gio ->
        # khong co co nay thi -t se cat mat toan bo am thanh
        "-use_wallclock_as_timestamps", "1",
        "-i", _rtsp_url(cam, src),
        "-t", str(seconds),
        "-vf", "scale='min(1280,iw)':-2",
        # Ep H.264 vi Telegram/trinh duyet khong phat truc tiep duoc HEVC
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
        log.warning("Khong quay duoc clip camera %s: %s", cam["name"], err.splitlines()[-1] if err else "?")
    except subprocess.TimeoutExpired:
        log.warning("Quay clip camera %s qua thoi gian cho", cam["name"])
    try:
        os.remove(path)
    except OSError:
        pass
    return None


def _send_photo_album(settings, photos, stamp):
    """Gui tat ca anh trong 1 lan (album). Telegram gioi han 10 anh/album
    nen neu nhieu camera hon thi chia thanh nhieu album lien tiep.
    Toan bo anh (moi camera) gui chung vao 1 topic "Ảnh"."""
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
    """Chup anh TAT CA camera roi gom gui chung 1 luot (album).
    Tra ve ket qua theo tung camera (dict cam_id -> row)."""
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
            rows[cam["id"]]["error"] = "không chụp được ảnh"
            log.warning("Telegram: khong chup duoc anh camera %s", cam["name"])
    if photos:
        try:
            _send_photo_album(settings, photos, stamp)
            for cam, _ in photos:
                rows[cam["id"]]["sent"].append("ảnh")
        except Exception as exc:
            for cam, _ in photos:
                rows[cam["id"]]["ok"] = False
                rows[cam["id"]]["error"] = str(exc)
            log.warning("Telegram: loi gui album anh: %s", exc)
    failed_snap = [r["camera"] for r in rows.values() if r.get("error") == "không chụp được ảnh"]
    if failed_snap:
        try:
            data_form = {
                "chat_id": settings["chat_id"],
                "text": f"⚠️ Không chụp được ảnh từ: {', '.join(failed_snap)} ({stamp})",
            }
            thread_id = _photos_topic_thread_id(settings)
            if thread_id:
                data_form["message_thread_id"] = thread_id
            _tg_send(settings, "sendMessage", data_form)
        except Exception as exc:
            log.warning("Telegram: loi gui canh bao: %s", exc)
    log.info("Telegram [anh]: da gui album %d/%d camera", len(photos), len(cams))
    return rows


def _send_all_reports():
    """Gui ngay album anh cua tat ca camera (dung cho nut 'Gui Telegram' tren
    web). Video khong con gui theo yeu cau - chi gui tu dong khi phat hien
    nguoi trong khung hinh, xem phan "Phat hien nguoi" ben duoi."""
    settings = _telegram_settings()
    if not settings["enabled"]:
        return {"enabled": False, "results": []}
    with _telegram_report_lock:
        photo_rows = _send_photo_batch(settings)
    return {"enabled": True, "results": list(photo_rows.values())}


def _send_person_clip(settings, cam, clip_path, seconds):
    """Gui 1 doan clip da quay duoc (co nguoi) len Telegram roi xoa file tam.
    Moi camera gui vao topic rieng cua no (neu nhom da bat Topics)."""
    stamp = time.strftime("%d/%m/%Y %H:%M:%S")
    try:
        with open(clip_path, "rb") as f:
            data_form = {
                "chat_id": settings["chat_id"],
                "caption": f"\U0001F6A8 Phát hiện người • {cam['name']} • {stamp} ({seconds}s)",
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
        log.info("Telegram [người]: đã gửi clip camera %s (%ds)", cam["name"], seconds)
    except Exception as exc:
        log.warning("Telegram: lỗi gửi clip phát hiện người camera %s: %s", cam["name"], exc)
    finally:
        try:
            os.remove(clip_path)
        except OSError:
            pass


def _run_periodic(kind, batch_fn, interval_key, first_delay):
    """Vong lap gui dinh ky. Chu ky tinh tu LUC BAT DAU moi dot de giu dung
    nhip ke ca khi mot dot keo dai; toi thieu nghi 20s giua 2 dot."""
    time.sleep(first_delay)
    while True:
        started = time.monotonic()
        settings = _telegram_settings()
        if settings["enabled"]:
            try:
                with _telegram_report_lock:
                    batch_fn(settings)
            except Exception as exc:
                log.warning("Telegram [%s]: loi dot gui: %s", kind, exc)
        elapsed = time.monotonic() - started
        time.sleep(max(20, settings[interval_key] * 60 - elapsed))


def _photo_worker():
    _run_periodic("anh", _send_photo_batch, "photo_interval_minutes", first_delay=15)


def _start_telegram_workers():
    settings = _telegram_settings()
    if settings["enabled"]:
        log.info(
            "Telegram: anh moi %.0f phut (album) | video: tu dong khi phat hien nguoi (nguon %s) | chat %s",
            settings["photo_interval_minutes"], settings["video_source"], settings["chat_id"],
        )
    else:
        log.info("Telegram: chua cau hinh TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID - tat gui dinh ky")
    threading.Thread(target=_ensure_all_topics, daemon=True).start()
    threading.Thread(target=_photo_worker, daemon=True).start()


# ---------------------------------------------------------------------------
# Phat hien nguoi (YOLOv8n, CPU) -> ghi clip -> gui Telegram
#
# Moi camera co 2 thread:
#  - _detection_loop: doc luong phu fps thap, chi chay YOLO khi co chuyen
#    dong (hoac da qua lau chua kiem tra) de khong ngon CPU khi khung hinh
#    dung yen - von la phan lon thoi gian voi camera an ninh.
#  - _recorder_loop: cho toi khi co nguoi xuat hien roi quay 1 doan clip
#    (mac dinh 20s) va gui Telegram; neu nguoi con xuat hien lien tuc thi
#    noi tiep cac doan clip dai hon (mac dinh 60s) cho toi khi nguoi roi
#    khoi khung hinh.
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
    """Nap model YOLO 1 lan duy nhat, dung chung cho moi camera (khoa toan
    cuc cung serialize luon cac lan chay inference - chi 1 camera chay AI
    tai 1 thoi diem, tranh nhieu tien trinh AI tranh CPU cung luc)."""
    global _yolo_model
    with _yolo_lock:
        if _yolo_model is None:
            weight_path = model_name
            if not os.path.isabs(weight_path) and os.sep not in weight_path:
                # Luu trong DATA_DIR (thu muc ben vung) de khong tai lai
                # tu mang moi khi container khoi dong lai
                weight_path = os.path.join(DATA_DIR, model_name)
            log.info("Dang nap model phat hien nguoi: %s", weight_path)
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
    """Thoi diem gan nhat phat hien nguoi cho 1 camera - doc/ghi tu 2 thread
    khac nhau (detection & recorder) nen can khoa."""

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
        log.warning("Khong nap duoc model phat hien nguoi cho camera %s: %s", cam["name"], exc)
        return
    state = _get_person_state(cam_id)

    # Vong lap ngoai: tu ket noi lai neu luong RTSP bi rot (camera reset,
    # mat mang...) thay vi thoat han va bo giam sat camera nay vinh vien.
    while not stop_event.is_set():
        cam = _get_camera(cam_id)
        if cam is None:
            return
        key, worker = _acquire_worker(cam, "sub", fps=settings["fps"])
        prev_small = None
        last_checked = 0.0
        last_id = 0
        got_first_frame = False
        log.info("Giam sat nguoi: da bat dau cho camera %s", cam["name"])
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
                    log.info("Giam sat nguoi: da nhan frame dau tien tu camera %s", cam["name"])
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
                    log.warning("Loi chay AI phat hien nguoi camera %s: %s", cam["name"], exc)
        finally:
            _release_worker(key, worker)

        if stop_event.is_set():
            break
        log.warning("Giam sat nguoi: mat ket noi camera %s, thu ket noi lai sau 5s", cam["name"])
        stop_event.wait(5)


def _recorder_loop(cam_id, stop_event):
    state = _get_person_state(cam_id)
    while not stop_event.is_set():
        if not state.recently_seen(1.0):
            time.sleep(0.5)
            continue
        # Vua phat hien nguoi -> bat dau 1 phien ghi hinh, doan dau 20s
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
    """Dong bo danh sach camera dang duoc giam sat nguoi voi cameras.json
    va trang thai bat/tat Telegram - kiem tra lai moi 20s."""
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
            return jsonify({"error": "Không tìm thấy camera"}), 404
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
        # Idempotent: xoa camera khong ton tai van tra ok (tranh bao loi
        # khi bam nut xoa 2 lan lien tiep)
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
    """Kiem tra ket noi truoc khi luu camera."""
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
        return jsonify({"error": "Không tìm thấy camera"}), 404
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
        return jsonify({"error": "Không tìm thấy camera"}), 404
    data = _snapshot(cam, "main")
    if data is None:
        return jsonify({"error": "Không chụp được ảnh từ camera"}), 502
    headers = {"Cache-Control": "no-store"}
    if request.args.get("download"):
        ts = time.strftime("%Y%m%d_%H%M%S")
        headers["Content-Disposition"] = f'attachment; filename="{cam["host"]}_{ts}.jpg"'
    return Response(data, mimetype="image/jpeg", headers=headers)


@app.route("/api/telegram/send-now", methods=["POST"])
def api_telegram_send_now():
    """Gui ngay album anh cua tat ca camera vao nhom Telegram (video chi
    gui tu dong khi phat hien nguoi)."""
    settings = _telegram_settings()
    if not settings["enabled"]:
        return jsonify({"error": "Chưa cấu hình Telegram (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID trong .env)"}), 400
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
