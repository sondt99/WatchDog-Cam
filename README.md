# Camera Hub

Web app xem camera RTSP (Hikvision / EZVIZ và các camera tương tự) trực tiếp
trên trình duyệt + tự động gửi ảnh định kỳ và **video khi phát hiện người**
lên Telegram.

## Tính năng

- **Thêm camera cực dễ**: chỉ cần nhập **địa chỉ IP + mật khẩu** (pass security
  in trên tem camera). App tự dựng URL RTSP theo chuẩn Hikvision/EZVIZ:

  ```
  rtsp://admin:<MẬT_KHẨU>@<IP>:554/Streaming/Channels/101   (luồng chính)
  rtsp://admin:<MẬT_KHẨU>@<IP>:554/Streaming/Channels/102   (luồng phụ)
  ```

- **Xem trực tiếp trên web**: video được ffmpeg đọc qua RTSP/TCP và chuyển thành
  MJPEG nên xem được trên mọi trình duyệt, kể cả camera H.265/HEVC.
- **Lưới camera**: xem nhiều camera cùng lúc (luồng phụ cho nhẹ máy),
  bấm vào để phóng to xem luồng chính chất lượng cao.
- **Gửi ảnh Telegram định kỳ**: mỗi chu kỳ bot chụp ảnh **tất cả camera gom
  thành 1 album gửi 1 lượt**. Có nút **📤 Gửi Telegram** trên giao diện để
  gửi ngay.
- **Gửi video khi phát hiện người**: mỗi camera tự chạy model YOLOv8n (CPU,
  không cần GPU) để phát hiện người trong luồng phụ. Vừa thấy người là ghi
  1 đoạn clip (mặc định 20s) và gửi Telegram; nếu người còn xuất hiện liên
  tục thì ghi nối tiếp các đoạn dài hơn (mặc định 60s, ~1 phút/lần) cho tới
  khi người rời khỏi khung hình. Không có người thì không gửi video.
- **Chụp ảnh** tải về máy; **kiểm tra kết nối** trước khi lưu camera.
- Camera khác chuẩn (Dahua, ONVIF...): chỉnh tài khoản/cổng/đường dẫn luồng
  trong mục **Cài đặt nâng cao**.

## Chạy bằng Docker (khuyên dùng)

```bash
# 1. Tạo file cấu hình Telegram
cp .env.example .env   # rồi điền token + chat ID

# 2. Build và chạy
docker compose up -d --build

# 3. Mở http://localhost:8000
```

Dữ liệu camera lưu ở `./data/cameras.json`. File `.env` được mount vào
container nên **sửa cấu hình Telegram không cần build lại** — áp dụng ở chu kỳ
gửi kế tiếp.

Xem log: `docker logs -f camera-hub` • Dừng: `docker compose down`

## Chạy trực tiếp bằng Python

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# cần có ffmpeg: sudo apt install ffmpeg
python3 app.py   # mở http://localhost:8000
```

Cài `torch`/`torchvision` bản CPU trước (dòng đầu) để tránh `pip` tự kéo bản
CUDA mặc định — nặng hơn nhiều GB mà máy không có GPU không dùng tới.
Khi chạy trực tiếp, danh sách camera lưu ở `cameras.json` cạnh mã nguồn.
Model `yolov8n.pt` (~6MB) sẽ tự tải về `data/` (hoặc `DATA_DIR`) ở lần chạy
đầu tiên — cần có mạng internet lúc đó, các lần sau dùng lại file đã tải.

## Cấu hình Telegram (.env)

```
TELEGRAM_BOT_TOKEN=<token từ @BotFather>
TELEGRAM_CHAT_ID=<ID nhóm, ví dụ -100xxxxxxxxxx>
TELEGRAM_PHOTO_INTERVAL_MINUTES=10   # chu kỳ gửi ảnh (phút)
```

Lưu ý:

- Bot phải được **thêm vào nhóm** trước.
- Nếu nhóm được Telegram nâng cấp lên supergroup, chat ID sẽ đổi thành dạng
  `-100...` — lấy ID mới qua `https://api.telegram.org/bot<TOKEN>/getUpdates`.
- Video được chuyển sang H.264 trước khi gửi để phát trực tiếp được trong Telegram.
- Nhiều hơn 10 camera: album ảnh tự chia thành nhiều đợt (giới hạn Telegram
  10 ảnh/album).

## Phát hiện người & gửi video (YOLOv8n, chạy CPU)

Không cần GPU. Mỗi camera có 1 luồng đọc bản rút gọn của video (fps thấp,
mặc định 2 khung hình/giây) và chỉ chạy AI khi phát hiện có **chuyển động**
(hoặc tối đa mỗi `PERSON_FORCE_CHECK_SECONDS` giây phải kiểm tra 1 lần dù
không có chuyển động) — nhờ vậy máy gần như rảnh khi khung hình đứng yên,
đỡ tốn CPU cho các việc khác đang chạy song song.

Khi phát hiện người: ghi 1 đoạn clip `PERSON_FIRST_CLIP_SECONDS` giây (mặc
định 20s) rồi gửi Telegram ngay. Nếu người vẫn còn trong khung hình, ghi nối
tiếp đoạn `PERSON_CONTINUOUS_CLIP_SECONDS` giây (mặc định 60s, ~1 phút/lần)
và cứ thế cho tới khi không còn thấy người trong `PERSON_GRACE_SECONDS` giây
(mặc định 8s). Toàn bộ có thể chỉnh trong `.env`, xem `.env.example` để biết
đầy đủ các biến (`PERSON_DETECT_MODEL`, `PERSON_DETECT_CONF`,
`PERSON_DETECT_FPS`, `PERSON_MOTION_THRESHOLD`...). Đặt
`PERSON_DETECT_ENABLED=false` để tắt hoàn toàn tính năng này.

## Kiểm tra camera bằng dòng lệnh

```bash
ffplay -rtsp_transport tcp "rtsp://admin:<MẬT_KHẨU>@<IP>:554/Streaming/Channels/101"
```

## Ghi chú

Các tính năng AI trước đây của repo này đã được gỡ bỏ — tóm tắt kỹ thuật đầy đủ
nằm trong [AI_FEATURES_NOTES.txt](./AI_FEATURES_NOTES.txt).
