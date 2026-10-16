# VitalsHub - HỆ THỐNG GIÁM SÁT SINH HIỆU KHÔNG TIẾP XÚC TỔNG HỢP (rPPG & YOLO CHEST POSE)

Dự án hợp nhất toàn diện 2 hệ thống đo sinh hiệu không tiếp xúc:
1. **Facial rPPG / rBBG Engine**: Đo nhịp tim (**Heart Rate - HR BPM**) và nhịp thở (**Respiration Rate - RR RPM**) qua phân tích dao động quang phổ mao mạch khuôn mặt sử dụng **MediaPipe Face Mesh (468 điểm 3D)**, **Affine Warping** và thuật toán sắc độ **CHROM (De Haan & Jeanne)**.
2. **Chest Motion Tracking Engine (YOLO Pose)**: Đo nhịp thở (**Breathing Rate - BPM**) qua nhận diện chuyển động lồng ngực sử dụng **YOLO11 Pose Estimation**, khóa cứng vùng ngực (Chest ROI) và luồng quang học **Gunnar Farneback Optical Flow**.
3. **Unified Real-time Dashboard**: Bảng điều khiển trung tâm y tế hiện đại (Medical-grade Dark UI) hiển thị video streaming thời gian thực (MJPEG), biểu đồ sóng xung mạch BVP, sóng hô hấp, phổ tần số FFT, cảnh báo âm thanh lâm sàng và xuất báo cáo CSV.

---

## 1. Cấu trúc thư mục dự án sau khi hợp nhất (`son/`)

```
d:\aaaaa\son/
├── backend/                             # Máy chủ Flask Backend & REST/SSE API
│   ├── app.py                           # Ứng dụng Flask trung tâm phục vụ Dashboard & API
│   ├── video_stream.py                  # Điều phối luồng video MJPEG & Pipeline rPPG
│   ├── local_db.py                      # CSDL SQLite cục bộ (chạy độc lập 100% không cần setup)
│   ├── alarm_manager.py                 # Bộ giám sát cảnh báo sinh hiệu & phân loại nguy cơ
│   ├── email_sender.py                  # Module gửi email cảnh báo vi phạm qua SMTP Gmail
│   ├── firebase_db.py                   # Module kết nối Firebase Realtime Database (Tùy chọn)
│   ├── config.py                        # Cấu hình ngưỡng cảnh báo & email
│   └── serviceAccountKey.json.example   # Mẫu cấu hình Firebase Service Account
│
├── dashboard/                           # Giao diện Web Dashboard thời gian thực (SPA)
│   ├── templates/
│   │   └── index.html                   # Giao diện HTML5 đa tab (Tổng quan, rPPG, Ngực, Lịch sử, Cài đặt)
│   └── static/
│       ├── css/style.css                # CSS thiết kế Dark Theme y tế, Glassmorphism, Neon glow
│       └── js/dashboard.js              # JavaScript điều khiển SSE, đồ thị Chart.js, bộ tạo âm thanh
│
├── chest_breathing/                     # Module đo nhịp thở qua cử động lồng ngực (YOLO)
│   ├── ai/
│   │   ├── roi_extractor.py             # Trích xuất Chest ROI từ tọa độ vai & hông
│   │   ├── motion_tracker.py            # Tính toán vector chuyển động Farneback
│   │   ├── signal_filter.py             # Bộ lọc Butterworth Bandpass 0.1 - 0.5 Hz
│   │   ├── breathing_rate.py            # Đếm đỉnh sóng (Peak finding) và tính BPM
│   │   └── pose_detector.py
│   ├── webcam_worker.py                 # Worker chạy trực tiếp qua Webcam laptop
│   ├── breathing_worker.py              # Worker nhận luồng MJPEG từ ESP32-CAM qua WiFi
│   ├── yolo11n-pose.onnx                # Trọng số mô hình YOLO Pose dạng ONNX tối ưu CPU
│   └── yolo11n-pose.pt                  # Trọng số mô hình PyTorch
│
├── hardware/                            # Phần cứng IoT
│   └── esp32cam_stream.ino              # Mã nguồn nạp vi điều khiển ESP32-CAM (Arduino IDE)
│
├── tests/                               # Bộ kiểm thử tự động (Unit Tests)
│   ├── test_pipeline.py                 # Kiểm thử giải thuật rPPG CHROM & FFT
│   ├── test_keypoint.py                 # Kiểm thử trích xuất điểm xương YOLO
│   ├── test_pose.py
│   └── test_roi.py
│
├── venv/                                # Môi trường ảo Python 3.9 đã cài đặt đủ thư viện
├── chrom_processor.py                   # Module giải thuật CHROM, bộ lọc Butterworth & FFT
├── roi_extractor.py                     # MediaPipe FaceMesh & Affine Warping trán và 2 má
├── rppg_pipeline.py                     # Pipeline cửa sổ trượt (Sliding Window) thời gian thực
├── ui_visualizer.py                     # Giao diện HUD trực quan trên OpenCV
├── demo.py                              # Ứng dụng chạy demo OpenCV độc lập (Interactive/Webcam/Video)
├── main.py                              # CLI phân tích video file, xuất CSV, xuất video
├── demo_video.mp4                       # Video mẫu chuẩn phục vụ thử nghiệm
├── demo_face.jpg                        # Ảnh mẫu chuẩn phục vụ mô phỏng tương tác
├── requirements.txt                     # Danh sách toàn bộ thư viện phụ thuộc
├── run_dashboard.bat                    # [1-CLICK] Khởi chạy máy chủ Web Dashboard
├── run_webcam_breathing.bat             # [1-CLICK] Chạy worker đo nhịp thở ngực (Webcam)
├── run_esp32_breathing.bat              # [1-CLICK] Chạy worker đo nhịp thở ngực (ESP32-CAM)
├── run_demo.bat                         # Chạy demo rPPG video mẫu
├── run_interactive.bat                  # Chạy demo rPPG tương tác phím mũi tên
└── run_webcam.bat                       # Chạy demo rPPG webcam trực tiếp
```

---

## 2. Hướng dẫn khởi chạy nhanh (Quick Start)

### Cách 1: Khởi chạy Web Dashboard tổng hợp (Khuyên dùng)
Chỉ cần nhấp đúp vào file:
```
run_dashboard.bat
```
Hoặc chạy bằng dòng lệnh PowerShell:
```powershell
.\venv\Scripts\python backend\app.py
```
Trình duyệt sẽ tự động mở tại địa chỉ: **`http://localhost:5000`**

---

### Cách 2: Khởi chạy Worker đo nhịp thở cử động ngực (Webcam laptop)
Để bật thêm camera ngực gửi dữ liệu thời gian thực lên Dashboard:
```
run_webcam_breathing.bat
```
Hoặc:
```powershell
.\venv\Scripts\python chest_breathing\webcam_worker.py --api http://127.0.0.1:5000
```

---

### Cách 3: Khởi chạy Worker đo nhịp thở qua ESP32-CAM
Nạp file `hardware/esp32cam_stream.ino` vào board ESP32-CAM, sau đó chạy:
```
run_esp32_breathing.bat
```
Hoặc:
```powershell
.\venv\Scripts\python chest_breathing\breathing_worker.py --api http://127.0.0.1:5000 --espcam http://<IP_ESP32>/stream
```

---

### Cách 4: Chạy Demo rPPG độc lập trên OpenCV Window
- Chạy video mẫu chuẩn: `run_demo.bat`
- Chạy tương tác phím mũi tên (Interactive): `run_interactive.bat` (Bấm `Mũi tên lên/xuống` để đổi nhịp tim, `Mũi tên trái/phải` để đổi nhịp thở).
- Chạy qua webcam thật: `run_webcam.bat`

---

## 3. Các tính năng nổi bật trên Web Dashboard

1. **Tổng quan Sinh hiệu (Overview Hub)**:
   - 4 Vital Hero Cards hiển thị lớn: **Nhịp tim (BPM)**, **Nhịp thở rPPG (RPM)**, **Nhịp thở ngực (BPM)**, **Đánh giá Lâm sàng**.
   - Icon trái tim nhấp nháy đồng bộ theo tần số tim thực tế.
   - 2 đồ thị sóng thời gian thực (Oscilloscope): Sóng xung mạch BVP tim và Sóng hô hấp tuần hoàn.
   - Danh sách sự kiện cảnh báo trực tiếp (Live Alert Log).

2. **Trực quan hóa rPPG Khuôn mặt (Camera & Spectrum)**:
   - Luồng video MJPEG khuôn mặt với lưới 3D Face Mesh và 3 khung trán + má.
   - Bảng điều khiển mô phỏng tương tác: 2 thanh trượt điều chỉnh nhịp tim (45 - 150 BPM) và nhịp thở (8 - 35 RPM) với phản hồi thuật toán tức thì.
   - Đồ thị phổ năng lượng FFT hiển thị đỉnh tần số $f_{\text{pulse}}$ và $f_{\text{resp}}$.
   - Nút bật/tắt Face Mesh, đổi vùng ROI và chụp ảnh HUD.

3. **Giám sát Ngực & Cử động (Chest & YOLO)**:
   - Sơ đồ kiến trúc 5 bước từ Camera -> YOLO Pose -> Chest ROI -> Farneback Optical Flow -> Lọc dải tần.
   - Công cụ kiểm tra kết nối luồng ESP32-CAM IP.
   - Hướng dẫn chạy các worker độc lập.

4. **Lịch sử & Báo cáo Dữ liệu (Records Management)**:
   - Đồ thị xu hướng biến thiên sinh hiệu theo thời gian (Hỗ trợ kéo và phóng to/thu nhỏ bằng chuột).
   - Bảng dữ liệu chi tiết có bộ lọc theo nguồn đo (Tất cả, rPPG, Cử động ngực).
   - Nút **Xuất Báo cáo CSV (Export CSV)** tải về file báo cáo lâm sàng chuẩn cho bác sĩ.
   - Nút xóa lịch sử bản ghi.

5. **Cài đặt Hệ thống & Ngưỡng Cảnh báo**:
   - Tùy chỉnh ngưỡng nhịp tim tối thiểu / tối đa (HR Min/Max).
   - Tùy chỉnh ngưỡng nhịp thở tối thiểu / tối đa (RR Min/Max).
   - Bật/tắt còi báo động âm thanh (Audio Beep synthesizer) kèm nút kiểm tra âm thanh.
   - Tích hợp email cảnh báo tự động gửi qua SMTP Gmail.
   - Bảng chẩn đoán trạng thái hoạt động của từng module.

---

## 4. Danh mục API RESTful & Server-Sent Events (SSE)

| Phương thức | Đường dẫn | Chức năng |
| :--- | :--- | :--- |
| `GET` | `/` | Giao diện Dashboard trung tâm (SPA) |
| `GET` | `/api/video_feed/rppg` | Luồng video MJPEG khuôn mặt rPPG kèm HUD |
| `GET` | `/api/latest` | Lấy chỉ số sinh hiệu mới nhất (HR, RR, Chest BPM, Status) |
| `GET` | `/api/history` | Lấy danh sách lịch sử đo lường (`?limit=100&type=all`) |
| `GET` | `/api/statistics` | Thống kê tổng hợp (Trung bình, Min, Max của HR & RR) |
| `POST` | `/api/breathing` | Endpoint nhận BPM từ worker cử động ngực (tương thích 100% mã cũ) |
| `POST` | `/api/vitals` | Endpoint nhận đầy đủ gói sinh hiệu tổng hợp |
| `POST` | `/api/control/rppg/mode` | Đổi nguồn phát rPPG (`simulation`, `demo_video`, `webcam`) |
| `POST` | `/api/control/rppg/params` | Điều chỉnh HR & RR mô phỏng từ thanh trượt UI |
| `POST` | `/api/control/rppg/mesh` | Bật / tắt lưới Face Mesh 468 điểm |
| `POST` | `/api/control/rppg/roi` | Đổi chế độ ROI (`all`, `forehead`, `cheeks`) |
| `GET` / `POST` | `/api/settings` | Lấy hoặc cập nhật cấu hình ngưỡng cảnh báo |
| `GET` | `/api/export/csv` | Tải xuống toàn bộ lịch sử bản ghi dạng file CSV |
| `POST` | `/api/history/clear` | Xóa dữ liệu lịch sử đo lường |
| `GET` | `/api/system/status` | Thông tin chẩn đoán trạng thái các thành phần hệ thống |
| `GET` | `/api/stream` | Luồng sự kiện SSE (Server-Sent Events) đẩy dữ liệu liên tục |

---

## 5. Cơ chế lưu trữ CSDL Kép (Dual Storage)

Hệ thống được thiết kế theo cơ chế dự phòng an toàn (Failover):
1. **Cơ sở dữ liệu SQLite cục bộ (`backend/vital_signs.db`)**: Hoạt động mặc định, độc lập 100%, không cần kết nối mạng internet hay cấu hình tài khoản cloud. Đảm bảo hệ thống luôn ghi nhận số liệu ổn định.
2. **Firebase Realtime Database**: Khi người dùng đặt file `serviceAccountKey.json` vào thư mục `backend/`, hệ thống sẽ tự động đồng bộ song song dữ liệu lên Firebase Cloud theo thời gian thực.
