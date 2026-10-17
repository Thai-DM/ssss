"""
backend/video_stream.py
-----------------------
Quản lý luồng video thời gian thực và xử lý AI trực tiếp trên máy chủ:
1. CameraReader (Bộ thu nhận camera dùng chung):
   - Mở Webcam Laptop (Camera 0) 1 lần duy nhất, tránh lỗi xung đột phần cứng (Device Busy).
   - Cung cấp frame đồng thời cho cả luồng rPPG khuôn mặt và luồng YOLO ngực.
2. RPPGStreamManager:
   - Xử lý MediaPipe Face Mesh, CHROM, đo nhịp tim (HR) và nhịp thở (RR).
   - Cung cấp luồng video MJPEG tại: /api/video_feed/rppg
3. ChestStreamManager:
   - Xử lý YOLO Pose (17 điểm xương), trích xuất Chest ROI, đo luồng chuyển động Farneback.
   - Tính toán nhịp thở cử động ngực thời gian thực.
   - Cung cấp luồng video MJPEG tại: /api/video_feed/chest
"""

import os
import sys
import time
import threading
from typing import Optional, Dict, Any, Tuple
from collections import deque
import cv2
import numpy as np

# Thêm đường dẫn gốc và thư mục backend vào sys.path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
SON_DIR = os.path.dirname(BACKEND_DIR)
CHEST_DIR = os.path.join(SON_DIR, "chest_breathing")

for p in (SON_DIR, BACKEND_DIR, CHEST_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from rppg_pipeline import RPPGPipeline
from ui_visualizer import UIVisualizer
from ultralytics import YOLO
from ai.roi_extractor import ROIExtractor as ChestROIExtractor
from ai.motion_tracker import MotionTracker as ChestMotionTracker
from ai.signal_filter import SignalFilter as ChestSignalFilter
from ai.breathing_rate import BreathingRateEstimator as ChestBREstimator

import local_db
import alarm_manager


# ==============================================================================
# 1. CAMERA CAPTURE DÙNG CHUNG (SINGLETON CAMERA READER)
# ==============================================================================
class CameraReader:
    """Đọc camera laptop 0 ở tiến trình nền để chia sẻ frame cho cả rPPG và Chest YOLO."""

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.last_frame_time = 0.0

    def start(self):
        with self.lock:
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self._capture_loop, daemon=True)
                self.thread.start()

    def stop(self):
        with self.lock:
            self.running = False

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def _capture_loop(self):
        print(f"[CameraReader] Đang mở Webcam Laptop (Index={self.camera_index})...")
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            print("[CameraReader] Đã mở thành công Webcam Laptop (640x480 @ 30 FPS).")
        else:
            print("[CameraReader] CẢNH BÁO: Không mở được Webcam Laptop.")

        while self.running:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    resized = cv2.resize(frame, (640, 480))
                    with self.lock:
                        self.latest_frame = resized
                        self.last_frame_time = time.time()
                else:
                    time.sleep(0.02)
            else:
                time.sleep(0.1)

            time.sleep(0.015)  # ~60 fps capture limit

        if self.cap is not None:
            self.cap.release()


# Khởi tạo camera reader toàn cục
camera_reader = CameraReader(0)
camera_reader.start()


# ==============================================================================
# 2. RPPG STREAM MANAGER (KHUÔN MẶT rPPG)
# ==============================================================================
class RPPGStreamManager:
    """Quản lý luồng video rPPG phân tích khuôn mặt."""

    def __init__(self):
        self.mode = "webcam"  # 'webcam', 'simulation', 'demo_video'
        self.running = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

        # Tham số mô phỏng
        self.sim_hr = 74.0
        self.sim_rr = 16.0
        self.show_mesh = False
        self.roi_mode = "all"

        self.face_img_path = os.path.join(SON_DIR, "demo_face.jpg")
        self.demo_video_path = os.path.join(SON_DIR, "demo_video.mp4")

        self.pipeline = RPPGPipeline(window_sec=8.0, target_fs=30.0, roi_mode=self.roi_mode)
        self.visualizer = UIVisualizer(show_mesh=self.show_mesh)

        self.latest_frame_jpeg: Optional[bytes] = None
        self.latest_data: Dict[str, Any] = {
            "heart_rate": 74.0,
            "respiration_rate": 16.0,
            "chest_bpm": 16.0,
            "fps": 30.0,
            "status": "Đang chạy Webcam Laptop",
            "is_stable": True,
            "bvp_wave": [],
            "resp_wave": [],
            "mode": "webcam",
        }
        self.active_clients = 0
        self.last_db_save_time = time.time()

    def start(self):
        with self.lock:
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self._run_loop, daemon=True)
                self.thread.start()

    def stop(self):
        with self.lock:
            self.running = False

    def set_mode(self, mode: str):
        with self.lock:
            if mode in ("simulation", "demo_video", "webcam"):
                self.mode = mode
                self.pipeline = RPPGPipeline(window_sec=8.0, target_fs=30.0, roi_mode=self.roi_mode)
                print(f"[RPPGStream] Đã đổi chế độ sang: {self.mode}")

    def set_sim_targets(self, hr: float, rr: float):
        with self.lock:
            self.sim_hr = max(40.0, min(160.0, float(hr)))
            self.sim_rr = max(6.0, min(40.0, float(rr)))

    def toggle_mesh(self, enable: Optional[bool] = None):
        with self.lock:
            if enable is None:
                self.show_mesh = not self.show_mesh
            else:
                self.show_mesh = bool(enable)
            self.visualizer.show_mesh = self.show_mesh

    def set_roi_mode(self, roi_mode: str):
        with self.lock:
            if roi_mode in ("all", "forehead", "cheeks"):
                self.roi_mode = roi_mode
                self.pipeline.roi_mode = roi_mode

    def get_latest_data(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.latest_data)

    def _run_loop(self):
        fps = 30.0
        frame_time = 1.0 / fps

        base_img = None
        if os.path.exists(self.face_img_path):
            base_img = cv2.imread(self.face_img_path)
            if base_img is not None:
                base_img = cv2.resize(base_img, (640, 480))

        frame_idx = 0
        cap_demo = None

        while self.running:
            if self.active_clients <= 0:
                time.sleep(0.08)
                continue

            loop_start = time.perf_counter()
            frame = None

            if self.mode == "webcam":
                frame = camera_reader.get_frame()
                if frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "DANG KET NOI WEBCAM LAPTOP...", (120, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 210, 180), 2)

            elif self.mode == "demo_video":
                if cap_demo is None or not cap_demo.isOpened():
                    cap_demo = cv2.VideoCapture(self.demo_video_path)

                ret, read_frame = cap_demo.read()
                if not ret:
                    cap_demo.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, read_frame = cap_demo.read()

                if ret and read_frame is not None:
                    frame = cv2.resize(read_frame, (640, 480))
                else:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)

            elif self.mode == "simulation":
                if base_img is not None:
                    t = frame_idx / fps
                    with self.lock:
                        sim_hr = self.sim_hr
                        sim_rr = self.sim_rr

                    f_pulse = sim_hr / 60.0
                    f_resp = sim_rr / 60.0

                    pulse = np.sin(2 * np.pi * f_pulse * t)
                    breath = np.sin(2 * np.pi * f_resp * t)

                    mod_r = 1.0 + 0.006 * pulse + 0.008 * breath
                    mod_g = 1.0 - 0.024 * pulse + 0.008 * breath
                    mod_b = 1.0 - 0.004 * pulse + 0.008 * breath

                    frame_mod = base_img.astype(np.float32)
                    frame_mod[:, :, 0] = np.clip(frame_mod[:, :, 0] * mod_b, 0, 255)
                    frame_mod[:, :, 1] = np.clip(frame_mod[:, :, 1] * mod_g, 0, 255)
                    frame_mod[:, :, 2] = np.clip(frame_mod[:, :, 2] * mod_r, 0, 255)

                    angle = 3.0 * np.sin(2 * np.pi * 0.15 * t)
                    dx = 2.5 * np.sin(2 * np.pi * 0.2 * t)
                    dy = 1.5 * np.cos(2 * np.pi * 0.15 * t)

                    M = cv2.getRotationMatrix2D((320.0, 240.0), angle, 1.0)
                    M[0, 2] += dx
                    M[1, 2] += dy

                    frame = cv2.warpAffine(
                        frame_mod.astype(np.uint8),
                        M,
                        (640, 480),
                        borderMode=cv2.BORDER_REFLECT,
                    )
                else:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)

            if frame is not None:
                pipeline_data = self.pipeline.process_frame(frame, timestamp=time.perf_counter())
                display_frame = self.visualizer.render(frame, pipeline_data)

                _, jpeg_buf = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                jpeg_bytes = jpeg_buf.tobytes()

                with self.lock:
                    self.latest_frame_jpeg = jpeg_bytes

                    hr_val = pipeline_data.get("hr_bpm", 0.0)
                    rr_val = pipeline_data.get("rr_rpm", 0.0)

                    if self.mode == "simulation" and hr_val == 0.0:
                        hr_val = self.sim_hr
                        rr_val = self.sim_rr

                    self.latest_data = {
                        "heart_rate": round(float(hr_val), 1),
                        "respiration_rate": round(float(rr_val), 1),
                        "chest_bpm": round(float(rr_val), 1),
                        "bpm": round(float(rr_val), 1),
                        "fps": round(float(pipeline_data.get("fps", 30.0)), 1),
                        "status": pipeline_data.get("status", "Đang xử lý"),
                        "is_stable": pipeline_data.get("is_stable", False),
                        "bvp_wave": pipeline_data.get("bvp_wave", []),
                        "resp_wave": pipeline_data.get("resp_wave", []),
                        "detected": pipeline_data.get("detected", False),
                        "mode": self.mode,
                    }

                now_t = time.time()
                if now_t - self.last_db_save_time >= 5.0 and hr_val > 0:
                    self.last_db_save_time = now_t
                    severity, _ = alarm_manager.evaluate_vitals(heart_rate=hr_val, respiration_rate=rr_val)
                    local_db.insert_record(
                        heart_rate=hr_val,
                        respiration_rate=rr_val,
                        chest_bpm=rr_val,
                        source=f"rppg_{self.mode}",
                        status=severity,
                        note="Đo từ Camera rPPG"
                    )

            frame_idx += 1
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.001, frame_time - elapsed)
            time.sleep(sleep_time)

        if cap_demo is not None:
            cap_demo.release()

    def generate_mjpeg_stream(self):
        with self.lock:
            self.active_clients += 1
        self.start()
        try:
            while self.running:
                frame_bytes = None
                with self.lock:
                    frame_bytes = self.latest_frame_jpeg

                if frame_bytes is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    )
                time.sleep(0.033)
        finally:
            with self.lock:
                self.active_clients = max(0, self.active_clients - 1)


# ==============================================================================
# 3. CHEST STREAM MANAGER (NGỰC & YOLO POSE)
# ==============================================================================
class ChestStreamManager:
    """Quản lý luồng video và AI theo dõi cử động ngực bằng YOLO Pose."""

    def __init__(self):
        self.source = "webcam"  # 'webcam', 'esp32'
        self.esp32_url = "http://192.168.143.249/stream"
        self.running = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

        self.show_skeleton = True
        self.show_roi = True

        # Load YOLO model
        model_path = os.path.join(CHEST_DIR, "yolo11n-pose.pt")
        print(f"[ChestStream] Đang nạp mô hình YOLO Pose ({model_path})...")
        self.yolo_model = YOLO(model_path)
        self.roi_extractor = ChestROIExtractor()
        self.motion_tracker = ChestMotionTracker(blur_kernel=(5, 5))

        # Signal Buffers
        self.motion_buffer = deque(maxlen=300)
        self.timestamps = deque(maxlen=300)
        self.current_bpm = 16.0
        self.current_roi = None
        self.last_skeleton_kpts = None
        self.last_seen_person_time = 0.0
        self.latest_frame_jpeg: Optional[bytes] = None
        self.latest_motion_val = 0.0

        self.latest_data: Dict[str, Any] = {
            "chest_bpm": 16.0,
            "motion": 0.0,
            "detected": False,
            "status": "Đang theo dõi ngực",
            "fps": 30.0,
            "source": "webcam",
        }
        self.active_clients = 0
        self.last_db_save = time.time()

    def start(self):
        with self.lock:
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self._run_loop, daemon=True)
                self.thread.start()

    def stop(self):
        with self.lock:
            self.running = False

    def set_source(self, source: str, esp32_url: Optional[str] = None):
        with self.lock:
            if source in ("webcam", "esp32"):
                self.source = source
                if esp32_url:
                    self.esp32_url = esp32_url
                print(f"[ChestStream] Đã đổi nguồn sang: {self.source}")

    def toggle_skeleton(self, enable: Optional[bool] = None):
        with self.lock:
            self.show_skeleton = not self.show_skeleton if enable is None else bool(enable)

    def toggle_roi(self, enable: Optional[bool] = None):
        with self.lock:
            self.show_roi = not self.show_roi if enable is None else bool(enable)

    def get_latest_data(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.latest_data)

    def _run_loop(self):
        frame_idx = 0
        fps = 30.0
        cap_esp = None
        curr_source = None

        # Connections for skeleton drawing (COCO pose pairs)
        SKELETON_PAIRS = [
            (5, 6),   # shoulders
            (5, 7), (7, 9),    # left arm
            (6, 8), (8, 10),   # right arm
            (5, 11), (6, 12),  # torso
            (11, 12),          # hips
        ]

        while self.running:
            if self.active_clients <= 0:
                time.sleep(0.08)
                continue

            loop_start = time.perf_counter()
            frame = None

            if self.source == "webcam":
                if cap_esp is not None:
                    cap_esp.release()
                    cap_esp = None
                frame = camera_reader.get_frame()

            elif self.source == "esp32":
                if curr_source != "esp32" or cap_esp is None or not cap_esp.isOpened():
                    if cap_esp is not None: cap_esp.release()
                    cap_esp = cv2.VideoCapture(self.esp32_url)
                    curr_source = "esp32"

                ret, read_frame = cap_esp.read()
                if ret and read_frame is not None:
                    frame = cv2.resize(read_frame, (640, 480))

            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "DANG KET NOI CAMERA NGUC (LAPTOP WEBCAM)...", (80, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 180), 2)
                time.sleep(0.03)
                continue

            display_frame = frame.copy()
            h, w = frame.shape[:2]

            # 1. Chạy YOLO Pose định kỳ mỗi 8 frame (~0.25s) với imgsz=256
            if frame_idx % 8 == 0:
                try:
                    res = self.yolo_model(frame, imgsz=256, verbose=False)[0]
                    if res.boxes is not None and len(res.boxes) > 0 and float(res.boxes.conf[0]) >= 0.35:
                        if res.keypoints is not None and len(res.keypoints.xy) > 0:
                            person = res.keypoints.xy[0]
                            self.last_skeleton_kpts = person.cpu().numpy()
                            self.last_seen_person_time = time.time()

                            ls, rs, lh, rh = person[5], person[6], person[11], person[12]
                            
                            # Tính ROI ngực thích ứng
                            rx1, ry1, rx2, ry2 = self.roi_extractor.get_chest_roi(frame, ls, rs, lh, rh)
                            rx1, ry1 = max(0, int(rx1)), max(0, int(ry1))
                            rx2, ry2 = min(w, int(rx2)), min(h, int(ry2))

                            if rx2 > rx1 + 25 and ry2 > ry1 + 25:
                                new_roi = (rx1, ry1, rx2, ry2)
                                if self.current_roi is None:
                                    self.current_roi = new_roi
                                    self.motion_tracker.prev_gray = None
                                else:
                                    # KHÓA CỨNG ROI (Anti-jitter lock):
                                    # Chỉ dịch chuyển khung ngực khi người dùng thực sự thay đổi tư thế (dist >= 14px).
                                    # Nếu chỉ rung lắc nhẹ do hít thở thì giữ nguyên khung cũ để triệt tiêu nhấp nháy.
                                    cx_old = (self.current_roi[0] + self.current_roi[2]) / 2.0
                                    cy_old = (self.current_roi[1] + self.current_roi[3]) / 2.0
                                    cx_new = (new_roi[0] + new_roi[2]) / 2.0
                                    cy_new = (new_roi[1] + new_roi[3]) / 2.0
                                    dist = ((cx_new - cx_old)**2 + (cy_new - cy_old)**2)**0.5

                                    if dist >= 14.0:
                                        alpha = 0.35
                                        self.current_roi = tuple(
                                            int(self.current_roi[i] + alpha * (new_roi[i] - self.current_roi[i]))
                                            for i in range(4)
                                        )
                                        # Reset frame tham chiếu optical flow khi ROI đổi vị trí để chống nhảy vọt tín hiệu
                                        self.motion_tracker.prev_gray = None
                except Exception:
                    pass

            # Trạng thái phát hiện người ổn định (không chớp tắt theo từng frame)
            person_detected = (time.time() - self.last_seen_person_time) < 1.5

            # 2. Vẽ khung xương Pose trên TẤT CẢ các frame (chống nhấp nháy / chớp tắt)
            if self.show_skeleton and person_detected and self.last_skeleton_kpts is not None:
                kpts = self.last_skeleton_kpts
                for p1, p2 in SKELETON_PAIRS:
                    if p1 < len(kpts) and p2 < len(kpts):
                        x1_s, y1_s = int(kpts[p1][0]), int(kpts[p1][1])
                        x2_s, y2_s = int(kpts[p2][0]), int(kpts[p2][1])
                        if x1_s > 0 and y1_s > 0 and x2_s > 0 and y2_s > 0:
                            cv2.line(display_frame, (x1_s, y1_s), (x2_s, y2_s), (99, 102, 241), 2)
                for pt in [5, 6, 11, 12]:
                    if pt < len(kpts):
                        px, py = int(kpts[pt][0]), int(kpts[pt][1])
                        if px > 0 and py > 0:
                            cv2.circle(display_frame, (px, py), 5, (0, 210, 180), -1)

            # 3. Xử lý luồng chuyển động quang học trên Chest ROI
            motion_val = 0.0
            if self.current_roi is not None:
                x1, y1, x2, y2 = self.current_roi
                chest_crop = frame[y1:y2, x1:x2]

                if chest_crop.size > 0:
                    raw_motion = self.motion_tracker.process(chest_crop)
                    if raw_motion is not None:
                        # Triệt tiêu gai đột biến bất thường (Outlier clipping)
                        motion_val = float(np.clip(raw_motion, -1.8, 1.8))
                        self.latest_motion_val = motion_val
                        self.motion_buffer.append(motion_val)
                        self.timestamps.append(time.time())

                    # Vẽ khung Chest ROI lên khung hình (vững chắc, không rung giật)
                    if self.show_roi:
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 210, 180), 2)
                        cv2.putText(display_frame, f"CHEST ROI ({x2-x1}x{y2-y1})", (x1, max(18, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 180), 1)

            # 4. Tính toán BPM nhịp thở ổn định (cửa sổ trượt >= 5.0 giây, tránh nhảy vọt)
            if len(self.motion_buffer) >= 100 and frame_idx % 12 == 0:
                duration = self.timestamps[-1] - self.timestamps[0]
                if duration >= 5.0:
                    sig_arr = np.array(list(self.motion_buffer))
                    t_arr = np.array(list(self.timestamps))
                    smooth_sig = ChestSignalFilter.smooth(sig_arr, timestamps=t_arr)
                    calc_bpm, _ = ChestBREstimator.estimate(smooth_sig, duration)
                    if 7.0 <= calc_bpm <= 32.0:
                        # EMA làm mượt: 12% giá trị mới + 88% giá trị cũ -> chỉ số êm ái, vững vàng
                        self.current_bpm = 0.12 * calc_bpm + 0.88 * self.current_bpm

            # Vẽ HUD thông tin lâm sàng lên màn hình
            cv2.rectangle(display_frame, (10, 10), (320, 85), (15, 23, 42), -1)
            cv2.rectangle(display_frame, (10, 10), (320, 85), (255, 255, 255), 1)

            bpm_text = f"NHIP THO: {self.current_bpm:.1f} BPM"
            cv2.putText(display_frame, bpm_text, (20, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 210, 180), 2)

            sub_info = f"Nguon: {'Webcam Laptop' if self.source == 'webcam' else 'ESP32-CAM'} | FPS: 30.0"
            cv2.putText(display_frame, sub_info, (20, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1)

            # Nén JPEG để phát stream
            _, jpeg_buf = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            jpeg_bytes = jpeg_buf.tobytes()

            with self.lock:
                self.latest_frame_jpeg = jpeg_bytes
                self.latest_data = {
                    "chest_bpm": round(float(self.current_bpm), 1),
                    "motion": round(float(self.latest_motion_val), 3),
                    "detected": person_detected,
                    "status": "Đang theo dõi cử động ngực",
                    "fps": 30.0,
                    "source": self.source,
                }

            # Lưu định kỳ vào CSDL
            now_t = time.time()
            if now_t - self.last_db_save >= 5.0 and self.current_bpm > 0:
                self.last_db_save = now_t
                severity, _ = alarm_manager.evaluate_vitals(chest_bpm=self.current_bpm)
                local_db.insert_record(
                    chest_bpm=self.current_bpm,
                    bpm=self.current_bpm,
                    source=f"chest_{self.source}",
                    status=severity,
                    note="Đo từ YOLO Pose Chest Motion"
                )

            frame_idx += 1
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.001, (1.0 / fps) - elapsed))

        if cap_esp is not None:
            cap_esp.release()

    def generate_mjpeg_stream(self):
        with self.lock:
            self.active_clients += 1
        self.start()
        try:
            while self.running:
                frame_bytes = None
                with self.lock:
                    frame_bytes = self.latest_frame_jpeg

                if frame_bytes is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    )
                time.sleep(0.033)
        finally:
            with self.lock:
                self.active_clients = max(0, self.active_clients - 1)


# ==============================================================================
# 4. UNIFIED STREAM MANAGER (1 CAMERA ĐO TOÀN BỘ: rPPG MẶT + CHEST POSE NGỰC)
# ==============================================================================
class UnifiedStreamManager:
    """
    Quản lý luồng video duy nhất All-in-One: trên cùng 1 camera đo đồng thời cả
    Nhịp tim (HR), Nhịp thở khuôn mặt (RR) và Nhịp thở cử động lồng ngực (Chest BPM).
    """

    def __init__(self):
        self.running = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

        # Pipeline rPPG khuôn mặt
        self.rppg_pipeline = RPPGPipeline(window_sec=8.0, target_fs=30.0, roi_mode="all")

        # Pipeline nhận diện lồng ngực & Pose
        model_path = os.path.join(CHEST_DIR, "yolo11n-pose.pt")
        self.yolo_model = YOLO(model_path)
        self.chest_roi_extractor = ChestROIExtractor()
        self.motion_tracker = ChestMotionTracker(blur_kernel=(5, 5))

        # Bộ đệm tín hiệu chuyển động ngực
        self.motion_buffer = deque(maxlen=300)
        self.motion_timestamps = deque(maxlen=300)

        # Biến lưu trữ trạng thái
        self.current_chest_roi = None
        self.last_skeleton_kpts = None
        self.last_seen_person_time = 0.0

        self.current_hr = 72.0
        self.current_rr = 16.0
        self.current_chest_bpm = 16.0

        self.show_skeleton = True
        self.show_chest_roi = True
        self.show_face_roi = True

        self.latest_frame_jpeg: Optional[bytes] = None
        self.latest_data: Dict[str, Any] = {
            "heart_rate": 72.0,
            "respiration_rate": 16.0,
            "chest_bpm": 16.0,
            "status": "Camera All-in-One Live",
            "fps": 30.0,
        }
        self.active_clients = 0
        self.last_db_save = time.time()

    def start(self):
        with self.lock:
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self._run_loop, daemon=True)
                self.thread.start()

    def stop(self):
        with self.lock:
            self.running = False

    def get_latest_data(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.latest_data)

    def _run_loop(self):
        fps = 30.0
        frame_time = 1.0 / fps
        frame_idx = 0

        SKELETON_PAIRS = [
            (5, 6),   # shoulders
            (5, 7), (7, 9),    # left arm
            (6, 8), (8, 10),   # right arm
            (5, 11), (6, 12),  # torso
            (11, 12),          # hips
        ]

        while self.running:
            if self.active_clients <= 0:
                time.sleep(0.08)
                continue

            loop_start = time.perf_counter()
            frame = camera_reader.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            h, w = frame.shape[:2]
            display_frame = frame.copy()

            # 1. XỬ LÝ rPPG KHUÔN MẶT
            rppg_data = self.rppg_pipeline.process_frame(frame, timestamp=time.perf_counter())
            face_detected = rppg_data.get("detected", False)

            hr_res = rppg_data.get("hr_bpm", 0.0)
            rr_res = rppg_data.get("rr_rpm", 0.0)
            if hr_res > 0:
                self.current_hr = round(float(hr_res), 1)
            if rr_res > 0:
                self.current_rr = round(float(rr_res), 1)

            # 2. XỬ LÝ LỒNG NGỰC (YOLO Pose định kỳ mỗi 8 frame)
            if frame_idx % 8 == 0:
                try:
                    res = self.yolo_model(frame, imgsz=256, verbose=False)[0]
                    if res.boxes is not None and len(res.boxes) > 0 and float(res.boxes.conf[0]) >= 0.35:
                        if res.keypoints is not None and len(res.keypoints.xy) > 0:
                            person = res.keypoints.xy[0]
                            self.last_skeleton_kpts = person.cpu().numpy()
                            self.last_seen_person_time = time.time()
                            ls, rs, lh, rh = person[5], person[6], person[11], person[12]
                            
                            rx1, ry1, rx2, ry2 = self.chest_roi_extractor.get_chest_roi(frame, ls, rs, lh, rh)
                            rx1, ry1 = max(0, int(rx1)), max(0, int(ry1))
                            rx2, ry2 = min(w, int(rx2)), min(h, int(ry2))

                            if rx2 > rx1 + 25 and ry2 > ry1 + 25:
                                new_roi = (rx1, ry1, rx2, ry2)
                                if self.current_chest_roi is None:
                                    self.current_chest_roi = new_roi
                                    self.motion_tracker.prev_gray = None
                                else:
                                    cx_old = (self.current_chest_roi[0] + self.current_chest_roi[2]) / 2.0
                                    cy_old = (self.current_chest_roi[1] + self.current_chest_roi[3]) / 2.0
                                    cx_new = (new_roi[0] + new_roi[2]) / 2.0
                                    cy_new = (new_roi[1] + new_roi[3]) / 2.0
                                    dist = ((cx_new - cx_old)**2 + (cy_new - cy_old)**2)**0.5
                                    if dist >= 14.0:
                                        self.current_chest_roi = tuple(
                                            int(self.current_chest_roi[i] + 0.35 * (new_roi[i] - self.current_chest_roi[i]))
                                            for i in range(4)
                                        )
                                        self.motion_tracker.prev_gray = None
                except Exception:
                    pass

            # 3. TÍNH CHUYỂN ĐỘNG NGỰC & NHỊP THỞ
            if self.current_chest_roi is not None:
                x1, y1, x2, y2 = self.current_chest_roi
                chest_crop = frame[y1:y2, x1:x2]
                if chest_crop.size > 0:
                    raw_m = self.motion_tracker.process(chest_crop)
                    if raw_m is not None:
                        m_val = float(np.clip(raw_m, -1.8, 1.8))
                        self.motion_buffer.append(m_val)
                        self.motion_timestamps.append(time.time())

            if len(self.motion_buffer) >= 100 and frame_idx % 12 == 0:
                dur = self.motion_timestamps[-1] - self.motion_timestamps[0]
                if dur >= 5.0:
                    s_arr = np.array(list(self.motion_buffer))
                    t_arr = np.array(list(self.motion_timestamps))
                    smooth_sig = ChestSignalFilter.smooth(s_arr, timestamps=t_arr)
                    calc_bpm, _ = ChestBREstimator.estimate(smooth_sig, dur)
                    if 7.0 <= calc_bpm <= 32.0:
                        self.current_chest_bpm = round(0.12 * calc_bpm + 0.88 * self.current_chest_bpm, 1)

            # 4. VẼ HUD ĐỒ HỌA HỢP NHẤT TRÊN DISPLAY FRAME
            # A. Vẽ hộp vùng mặt rPPG (Trán & Má)
            if self.show_face_roi and face_detected:
                roi_res = rppg_data.get("roi_result", {})
                rois = roi_res.get("rois", {})
                if "forehead" in rois:
                    pts = rois["forehead"].get("box_points")
                    if pts is not None:
                        cv2.polylines(display_frame, [pts], True, (0, 215, 255), 2)
                        cv2.putText(display_frame, "TRAN (rPPG)", (pts[0][0], max(16, pts[0][1] - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 215, 255), 1)
                for cheek in ("left_cheek", "right_cheek"):
                    if cheek in rois:
                        pts = rois[cheek].get("box_points")
                        if pts is not None:
                            cv2.polylines(display_frame, [pts], True, (0, 210, 180), 2)

            # B. Vẽ khung xương Pose ngực (liên tục 100% frame)
            person_seen = (time.time() - self.last_seen_person_time) < 1.5
            if self.show_skeleton and person_seen and self.last_skeleton_kpts is not None:
                kpts = self.last_skeleton_kpts
                for p1, p2 in SKELETON_PAIRS:
                    if p1 < len(kpts) and p2 < len(kpts):
                        x1_s, y1_s = int(kpts[p1][0]), int(kpts[p1][1])
                        x2_s, y2_s = int(kpts[p2][0]), int(kpts[p2][1])
                        if x1_s > 0 and y1_s > 0 and x2_s > 0 and y2_s > 0:
                            cv2.line(display_frame, (x1_s, y1_s), (x2_s, y2_s), (99, 102, 241), 2)
                for pt in [5, 6, 11, 12]:
                    if pt < len(kpts):
                        px, py = int(kpts[pt][0]), int(kpts[pt][1])
                        if px > 0 and py > 0:
                            cv2.circle(display_frame, (px, py), 5, (0, 210, 180), -1)

            # C. Vẽ khung ngực Chest ROI (xanh neon)
            if self.show_chest_roi and self.current_chest_roi is not None:
                x1, y1, x2, y2 = self.current_chest_roi
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 210, 180), 2)
                cv2.putText(display_frame, f"CHEST ROI ({x2-x1}x{y2-y1})", (x1, max(18, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 180), 1)

            # D. HUD Thông tin Hợp nhất trên đỉnh video (Medical-Grade Overlay)
            cv2.rectangle(display_frame, (10, 10), (w - 10, 78), (15, 23, 42), -1)
            cv2.rectangle(display_frame, (10, 10), (w - 10, 78), (255, 255, 255), 1)

            # Cột 1: NHỊP TIM
            cv2.putText(display_frame, "NHIP TIM (HR)", (25, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 75, 120), 1)
            cv2.putText(display_frame, f"{self.current_hr:.1f} BPM", (25, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 75, 120), 2)

            # Cột 2: THỞ KHUÔN MẶT
            cv2.putText(display_frame, "THO MAT (rPPG)", (215, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 210, 180), 1)
            cv2.putText(display_frame, f"{self.current_rr:.1f} RPM", (215, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 210, 180), 2)

            # Cột 3: THỞ LỒNG NGỰC
            cv2.putText(display_frame, "THO NGUC (YOLO)", (410, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 120, 255), 1)
            cv2.putText(display_frame, f"{self.current_chest_bpm:.1f} BPM", (410, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (160, 120, 255), 2)

            # Nén JPEG chất lượng 70
            _, jpeg_buf = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            jpeg_bytes = jpeg_buf.tobytes()

            with self.lock:
                self.latest_frame_jpeg = jpeg_bytes
                self.latest_data = {
                    "heart_rate": self.current_hr,
                    "respiration_rate": self.current_rr,
                    "chest_bpm": self.current_chest_bpm,
                    "detected": face_detected or person_seen,
                    "status": "Đang giám sát All-in-One",
                    "fps": 30.0,
                }

            # Lưu định kỳ CSDL
            now_t = time.time()
            if now_t - self.last_db_save >= 5.0 and self.current_hr > 0:
                self.last_db_save = now_t
                sev, _ = alarm_manager.evaluate_vitals(
                    heart_rate=self.current_hr,
                    respiration_rate=self.current_rr,
                    chest_bpm=self.current_chest_bpm,
                )
                local_db.insert_record(
                    heart_rate=self.current_hr,
                    respiration_rate=self.current_rr,
                    chest_bpm=self.current_chest_bpm,
                    source="unified_webcam",
                    status=sev,
                    note="Giám sát Camera All-in-One"
                )

            frame_idx += 1
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.001, frame_time - elapsed))

    def generate_mjpeg_stream(self):
        with self.lock:
            self.active_clients += 1
        self.start()
        try:
            while self.running:
                frame_bytes = None
                with self.lock:
                    frame_bytes = self.latest_frame_jpeg

                if frame_bytes is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    )
                time.sleep(0.033)
        finally:
            with self.lock:
                self.active_clients = max(0, self.active_clients - 1)


# Khởi tạo singleton managers
rppg_streamer = RPPGStreamManager()
chest_streamer = ChestStreamManager()
unified_streamer = UnifiedStreamManager()
