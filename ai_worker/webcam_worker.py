"""
webcam_worker.py
--------------------
Phien ban dung WEBCAM LAPTOP (camera tich hop) thay vi ESP32-CAM.

Cach dung:
    python webcam_worker.py

Tham so tuy chon:
    --api         URL backend Flask  (mac dinh http://127.0.0.1:5000)
    --camera      Index camera        (mac dinh 0 = webcam chinh)
    --interval    Do dai khoi (giay)  (mac dinh 60)

So voi breathing_worker.py (ESP32-CAM):
    - Dung cv2.VideoCapture(index) thay vi URL stream HTTP.
    - Blur 5x5 (thay vi 7x7) vi webcam cho anh raw, khong bi JPEG artifact.
    - Khong can reconnect logic phuc tap.
    - FPS cao va on dinh hon (~30 fps).
"""

import argparse
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
from ultralytics import YOLO
import os
from datetime import datetime
from ai.roi_extractor import ROIExtractor
from ai.motion_tracker import MotionTracker
from ai.signal_filter import SignalFilter
from ai.breathing_rate import BreathingRateEstimator


CAPTURE_DIR = "captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)


# ----------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Breathing rate worker (WEBCAM laptop) -> push to API"
    )
    parser.add_argument(
        "--api", default="http://127.0.0.1:5000",
        help="Base URL cua Flask backend (mac dinh http://127.0.0.1:5000)"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Index cua webcam (mac dinh 0 = camera chinh cua laptop)"
    )
    parser.add_argument(
        "--interval", type=float, default=60.0,
        help="Do dai MOI khoi tin hieu KHONG chong lap, don vi giay (mac dinh 60)"
    )
    parser.add_argument(
        "--roi-update-every", type=int, default=10,
        help="So frame giua 2 lan chay lai YOLO de cap nhat vi tri ROI (mac dinh 10)"
    )
    parser.add_argument(
        "--roi-smoothing", type=float, default=0.3,
        help="He so lam muot EMA khi ROI di chuyen, 0-1 (mac dinh 0.3)"
    )
    return parser.parse_args()


# ----------------------------------------------------------------
# Mo webcam laptop
# ----------------------------------------------------------------
def open_webcam(camera_index):
    """Mo webcam laptop theo index (0 = camera chinh)."""
    print(f"[worker] Dang mo webcam (index={camera_index})...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # CAP_DSHOW nhanh hon tren Windows
    if not cap.isOpened():
        # Thu lai khong co CAP_DSHOW (Linux / macOS)
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Khong mo duoc webcam voi index={camera_index}.\n"
            "Kiem tra: (1) Laptop co camera khong? "
            "(2) Camera co dang bi ung dung khac chiem khong? "
            "(3) Thu --camera 1 neu co nhieu camera."
        )
    # Dat do phan giai 640x480 de tiet kiem CPU
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[worker] Da mo webcam: {actual_w}x{actual_h}")
    return cap


# ----------------------------------------------------------------
# [FIX] Cập nhật hàm smooth_roi: Thêm tham số threshold để chống lỗi "ROI jitter"
# Lỗi cũ: YOLO cập nhật sai số 1-2 pixel khiến Optical Flow nhận diện thành luồng chuyển động lớn (khuếch đại motion).
# Cách sửa: Nếu khoảng cách dịch chuyển tâm của khung ngực (dist) < threshold (10.0), ta giữ nguyên khung cũ (khóa cứng ROI).
def smooth_roi(old_roi, new_roi, alpha, threshold=10.0):
    if old_roi is None:
        return new_roi
        
    # Tính tâm của ROI cũ và mới
    cx_old = (old_roi[0] + old_roi[2]) / 2
    cy_old = (old_roi[1] + old_roi[3]) / 2
    cx_new = (new_roi[0] + new_roi[2]) / 2
    cy_new = (new_roi[1] + new_roi[3]) / 2
    
    # Tính khoảng cách dịch chuyển của tâm vùng ngực
    dist = ((cx_new - cx_old)**2 + (cy_new - cy_old)**2)**0.5
    
    # Nếu dịch chuyển quá nhỏ (do YOLO rung), khóa chặt khung cũ
    if dist < threshold:
        return old_roi
        
    return tuple(int(old_roi[i] + alpha * (new_roi[i] - old_roi[i])) for i in range(4))


def detect_chest_roi(model, roi_extractor, frame):
    result = model(frame, verbose=False)[0]
    
    # Kiem tra xem co box va confidence cua box dau tien > 0.5 khong
    if result.boxes is None or len(result.boxes) == 0 or float(result.boxes.conf[0]) < 0.5:
        return None
        
    if result.keypoints is None or len(result.keypoints.xy) == 0:
        return None
    person = result.keypoints.xy[0]
    ls, rs, lh, rh = person[5], person[6], person[11], person[12]
    x1, y1, x2, y2 = roi_extractor.get_chest_roi(frame, ls, rs, lh, rh)
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 > x1 and y2 > y1:
        return (x1, y1, x2, y2)
    return None


def push_bpm(api_base, bpm_value, frame):
    url = f"{api_base}/api/breathing"

    image_path = None

    # Chỉ lưu ảnh khi BPM vượt ngưỡng
    if bpm_value > 25 or bpm_value < 10:
        filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
        image_path = os.path.join(CAPTURE_DIR, filename)
        cv2.imwrite(image_path, frame)

    detect_time = datetime.now()

    print("=" * 60)
    print(f"[Worker] Phát hiện cảnh báo : {detect_time:%Y-%m-%d %H:%M:%S}")
    print(f"[Worker] BPM              : {bpm_value:.2f}")
    print(f"[Worker] Ảnh              : {image_path}")
    print("=" * 60)

    payload = {
        "bpm": round(bpm_value, 2),
        "source": "camera_ai",
        "image": image_path
    }

    try:
        resp = requests.post(
            url,
            json=payload,
            timeout=5,
        )

        if resp.status_code == 201:
            print(f"[worker] Da gui BPM={bpm_value:.2f}")

        else:
            print(resp.text)

    except requests.exceptions.RequestException as e:
        print(e)


# ----------------------------------------------------------------
def main():
    args = parse_args()
    print(f"[worker] >>> Nguon camera: WEBCAM (index={args.camera})")

    cap = open_webcam(args.camera)

    # Dung duong dan tuyet doi de tranh loi FileNotFoundError khi chay tu thu muc khac
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo11n-pose.onnx")
    model = YOLO(model_path)
    roi_extractor = ROIExtractor()

    # Webcam cho anh raw (khong bi JPEG artifact nhu ESP32-CAM)
    # nen dung blur 5x5 la du, nhe hon 7x7 -> giu duoc nhieu chi tiet hon.
    tracker = MotionTracker(blur_kernel=(5, 5))

    current_roi = None
    frame_idx = 0

    # --- Cua so matplotlib ---
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    raw_line, = ax.plot([], [], alpha=0.5, label="Raw Signal")
    filtered_line, = ax.plot([], [], linewidth=2, label="Filtered Signal")
    peaks_line, = ax.plot([], [], "x", color='red', markersize=8, markeredgewidth=2, label="Peaks")
    ax.set_title("Respiration Signal Analysis")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vertical Motion")
    ax.legend()
    ax.grid(True)
    try:
        fig.canvas.manager.set_window_title("Breathing Signal Analysis")
    except Exception:
        pass
    fig.show()

    block_signal = []
    block_start_time = time.time()

    print(f"[worker] Bat dau do nhip tho qua WEBCAM (index={args.camera})")
    print(f"[worker] Khoi {args.interval:.0f} giay, Esc de dung")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("[worker] Khong doc duoc frame tu webcam, dang thu lai...")
                time.sleep(0.1)
                continue

            frame_idx += 1

            display_frame = frame
            should_update_roi = (current_roi is None) or (frame_idx % args.roi_update_every == 0)

            if should_update_roi:
                detected = detect_chest_roi(model, roi_extractor, frame)
                if detected is not None:
                    was_none = current_roi is None
                    current_roi = smooth_roi(current_roi, detected, args.roi_smoothing)
                    if was_none:
                        print("[worker] Da phat hien vung nguc, bat dau theo doi")
                else:
                    if current_roi is not None:
                        print("[worker] Khong thay nguoi, tam dung theo doi...")
                        current_roi = None
                        tracker.reset()

            if current_roi is not None:
                x1, y1, x2, y2 = current_roi
                roi = frame[y1:y2, x1:x2]
                motion = tracker.process(roi)

                if motion is not None:
                    block_signal.append((time.time(), motion))
                    cv2.putText(
                        display_frame, f"Motion: {motion:.6f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )

                tracker.draw_flow_particles(display_frame, current_roi, step=8)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                display_frame, "ESP32-CAM", (20, display_frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
            )

            cv2.imshow("Chest ROI", display_frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

            if plt.fignum_exists(fig.number):
                fig.canvas.flush_events()

            # --- Tinh BPM sau moi khoi --interval giay ---
            now = time.time()
            if now - block_start_time >= args.interval:
                if len(block_signal) >= 10:
                    timestamps = [t for t, _ in block_signal]
                    values = [v for _, v in block_signal]
                    duration = timestamps[-1] - timestamps[0]
                    actual_fps_block = len(values) / duration if duration > 0 else 0

                    try:
                        filtered_signal = SignalFilter.smooth(
                            values, timestamps=timestamps, duration_seconds=duration
                        )
                        bpm, _peaks = BreathingRateEstimator.estimate(filtered_signal, duration)
                    except Exception as e:
                        print(
                            f"[worker] LOI khi tinh BPM cho khoi nay, BO QUA khoi "
                            f"va tiep tuc: {type(e).__name__}: {e}"
                        )
                        block_signal = []
                        block_start_time = now
                        continue

                    push_bpm(
                        args.api,
                        bpm,
                        display_frame.copy()
                    )
                    print(
                        f"[worker] Respiration Rate: {bpm:.2f} BPM "
                        f"(fps={actual_fps_block:.1f}, frames={len(values)}, duration={duration:.1f}s)"
                    )

                    if plt.fignum_exists(fig.number):
                        x_raw = [t - timestamps[0] for t in timestamps]
                        if len(filtered_signal) == len(x_raw):
                            x_filtered = x_raw
                        else:
                            x_filtered = np.linspace(0, duration, len(filtered_signal))
                        # Can tru trung binh de ve do thi raw va filtered cung truc Y, de so sanh
                        raw_values_for_plot = np.array(values) - np.mean(values)
                        raw_line.set_data(x_raw, raw_values_for_plot)
                        filtered_line.set_data(x_filtered, filtered_signal)
                        
                        if len(_peaks) > 0:
                            peaks_line.set_data([x_filtered[p] for p in _peaks], [filtered_signal[p] for p in _peaks])
                        else:
                            peaks_line.set_data([], [])

                        ax.relim()
                        ax.autoscale_view()
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                else:
                    print(f"[worker] Khoi nay chi co {len(block_signal)} sample, bo qua")

                block_signal = []
                block_start_time = now

    except KeyboardInterrupt:
        print("[worker] Da dung theo yeu cau (Ctrl+C)")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        plt.close(fig)


if __name__ == "__main__":
    main()
