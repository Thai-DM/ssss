"""
breathing_worker.py
--------------------
Chi dung ESP32-CAM (khong con dung webcam laptop nua).

Cach dung:
    1. Sua bien ESPCAM_URL o duoi day thanh dia chi IP that cua
       ESP32-CAM nha ban, vd: "http://192.168.1.50/stream"
    2. Chay truc tiep, khong can them tham so gi ca:
           python breathing_worker.py

Cac tham so khac (API, interval, roi...) van co the truyen qua dong
lenh neu muon, nhung KHONG BAT BUOC, vi da co gia tri mac dinh hop ly.

Script tu dong:
    - Mo stream MJPEG cua ESP32-CAM qua HTTP.
    - Dung GaussianBlur 7x7 (thay vi 5x5) vi JPEG artifact tu ESP32-CAM
      nhieu hon ảnh raw tu webcam, Farneback rat nhay voi block artifact.
    - Tu dong ket noi lai neu ESP32-CAM mat mang hoac reboot.
    - In canh bao neu FPS trung binh qua thap (<5) de nguoi dung biet
      chat luong tin hieu co the bi anh huong.

Do chinh xac cua mo hinh (bandpass 0.1-0.5Hz, prominence, distance)
KHONG thay doi -- cac tham so nay tu tinh fps tu chinh du lieu, nen
tu dong thich nghi voi FPS thap cua ESP32-CAM ma khong can chinh tay.
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


# ==================================================================
# SUA DONG NAY: dia chi stream cua ESP32-CAM nha ban
# ==================================================================
ESPCAM_URL = "http://192.168.143.249/stream"

CAPTURE_DIR = "captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)
# ----------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Breathing rate worker (ESP32-CAM only) -> push to API"
    )
    parser.add_argument(
        "--api", default="http://127.0.0.1:5000",
        help="Base URL cua Flask backend (mac dinh http://127.0.0.1:5000)"
    )
    parser.add_argument(
        "--espcam", type=str, default=ESPCAM_URL,
        help=(
            "URL MJPEG stream cua ESP32-CAM. Mac dinh lay tu bien "
            f"ESPCAM_URL trong file ({ESPCAM_URL}). Chi can truyen "
            "tham so nay neu muon ghi de tam thoi."
        )
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
    args = parser.parse_args()

    args.espcam = args.espcam.strip()
    if not args.espcam or not (
        args.espcam.startswith("http://") or args.espcam.startswith("https://")
    ):
        parser.error(
            f"URL ESP32-CAM khong hop le: {args.espcam!r}\n"
            "Sua bien ESPCAM_URL o dau file, hoac truyen --espcam http://IP/stream"
        )

    return args


# ----------------------------------------------------------------
# Mo stream ESP32-CAM
# ----------------------------------------------------------------
def open_capture(url):
    print(f"[worker] Dang ket noi toi ESP32-CAM: {url}")
    cap = cv2.VideoCapture(url)
    # Giam buffer cua OpenCV xuong 2 frame: tranh doc frame cu tich
    # dong trong hang cho khi xu ly cham (YOLO nang) -> giam do tre.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not cap.isOpened():
        raise RuntimeError(
            f"Khong mo duoc stream ESP32-CAM tai {url}\n"
            "Kiem tra: (1) ESP32-CAM da ket noi WiFi chua? "
            "(2) Dung IP dung chua? (3) Mo http://IP/ping tren browser xem co tra 'OK' khong?"
        )
    print(f"[worker] Da ket noi ESP32-CAM: {url}")
    return cap


def reopen_espcam(url, max_retries=10, wait_sec=3.0):
    """Thu mo lai stream ESP32-CAM sau khi mat ket noi."""
    for attempt in range(1, max_retries + 1):
        print(f"[worker] Thu ket noi lai ESP32-CAM lan {attempt}/{max_retries}...")
        time.sleep(wait_sec)
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print("[worker] Da ket noi lai ESP32-CAM thanh cong")
                return cap
        cap.release()
    raise RuntimeError(
        f"Khong the ket noi lai ESP32-CAM sau {max_retries} lan thu. "
        "Kiem tra nguon dien va WiFi cua ESP32-CAM."
    )


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

    # Kiem tra box confidence > 0.5 (giong webcam_worker) de loai bo
    # truong hop YOLO phat hien nguoi voi do tin cay thap khi anh ESP32-CAM bi mo/nhieu JPEG
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
    if bpm_value > 25 or bpm_value < 5:
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
    print(f"[worker] >>> Nguon camera: ESP32-CAM ({args.espcam})")

    cap = open_capture(args.espcam)

    # Dung duong dan tuyet doi de tranh loi FileNotFoundError khi chay tu thu muc khac
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo11n-pose.onnx")
    model = YOLO(model_path)
    roi_extractor = ROIExtractor()

    # Dung blur manh hon (7x7) vi JPEG artifact tu ESP32-CAM nhieu hon
    # anh raw tu webcam, Farneback rat nhay voi block artifact JPEG.
    tracker = MotionTracker(blur_kernel=(7, 7))

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

    # Theo doi FPS thuc te cua ESP32-CAM de canh bao nguoi dung
    fps_counter = 0
    fps_window_start = time.time()
    MIN_ACCEPTABLE_FPS = 5.0

    print(f"[worker] Bat dau do nhip tho qua ESP32-CAM ({args.espcam})")
    print(f"[worker] Khoi {args.interval:.0f} giay, Esc de dung")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("[worker] Mat ket noi ESP32-CAM, dang thu ket noi lai...")
                cap.release()
                cap = reopen_espcam(args.espcam)
                fps_counter = 0
                fps_window_start = time.time()
                continue

            frame_idx += 1
            fps_counter += 1

            # Canh bao neu FPS trung binh trong 10 giay < 5
            elapsed_fps = time.time() - fps_window_start
            if elapsed_fps >= 10.0:
                actual_fps = fps_counter / elapsed_fps
                if actual_fps < MIN_ACCEPTABLE_FPS:
                    print(
                        f"[worker] CANH BAO: FPS trung binh {actual_fps:.1f} < {MIN_ACCEPTABLE_FPS}. "
                        "Tin hieu co the khong du de loc bandpass chinh xac. "
                        "Thu: (1) giam jpeg_quality trong firmware, (2) chuyen sang QVGA."
                    )
                fps_counter = 0
                fps_window_start = time.time()

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

                # Bỏ vẽ hạt khi dùng ESP32-CAM để đỡ lag
                # tracker.draw_flow_particles(display_frame, current_roi, step=8)
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

                    # Kiem tra FPS cua khoi: neu qua thap, canh bao truoc khi tinh
                    if actual_fps_block < MIN_ACCEPTABLE_FPS:
                        print(
                            f"[worker] CANH BAO: FPS khoi nay chi {actual_fps_block:.1f} fps "
                            f"({len(values)} frame / {duration:.1f}s). "
                            "BPM se duoc tinh nhung do chinh xac co the thap hon binh thuong."
                        )

                    # FIX: boc try/except quanh buoc loc + tinh BPM. Truoc
                    # day, neu gap truong hop bien (vd fps qua thap/khong
                    # on dinh khien bandpass tra ve sai dang du lieu), loi
                    # se CRASH toan bo worker. Gio neu 1 khoi loi, chi in
                    # canh bao va BO QUA khoi do, worker van chay tiep.
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
                        # filtered_signal co the da bi resample nen do dai
                        # co the khac voi timestamps goc.
                        x_raw = [t - timestamps[0] for t in timestamps]
                        if len(filtered_signal) == len(x_raw):
                            x_filtered = x_raw
                        else:
                            x_filtered = np.linspace(0, duration, len(filtered_signal))
                        raw_line.set_data(x_raw, values)
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
