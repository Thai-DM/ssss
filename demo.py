"""
demo.py
-------
Chương trình DEMO TEST trực quan toàn diện hệ thống rPPG / rBBG:
1. Mặc định (Default): Chạy video mẫu chuẩn 'demo_video.mp4' lặp vô tận (Looping),
   hiển thị đầy đủ giao diện HUD thời gian thực, khung bao nghiêng 5 tham số,
   ảnh ROI nắn xoay Affine, sóng dao động BVP tim & thở, phổ FFT.
2. Chế độ Webcam (--webcam): Mở webcam trực tiếp để đo nhịp tim & thở của người dùng.
3. Chế độ Tương tác Live (--interactive): Cho phép dùng phím mũi tên [LÊN/XUỐNG]
   để thay đổi nhịp tim mô phỏng tức thời (60 - 120 BPM) và xem thuật toán thích ứng.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import os
import time
import cv2
import numpy as np

from rppg_pipeline import RPPGPipeline
from ui_visualizer import UIVisualizer


def run_interactive_simulation(window_sec: float = 8.0):
    """
    Chế độ tương tác thời gian thực:
    - Hiển thị khuôn mặt 'demo_face.jpg' với hiệu ứng xoay và dao động mạch máu sống động.
    - Người dùng bấm phím mũi tên [LÊN / XUỐNG] để đổi nhịp tim (BPM).
    - Người dùng bấm phím mũi tên [TRÁI / PHẢI] để đổi nhịp thở (RPM).
    - Hệ thống xử lý qua MediaPipe Face Mesh, tính Affine Warping và CHROM trực tiếp trên từng frame!
    """
    img_path = "demo_face.jpg"
    if not os.path.exists(img_path):
        print(f"LỖI: Không tìm thấy file ảnh {img_path}")
        return

    base_img = cv2.imread(img_path)
    base_img = cv2.resize(base_img, (640, 480))
    target_w, target_h = 640, 480
    center_x, center_y = target_w / 2.0, target_h / 2.0

    fps = 30.0
    pipeline = RPPGPipeline(window_sec=window_sec, target_fs=fps)
    visualizer = UIVisualizer(show_mesh=False)

    sim_hr = 75.0  # BPM khởi đầu
    sim_rr = 16.0  # RPM khởi đầu

    print("=" * 65)
    print("CHẾ ĐỘ DEMO TEST TƯƠNG TÁC THỜI GIAN THỰC (INTERACTIVE DEMO)")
    print("=" * 65)
    print("Hướng dẫn điều khiển:")
    print("  [MŨI TÊN LÊN / XUỐNG]  : Tăng / Giảm nhịp tim mô phỏng (+/- 5 BPM)")
    print("  [MŨI TÊN PHẢI / TRÁI]  : Tăng / Giảm nhịp thở mô phỏng (+/- 2 RPM)")
    print("  [M]                    : Bật/tắt lưới tọa độ MediaPipe Face Mesh")
    print("  [R]                    : Đặt lại (Reset) bộ đệm")
    print("  [S]                    : Lưu ảnh chụp màn hình HUD")
    print("  [1/2/3]                : Đổi vùng ROI (1: Trán+Má | 2: Trán | 3: Má)")
    print("  [Q] hoặc [ESC]         : Dừng và thoát demo")
    print("=" * 65)

    frame_idx = 0
    start_t = time.perf_counter()

    cv2.namedWindow("DEMO TEST rPPG / rBBG - CHROM Algorithm", cv2.WINDOW_AUTOSIZE)

    while True:
        loop_start = time.perf_counter()
        t = frame_idx / fps

        f_pulse = sim_hr / 60.0
        f_resp = sim_rr / 60.0

        # Mô phỏng hấp thụ mao mạch da
        pulse = np.sin(2 * np.pi * f_pulse * t)
        breath = np.sin(2 * np.pi * f_resp * t)

        mod_r = 1.0 + 0.006 * pulse + 0.008 * breath
        mod_g = 1.0 - 0.024 * pulse + 0.008 * breath
        mod_b = 1.0 - 0.004 * pulse + 0.008 * breath

        frame_mod = base_img.astype(np.float32)
        frame_mod[:, :, 0] = np.clip(frame_mod[:, :, 0] * mod_b, 0, 255)
        frame_mod[:, :, 1] = np.clip(frame_mod[:, :, 1] * mod_g, 0, 255)
        frame_mod[:, :, 2] = np.clip(frame_mod[:, :, 2] * mod_r, 0, 255)

        # Mô phỏng nghiêng đầu nhẹ
        angle = 3.5 * np.sin(2 * np.pi * 0.15 * t)
        dx = 3.0 * np.sin(2 * np.pi * 0.2 * t)
        dy = 2.0 * np.cos(2 * np.pi * 0.15 * t)

        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        M[0, 2] += dx
        M[1, 2] += dy

        frame = cv2.warpAffine(
            frame_mod.astype(np.uint8),
            M,
            (target_w, target_h),
            borderMode=cv2.BORDER_REFLECT,
        )

        # Đưa vào pipeline xử lý rPPG
        pipeline_data = pipeline.process_frame(frame, timestamp=start_t + t)

        # Render giao diện HUD
        display = visualizer.render(frame, pipeline_data)

        # Vẽ thanh thông tin nhịp tim mục tiêu đang mô phỏng ở góc trên
        sim_banner = f"DANG MO PHONG: HR = {sim_hr:.0f} BPM | RR = {sim_rr:.0f} RPM (Dung mui ten de dieu chinh)"
        cv2.rectangle(display, (10, 480 - 40), (630, 480 - 15), (0, 0, 0), -1)
        cv2.putText(
            display,
            sim_banner,
            (15, 480 - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("DEMO TEST rPPG / rBBG - CHROM Algorithm", display)

        # Xử lý phím bấm (delay để duy trì 30 FPS)
        elapsed = time.perf_counter() - loop_start
        wait_ms = max(1, int((1.0 / fps - elapsed) * 1000))
        key = cv2.waitKeyEx(wait_ms)

        if key in [ord("q"), ord("Q"), 27]:
            break
        elif key in [ord("m"), ord("M")]:
            visualizer.toggle_mesh()
        elif key in [ord("r"), ord("R")]:
            pipeline.reset()
        elif key in [ord("s"), ord("S")]:
            fname = f"demo_snapshot_{int(time.time())}.png"
            cv2.imwrite(fname, display)
            print(f"Đã lưu ảnh snapshot: {fname}")
        elif key in [ord("1")]:
            pipeline.roi_mode = "all"
        elif key in [ord("2")]:
            pipeline.roi_mode = "forehead"
        elif key in [ord("3")]:
            pipeline.roi_mode = "cheeks"
        # Mã phím mũi tên trên Windows trong OpenCV waitKeyEx
        elif key in [2490368, 0x260000, 38]:  # UP ARROW
            sim_hr = min(150.0, sim_hr + 5.0)
            print(f">> Tăng nhịp tim mô phỏng: {sim_hr:.0f} BPM")
        elif key in [2621440, 0x280000, 40]:  # DOWN ARROW
            sim_hr = max(45.0, sim_hr - 5.0)
            print(f">> Giảm nhịp tim mô phỏng: {sim_hr:.0f} BPM")
        elif key in [2555904, 0x270000, 39]:  # RIGHT ARROW
            sim_rr = min(30.0, sim_rr + 2.0)
            print(f">> Tăng nhịp thở mô phỏng: {sim_rr:.0f} RPM")
        elif key in [2424832, 0x250000, 37]:  # LEFT ARROW
            sim_rr = max(8.0, sim_rr - 2.0)
            print(f">> Giảm nhịp thở mô phỏng: {sim_rr:.0f} RPM")

        frame_idx += 1

    pipeline.close()
    cv2.destroyAllWindows()
    print("Đã đóng Demo Test.")


def run_video_demo(video_path: str = "demo_video.mp4", window_sec: float = 8.0):
    """Chạy video demo mẫu lặp vô tận (Looping) với đầy đủ giao diện HUD."""
    if not os.path.exists(video_path):
        print(f"Video {video_path} chưa có. Đang tạo tự động...")
        from generate_demo_video import create_demo_video
        create_demo_video(output_video=video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"LỖI: Không mở được video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pipeline = RPPGPipeline(window_sec=window_sec, target_fs=fps)
    visualizer = UIVisualizer(show_mesh=False)

    print("=" * 65)
    print(f"ĐANG CHẠY VIDEO DEMO: '{video_path}' (Lặp liên tục)")
    print("=" * 65)
    print("Phím tắt:")
    print("  [M]            : Bật/tắt lưới tọa độ MediaPipe Face Mesh")
    print("  [R]            : Đặt lại (Reset) bộ đệm")
    print("  [S]            : Chụp và lưu ảnh màn hình HUD")
    print("  [Q] hoặc [ESC] : Dừng và thoát demo")
    print("=" * 65)

    cv2.namedWindow("DEMO TEST rPPG / rBBG - Video Looping", cv2.WINDOW_AUTOSIZE)

    frame_delay_ms = int(1000.0 / fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            # Quay lại đầu video để lặp vô tận
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        pipeline_data = pipeline.process_frame(frame)
        display = visualizer.render(frame, pipeline_data)

        cv2.imshow("DEMO TEST rPPG / rBBG - Video Looping", display)

        key = cv2.waitKey(frame_delay_ms) & 0xFF
        if key in [ord("q"), ord("Q"), 27]:
            break
        elif key in [ord("m"), ord("M")]:
            visualizer.toggle_mesh()
        elif key in [ord("r"), ord("R")]:
            pipeline.reset()
        elif key in [ord("s"), ord("S")]:
            fname = f"demo_snapshot_{int(time.time())}.png"
            cv2.imwrite(fname, display)
            print(f"Đã lưu ảnh snapshot: {fname}")

    cap.release()
    pipeline.close()
    cv2.destroyAllWindows()
    print("Đã dừng Video Demo.")


def main():
    parser = argparse.ArgumentParser(description="Chương trình DEMO TEST hệ thống rPPG / rBBG.")
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Mở webcam thực tế (camera 0) để chạy demo trực tiếp trên khuôn mặt bạn.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Chạy demo tương tác: dùng phím mũi tên [LÊN/XUỐNG/TRÁI/PHẢI] để tăng giảm nhịp tim và thở theo thời gian thực.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="demo_video.mp4",
        help="Đường dẫn video demo mẫu (mặc định: demo_video.mp4).",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=8.0,
        help="Độ dài cửa sổ thời gian trượt CHROM (giây).",
    )

    args = parser.parse_args()

    if args.webcam:
        from main import run_live
        # Giả lập arguments cho main.run_live
        class LiveArgs:
            source = "0"
            window = args.window
            roi = "all"
            save_csv = "webcam_demo_signals.csv"
            save_video = None
        run_live(LiveArgs())
    elif args.interactive:
        run_interactive_simulation(window_sec=args.window)
    else:
        run_video_demo(video_path=args.video, window_sec=args.window)


if __name__ == "__main__":
    main()
