"""
main.py
-------
Chương trình chính thực thi hệ thống rPPG / rBBG:
- Hỗ trợ chạy trực tiếp từ Webcam (default: 0).
- Hỗ trợ phân tích từ file video có sẵn (--source video.mp4).
- Hỗ trợ chế độ mô phỏng kiểm thử (--simulate) không cần camera.
- Xuất dữ liệu chuỗi thời gian ra file CSV (--save_csv signals.csv).
- Ghi hình video đầu ra (--save_video output.mp4).
"""

import argparse
import csv
import os
import sys
import time
import cv2
import numpy as np

# Đảm bảo in tiếng Việt không bị lỗi font trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rppg_pipeline import RPPGPipeline
from ui_visualizer import UIVisualizer


def run_simulation(args):
    """
    Chế độ mô phỏng: Tạo video tổng hợp khuôn mặt có dao động nhịp tim & nhịp thở
    để kiểm thử toàn bộ thuật toán CHROM khi không có camera thực tế.
    """
    print("=" * 60)
    print("BẮT ĐẦU CHẾ ĐỘ MÔ PHỎNG (SIMULATION TEST)")
    print("=" * 60)
    fps = 30.0
    duration = 15.0  # 15 giây
    total_frames = int(fps * duration)

    pipeline = RPPGPipeline(window_sec=args.window, target_fs=fps, roi_mode=args.roi)
    visualizer = UIVisualizer(show_mesh=False)

    # Nhịp tim mẫu: 72 BPM -> f = 1.20 Hz
    # Nhịp thở mẫu: 16 RPM -> f = 0.27 Hz
    target_hr = 72.0
    target_rr = 16.0
    f_pulse = target_hr / 60.0
    f_resp = target_rr / 60.0

    print(f"Tham số mô phỏng: HR={target_hr} BPM ({f_pulse:.2f} Hz), RR={target_rr} RPM ({f_resp:.2f} Hz)")

    # Tạo một khung hình khuôn mặt giả lập
    frame_w, frame_h = 640, 480
    csv_rows = []

    for i in range(total_frames):
        t = i / fps

        # Dao động PPG trong mạch máu (hấp thụ mạnh ở kênh Xanh lá G)
        pulse = 2.5 * np.sin(2 * np.pi * f_pulse * t)
        breath = 1.0 * np.sin(2 * np.pi * f_resp * t)
        noise = np.random.normal(0, 0.1)

        # Mô phỏng RGB da mặt
        r_val = float(180.0 + 0.4 * pulse + 0.6 * breath + noise)
        g_val = float(140.0 - 1.8 * pulse + 0.6 * breath + noise)
        b_val = float(115.0 - 0.3 * pulse + 0.6 * breath + noise)

        # Đẩy trực tiếp vào buffer pipeline
        pipeline.timestamps.append(t)
        pipeline.r_buffer.append(r_val)
        pipeline.g_buffer.append(g_val)
        pipeline.b_buffer.append(b_val)
        pipeline.frame_count += 1
        pipeline.measured_fps = fps

        min_frames = int(fps * 3.5)
        fft_result = None

        if len(pipeline.r_buffer) >= min_frames and (i % 2 == 0):
            r_arr = np.array(list(pipeline.r_buffer))
            g_arr = np.array(list(pipeline.g_buffer))
            b_arr = np.array(list(pipeline.b_buffer))
            res = pipeline.chrom_processor.process(r_arr, g_arr, b_arr, fs=fps)
            if res["success"]:
                new_hr = res["hr_bpm"]
                new_rr = res["rr_rpm"]
                if pipeline.current_hr == 0.0:
                    pipeline.current_hr = new_hr
                else:
                    pipeline.current_hr = 0.2 * new_hr + 0.8 * pipeline.current_hr

                if pipeline.current_rr == 0.0:
                    pipeline.current_rr = new_rr
                else:
                    pipeline.current_rr = 0.2 * new_rr + 0.8 * pipeline.current_rr

                pipeline.current_snr_hr = res["snr_hr"]
                pipeline.current_snr_rr = res["snr_rr"]
                pipeline.is_stable = True
                pipeline.status_message = "Mô phỏng ổn định"

                if len(res["bvp_cardiac"]) > 0:
                    pipeline.bvp_wave_display.append(float(res["bvp_cardiac"][-1]))
                if len(res["bvp_respiratory"]) > 0:
                    pipeline.resp_wave_display.append(float(res["bvp_respiratory"][-1]))

                fft_result = {
                    "freqs": res["freqs"],
                    "fft_hr": res["fft_hr"],
                    "fft_rr": res["fft_rr"],
                    "f_heart": res["f_heart"],
                    "f_resp": res["f_resp"],
                    "alpha": res["alpha"],
                }

        # Lưu dữ liệu
        csv_rows.append([t, r_val, g_val, b_val, pipeline.current_hr, pipeline.current_rr])

        if i % 30 == 0:
            print(f"Tiến độ: frame {i}/{total_frames} ({t:.1f}s) | HR: {pipeline.current_hr:.1f} BPM | RR: {pipeline.current_rr:.1f} RPM")

    print("=" * 60)
    print("KẾT QUẢ MÔ PHỎNG:")
    print(f"HR ước lượng: {pipeline.current_hr:.2f} BPM (Kỳ vọng: {target_hr} BPM)")
    print(f"RR ước lượng: {pipeline.current_rr:.2f} RPM (Kỳ vọng: {target_rr} RPM)")
    print(f"Sai số HR: {abs(pipeline.current_hr - target_hr):.2f} BPM")
    print(f"Sai số RR: {abs(pipeline.current_rr - target_rr):.2f} RPM")
    print("=" * 60)

    if args.save_csv:
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "R", "G", "B", "HR_BPM", "RR_RPM"])
            writer.writerows(csv_rows)
        print(f"Đã lưu kết quả ra file: {args.save_csv}")

    pipeline.close()


def run_live(args):
    """Chạy trực tiếp từ Webcam hoặc Video file với giao diện thời gian thực."""
    source = args.source
    if source.isdigit():
        source = int(source)

    print(f"Đang mở nguồn video: {source}...")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"LỖI: Không thể mở nguồn video {source}. Vui lòng kiểm tra lại thiết bị webcam!")
        sys.exit(1)

    # Thử cấu hình độ phân giải camera HD 720p hoặc VGA
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    pipeline = RPPGPipeline(
        window_sec=args.window,
        roi_mode=args.roi,
    )
    visualizer = UIVisualizer(show_mesh=False)

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, 30.0, (640 + 420, 480))

    csv_records = []
    start_time = time.time()

    print("=" * 60)
    print("HỆ THỐNG ĐÃ SẴN SÀNG.")
    print("Phím tắt điều khiển:")
    print("  [Q] hoặc [ESC] : Dừng và thoát")
    print("  [M]            : Bật/tắt hiển thị lưới MediaPipe Face Mesh")
    print("  [R]            : Đặt lại (Reset) bộ đệm")
    print("  [S]            : Chụp và lưu ảnh màn hình")
    print("  [1/2/3]        : Đổi ROI (1: Tất cả | 2: Trán | 3: Má)")
    print("=" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Hết video hoặc mất kết nối camera.")
                break

            cur_time = time.time() - start_time
            pipeline_data = pipeline.process_frame(frame)

            # Ghép hình hiển thị HUD
            rendered_display = visualizer.render(frame, pipeline_data)

            if writer:
                # Resize nếu kích thước canvas khác chuẩn ghi hình
                if rendered_display.shape[0] != 480 or rendered_display.shape[1] != (640 + 420):
                    rec_frame = cv2.resize(rendered_display, (640 + 420, 480))
                else:
                    rec_frame = rendered_display
                writer.write(rec_frame)

            # Lưu CSV nếu bật
            if args.save_csv and pipeline_data.get("detected"):
                mean_rgb = pipeline_data.get("roi_result", {}).get("mean_rgb", (0, 0, 0))
                csv_records.append([
                    round(cur_time, 3),
                    round(mean_rgb[0], 2),
                    round(mean_rgb[1], 2),
                    round(mean_rgb[2], 2),
                    round(pipeline_data.get("hr_bpm", 0.0), 2),
                    round(pipeline_data.get("rr_rpm", 0.0), 2),
                ])

            # Hiển thị cửa sổ OpenCV
            cv2.imshow("He thong do rPPG / rBBG - CHROM Algorithm", rendered_display)

            # Bắt sự kiện bàn phím
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # Q or ESC
                print("Đang dừng chương trình...")
                break
            elif key == ord("m") or key == ord("M"):
                visualizer.toggle_mesh()
                print("Chuyển trạng thái Face Mesh:", visualizer.show_mesh)
            elif key == ord("r") or key == ord("R"):
                pipeline.reset()
                print("Đã reset bộ đệm.")
            elif key == ord("s") or key == ord("S"):
                snap_name = f"snapshot_{int(time.time())}.png"
                cv2.imwrite(snap_name, rendered_display)
                print(f"Đã lưu ảnh màn hình: {snap_name}")
            elif key == ord("1"):
                pipeline.roi_mode = "all"
                print("Đã chuyển chế độ ROI: Kết hợp Trán & 2 Má")
            elif key == ord("2"):
                pipeline.roi_mode = "forehead"
                print("Đã chuyển chế độ ROI: Chỉ lấy vùng Trán")
            elif key == ord("3"):
                pipeline.roi_mode = "cheeks"
                print("Đã chuyển chế độ ROI: Chỉ lấy 2 bên Má")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        pipeline.close()

        if args.save_csv and len(csv_records) > 0:
            with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(["time_s", "R", "G", "B", "HR_BPM", "RR_RPM"])
                csv_writer.writerows(csv_records)
            print(f"Đã lưu nhật ký dữ liệu ra: {args.save_csv}")

        print("Đã đóng chương trình hoàn tất.")


def main():
    parser = argparse.ArgumentParser(
        description="Hệ thống trích xuất nhịp tim và nhịp thở qua video khuôn mặt bằng thuật toán CHROM (rPPG/rBBG)."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Nguồn video: chỉ số webcam (ví dụ 0, 1) hoặc đường dẫn file video (ví dụ video.mp4). Mặc định: 0.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Chạy chế độ mô phỏng tín hiệu để kiểm thử toàn bộ thuật toán không cần camera.",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default="all",
        choices=["all", "forehead", "cheeks"],
        help="Chế độ vùng ROI trích xuất: 'all' (Trán + Má), 'forehead' (Trán), 'cheeks' (2 bên Má).",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=8.0,
        help="Độ dài cửa sổ thời gian trượt (giây) dùng cho thuật toán CHROM (Mặc định: 8.0s).",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default=None,
        help="Đường dẫn file CSV để lưu chuỗi tín hiệu và chỉ số.",
    )
    parser.add_argument(
        "--save_video",
        type=str,
        default=None,
        help="Đường dẫn file MP4 để ghi lại video hiển thị trực quan.",
    )

    args = parser.parse_args()

    if args.simulate:
        run_simulation(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
