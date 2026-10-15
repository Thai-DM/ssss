"""
generate_demo_video.py
----------------------
Tạo video mẫu chuẩn 'demo_video.mp4' (15 giây, 30 FPS) từ ảnh chân dung:
- Áp dụng dao động huyết sắc tố (Photoplethysmography) với:
  + Nhịp tim chuẩn: 74 BPM (1.23 Hz) - biên độ hấp thụ ánh sáng lớn nhất ở kênh Green.
  + Nhịp thở chuẩn: 16 RPM (0.27 Hz).
- Mô phỏng chuyển động nhẹ đầu (dao động góc nghiêng theta và vị trí)
  để kiểm chứng thuật toán nắn xoay Affine và triệt tiêu nhiễu của CHROM.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np


def create_demo_video(image_path: str = "demo_face.jpg", output_video: str = "demo_video.mp4", duration_sec: float = 15.0, fps: float = 30.0):
    img = cv2.imread(image_path)
    if img is None:
        print(f"LỖI: Không tìm thấy ảnh {image_path}")
        return False

    # Resize về kích thước chuẩn 640x480
    target_w, target_h = 640, 480
    base_frame = cv2.resize(img, (target_w, target_h))

    total_frames = int(duration_sec * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (target_w, target_h))

    # Tần số mục tiêu
    target_hr = 74.0  # 74 BPM
    target_rr = 16.0  # 16 RPM
    f_pulse = target_hr / 60.0
    f_resp = target_rr / 60.0

    print(f"Đang khởi tạo '{output_video}' ({total_frames} frames, {duration_sec}s)...")
    print(f"Nhịp tim thiết lập: {target_hr} BPM ({f_pulse:.2f} Hz)")
    print(f"Nhịp thở thiết lập: {target_rr} RPM ({f_resp:.2f} Hz)")

    center_x = target_w / 2.0
    center_y = target_h / 2.0

    for i in range(total_frames):
        t = i / fps

        # 1. Dao động vi mạch máu: Kênh Green hấp thụ mạnh nhất, Red & Blue hấp thụ yếu hơn
        pulse = np.sin(2 * np.pi * f_pulse * t)
        breath = np.sin(2 * np.pi * f_resp * t)

        # Hệ số điều chế màu
        mod_r = 1.0 + 0.006 * pulse + 0.008 * breath
        mod_g = 1.0 - 0.024 * pulse + 0.008 * breath
        mod_b = 1.0 - 0.004 * pulse + 0.008 * breath

        frame_mod = base_frame.astype(np.float32)
        frame_mod[:, :, 0] = np.clip(frame_mod[:, :, 0] * mod_b, 0, 255)  # Blue
        frame_mod[:, :, 1] = np.clip(frame_mod[:, :, 1] * mod_g, 0, 255)  # Green
        frame_mod[:, :, 2] = np.clip(frame_mod[:, :, 2] * mod_r, 0, 255)  # Red

        # 2. Dao động lắc nhẹ đầu (góc nghiêng theta từ -3 đến +3 độ, độ trôi vị trí 2-4 px)
        angle = 3.5 * np.sin(2 * np.pi * 0.15 * t)
        dx = 3.0 * np.sin(2 * np.pi * 0.2 * t)
        dy = 2.0 * np.cos(2 * np.pi * 0.15 * t)

        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        M[0, 2] += dx
        M[1, 2] += dy

        warped_frame = cv2.warpAffine(
            frame_mod.astype(np.uint8),
            M,
            (target_w, target_h),
            borderMode=cv2.BORDER_REFLECT,
        )

        out.write(warped_frame)

    out.release()
    print(f"Đã xuất thành công video demo: {output_video}")
    return True


if __name__ == "__main__":
    create_demo_video()
