"""
ui_visualizer.py
----------------
Giao diện trực quan hóa HUD (Heads-Up Display) trên OpenCV:
- Vẽ 5 tham số khung bao nghiêng (xc, yc, w, h, theta) ôm sát trán và 2 má.
- Hiển thị ảnh các ROI đã được chuẩn hóa Affine Warp về trục ngang.
- Vẽ đồ thị dạng sóng dao động (Oscilloscope) cho mạch máu BVP và nhịp thở.
- Vẽ đồ thị phổ tần số FFT với đỉnh tần số f_heart và f_resp.
- Hiển thị bảng đồng hồ kỹ thuật số: Nhịp tim (HR BPM), Nhịp thở (RR RPM), FPS, alpha.
"""

from typing import Dict, List, Optional
import cv2
import numpy as np


class UIVisualizer:
    """Tạo giao diện hiển thị chuyên nghiệp thời gian thực bằng OpenCV."""

    # Bảng màu HUD hiện đại (BGR)
    COLOR_BG = (20, 24, 28)
    COLOR_PANEL_BG = (30, 36, 44)
    COLOR_ACCENT_GREEN = (80, 230, 110)
    COLOR_ACCENT_RED = (80, 80, 255)
    COLOR_ACCENT_CYAN = (235, 205, 50)
    COLOR_ACCENT_ORANGE = (40, 150, 255)
    COLOR_TEXT_WHITE = (245, 245, 245)
    COLOR_TEXT_MUTED = (160, 165, 175)
    COLOR_BOX_FOREHEAD = (0, 220, 255)   # Vàng neon
    COLOR_BOX_CHEEK_L = (255, 120, 50)   # Xanh dương neon
    COLOR_BOX_CHEEK_R = (50, 220, 120)   # Xanh lục neon

    def __init__(self, show_mesh: bool = False):
        self.show_mesh = show_mesh

    def toggle_mesh(self):
        """Bật/tắt hiển thị lưới 468 landmarks."""
        self.show_mesh = not self.show_mesh

    def _draw_waveform(
        self,
        canvas: np.ndarray,
        data: List[float],
        x: int,
        y: int,
        w: int,
        h: int,
        color: tuple,
        title: str,
        unit: str = "",
    ):
        """Vẽ biểu đồ dạng sóng dao động (Oscilloscope)."""
        # Nền biểu đồ
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (20, 24, 28), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (55, 65, 75), 1)

        # Tiêu đề
        cv2.putText(
            canvas,
            f"{title} {unit}",
            (x + 8, y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            self.COLOR_TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

        if len(data) < 2:
            cv2.putText(
                canvas,
                "Dang thu thap du lieu song...",
                (x + 20, y + h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                self.COLOR_TEXT_MUTED,
                1,
                cv2.LINE_AA,
            )
            return

        data_arr = np.array(data, dtype=np.float32)
        min_val = np.min(data_arr)
        max_val = np.max(data_arr)
        span = max_val - min_val
        if span < 1e-6:
            span = 1.0

        num_pts = len(data_arr)
        pts = []
        for i, val in enumerate(data_arr):
            pt_x = int(x + 5 + (i / max(1, num_pts - 1)) * (w - 10))
            norm_y = (val - min_val) / span
            pt_y = int(y + h - 8 - norm_y * (h - 28))
            pts.append((pt_x, pt_y))

        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i + 1], color, 2, cv2.LINE_AA)

    def _draw_fft_spectrum(
        self,
        canvas: np.ndarray,
        freqs: np.ndarray,
        fft_vals: np.ndarray,
        band: tuple,
        peak_f: float,
        x: int,
        y: int,
        w: int,
        h: int,
        color: tuple,
        title: str,
    ):
        """Vẽ biểu đồ phổ tần số FFT với thanh chỉ vị trí đỉnh peak."""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (20, 24, 28), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (55, 65, 75), 1)

        cv2.putText(
            canvas,
            f"{title} (f_peak: {peak_f:.2f} Hz)",
            (x + 8, y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            self.COLOR_TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

        if freqs is None or fft_vals is None or len(freqs) == 0:
            return

        mask = (freqs >= band[0]) & (freqs <= band[1])
        if not np.any(mask):
            return

        sub_f = freqs[mask]
        sub_v = fft_vals[mask]
        max_v = np.max(sub_v)
        if max_v < 1e-6:
            max_v = 1.0

        num_bars = len(sub_f)
        step = max(1, num_bars // 60)
        bar_w = max(2, (w - 16) // (num_bars // step + 1))

        draw_idx = 0
        for i in range(0, num_bars, step):
            f_val = sub_f[i]
            val = sub_v[i]
            bx = x + 8 + draw_idx * (bar_w + 1)
            bar_h = int((val / max_v) * (h - 32))
            by = y + h - 6 - bar_h

            is_peak = abs(f_val - peak_f) < (step * (freqs[1] - freqs[0]) * 1.5)
            bar_col = self.COLOR_ACCENT_RED if is_peak else (80, 95, 110)

            cv2.rectangle(canvas, (bx, by), (bx + bar_w, y + h - 6), bar_col, -1)
            draw_idx += 1

        # Nhãn tần số biên
        cv2.putText(
            canvas,
            f"{band[0]:.1f}Hz",
            (x + 8, y + h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            self.COLOR_TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{band[1]:.1f}Hz",
            (x + w - 45, y + h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            self.COLOR_TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

    def render(self, frame: np.ndarray, pipeline_data: Dict) -> np.ndarray:
        """
        Ghép hình ảnh webcam và bảng điều khiển HUD thành khung hình hoàn chỉnh.
        """
        frame_h, frame_w, _ = frame.shape
        panel_w = 420
        total_w = frame_w + panel_w
        canvas = np.zeros((frame_h, total_w, 3), dtype=np.uint8)

        # 1. Vẽ khung video chính bên trái
        main_video = frame.copy()
        roi_result = pipeline_data.get("roi_result", {})
        rois = roi_result.get("rois", {})

        # Vẽ lưới landmarks nếu bật chế độ show_mesh
        if self.show_mesh and roi_result.get("landmarks_pixel") is not None:
            lm_pts = roi_result["landmarks_pixel"]
            for pt in lm_pts:
                cv2.circle(main_video, (int(pt[0]), int(pt[1])), 1, (100, 200, 255), -1)

        # Vẽ các khung bao nghiêng với 5 tham số
        colors = {
            "forehead": self.COLOR_BOX_FOREHEAD,
            "left_cheek": self.COLOR_BOX_CHEEK_L,
            "right_cheek": self.COLOR_BOX_CHEEK_R,
        }
        names_vn = {
            "forehead": "Tran",
            "left_cheek": "Ma trai",
            "right_cheek": "Ma phai",
        }

        inset_y_offset = 20
        for name, r_info in rois.items():
            pts = r_info["box_points"]
            (xc, yc) = r_info["center"]
            (rw, rh) = r_info["size"]
            angle = r_info["angle"]
            col = colors.get(name, (0, 255, 0))

            # Vẽ khung chữ nhật nghiêng
            cv2.polylines(main_video, [pts], isClosed=True, color=col, thickness=2, lineType=cv2.LINE_AA)
            # Vẽ tâm (xc, yc)
            cv2.circle(main_video, (int(xc), int(yc)), 3, col, -1)

            # Nhãn 5 tham số: (xc, yc, w, h, theta)
            param_label = f"{names_vn.get(name, name)}: w={rw:.0f}, h={rh:.0f}, {angle:.1f}deg"
            label_y = int(max(20, yc - rh / 2 - 8))
            cv2.putText(
                main_video,
                param_label,
                (int(max(10, xc - rw / 2)), label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                col,
                1,
                cv2.LINE_AA,
            )

            # Hiển thị ảnh ROI chuẩn hóa (Affine Warped) thu nhỏ ở góc trái
            warped = r_info.get("warped_roi")
            if warped is not None and warped.size > 0:
                try:
                    mini_h = 36
                    aspect = warped.shape[1] / max(1, warped.shape[0])
                    mini_w = max(10, min(100, int(mini_h * aspect)))
                    resized_roi = cv2.resize(warped, (mini_w, mini_h))
                    
                    # Nền bao quanh thumbnail
                    cv2.rectangle(
                        main_video,
                        (15, inset_y_offset - 2),
                        (15 + mini_w + 4, inset_y_offset + mini_h + 2),
                        (0, 0, 0),
                        -1,
                    )
                    main_video[inset_y_offset : inset_y_offset + mini_h, 17 : 17 + mini_w] = resized_roi
                    cv2.rectangle(
                        main_video,
                        (17, inset_y_offset),
                        (17 + mini_w, inset_y_offset + mini_h),
                        col,
                        1,
                    )
                    cv2.putText(
                        main_video,
                        f"Warped {names_vn.get(name, name)}",
                        (17 + mini_w + 8, inset_y_offset + 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        self.COLOR_TEXT_WHITE,
                        1,
                        cv2.LINE_AA,
                    )
                    inset_y_offset += mini_h + 10
                except Exception:
                    pass

        # Đặt video chính vào canvas
        canvas[:, :frame_w] = main_video

        # 2. Bảng điều khiển HUD bên phải (Telemetry Panel)
        panel = canvas[:, frame_w:]
        panel[:] = self.COLOR_PANEL_BG

        # Đường viền phân cách
        cv2.line(canvas, (frame_w, 0), (frame_w, frame_h), (50, 60, 72), 2)

        # Header
        cv2.putText(
            panel,
            "HE THONG DO rPPG / rBBG",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            self.COLOR_TEXT_WHITE,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "Thuat toan CHROM & MediaPipe Face Mesh",
            (20, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            self.COLOR_TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )
        cv2.line(panel, (20, 64), (panel_w - 20, 64), (50, 60, 72), 1)

        # Thông số trạng thái
        fps = pipeline_data.get("fps", 0.0)
        status = pipeline_data.get("status", "Dang khoi dong")
        is_stable = pipeline_data.get("is_stable", False)
        status_color = self.COLOR_ACCENT_GREEN if is_stable else self.COLOR_ACCENT_ORANGE

        cv2.putText(
            panel,
            f"FPS: {fps:.1f}",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            self.COLOR_TEXT_WHITE,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"Trang thai: {status}",
            (110, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            status_color,
            1,
            cv2.LINE_AA,
        )

        # Thanh tiến trình tích lũy bộ đệm
        prog = pipeline_data.get("buffer_progress", 0.0)
        prog_bar_w = panel_w - 40
        cv2.rectangle(panel, (20, 96), (20 + prog_bar_w, 104), (45, 52, 62), -1)
        fill_w = int(prog * prog_bar_w)
        cv2.rectangle(panel, (20, 96), (20 + fill_w, 104), self.COLOR_ACCENT_CYAN, -1)

        # ==========================================
        # CARD 1: NHỊP TIM (HEART RATE - HR BPM)
        # ==========================================
        card1_y = 118
        card1_h = 150
        cv2.rectangle(panel, (15, card1_y), (panel_w - 15, card1_y + card1_h), (24, 29, 36), -1)
        cv2.rectangle(panel, (15, card1_y), (panel_w - 15, card1_y + card1_h), (55, 65, 78), 1)

        cv2.putText(
            panel,
            "5.1. NHIP TIM (HEART RATE)",
            (25, card1_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            self.COLOR_ACCENT_RED,
            1,
            cv2.LINE_AA,
        )

        hr_val = pipeline_data.get("hr_bpm", 0.0)
        hr_str = f"{hr_val:.1f}" if hr_val > 0 else "--"
        cv2.putText(
            panel,
            hr_str,
            (25, card1_y + 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            self.COLOR_TEXT_WHITE,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "BPM",
            (150, card1_y + 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.COLOR_ACCENT_RED,
            2,
            cv2.LINE_AA,
        )

        fft_info = pipeline_data.get("fft_result")
        f_heart = fft_info.get("f_heart", 0.0) if fft_info else (hr_val / 60.0 if hr_val > 0 else 0.0)
        snr_h = pipeline_data.get("snr_hr", 0.0)
        cv2.putText(
            panel,
            f"f_heart: {f_heart:.2f} Hz | SNR: {snr_h:.1f} dB",
            (25, card1_y + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            self.COLOR_TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

        # Sóng BVP
        bvp_wave = pipeline_data.get("bvp_wave", [])
        self._draw_waveform(
            panel,
            bvp_wave,
            x=25,
            y=card1_y + 98,
            w=panel_w - 50,
            h=42,
            color=self.COLOR_ACCENT_GREEN,
            title="Song mach BVP (0.75 - 2.5 Hz)",
        )

        # ==========================================
        # CARD 2: NHỊP THỞ (RESPIRATION RATE - RR RPM)
        # ==========================================
        card2_y = card1_y + card1_h + 12
        card2_h = 150
        cv2.rectangle(panel, (15, card2_y), (panel_w - 15, card2_y + card2_h), (24, 29, 36), -1)
        cv2.rectangle(panel, (15, card2_y), (panel_w - 15, card2_y + card2_h), (55, 65, 78), 1)

        cv2.putText(
            panel,
            "5.2. NHIP THO (RESPIRATION RATE)",
            (25, card2_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            self.COLOR_ACCENT_CYAN,
            1,
            cv2.LINE_AA,
        )

        rr_val = pipeline_data.get("rr_rpm", 0.0)
        rr_str = f"{rr_val:.1f}" if rr_val > 0 else "--"
        cv2.putText(
            panel,
            rr_str,
            (25, card2_y + 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            self.COLOR_TEXT_WHITE,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "RPM (lan/phut)",
            (130, card2_y + 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            self.COLOR_ACCENT_CYAN,
            2,
            cv2.LINE_AA,
        )

        f_resp = fft_info.get("f_resp", 0.0) if fft_info else (rr_val / 60.0 if rr_val > 0 else 0.0)
        snr_r = pipeline_data.get("snr_rr", 0.0)
        cv2.putText(
            panel,
            f"f_resp: {f_resp:.2f} Hz | SNR: {snr_r:.1f} dB",
            (25, card2_y + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            self.COLOR_TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

        # Sóng nhịp thở
        resp_wave = pipeline_data.get("resp_wave", [])
        self._draw_waveform(
            panel,
            resp_wave,
            x=25,
            y=card2_y + 98,
            w=panel_w - 50,
            h=42,
            color=self.COLOR_ACCENT_CYAN,
            title="Song ho hap (0.1 - 0.5 Hz)",
        )

        # ==========================================
        # CARD 3: PHỔ TẦN SỐ FFT (SPECTRAL ANALYSIS)
        # ==========================================
        card3_y = card2_y + card2_h + 12
        card3_h = 100
        if card3_y + card3_h <= frame_h - 25:
            if fft_info:
                self._draw_fft_spectrum(
                    panel,
                    freqs=fft_info.get("freqs"),
                    fft_vals=fft_info.get("fft_hr"),
                    band=(0.75, 2.5),
                    peak_f=f_heart,
                    x=15,
                    y=card3_y,
                    w=panel_w - 30,
                    h=card3_h,
                    color=self.COLOR_ACCENT_RED,
                    title="Pho FFT Nhip tim",
                )

        # Footer hướng dẫn phím tắt
        cv2.line(panel, (15, frame_h - 26), (panel_w - 15, frame_h - 26), (50, 60, 72), 1)
        cv2.putText(
            panel,
            "[Q] Thoat | [M] Mesh | [R] Reset | [S] Luu anh",
            (20, frame_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            self.COLOR_TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

        return canvas
