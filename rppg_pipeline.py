"""
rppg_pipeline.py
----------------
Pipeline điều phối luồng xử lý rPPG thời gian thực:
- Quản lý bộ đệm cửa sổ trượt (Sliding Window Buffer) cho các tín hiệu màu R, G, B.
- Đo đạc tốc độ khung hình thực tế (Dynamic FPS estimation).
- Kích hoạt thuật toán CHROM theo chu kỳ (mỗi frame hoặc mỗi k frame).
- Lọc làm mượt (EMA) chỉ số HR (BPM) và RR (RPM) hiển thị.
- Quản lý trạng thái chất lượng tín hiệu (Ổn định, Đang hiệu chuẩn, Mất dấu khuôn mặt).
"""

from collections import deque
import time
from typing import Dict, Optional, Tuple
import numpy as np

from chrom_processor import CHROMProcessor
from roi_extractor import FaceROIExtractor


class RPPGPipeline:
    """Điều phối toàn bộ quy trình đo nhịp tim & nhịp thở không tiếp xúc qua video/camera."""

    def __init__(
        self,
        window_sec: float = 8.0,
        update_interval_frames: int = 2,
        ema_alpha: float = 0.25,
        target_fs: float = 30.0,
        roi_mode: str = "all",  # 'all', 'forehead', 'cheeks'
    ):
        self.window_sec = window_sec
        self.update_interval_frames = update_interval_frames
        self.ema_alpha = ema_alpha
        self.target_fs = target_fs
        self.roi_mode = roi_mode

        # Khởi tạo extractor và processor
        self.roi_extractor = FaceROIExtractor()
        self.chrom_processor = CHROMProcessor(default_fs=target_fs)

        # Bộ đệm dữ liệu chuỗi thời gian (hỗ trợ tốc độ khung hình cao lên tới 60+ FPS)
        self.max_buffer_len = int(window_sec * 65)  # buffer tối đa
        self.timestamps = deque(maxlen=self.max_buffer_len)
        self.r_buffer = deque(maxlen=self.max_buffer_len)
        self.g_buffer = deque(maxlen=self.max_buffer_len)
        self.b_buffer = deque(maxlen=self.max_buffer_len)

        # Bộ đệm hiển thị sóng thời gian thực (độ dài 150 mẫu ~ 5 giây)
        self.wave_buffer_len = 150
        self.bvp_wave_display = deque(maxlen=self.wave_buffer_len)
        self.resp_wave_display = deque(maxlen=self.wave_buffer_len)

        # Trạng thái hiện tại
        self.frame_count = 0
        self.last_process_result: Optional[Dict] = None
        self.current_hr: float = 0.0
        self.current_rr: float = 0.0
        self.current_snr_hr: float = 0.0
        self.current_snr_rr: float = 0.0
        self.measured_fps: float = target_fs
        self.is_stable: bool = False
        self.status_message: str = "Đang khởi động..."

    def _select_rgb_by_mode(self, roi_result: Dict) -> Tuple[float, float, float]:
        """Chọn giá trị màu R, G, B theo chế độ ROI: 'all', 'forehead', hoặc 'cheeks'."""
        rois = roi_result.get("rois", {})
        if self.roi_mode == "forehead" and "forehead" in rois:
            return rois["forehead"]["mean_rgb"]

        if self.roi_mode == "cheeks":
            total_px = 0
            wr, wg, wb = 0.0, 0.0, 0.0
            for name in ["left_cheek", "right_cheek"]:
                if name in rois:
                    px = rois[name]["pixel_count"]
                    r, g, b = rois[name]["mean_rgb"]
                    wr += r * px
                    wg += g * px
                    wb += b * px
                    total_px += px
            if total_px > 0:
                return (wr / total_px, wg / total_px, wb / total_px)

        # Mặc định: 'all' (kết hợp cả trán và 2 má)
        return roi_result.get("mean_rgb", (0.0, 0.0, 0.0))

    def process_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> Dict:
        """
        Xử lý từng khung hình video:
        1. Trích xuất ROI và tính cường độ màu R(t), G(t), B(t).
        2. Cập nhật bộ đệm trượt.
        3. Tính toán BVP, HR, RR qua thuật toán CHROM khi đủ dữ liệu.
        """
        if timestamp is None:
            timestamp = time.perf_counter()

        self.frame_count += 1
        h, w, _ = frame.shape

        # 1. Trích xuất ROI
        roi_result = self.roi_extractor.extract_rois(frame)
        face_detected = roi_result["detected"]

        if not face_detected:
            self.status_message = "Không tìm thấy khuôn mặt"
            self.is_stable = False
            return {
                "detected": False,
                "roi_result": roi_result,
                "hr_bpm": self.current_hr,
                "rr_rpm": self.current_rr,
                "fps": self.measured_fps,
                "status": self.status_message,
                "is_stable": self.is_stable,
                "buffer_progress": min(1.0, len(self.r_buffer) / max(1, int(self.measured_fps * 4.0))),
                "bvp_wave": list(self.bvp_wave_display),
                "resp_wave": list(self.resp_wave_display),
                "fft_result": None,
            }

        # 2. Lấy giá trị màu trung bình theo ROI đã chọn
        r_val, g_val, b_val = self._select_rgb_by_mode(roi_result)

        # Cập nhật buffer
        self.timestamps.append(timestamp)
        self.r_buffer.append(r_val)
        self.g_buffer.append(g_val)
        self.b_buffer.append(b_val)

        # Đo FPS thực tế dựa trên timestamps
        if len(self.timestamps) >= 10:
            time_span = self.timestamps[-1] - self.timestamps[0]
            if time_span > 0.01:
                self.measured_fps = float((len(self.timestamps) - 1) / time_span)

        # 3. Kiểm tra số lượng khung hình để tính toán CHROM
        window_frames = int(self.measured_fps * self.window_sec)
        min_frames = max(60, int(self.measured_fps * 3.5))

        # Chỉ tính toán lại sau mỗi update_interval_frames
        should_update = (
            len(self.r_buffer) >= min_frames
            and (self.frame_count % self.update_interval_frames == 0)
        )

        fft_result = None

        if should_update:
            # Lấy dữ liệu trong cửa sổ trượt gần nhất
            cur_len = len(self.r_buffer)
            use_len = min(cur_len, window_frames)
            
            r_arr = np.array(list(self.r_buffer)[-use_len:])
            g_arr = np.array(list(self.g_buffer)[-use_len:])
            b_arr = np.array(list(self.b_buffer)[-use_len:])

            res = self.chrom_processor.process(r_arr, g_arr, b_arr, fs=self.measured_fps)
            self.last_process_result = res

            if res["success"]:
                new_hr = res["hr_bpm"]
                new_rr = res["rr_rpm"]
                snr_h = res["snr_hr"]
                snr_r = res["snr_rr"]

                # Kiểm tra tính hợp lệ sinh học (HR: 45-160 BPM, RR: 6-32 RPM)
                if 45.0 <= new_hr <= 160.0:
                    if self.current_hr == 0.0:
                        self.current_hr = new_hr
                    else:
                        # Làm mượt chỉ số bằng Exponential Moving Average (EMA)
                        self.current_hr = (
                            self.ema_alpha * new_hr + (1.0 - self.ema_alpha) * self.current_hr
                        )

                if 6.0 <= new_rr <= 32.0:
                    if self.current_rr == 0.0:
                        self.current_rr = new_rr
                    else:
                        self.current_rr = (
                            self.ema_alpha * new_rr + (1.0 - self.ema_alpha) * self.current_rr
                        )

                self.current_snr_hr = snr_h
                self.current_snr_rr = snr_r

                # Cập nhật điểm sóng cuối cùng vào wave buffer
                if len(res["bvp_cardiac"]) > 0:
                    self.bvp_wave_display.append(float(res["bvp_cardiac"][-1]))
                if len(res["bvp_respiratory"]) > 0:
                    self.resp_wave_display.append(float(res["bvp_respiratory"][-1]))

                self.is_stable = (snr_h >= 3.0) and (len(self.r_buffer) >= min_frames)
                if self.is_stable:
                    self.status_message = "Đang đo ổn định"
                else:
                    self.status_message = "Đang hiệu chuẩn tín hiệu..."

                fft_result = {
                    "freqs": res["freqs"],
                    "fft_hr": res["fft_hr"],
                    "fft_rr": res["fft_rr"],
                    "f_heart": res["f_heart"],
                    "f_resp": res["f_resp"],
                    "alpha": res["alpha"],
                }
            else:
                self.status_message = res.get("error", "Đang tích lũy khung hình...")
        else:
            if len(self.r_buffer) < min_frames:
                self.status_message = f"Thu thập khung hình ({len(self.r_buffer)}/{min_frames})..."
            else:
                self.status_message = "Đang đo ổn định" if self.is_stable else "Đang phân tích..."

        buffer_progress = min(1.0, len(self.r_buffer) / float(min_frames))

        return {
            "detected": True,
            "roi_result": roi_result,
            "hr_bpm": self.current_hr,
            "rr_rpm": self.current_rr,
            "snr_hr": self.current_snr_hr,
            "snr_rr": self.current_snr_rr,
            "fps": self.measured_fps,
            "status": self.status_message,
            "is_stable": self.is_stable,
            "buffer_progress": buffer_progress,
            "bvp_wave": list(self.bvp_wave_display),
            "resp_wave": list(self.resp_wave_display),
            "fft_result": fft_result or (
                {
                    "freqs": self.last_process_result["freqs"],
                    "fft_hr": self.last_process_result["fft_hr"],
                    "fft_rr": self.last_process_result["fft_rr"],
                    "f_heart": self.last_process_result["f_heart"],
                    "f_resp": self.last_process_result["f_resp"],
                    "alpha": self.last_process_result["alpha"],
                }
                if self.last_process_result and self.last_process_result.get("success")
                else None
            ),
        }

    def reset(self):
        """Đặt lại toàn bộ bộ đệm khi đổi góc nhìn hoặc bắt đầu phiên đo mới."""
        self.timestamps.clear()
        self.r_buffer.clear()
        self.g_buffer.clear()
        self.b_buffer.clear()
        self.bvp_wave_display.clear()
        self.resp_wave_display.clear()
        self.current_hr = 0.0
        self.current_rr = 0.0
        self.is_stable = False
        self.status_message = "Đã đặt lại dữ liệu"

    def close(self):
        """Giải phóng MediaPipe."""
        self.roi_extractor.close()
