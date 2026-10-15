"""
chrom_processor.py
------------------
Cài đặt thuật toán CHROM (Chrominance-based method by De Haan & Jeanne, 2013)
và tính toán các chỉ số sinh trắc học:
- 5.1. Nhịp tim HR (Heart Rate, đơn vị BPM)
- 5.2. Nhịp thở RR (Respiration Rate, đơn vị RPM / lần trên phút)

Các bước theo đúng tài liệu rBBG.docx:
1. Thu thập chuỗi 1D màu R(t), G(t), B(t) độ dài N.
2. Làm mịn sơ bộ qua bộ lọc làm mịn để loại bỏ nhiễu tần số cao của cảm biến camera: Rf, Gf, Bf.
3. Chia từng kênh màu cho giá trị trung bình động của chính nó:
   Rn(t) = Rf(t) / mu_R
   Gn(t) = Gf(t) / mu_G
   Bn(t) = Bf(t) / mu_B
4. Chiếu sắc độ:
   X(t) = 3 * Rn(t) - 2 * Gn(t)
   Y(t) = 1.5 * Rn(t) + 1.5 * Gn(t) - 3 * Bn(t)
5. Tính độ lệch chuẩn sigma_X và sigma_Y, hệ số alpha = sigma_X / sigma_Y:
   S_CHROM(t) = X(t) - alpha * Y(t)
6. 5.1 Đo nhịp tim:
   Butterworth Bandpass Filter dải [0.75Hz - 2.5Hz] (45 - 150 BPM).
   FFT tìm tần số f_heart có biên độ lớn nhất -> HR (BPM) = f_heart * 60.
7. 5.2 Đo nhịp thở:
   Bandpass Filter dải tần thấp [0.1Hz - 0.5Hz] (6 - 30 lần/phút).
   FFT tìm tần số f_resp có biên độ lớn nhất -> RR (RPM) = f_resp * 60.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import scipy.ndimage
import scipy.signal


class CHROMProcessor:
    """Xử lý tín hiệu màu RGB bằng thuật toán CHROM để trích xuất HR và RR."""

    def __init__(
        self,
        default_fs: float = 30.0,
        smoothing_size: int = 5,
        hr_band: Tuple[float, float] = (0.75, 2.5),  # 45 - 150 BPM
        rr_band: Tuple[float, float] = (0.1, 0.5),   # 6 - 30 RPM
        fft_n_points: int = 4096,
    ):
        self.default_fs = default_fs
        self.smoothing_size = smoothing_size
        self.hr_band = hr_band
        self.rr_band = rr_band
        self.fft_n_points = fft_n_points

    def butter_bandpass_filter(
        self,
        data: np.ndarray,
        lowcut: float,
        highcut: float,
        fs: float,
        order: int = 3,
    ) -> np.ndarray:
        """Bộ lọc Butterworth Bandpass Filter hai chiều zero-phase (filtfilt)."""
        nyq = 0.5 * fs
        low = max(0.01, lowcut / nyq)
        high = min(0.99, highcut / nyq)
        if low >= high:
            raise ValueError(f"Tần số cắt không hợp lệ: lowcut={lowcut}, highcut={highcut}, nyq={nyq}")

        b, a = scipy.signal.butter(order, [low, high], btype="bandpass")
        # Kiểm tra độ dài dữ liệu tối thiểu cho filtfilt
        padlen = 3 * max(len(a), len(b))
        if len(data) <= padlen:
            # Nếu tín hiệu quá ngắn, dùng lfilter
            return scipy.signal.lfilter(b, a, data)
        return scipy.signal.filtfilt(b, a, data)

    def extract_bvp_chrom(
        self,
        R: np.ndarray,
        G: np.ndarray,
        B: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Thực hiện thuật toán CHROM từ 3 chuỗi tín hiệu màu R, G, B:
        Returns:
            S_chrom: tín hiệu BVP thô
            X: chuỗi sắc độ X
            Y: chuỗi sắc độ Y
            alpha: hệ số cân bằng biên độ
        """
        N = len(R)
        if N < self.smoothing_size:
            raise ValueError(f"Độ dài tín hiệu N={N} quá ngắn so với smoothing_size={self.smoothing_size}")

        # Bước 1: Làm mịn sơ bộ loại bỏ nhiễu tần số cao của cảm biến camera
        Rf = scipy.ndimage.uniform_filter1d(R.astype(np.float64), size=self.smoothing_size)
        Gf = scipy.ndimage.uniform_filter1d(G.astype(np.float64), size=self.smoothing_size)
        Bf = scipy.ndimage.uniform_filter1d(B.astype(np.float64), size=self.smoothing_size)

        # Bước 2: Chuẩn hóa theo giá trị trung bình động (mu_R, mu_G, mu_B)
        mu_R = np.mean(Rf) + 1e-8
        mu_G = np.mean(Gf) + 1e-8
        mu_B = np.mean(Bf) + 1e-8

        Rn = Rf / mu_R
        Gn = Gf / mu_G
        Bn = Bf / mu_B

        # Bước 3: Chiếu tín hiệu sắc độ CHROM
        # X(t) = 3 * Rn(t) - 2 * Gn(t)
        # Y(t) = 1.5 * Rn(t) + 1.5 * Gn(t) - 3 * Bn(t)
        X = 3.0 * Rn - 2.0 * Gn
        Y = 1.5 * Rn + 1.5 * Gn - 3.0 * Bn

        # Bước 4: Tính độ lệch chuẩn sigma_X và sigma_Y
        # X_bar = 1/N * sum(X)
        # sigma_X = sqrt( 1/N * sum((X - X_bar)^2) )
        sigma_x = float(np.std(X))
        sigma_y = float(np.std(Y))

        # alpha là hệ số cân bằng biên độ: alpha = sigma_X / sigma_Y
        alpha = sigma_x / (sigma_y + 1e-8)

        # Tín hiệu BVP thô: S_CHROM(t) = X(t) - alpha * Y(t)
        S_chrom = X - alpha * Y

        return S_chrom, X, Y, alpha

    def compute_spectrum_peak(
        self,
        signal_data: np.ndarray,
        fs: float,
        band: Tuple[float, float],
    ) -> Tuple[float, np.ndarray, np.ndarray, float]:
        """
        Áp dụng FFT với cửa sổ Hanning và zero-padding để tìm đỉnh tần số lớn nhất.

        Returns:
            peak_freq: Tần số đỉnh (Hz)
            freqs: Mảng tần số
            psd: Mảng mật độ phổ công suất / biên độ FFT
            snr: Tỷ số tín hiệu trên nhiễu ước lượng (dB)
        """
        N = len(signal_data)
        # Nhân với cửa sổ Hanning để giảm rò rỉ phổ (spectral leakage)
        windowed = (signal_data - np.mean(signal_data)) * np.hanning(N)

        nfft = max(self.fft_n_points, N * 4)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
        fft_vals = np.abs(np.fft.rfft(windowed, n=nfft))

        # Mặt nạ dải tần mong muốn
        mask = (freqs >= band[0]) & (freqs <= band[1])
        if not np.any(mask):
            return 0.0, freqs, fft_vals, 0.0

        sub_freqs = freqs[mask]
        sub_fft = fft_vals[mask]

        max_idx = np.argmax(sub_fft)
        peak_freq = float(sub_freqs[max_idx])
        peak_amp = float(sub_fft[max_idx])

        # Ước lượng SNR: tỉ lệ năng lượng đỉnh so với trung bình nền nhiễu
        noise_mean = float(np.mean(sub_fft) + 1e-8)
        snr_db = float(20.0 * np.log10((peak_amp + 1e-8) / noise_mean))

        return peak_freq, freqs, fft_vals, snr_db

    def process(
        self,
        R: np.ndarray,
        G: np.ndarray,
        B: np.ndarray,
        fs: Optional[float] = None,
    ) -> Dict:
        """
        Quy trình xử lý hoàn chỉnh từ 3 chuỗi màu R, G, B thu được từ ROI:
        - Tính BVP thô bằng CHROM.
        - 5.1 Đo nhịp tim HR (BPM).
        - 5.2 Đo nhịp thở RR (RPM).
        """
        if fs is None:
            fs = self.default_fs

        N = len(R)
        min_required_frames = max(45, int(fs * 2.0))
        if N < min_required_frames:
            return {
                "success": False,
                "error": f"Cần ít nhất {min_required_frames} khung hình (hiện có {N})",
                "hr_bpm": 0.0,
                "rr_rpm": 0.0,
            }

        # 1. Trích xuất BVP thô từ thuật toán CHROM
        S_chrom, X, Y, alpha = self.extract_bvp_chrom(R, G, B)

        # 2. 5.1 Đo nhịp tim HR
        # Lọc Butterworth Bandpass [0.75Hz - 2.5Hz] (45 - 150 BPM)
        try:
            bvp_cardiac = self.butter_bandpass_filter(
                S_chrom,
                lowcut=self.hr_band[0],
                highcut=self.hr_band[1],
                fs=fs,
                order=3,
            )
        except Exception:
            bvp_cardiac = S_chrom - np.mean(S_chrom)

        f_heart, freqs_hr, fft_hr, snr_hr = self.compute_spectrum_peak(
            bvp_cardiac, fs, self.hr_band
        )
        # Công thức: HR(BPM) = f_heart * 60 (nhịp/phút)
        hr_bpm = f_heart * 60.0

        # 3. 5.2 Đo nhịp thở RR
        # Cho S_chrom thô qua bộ lọc Bandpass dải tần số thấp [0.1Hz - 0.5Hz] (6 - 30 nhịp/phút)
        try:
            bvp_respiratory = self.butter_bandpass_filter(
                S_chrom,
                lowcut=self.rr_band[0],
                highcut=self.rr_band[1],
                fs=fs,
                order=2,
            )
        except Exception:
            bvp_respiratory = S_chrom - np.mean(S_chrom)

        f_resp, freqs_rr, fft_rr, snr_rr = self.compute_spectrum_peak(
            bvp_respiratory, fs, self.rr_band
        )
        # Công thức: RR(RPM) = f_resp * 60 (lần/phút)
        rr_rpm = f_resp * 60.0

        return {
            "success": True,
            "fs": fs,
            "N": N,
            "alpha": alpha,
            # Chỉ số sinh trắc học
            "f_heart": f_heart,
            "hr_bpm": hr_bpm,
            "snr_hr": snr_hr,
            "f_resp": f_resp,
            "rr_rpm": rr_rpm,
            "snr_rr": snr_rr,
            # Tín hiệu miền thời gian
            "bvp_raw": S_chrom,
            "bvp_cardiac": bvp_cardiac,
            "bvp_respiratory": bvp_respiratory,
            # Phổ tần số
            "freqs": freqs_hr,
            "fft_hr": fft_hr,
            "fft_rr": fft_rr,
        }
