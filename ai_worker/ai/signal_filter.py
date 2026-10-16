import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter


class SignalFilter:
    """
    Loc tin hieu motion ve dung dai tan nhip tho nguoi thuc te:
    ~6-30 lan/phut tuong duong 0.1-0.5 Hz.

    Ban cu dung Savitzky-Golay voi window_length=21 SAMPLE CO DINH.
    O ~20-30 FPS, 21 sample chi la ~0.7-1 giay -- ngan hon ca 1/4 chu
    ky tho thuc (3-5 giay) -- nen gan nhu khong loc duoc nhieu tan so
    cao, va cung khong loai duoc troi nen tan so thap. Ban moi nay
    dung bandpass Butterworth, tu tinh sample rate thuc te (fps) tu
    chinh du lieu thay vi gia dinh co dinh.

    FIX (quan trong): mot so phien ban scipy/numpy, khi filtfilt()
    chay voi input rat ngan sat nguong padlen, co the tra ve mang
    0-chieu (0-d array) hoac gia tri khong dung dang mang 1 chieu ma
    ham goi phia sau (BreathingRateEstimator.estimate) khong luong
    truoc -- gay loi "TypeError: len() of unsized object". Ham nay
    now LUON validate ket qua bang np.atleast_1d() va kiem tra
    shape/length khop voi input truoc khi tra ve, neu khong hop le se
    coi nhu that bai va roi ve nhanh fallback an toan hon.
    """

    LOW_HZ = 0.1   # ~6 lan/phut
    HIGH_HZ = 0.5  # ~30 lan/phut

    @staticmethod
    def smooth(signal, timestamps=None, fps=None, duration_seconds=None):
        """
        signal: tin hieu motion thoi gian.
        timestamps: (tuy chon nhung KHUYEN NGHI dung) mang thoi diem
            thuc te (giay, vd time.time()) cua tung sample. Neu co,
            se RESAMPLE tin hieu ve luoi thoi gian DEU truoc khi loc
            bandpass -- vi fps thuc te thuong khong deu (do YOLO chay
            xen ke, mang WiFi cua ESP32-CAM giat cuc), va bo loc tan
            so (bandpass) chi dung khi sample cach deu nhau.
        fps: so sample/giay (uu tien dung truc tiep neu co va khong
            co timestamps).
        duration_seconds: neu khong co fps/timestamps, se tu tinh
            fps = len(signal) / duration_seconds (kem chinh xac hon
            vi gia dinh sample deu nhau).

        Neu khong du du lieu de loc bandpass on dinh, fallback ve
        Savitzky-Golay (nhu ban cu) hoac tra nguyen tin hieu goc.
        Luon tra ve mang numpy 1 chieu co CUNG DO DAI voi tin hieu
        dau vao (sau khi resample neu co).
        """
        signal = np.atleast_1d(np.asarray(signal, dtype=float))

        if signal.size == 0:
            return signal

        # --- Loại bỏ đột biến bất thường (outlier / transient jump) ---
        if signal.size > 10:
            q25, q75 = np.percentile(signal, [25, 75])
            iqr = q75 - q25
            if iqr > 1e-6:
                low_limit = q25 - 3.5 * iqr
                high_limit = q75 + 3.5 * iqr
                signal = np.clip(signal, low_limit, high_limit)

        # --- Resample ve luoi thoi gian deu neu co timestamps ---
        if timestamps is not None:
            timestamps = np.atleast_1d(np.asarray(timestamps, dtype=float))
            if timestamps.size == signal.size and timestamps.size >= 2:
                duration = float(timestamps[-1] - timestamps[0])
                if duration > 0:
                    target_fps = max(5.0, signal.size / duration)
                    n_uniform = max(2, int(duration * target_fps))
                    uniform_t = np.linspace(timestamps[0], timestamps[-1], n_uniform)
                    signal = np.interp(uniform_t, timestamps, signal)
                    fps = target_fps

        if fps is None and duration_seconds:
            fps = signal.size / duration_seconds

        if fps and fps > 0 and signal.size > 30:
            filtered = SignalFilter._bandpass(signal, fps)
            if filtered is not None:
                filtered = np.atleast_1d(filtered)
                # FIX: chi chap nhan ket qua neu la mang 1 chieu hop le
                # co do dai bang tin hieu dau vao. Neu khong, coi nhu
                # loc that bai va roi xuong fallback ben duoi thay vi
                # tra ve gia tri sai dang (0-d array / do dai sai).
                if filtered.ndim == 1 and filtered.size == signal.size:
                    return filtered

        # --- Fallback: khong biet fps, khong du sample, hoac bandpass loi ---
        if signal.size < 21:
            return signal

        smoothed = savgol_filter(signal, window_length=21, polyorder=3)
        return np.atleast_1d(smoothed)

    @staticmethod
    def _bandpass(signal, fps):
        nyquist = fps / 2.0
        low = SignalFilter.LOW_HZ / nyquist
        high = SignalFilter.HIGH_HZ / nyquist

        if high >= 1.0:
            high = 0.99
        if low <= 0:
            low = 1e-4
        if low >= high:
            return None  # fps qua thap so voi dai tan can loc

        try:
            b, a = butter(N=3, Wn=[low, high], btype="band")
            padlen = 3 * max(len(a), len(b))
            if signal.size <= padlen:
                return None  # qua it sample de filtfilt chay on dinh
            result = filtfilt(b, a, signal)
            result = np.atleast_1d(result)
            if result.ndim != 1 or result.size != signal.size:
                return None
            return result
        except Exception:
            return None