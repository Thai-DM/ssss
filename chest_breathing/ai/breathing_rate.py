import numpy as np
from scipy.signal import find_peaks


class BreathingRateEstimator:
    """
    Dem peak tren tin hieu da loc de tinh BPM.

    Ban cu dung distance=50 CO DINH THEO SAMPLE (khong theo thoi gian),
    va KHONG co nguong bien do (prominence/height) -- nen bat ky gon
    nhieu nho nao cung duoc tinh ngang hang voi 1 nhip tho thuc, gay
    "nhay so" giua cac lan do.

    Ban moi: distance tinh theo THOI GIAN THUC (tu fps = len(signal)/
    duration_seconds), va them prominence -- chi tinh la peak khi noi
    bat ro rang so voi muc dao dong chung cua tin hieu.

    FIX: dung np.atleast_1d() thay vi np.asarray() truc tiep de tranh
    loi "TypeError: len() of unsized object" khi tham so signal truyen
    vao vo tinh la scalar hoac mang 0-chieu (co the xay ra neu ham loc
    phia truoc tra ve gia tri khong dung dang, dac biet o fps rat
    thap/khong on dinh nhu ESP32-CAM qua WiFi yeu).
    """

    # Nguoi thuc te tho khoang 5-40 lan/phut -> 1 nhip cach nhau toi
    # thieu 60/40 = 1.5 giay. Dung de chan dem trung nhieu peak trong
    # cung 1 chu ky tho.
    MAX_PLAUSIBLE_BPM = 40
    MIN_SECONDS_BETWEEN_BREATHS = 40.0 / MAX_PLAUSIBLE_BPM  # = 1.5s

    # He so nguong noi bat cua peak so voi do lech chuan tin hieu.
    # Tang len (vd 0.4-0.5) neu van con nhay so do nhieu; giam xuong
    # (vd 0.15-0.2) neu BPM bi dem THIEU nhip thuc.
    PROMINENCE_FACTOR = 0.3

    @staticmethod
    def estimate(signal, duration_seconds):
        # FIX: atleast_1d dam bao signal luon la mang >=1 chieu, tranh
        # crash "len() of unsized object" neu dau vao la scalar/0-d.
        signal = np.atleast_1d(np.asarray(signal, dtype=float))

        if signal.size == 0 or duration_seconds is None or duration_seconds <= 0:
            return 0.0, []

        # Bo cac gia tri NaN/Inf neu co (vd loc bandpass loi cuc bo)
        if not np.all(np.isfinite(signal)):
            signal = signal[np.isfinite(signal)]
            if signal.size == 0:
                return 0.0, []

        fps = signal.size / duration_seconds
        min_distance_samples = max(
            1,
            int(fps * BreathingRateEstimator.MIN_SECONDS_BETWEEN_BREATHS)
        )
        # find_peaks yeu cau distance < do dai tin hieu
        min_distance_samples = min(min_distance_samples, max(1, signal.size - 1))

        signal_std = float(np.std(signal))
        #signal_mean = float(np.mean(signal))

        # Nếu độ lệch chuẩn quá nhỏ (tín hiệu đi ngang hoàn toàn), coi như nín thở / không có chuyển động
        if signal_std < 1e-4:
            return 0.0, []

        # Ngưỡng nổi bật: tỷ lệ với độ lệch chuẩn
        prominence = BreathingRateEstimator.PROMINENCE_FACTOR * signal_std

        peaks, _ = find_peaks(
            signal,
            distance=min_distance_samples,
            prominence=prominence,
        )

        breaths = len(peaks)

        bpm = (
            breaths / duration_seconds
        ) * 60

        return bpm, peaks