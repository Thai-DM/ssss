"""
backend/alarm_manager.py
------------------------
Bộ quản lý cảnh báo sinh hiệu thông minh (Vital Signs Alarm Manager):
- Giám sát đồng thời Nhịp tim (HR - Heart Rate) và Nhịp thở (RR - Respiration Rate / Chest BPM).
- Tự động phân loại mức độ nghiêm trọng: 'normal' (bình thường), 'warning' (cảnh báo nhẹ), 'critical' (nguy kịch).
- Phát âm thanh cảnh báo trên server (winsound) và gửi tín hiệu âm thanh tới trình duyệt dashboard.
- Gửi email cảnh báo qua SMTP Gmail kèm ảnh chụp vi phạm (nếu cấu hình).
- Duy trì bộ đệm danh sách sự kiện cảnh báo thời gian thực (In-memory alert log).
"""

from datetime import datetime
import os
import threading
from typing import Dict, List, Optional, Tuple

try:
    import winsound
except ImportError:
    winsound = None

# Ngưỡng mặc định lâm sàng
DEFAULT_THRESHOLDS = {
    "hr_low": 50.0,    # Nhịp tim chậm (< 50 BPM)
    "hr_high": 100.0,  # Nhịp tim nhanh (> 100 BPM)
    "rr_low": 10.0,    # Nhịp thở chậm (< 10 RPM)
    "rr_high": 24.0,   # Nhịp thở nhanh (> 24 RPM)
}

# Cấu hình hiện tại (có thể thay đổi động từ dashboard)
current_thresholds = DEFAULT_THRESHOLDS.copy()
sound_alert_enabled = True
email_alert_enabled = False

# Hàng đợi lưu trữ 50 cảnh báo gần nhất
_alert_history: List[Dict] = []
_alert_lock = threading.Lock()


def update_thresholds(hr_low=None, hr_high=None, rr_low=None, rr_high=None, sound=None, email=None):
    """Cập nhật các ngưỡng từ giao diện web."""
    global sound_alert_enabled, email_alert_enabled
    if hr_low is not None: current_thresholds["hr_low"] = float(hr_low)
    if hr_high is not None: current_thresholds["hr_high"] = float(hr_high)
    if rr_low is not None: current_thresholds["rr_low"] = float(rr_low)
    if rr_high is not None: current_thresholds["rr_high"] = float(rr_high)
    if sound is not None: sound_alert_enabled = bool(sound)
    if email is not None: email_alert_enabled = bool(email)


def get_thresholds():
    """Lấy các ngưỡng và cài đặt cảnh báo hiện tại."""
    return {
        **current_thresholds,
        "sound_alert": sound_alert_enabled,
        "email_alert": email_alert_enabled,
    }


def evaluate_vitals(
    heart_rate: Optional[float] = None,
    respiration_rate: Optional[float] = None,
    chest_bpm: Optional[float] = None,
) -> Tuple[str, List[str]]:
    """
    Đánh giá trạng thái lâm sàng của các chỉ số sinh hiệu:
    Trả về: (severity: 'normal'|'warning'|'critical', messages: list of str)
    """
    severity = "normal"
    messages = []

    # 1. Đánh giá nhịp tim
    if heart_rate is not None and heart_rate > 0:
        if heart_rate < current_thresholds["hr_low"] - 5.0 or heart_rate > current_thresholds["hr_high"] + 25.0:
            severity = "critical"
            msg = f"Nguy kịch: Nhịp tim bất thường nghiêm trọng ({heart_rate:.1f} BPM)"
            messages.append(msg)
        elif heart_rate < current_thresholds["hr_low"]:
            if severity != "critical": severity = "warning"
            messages.append(f"Cảnh báo: Nhịp tim chậm ({heart_rate:.1f} BPM < {current_thresholds['hr_low']:.0f})")
        elif heart_rate > current_thresholds["hr_high"]:
            if severity != "critical": severity = "warning"
            messages.append(f"Cảnh báo: Nhịp tim nhanh ({heart_rate:.1f} BPM > {current_thresholds['hr_high']:.0f})")

    # 2. Đánh giá nhịp thở (ưu tiên chest_bpm nếu có, hoặc respiration_rate)
    rr_val = chest_bpm if chest_bpm is not None and chest_bpm > 0 else respiration_rate
    if rr_val is not None and rr_val > 0:
        if rr_val < current_thresholds["rr_low"] - 3.0 or rr_val > current_thresholds["rr_high"] + 8.0:
            severity = "critical"
            messages.append(f"Nguy kịch: Nhịp thở cực kỳ bất thường ({rr_val:.1f} RPM)")
        elif rr_val < current_thresholds["rr_low"]:
            if severity != "critical": severity = "warning"
            messages.append(f"Cảnh báo: Nhịp thở quá chậm ({rr_val:.1f} RPM < {current_thresholds['rr_low']:.0f})")
        elif rr_val > current_thresholds["rr_high"]:
            if severity != "critical": severity = "warning"
            messages.append(f"Cảnh báo: Nhịp thở quá nhanh ({rr_val:.1f} RPM > {current_thresholds['rr_high']:.0f})")

    if not messages:
        messages.append("Chỉ số sinh hiệu bình thường, ổn định.")

    return severity, messages


def _play_alarm_sound(freq=2500, duration_ms=1000):
    """Phát âm thanh cảnh báo ngầm trên hệ thống."""
    if winsound is not None:
        try:
            winsound.Beep(freq, duration_ms)
        except Exception:
            pass


def process(bpm: Optional[float] = None, image_path: Optional[str] = None, heart_rate: Optional[float] = None) -> Dict:
    """
    Hàm xử lý cảnh báo (tương thích 100% với cả mã cũ Nhiptho_project và luồng mới).
    """
    chest_val = bpm
    severity, messages = evaluate_vitals(heart_rate=heart_rate, chest_bpm=chest_val)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    alert_event = {
        "timestamp": now_str,
        "severity": severity,
        "messages": messages,
        "heart_rate": heart_rate,
        "bpm": bpm,
        "image_path": image_path,
    }

    with _alert_lock:
        _alert_history.insert(0, alert_event)
        if len(_alert_history) > 50:
            _alert_history.pop()

    # Log ra terminal
    print("=" * 60)
    print(f"[Alarm] Thoi gian      : {now_str}")
    if heart_rate is not None:
        print(f"[Alarm] Nhip tim (HR)  : {heart_rate:.1f} BPM")
    if bpm is not None:
        print(f"[Alarm] Nhip tho (BPM) : {bpm:.1f}")
    print(f"[Alarm] Trang thai     : {severity.upper()} -> {'; '.join(messages)}")
    print("=" * 60)

    # Phát chuông nếu là warning hoặc critical
    if severity in ("warning", "critical") and sound_alert_enabled:
        freq = 3000 if severity == "critical" else 2000
        dur = 1500 if severity == "critical" else 800
        threading.Thread(target=_play_alarm_sound, args=(freq, dur), daemon=True).start()

    # Gửi email cảnh báo (nếu bật)
    if severity in ("warning", "critical") and email_alert_enabled:
        def _send_email_async():
            try:
                from email_sender import send_breathing_alert
                send_breathing_alert(bpm or heart_rate, image_path)
            except Exception as e:
                print(f"[Alarm] Gui email that bai: {e}")
        threading.Thread(target=_send_email_async, daemon=True).start()

    return alert_event


def get_recent_alerts(limit=20) -> List[Dict]:
    """Lấy danh sách cảnh báo gần đây."""
    with _alert_lock:
        return list(_alert_history[:limit])