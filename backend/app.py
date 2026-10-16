"""
backend/app.py
--------------
Ứng dụng Flask trung tâm hợp nhất toàn bộ hệ thống Contactless Vital Signs:
1. rPPG / rBBG: Nhịp tim (HR) & Nhịp thở (RR) qua khuôn mặt (MediaPipe Face Mesh + CHROM).
2. Chest Motion: Nhịp thở qua cử động ngực (YOLO Pose + Optical Flow) từ Webcam / ESP32-CAM.
3. Dashboard thời gian thực: Video stream MJPEG, biểu đồ sóng BVP, FFT spectrum, phân tích lâm sàng.
4. Quản lý dữ liệu kép: Cơ sở dữ liệu SQLite cục bộ (chạy độc lập 100%) + Hỗ trợ Firebase Realtime DB.
5. Cảnh báo thông minh: Còi báo động (Audio/Beep), Email cảnh báo (SMTP), phân loại nguy cơ.
"""

import os
import sys

# Đảm bảo in tiếng Việt an toàn trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import time
import io
import csv
import json
import queue
import threading
from datetime import datetime
from typing import Dict, Any

from flask import Flask, render_template, jsonify, request, Response, send_file

# Đường dẫn thư mục
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BACKEND_DIR)
for p in (BASE_DIR, BACKEND_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

TEMPLATE_DIR = os.path.join(BASE_DIR, "dashboard", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "dashboard", "static")

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR,
)

# Import các module nội bộ
import local_db
import alarm_manager
from video_stream import rppg_streamer, chest_streamer

# Thử khởi tạo Firebase (nếu có cấu hình, nếu không dùng SQLite)
firebase_available = False
db_firebase = None
try:
    from firebase_db import db as rtdb
    if rtdb is not None:
        db_firebase = rtdb
        firebase_available = True
        print("[App] Kết nối Firebase Realtime Database thành công!")
except Exception as e:
    print(f"[App] Chạy chế độ CSDL cục bộ SQLite (Firebase: {e})")

# Khởi động sẵn luồng rPPG Streamer và Chest Streamer nền
rppg_streamer.start()
chest_streamer.start()

# SSE clients queue
_sse_clients = []
_sse_lock = threading.Lock()


def broadcast_sse(event_type: str, data: Dict[str, Any]):
    """Gửi sự kiện SSE tới tất cả trình duyệt client đang kết nối."""
    payload = json.dumps(data)
    msg = f"event: {event_type}\ndata: {payload}\n\n"
    with _sse_lock:
        for q in list(_sse_clients):
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass


# ==============================================================================
# 1. TRANG DASHBOARD CHÍNH
# ==============================================================================
@app.route("/")
def dashboard():
    """Giao diện Dashboard tổng thể quản lý toàn bộ nội dung của hai dự án."""
    return render_template("index.html")


# ==============================================================================
# 2. VIDEO STREAMING ENDPOINTS (MJPEG)
# ==============================================================================
@app.route("/api/video_feed/rppg")
def video_feed_rppg():
    """Luồng video MJPEG khuôn mặt rPPG kèm HUD trực quan hóa thời gian thực."""
    return Response(
        rppg_streamer.generate_mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/video_feed/chest")
def video_feed_chest():
    """Luồng video MJPEG nhận diện ngực & cử động YOLO Pose thời gian thực."""
    return Response(
        chest_streamer.generate_mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/control/chest/source", methods=["POST"])
def control_chest_source():
    """Đổi nguồn camera cho tab ngực: 'webcam' hoặc 'esp32'."""
    data = request.get_json(silent=True) or {}
    source = data.get("source", "webcam")
    esp32_url = data.get("esp32_url")
    chest_streamer.set_source(source, esp32_url)
    return jsonify({"success": True, "source": source}), 200


@app.route("/api/control/chest/skeleton", methods=["POST"])
def control_chest_skeleton():
    """Bật/tắt hiển thị khung xương Pose trên video ngực."""
    data = request.get_json(silent=True) or {}
    chest_streamer.toggle_skeleton(data.get("show_skeleton"))
    return jsonify({"success": True, "show_skeleton": chest_streamer.show_skeleton}), 200


@app.route("/api/control/chest/roi", methods=["POST"])
def control_chest_roi():
    """Bật/tắt hiển thị hộp Chest ROI trên video ngực."""
    data = request.get_json(silent=True) or {}
    chest_streamer.toggle_roi(data.get("show_roi"))
    return jsonify({"success": True, "show_roi": chest_streamer.show_roi}), 200


# ==============================================================================
# 3. API DỮ LIỆU SINH HIỆU (LATEST, HISTORY, STATS)
# ==============================================================================
@app.route("/api/latest", methods=["GET"])
def get_latest():
    """Lấy số liệu sinh hiệu mới nhất từ cả rPPG và Chest Tracker."""
    live_rppg = rppg_streamer.get_latest_data()
    live_chest = chest_streamer.get_latest_data()
    db_rec = local_db.get_latest_record() or {}

    hr = live_rppg.get("heart_rate") or db_rec.get("heart_rate", 72.0)
    rr = live_rppg.get("respiration_rate") or db_rec.get("respiration_rate", 16.0)
    chest = live_chest.get("chest_bpm") or db_rec.get("chest_bpm", rr)
    bpm = db_rec.get("bpm", chest)

    severity, messages = alarm_manager.evaluate_vitals(heart_rate=hr, respiration_rate=rr, chest_bpm=chest)

    response_data = {
        "heart_rate": round(float(hr), 1),
        "respiration_rate": round(float(rr), 1),
        "chest_bpm": round(float(chest), 1),
        "chest_motion": live_chest.get("motion", 0.0),
        "bpm": round(float(bpm), 1),
        "status": severity,
        "status_messages": messages,
        "is_stable": live_rppg.get("is_stable", True),
        "fps": live_rppg.get("fps", 30.0),
        "source": live_rppg.get("mode", db_rec.get("source", "camera_ai")),
        "recorded_at": datetime.now().isoformat(),
        "bvp_wave": live_rppg.get("bvp_wave", []),
        "resp_wave": live_rppg.get("resp_wave", []),
    }
    return jsonify(response_data), 200


@app.route("/api/history", methods=["GET"])
def get_history():
    """Lấy danh sách lịch sử đo lường với bộ lọc và phân trang."""
    limit = request.args.get("limit", default=100, type=int)
    metric_type = request.args.get("type", default="all", type=str)

    records = local_db.get_history_records(limit=limit, metric_type=metric_type)
    return jsonify(records), 200


@app.route("/api/statistics", methods=["GET"])
def get_statistics():
    """Lấy các chỉ số thống kê tổng hợp: Trung bình, Thấp nhất, Cao nhất."""
    stats = local_db.get_statistics()
    return jsonify(stats), 200


# ==============================================================================
# 4. API GHI NHẬN DỮ LIỆU (INGESTION - BACKWARD COMPATIBLE)
# ==============================================================================
@app.route("/api/breathing", methods=["POST"])
def ingest_breathing():
    """
    Endpoint tương thích 100% với worker cũ (breathing_worker.py & webcam_worker.py).
    Nhận BPM từ cử động ngực / YOLO pose và lưu trữ, phát chuông cảnh báo.
    """
    data = request.get_json(silent=True) or {}
    if "bpm" not in data:
        return jsonify({"error": "Thiếu trường 'bpm'"}), 400

    try:
        bpm_val = float(data["bpm"])
    except (ValueError, TypeError):
        return jsonify({"error": "'bpm' phải là số"}), 400

    source = data.get("source", "chest_yolo")
    note = data.get("note", "")
    image_path = data.get("image")

    # Đánh giá cảnh báo
    severity, messages = alarm_manager.evaluate_vitals(chest_bpm=bpm_val)

    # Lưu vào SQLite
    record = local_db.insert_record(
        chest_bpm=bpm_val,
        bpm=bpm_val,
        source=source,
        status=severity,
        note=note,
    )

    # Lưu Firebase (nếu có kết nối)
    if firebase_available and db_firebase is not None:
        try:
            db_firebase.child("breathing_records").push({
                "bpm": bpm_val,
                "chest_bpm": bpm_val,
                "source": source,
                "status": severity,
                "note": note,
                "created_at": record["created_at"],
            })
        except Exception as e:
            print(f"[Firebase] Ghi thất bại: {e}")

    # Xử lý âm thanh và log cảnh báo
    alarm_manager.process(bpm=bpm_val, image_path=image_path)

    # Broadcast tới các dashboard qua SSE
    broadcast_sse("update", record)

    return jsonify(record), 201


@app.route("/api/vitals", methods=["POST"])
def ingest_vitals():
    """Endpoint mở rộng để nhận đầy đủ gói sinh hiệu (Heart Rate, Respiration Rate, Chest BPM)."""
    data = request.get_json(silent=True) or {}

    hr = data.get("heart_rate")
    rr = data.get("respiration_rate")
    chest_bpm = data.get("chest_bpm")
    source = data.get("source", "rppg_client")
    note = data.get("note", "")

    severity, _ = alarm_manager.evaluate_vitals(heart_rate=hr, respiration_rate=rr, chest_bpm=chest_bpm)

    record = local_db.insert_record(
        heart_rate=hr,
        respiration_rate=rr,
        chest_bpm=chest_bpm,
        source=source,
        status=severity,
        note=note
    )

    broadcast_sse("update", record)
    return jsonify(record), 201


# ==============================================================================
# 5. API ĐIỀU KHIỂN & CẤU HÌNH HỆ THỐNG
# ==============================================================================
@app.route("/api/control/rppg/mode", methods=["POST"])
def control_rppg_mode():
    """Chuyển đổi nguồn phát rPPG: 'simulation', 'demo_video', 'webcam'."""
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "simulation")
    rppg_streamer.set_mode(mode)
    return jsonify({"success": True, "mode": mode}), 200


@app.route("/api/control/rppg/params", methods=["POST"])
def control_rppg_params():
    """Điều chỉnh nhịp tim và nhịp thở mô phỏng tức thời từ thanh trượt UI."""
    data = request.get_json(silent=True) or {}
    hr = data.get("heart_rate", 74.0)
    rr = data.get("respiration_rate", 16.0)
    rppg_streamer.set_sim_targets(hr, rr)
    return jsonify({"success": True, "sim_hr": hr, "sim_rr": rr}), 200


@app.route("/api/control/rppg/mesh", methods=["POST"])
def control_rppg_mesh():
    """Bật/tắt lưới tọa độ 3D MediaPipe Face Mesh."""
    data = request.get_json(silent=True) or {}
    show = data.get("show_mesh")
    rppg_streamer.toggle_mesh(show)
    return jsonify({"success": True, "show_mesh": rppg_streamer.show_mesh}), 200


@app.route("/api/control/rppg/roi", methods=["POST"])
def control_rppg_roi():
    """Đổi chế độ ROI: 'all', 'forehead', 'cheeks'."""
    data = request.get_json(silent=True) or {}
    roi_mode = data.get("roi_mode", "all")
    rppg_streamer.set_roi_mode(roi_mode)
    return jsonify({"success": True, "roi_mode": roi_mode}), 200


@app.route("/api/settings", methods=["GET", "POST"])
def handle_settings():
    """Lấy hoặc cập nhật cấu hình ngưỡng cảnh báo & hệ thống."""
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        alarm_manager.update_thresholds(
            hr_low=data.get("hr_low"),
            hr_high=data.get("hr_high"),
            rr_low=data.get("rr_low"),
            rr_high=data.get("rr_high"),
            sound=data.get("sound_alert"),
            email=data.get("email_alert")
        )
        for k, v in data.items():
            local_db.update_setting(k, str(v))
        return jsonify({"success": True, "settings": alarm_manager.get_thresholds()}), 200

    return jsonify(alarm_manager.get_thresholds()), 200


@app.route("/api/export/csv", methods=["GET"])
def export_csv():
    """Xuất toàn bộ lịch sử đo lường thành file CSV để báo cáo y tế."""
    records = local_db.get_history_records(limit=10000, metric_type="all")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Thời gian", "Nhịp tim (BPM)", "Nhịp thở rPPG (RPM)", "Nhịp thở ngực (BPM)", "Nguồn", "Trạng thái", "Ghi chú"])

    for r in records:
        writer.writerow([
            r.get("id"),
            r.get("recorded_at") or r.get("created_at"),
            r.get("heart_rate") or "",
            r.get("respiration_rate") or "",
            r.get("chest_bpm") or r.get("bpm") or "",
            r.get("source"),
            r.get("status"),
            r.get("note")
        ])

    output.seek(0)
    filename = f"vitals_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8-sig")),
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename
    )


@app.route("/api/history/clear", methods=["POST", "DELETE"])
def clear_history_endpoint():
    """Xóa sạch lịch sử bản ghi dữ liệu."""
    deleted_count = local_db.clear_history()
    broadcast_sse("history_cleared", {"deleted": deleted_count})
    return jsonify({"success": True, "deleted_count": deleted_count}), 200


@app.route("/api/system/status", methods=["GET"])
def system_status():
    """Thông tin trạng thái hoạt động của các thành phần trong hệ thống."""
    return jsonify({
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "database": "Firebase & SQLite" if firebase_available else "Local SQLite (Auto-sync)",
        "rppg_worker": "Active",
        "rppg_mode": rppg_streamer.mode,
        "chest_worker_ready": True,
        "alarm_manager": "Active",
        "sse_clients_count": len(_sse_clients),
    }), 200


# ==============================================================================
# 6. SERVER-SENT EVENTS (SSE) REALTIME STREAM
# ==============================================================================
@app.route("/api/stream")
def sse_stream():
    """SSE endpoint cung cấp luồng đẩy dữ liệu realtime liên tục cho Dashboard."""
    client_q = queue.Queue(maxsize=100)
    with _sse_lock:
        _sse_clients.append(client_q)

    def event_generator():
        try:
            yield "event: connected\ndata: {}\n\n"
            last_pulse_time = time.time()

            while True:
                try:
                    msg = client_q.get(timeout=1.0)
                    yield msg
                except queue.Empty:
                    # Gửi nhịp cập nhật realtime mỗi giây kèm dữ liệu sóng
                    now = time.time()
                    if now - last_pulse_time >= 1.0:
                        last_pulse_time = now
                        live_rppg = rppg_streamer.get_latest_data()
                        live_chest = chest_streamer.get_latest_data()
                        payload = json.dumps({
                            "type": "live_pulse",
                            "heart_rate": live_rppg.get("heart_rate", 72.0),
                            "respiration_rate": live_rppg.get("respiration_rate", 16.0),
                            "chest_bpm": live_chest.get("chest_bpm", 16.0),
                            "chest_motion": live_chest.get("motion", 0.0),
                            "fps": live_rppg.get("fps", 30.0),
                            "status": live_rppg.get("status", "Bình thường"),
                            "is_stable": live_rppg.get("is_stable", True),
                            "timestamp": datetime.now().isoformat(),
                            "bvp_wave": live_rppg.get("bvp_wave", []),
                            "resp_wave": live_rppg.get("resp_wave", []),
                        })
                        yield f"event: live_vitals\ndata: {payload}\n\n"
        finally:
            with _sse_lock:
                if client_q in _sse_clients:
                    _sse_clients.remove(client_q)

    return Response(
        event_generator(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


if __name__ == "__main__":
    print("=" * 70)
    print("   KHỞI ĐỘNG HỆ THỐNG GIÁM SÁT SINH HIỆU TỔNG HỢP (RPPG + CHEST POSE)")
    print("   Bảng điều khiển (Dashboard): http://localhost:5000")
    print("=" * 70)
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False,
    )