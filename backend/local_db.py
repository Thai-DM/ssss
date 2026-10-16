"""
backend/local_db.py
-------------------
Cơ sở dữ liệu SQLite cục bộ (Local SQLite Database):
- Hoạt động 100% độc lập, không cần cài đặt MySQL hay cấu hình Firebase serviceAccountKey.
- Tự động tạo bảng khi khởi động (auto-migration).
- Lưu trữ bản ghi sinh hiệu (Heart Rate, Respiration Rate rPPG, Chest Motion BPM).
- Cung cấp các hàm truy vấn lịch sử, thống kê (avg, min, max), xuất CSV và xóa bản ghi.
"""

import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Any

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vital_signs.db")


def get_connection():
    """Tạo kết nối tới file SQLite cục bộ."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Khởi tạo bảng vitals_records và settings nếu chưa tồn tại."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vitals_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            heart_rate REAL,
            respiration_rate REAL,
            chest_bpm REAL,
            bpm REAL, -- Trường tương thích ngược với Nhiptho_project
            source TEXT DEFAULT 'camera_ai',
            status TEXT DEFAULT 'normal', -- normal, warning, critical
            note TEXT DEFAULT '',
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    # Gán các cài đặt mặc định nếu chưa có
    default_settings = {
        "hr_low": "50",
        "hr_high": "100",
        "rr_low": "10",
        "rr_high": "24",
        "sound_alert": "true",
        "email_alert": "false",
        "esp32_url": "http://192.168.1.50/stream",
        "storage_mode": "local",
    }
    for k, v in default_settings.items():
        cursor.execute("INSERT OR IGNORE INTO system_settings (key, value) VALUES (?, ?)", (k, v))

    # Nếu bảng trống, thêm một số dữ liệu mẫu ban đầu để dashboard hiển thị trực quan ngay lập tức
    cursor.execute("SELECT COUNT(*) FROM vitals_records")
    count = cursor.fetchone()[0]
    if count == 0:
        base_time = datetime.now()
        import random
        for i in range(25):
            t_offset = (25 - i) * 60
            past_time = datetime.fromtimestamp(base_time.timestamp() - t_offset).isoformat()
            hr = round(random.uniform(68.0, 78.0), 1)
            rr = round(random.uniform(14.0, 18.0), 1)
            chest_bpm = round(rr + random.uniform(-0.8, 0.8), 1)
            cursor.execute("""
                INSERT INTO vitals_records (heart_rate, respiration_rate, chest_bpm, bpm, source, status, note, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (hr, rr, chest_bpm, chest_bpm, "initial_sample", "normal", "Khởi tạo hệ thống", past_time))

    conn.commit()
    conn.close()


def insert_record(
    heart_rate: Optional[float] = None,
    respiration_rate: Optional[float] = None,
    chest_bpm: Optional[float] = None,
    bpm: Optional[float] = None,
    source: str = "camera_ai",
    status: str = "normal",
    note: str = "",
    created_at: Optional[str] = None
) -> Dict[str, Any]:
    """Thêm một bản ghi mới vào cơ sở dữ liệu."""
    if created_at is None:
        created_at = datetime.now().isoformat()

    # Đồng bộ trường tương thích ngược bpm
    if bpm is None:
        bpm = chest_bpm if chest_bpm is not None else respiration_rate

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO vitals_records (heart_rate, respiration_rate, chest_bpm, bpm, source, status, note, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (heart_rate, respiration_rate, chest_bpm, bpm, source, status, note, created_at))
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return {
        "id": new_id,
        "heart_rate": heart_rate,
        "respiration_rate": respiration_rate,
        "chest_bpm": chest_bpm,
        "bpm": bpm,
        "source": source,
        "status": status,
        "note": note,
        "recorded_at": created_at,
        "created_at": created_at
    }


def get_latest_record() -> Optional[Dict[str, Any]]:
    """Lấy bản ghi sinh hiệu mới nhất."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM vitals_records ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    res = dict(row)
    res["recorded_at"] = res.get("created_at")
    return res


def get_history_records(limit: int = 100, metric_type: str = "all") -> List[Dict[str, Any]]:
    """Lấy danh sách lịch sử theo giới hạn và bộ lọc."""
    conn = get_connection()
    cursor = conn.cursor()

    if metric_type == "rppg":
        query = "SELECT * FROM vitals_records WHERE heart_rate IS NOT NULL OR respiration_rate IS NOT NULL ORDER BY id DESC LIMIT ?"
        cursor.execute(query, (limit,))
    elif metric_type == "chest":
        query = "SELECT * FROM vitals_records WHERE chest_bpm IS NOT NULL OR bpm IS NOT NULL ORDER BY id DESC LIMIT ?"
        cursor.execute(query, (limit,))
    else:
        query = "SELECT * FROM vitals_records ORDER BY id DESC LIMIT ?"
        cursor.execute(query, (limit,))

    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in rows:
        d = dict(r)
        d["recorded_at"] = d.get("created_at")
        results.append(d)
    return results


def get_statistics() -> Dict[str, Any]:
    """Tính toán thống kê: Trung bình, Min, Max của Heart Rate và Respiration/Chest BPM."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(*) as total_count,
            AVG(heart_rate) as avg_hr,
            MIN(heart_rate) as min_hr,
            MAX(heart_rate) as max_hr,
            AVG(COALESCE(chest_bpm, bpm, respiration_rate)) as avg_rr,
            MIN(COALESCE(chest_bpm, bpm, respiration_rate)) as min_rr,
            MAX(COALESCE(chest_bpm, bpm, respiration_rate)) as max_rr
        FROM vitals_records
    """)
    row = cursor.fetchone()
    conn.close()

    if not row or row["total_count"] == 0:
        return {
            "total_count": 0,
            "avg_hr": 0.0, "min_hr": 0.0, "max_hr": 0.0,
            "avg_rr": 0.0, "min_rr": 0.0, "max_rr": 0.0,
        }

    return {
        "total_count": row["total_count"],
        "avg_hr": round(row["avg_hr"] or 0.0, 1),
        "min_hr": round(row["min_hr"] or 0.0, 1),
        "max_hr": round(row["max_hr"] or 0.0, 1),
        "avg_rr": round(row["avg_rr"] or 0.0, 1),
        "min_rr": round(row["min_rr"] or 0.0, 1),
        "max_rr": round(row["max_rr"] or 0.0, 1),
    }


def clear_history() -> int:
    """Xóa toàn bộ lịch sử bản ghi và trả về số bản ghi đã xóa."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM vitals_records")
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted


def get_all_settings() -> Dict[str, str]:
    """Lấy danh sách cài đặt hệ thống."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM system_settings")
    rows = cursor.fetchall()
    conn.close()
    return {row["key"]: row["value"] for row in rows}


def update_setting(key: str, value: str):
    """Cập nhật hoặc thêm một cài đặt."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO system_settings (key, value) VALUES (?, ?)", (key, str(value)))
    conn.commit()
    conn.close()


# Khởi tạo DB ngay khi module được import
init_db()
