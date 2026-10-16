import os
import sys

# Đảm bảo in tiếng Việt an toàn trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import firebase_admin
from firebase_admin import credentials, db as rtdb

# Đường dẫn tới file serviceAccountKey.json
SERVICE_ACCOUNT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "serviceAccountKey.json"
)

DATABASE_URL = "https://nhiptho001-default-rtdb.asia-southeast1.firebasedatabase.app"

def init_firebase():
    """Khởi tạo Firebase Admin SDK và trả về Realtime Database reference."""
    if not firebase_admin._apps:
        if not os.path.exists(SERVICE_ACCOUNT_FILE):
            print(f"[Firebase] Chua co file {SERVICE_ACCOUNT_FILE}, su dung Local SQLite.")
            try:
                firebase_admin.initialize_app(options={'databaseURL': DATABASE_URL})
            except Exception:
                pass
        else:
            try:
                cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
                firebase_admin.initialize_app(cred, {
                    'databaseURL': DATABASE_URL
                })
            except Exception as e:
                print(f"[Firebase] Khoi tao that bai: {e}")
            
    try:
        return rtdb.reference()
    except Exception:
        return None

# db ở đây sẽ là reference tới gốc của Realtime Database
db = init_firebase()
