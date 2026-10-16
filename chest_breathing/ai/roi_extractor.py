import cv2
import numpy as np


class ROIExtractor:
    """Trích xuất vùng ngực (Chest ROI) từ các điểm mốc giải phẫu YOLO Pose."""

    def get_chest_roi(
        self,
        frame,
        left_shoulder,
        right_shoulder,
        left_hip=None,
        right_hip=None,
    ):
        """
        Tính toán vùng ngực tối ưu cho Optical Flow:
        - Tự động thích ứng khi người dùng ngồi trước webcam laptop (không nhìn thấy hông).
        - Thu hẹp 12% biên vai để tránh nhiễu do vung tay hoặc cử động vai.
        - Khóa cứng tỷ lệ lồng ngực sinh học chuẩn xác.
        """
        h, w = frame.shape[:2]

        # Kiểm tra tọa độ 2 vai hợp lệ
        if left_shoulder is None or right_shoulder is None:
            return 0, 0, 0, 0

        ls_x, ls_y = float(left_shoulder[0]), float(left_shoulder[1])
        rs_x, rs_y = float(right_shoulder[0]), float(right_shoulder[1])

        if ls_x <= 0 or rs_x <= 0 or ls_y <= 0 or rs_y <= 0:
            return 0, 0, 0, 0

        sx1 = min(ls_x, rs_x)
        sx2 = max(ls_x, rs_x)
        shoulder_w = max(30.0, sx2 - sx1)
        shoulder_y = (ls_y + rs_y) / 2.0

        # Thu lề 12% mỗi bên để tập trung vào trung tâm lồng ngực (xương ức), loại bỏ cánh tay
        margin_x = shoulder_w * 0.12
        x1 = int(max(0, sx1 + margin_x))
        x2 = int(min(w, sx2 - margin_x))

        # Điểm bắt đầu phía trên: ngay dưới xương đòn
        y1 = int(max(0, shoulder_y + shoulder_w * 0.05))

        # Điểm kết thúc phía dưới:
        has_valid_hips = False
        if left_hip is not None and right_hip is not None:
            lh_y, rh_y = float(left_hip[1]), float(right_hip[1])
            if lh_y > shoulder_y + 30.0 and rh_y > shoulder_y + 30.0:
                hip_y = (lh_y + rh_y) / 2.0
                y2 = int(shoulder_y + (hip_y - shoulder_y) * 0.42)
                has_valid_hips = True

        # Khi ngồi trước webcam (hông khuất dưới bàn): tính theo tỷ lệ bề rộng vai
        if not has_valid_hips:
            y2 = int(shoulder_y + shoulder_w * 0.45)

        y2 = int(min(h, max(y1 + 30, y2)))

        return x1, y1, x2, y2