"""
roi_extractor.py
----------------
Phát hiện khuôn mặt & Trích xuất vùng ROI (Region of Interest)
theo quy trình chuẩn trong tài liệu rBBG.docx:
1. Sử dụng MediaPipe Face Mesh (468 điểm tọa độ 3D).
2. Trích xuất ROI:
   - Trán: Vùng nằm giữa 2 lông mày kéo dài lên viền tóc.
   - Hai bên má: Vùng dưới mắt, tránh môi và gò má.
3. Chuyển đổi mặt nạ thành khung bao nghiêng ôm sát (Oriented Bounding Box)
   với 5 tham số: tâm (xc, yc), chiều dài w, chiều rộng h, và góc nghiêng theta.
4. Chuẩn hóa ROI bằng ma trận biến đổi Affine M xoay về trục ngang và warpAffine.
5. Thu thập giá trị trung bình màu R(t), G(t), B(t) trên từng ROI.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np


class FaceROIExtractor:
    """Trích xuất và chuẩn hóa ROI khuôn mặt sử dụng MediaPipe Face Mesh."""

    # Tập hợp các chỉ số landmark trên MediaPipe 468 Face Mesh
    # 1. Vùng trán (Forehead): Giữa 2 lông mày kéo lên viền tóc
    FOREHEAD_LANDMARKS: List[int] = [
        70, 63, 105, 66, 107,   # Lông mày trái (phía trên)
        9, 8,                   # Điểm glabella (giữa 2 chân mày)
        336, 296, 334, 293, 300, # Lông mày phải (phía trên)
        284, 332, 297, 338, 10, # Đường viền tóc bên phải & đỉnh trán
        109, 67, 103, 54        # Đường viền tóc bên trái
    ]

    # 2. Vùng má trái (Left cheek - má bên trái của ảnh, tức má phải khuôn mặt)
    LEFT_CHEEK_LANDMARKS: List[int] = [
        116, 117, 118, 101, 205, 50, 123, 147, 213, 192, 138, 135, 136
    ]

    # 3. Vùng má phải (Right cheek - má bên phải của ảnh, tức má trái khuôn mặt)
    RIGHT_CHEEK_LANDMARKS: List[int] = [
        345, 346, 347, 330, 425, 280, 352, 376, 433, 416, 367, 364, 365
    ]

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def _get_oriented_box(self, pts: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], float, np.ndarray]:
        """
        Tính khung bao nghiêng với 5 tham số:
        (xc, yc), (w, h), theta và 4 đỉnh box_points.
        Chuẩn hóa trục ngang (w >= h) để góc theta phản ánh đúng độ nghiêng của đầu.
        """
        rect = cv2.minAreaRect(pts.astype(np.float32))
        (xc, yc), (w, h), angle = rect

        # Đảm bảo chiều rộng w luôn là trục dài nằm ngang (w >= h)
        if w < h:
            w, h = h, w
            angle += 90.0

        # Chuẩn hóa góc theta về dải [-90, 90] độ
        while angle > 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0

        norm_rect = ((xc, yc), (w, h), angle)
        box_points = cv2.boxPoints(norm_rect).astype(np.int32)
        return (xc, yc), (w, h), angle, box_points

    def _affine_warp_roi(
        self,
        frame: np.ndarray,
        center: Tuple[float, float],
        size: Tuple[float, float],
        angle: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Chuẩn hóa ROI:
        Dùng ma trận biến đổi Affine M để xoay ROI góc theta về trục ngang,
        sau đó warp hình ảnh để thu ảnh ROI dạng chuẩn w x h.
        """
        xc, yc = center
        w, h = size
        w_int = max(1, int(round(w)))
        h_int = max(1, int(round(h)))

        # Ma trận xoay quanh tâm (xc, yc) góc angle
        M = cv2.getRotationMatrix2D((xc, yc), angle, 1.0)
        # Dịch chuyển tâm (xc, yc) về chính giữa ảnh đích chuẩn hóa (w/2, h/2)
        M[0, 2] += (w_int / 2.0 - xc)
        M[1, 2] += (h_int / 2.0 - yc)

        # Warp ROI chuẩn hóa kích thước w_int x h_int
        warped_roi = cv2.warpAffine(
            frame,
            M,
            (w_int, h_int),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return warped_roi, M

    def extract_rois(
        self,
        frame: np.ndarray,
    ) -> Dict:
        """
        Phát hiện khuôn mặt và trích xuất các vùng ROI trán và 2 má.

        Returns:
            Dict chứa:
            - 'detected': bool
            - 'mean_rgb': tuple (R, G, B) - giá trị trung bình trên các vùng ROI hợp nhất
            - 'rois': dict chi tiết từng ROI ('forehead', 'left_cheek', 'right_cheek')
                      gồm center, size, angle, box_points, warped_roi, mean_rgb, pixel_count
            - 'landmarks_pixel': np.ndarray (468, 2)
        """
        h_frame, w_frame, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return {
                "detected": False,
                "mean_rgb": (0.0, 0.0, 0.0),
                "rois": {},
                "landmarks_pixel": None,
            }

        face_landmarks = results.multi_face_landmarks[0]
        # Chuyển đổi tọa độ chuẩn hóa [0, 1] sang tọa độ pixel thực
        landmarks_pixel = np.array([
            (p.x * w_frame, p.y * h_frame) for p in face_landmarks.landmark
        ], dtype=np.float32)

        roi_definitions = {
            "forehead": self.FOREHEAD_LANDMARKS,
            "left_cheek": self.LEFT_CHEEK_LANDMARKS,
            "right_cheek": self.RIGHT_CHEEK_LANDMARKS,
        }

        extracted_rois = {}
        total_pixels = 0
        weighted_r = 0.0
        weighted_g = 0.0
        weighted_b = 0.0

        for roi_name, landmark_indices in roi_definitions.items():
            pts = landmarks_pixel[landmark_indices]
            (xc, yc), (rw, rh), angle, box_points = self._get_oriented_box(pts)

            if rw < 2 or rh < 2:
                continue

            warped_roi, M = self._affine_warp_roi(frame, (xc, yc), (rw, rh), angle)

            # Thu thập cường độ sáng trung bình trên ROI theo công thức:
            # C(t) = 1/|R| * sum_{(x,y) in R} I_C(x, y, t)
            # frame trong OpenCV là BGR
            mean_b, mean_g, mean_r = cv2.mean(warped_roi)[:3]
            num_pixels = warped_roi.shape[0] * warped_roi.shape[1]

            extracted_rois[roi_name] = {
                "center": (xc, yc),
                "size": (rw, rh),
                "angle": angle,
                "box_points": box_points,
                "warped_roi": warped_roi,
                "affine_matrix": M,
                "mean_rgb": (mean_r, mean_g, mean_b),
                "pixel_count": num_pixels,
            }

            total_pixels += num_pixels
            weighted_r += mean_r * num_pixels
            weighted_g += mean_g * num_pixels
            weighted_b += mean_b * num_pixels

        if total_pixels > 0:
            combined_mean_rgb = (
                weighted_r / total_pixels,
                weighted_g / total_pixels,
                weighted_b / total_pixels,
            )
        else:
            combined_mean_rgb = (0.0, 0.0, 0.0)

        return {
            "detected": True,
            "mean_rgb": combined_mean_rgb,
            "rois": extracted_rois,
            "landmarks_pixel": landmarks_pixel,
        }

    def close(self):
        """Giải phóng tài nguyên MediaPipe."""
        if hasattr(self, "face_mesh") and self.face_mesh:
            self.face_mesh.close()
