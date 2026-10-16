import cv2


class ROIExtractor:

    def get_chest_roi(
            self,
            frame,
            left_shoulder,
            right_shoulder,
            left_hip,
            right_hip
    ):

        x1 = int(min(left_shoulder[0], right_shoulder[0]))
        x2 = int(max(left_shoulder[0], right_shoulder[0]))

        shoulder_y = (
            left_shoulder[1] +
            right_shoulder[1]
        ) / 2

        hip_y = (
            left_hip[1] +
            right_hip[1]
        ) / 2

        y1 = int(shoulder_y)

        y2 = int(
            shoulder_y +
            (hip_y - shoulder_y) * 0.30
        )

        return x1, y1, x2, y2