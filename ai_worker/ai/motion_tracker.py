import cv2
import numpy as np


class MotionTracker:

    def __init__(self, blur_kernel=(5, 5)):
        """
        blur_kernel: kich thuoc Gaussian blur truoc khi tinh optical flow.
        - (5, 5): phu hop cho webcam laptop (anh raw it artifact).
        - (7, 7): phu hop cho ESP32-CAM (JPEG artifact nhieu hon, can
          lam mo manh hon de Farneback khong "bam" vao block artifact).
        - KHONG nen tang qua (9,9) vi se lam mo canh, Farneback se kho
          bam duoc chuyen dong tinh te cua long nguc.
        """
        self.blur_kernel = blur_kernel
        self.prev_gray = None
        self.signal = []
        self.last_flow = None
        

    def process(self, roi):

        if roi is None:
            return None

        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (200, 200))


        # Gaussian blur: giam nhieu cam bien / JPEG artifact truoc khi
        # tinh optical flow. Kernel duoc truyen vao luc khoi tao de co
        # the chinh khac nhau cho webcam vs ESP32-CAM.
        gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            0.5,   # pyr_scale
            3,     # levels
            21,    # winsize: cua so tim kiem tuong quan
            3,     # iterations
            5,     # poly_n
            1.2,   # poly_sigma
            0
        )

        vertical_motion = flow[..., 1]

        # Dung MEDIAN thay vi MEAN: ben hon voi cac vector flow loi cuc bo
        # (vai/ao it texture lam Farneback uoc luong sai o mot so pixel).
        motion_value = float(np.median(vertical_motion))

        self.signal.append(motion_value)
        self.prev_gray = gray
        self.last_flow = flow

        return motion_value

    def reset(self):
        self.prev_gray = None
        self.signal.clear()
        self.last_flow = None

    def draw_flow_particles(self, frame, roi_bbox, step=16):
        """
        Vẽ các phần tử hạt (particles) trực quan hóa dense optical flow
        lên khung hình, màu sắc thay đổi dựa trên chuyển động dọc (lên/xuống).
        """
        if self.last_flow is None:
            return

        x1, y1, x2, y2 = roi_bbox
        # last_flow được tính trên kích thước (200, 200) của ROI
        flow = self.last_flow
        
        # Lấy kích thước thực tế của ROI trên frame
        h_roi, w_roi = y2 - y1, x2 - x1
        
        if h_roi <= 0 or w_roi <= 0:
            return

        # Resize flow về kích thước ROI để scale cho đúng tọa độ
        flow_resized = cv2.resize(flow, (w_roi, h_roi))
        
        # Scale giá trị flow theo tỷ lệ kích thước khung hình
        flow_resized[..., 0] *= w_roi / 200.0
        flow_resized[..., 1] *= h_roi / 200.0

        h, w = flow_resized.shape[:2]
        
        # Tính toán lưới (grid) bằng numpy để tối ưu tốc độ realtime
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        
        fx, fy = flow_resized[y, x].T
        
        # Phóng đại tín hiệu (chỉ áp dụng cho hiển thị hạt) để dễ nhìn bằng mắt thường
        visual_multiplier = 15.0
        fx_visual = fx * visual_multiplier
        fy_visual = fy * visual_multiplier
        
        # Tạo tọa độ bắt đầu và kết thúc của vector flow
        lines = np.vstack([x, y, x + fx_visual, y + fy_visual]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        
        # Dịch chuyển tọa độ theo x1, y1 của ROI trên frame chính
        offset = np.array([x1, y1])
        lines += offset
        
        # Vẽ từng hạt và vector
        for (x_start, y_start), (x_end, y_end) in lines:
            dy = y_end - y_start
            
            # Ngực chuyển động khi thở rất nhẹ, vector dy giữa 2 frame liên tiếp thường rất nhỏ.
            if dy < 0:
                color = (0, 255, 0) # Xanh lá (Lên)
            elif dy > 0:
                color = (0, 0, 255) # Đỏ (Xuống)
            else:
                color = (0, 255, 255) # Vàng (Đứng yên)
                
            # Vẽ đường chỉ hướng chuyển động
            cv2.line(frame, (x_start, y_start), (x_end, y_end), color, 1)
            # Vẽ phần tử hạt
            cv2.circle(frame, (x_start, y_start), 2, color, -1)
