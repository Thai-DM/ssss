# HỆ THỐNG ĐO NHỊP TIM & NHỊP THỞ QUA VIDEO (rPPG / rBBG)

Dự án tái lập toàn bộ thuật toán và kiến trúc từ tài liệu **`rBBG.docx`**, sử dụng mô hình **MediaPipe Face Mesh (468 điểm tọa độ 3D)** kết hợp thuật toán **CHROM (Chrominance-based method by De Haan & Jeanne)** để đo lường không tiếp xúc các chỉ số sinh trắc học:
- **5.1. Nhịp tim (Heart Rate - HR)**: Đơn vị **BPM** (nhịp/phút).
- **5.2. Nhịp thở (Respiration Rate - RR)**: Đơn vị **RPM** (lần/phút).

---

## 1. Cơ sở lý thuyết & Quy trình thuật toán từ `rBBG.docx`

### Bước 1: Phát hiện khuôn mặt & Cắt vùng ROI (Region of Interest)
- Sử dụng **MediaPipe Face Mesh** trích xuất 468 điểm tọa độ 3D khuôn mặt.
- **Vùng trán (Forehead ROI)**: Lấy vùng nằm giữa 2 lông mày kéo dài lên đường viền tóc.
- **Hai bên má (Cheek ROIs)**: Lấy vùng mô thịt dưới mắt, tránh mắt, viền mũi và vùng miệng.
- **Khung bao nghiêng ôm sát (Oriented Bounding Box)**: Mạng nhận diện mặt nạ da mặt được quy đổi thành hình chữ nhật xoay với 5 tham số:
  $$\text{ROI} = (x_c, y_c, w, h, \theta)$$
  Trong đó: $(x_c, y_c)$ là tọa độ tâm, $w$ là chiều dài, $h$ là chiều rộng, và $\theta$ là góc nghiêng.

### Bước 2: Chuẩn hóa ROI (Affine Warping)
- Do khuôn mặt có thể nghiêng một góc $\theta$, hệ thống xây dựng ma trận biến đổi Affine $M$ từ góc $\theta$ và tâm $(x_c, y_c)$:
  $$M = \begin{bmatrix} \cos\theta & -\sin\theta & x_c' \\ \sin\theta & \cos\theta & y_c' \end{bmatrix}$$
- Dùng `cv2.warpAffine` xoay ROI về trục ngang và chuẩn hóa kích thước về dạng chữ nhật phẳng $w \times h$.

### Bước 3: Thu thập tín hiệu màu theo thời gian
- Tại mỗi khung hình $t$, tính giá trị cường độ sáng trung bình cho từng kênh màu $R, G, B$:
  $$R(t) = \frac{1}{|R|} \sum_{(x,y) \in R} I_R(x, y, t)$$
  $$G(t) = \frac{1}{|R|} \sum_{(x,y) \in R} I_G(x, y, t)$$
  $$B(t) = \frac{1}{|R|} \sum_{(x,y) \in R} I_B(x, y, t)$$
  Với $|R|$ là tổng số pixel trong vùng ROI, $I$ là cường độ sáng.
- Sau $N$ khung hình (cửa sổ thời gian trượt từ 8 - 10 giây), thu được 3 chuỗi tín hiệu 1D: $S_R, S_G, S_B$ có độ dài $N$.

### Bước 4: Tách tín hiệu sóng mạch (BVP) bằng Thuật toán CHROM
Thuật toán dựa trên lý thuyết sắc độ, khai thác sự khác biệt về độ hấp thụ ánh sáng của huyết sắc tố (hemoglobin) ở các bước sóng để triệt tiêu nhiễu chuyển động và biến thiên ánh sáng:
1. **Làm mịn sơ bộ**: Đưa chuỗi $R(t), G(t), B(t)$ qua bộ lọc làm mịn để loại bỏ nhiễu tần số cao của cảm biến camera $\to R_f(t), G_f(t), B_f(t)$.
2. **Chuẩn hóa giá trị trung bình động**:
   $$R_n(t) = \frac{R_f(t)}{\mu_R}, \quad G_n(t) = \frac{G_f(t)}{\mu_G}, \quad B_n(t) = \frac{B_f(t)}{\mu_B}$$
   Với $\mu_R, \mu_G, \mu_B$ là giá trị trung bình toàn chuỗi màu trong cửa sổ $N$ khung hình.
3. **Chiếu sắc độ vuông góc với phản xạ gương (Specular reflection)**:
   $$X(t) = 3 \cdot R_n(t) - 2 \cdot G_n(t)$$
   $$Y(t) = 1.5 \cdot R_n(t) + 1.5 \cdot G_n(t) - 3 \cdot B_n(t)$$
4. **Cân bằng biên độ và tổng hợp BVP**:
   $$\sigma_X = \sqrt{\frac{1}{N} \sum_{t=1}^N (X(t) - \bar{X})^2}, \quad \sigma_Y = \sqrt{\frac{1}{N} \sum_{t=1}^N (Y(t) - \bar{Y})^2}$$
   $$\alpha = \frac{\sigma_X}{\sigma_Y}$$
   $$S_{\text{CHROM}}(t) = X(t) - \alpha \cdot Y(t)$$

### Bước 5: Tính toán chỉ số sinh trắc học
- **5.1. Đo nhịp tim (Heart Rate - HR)**:
  - Cho $S_{\text{CHROM}}(t)$ qua bộ lọc Butterworth Bandpass Filter dải $[0.75\,\text{Hz} - 2.5\,\text{Hz}]$ (tương ứng 45 - 150 BPM).
  - Áp dụng biến đổi Fourier (FFT), tìm tần số $f_{\text{heart}}$ có biên độ lớn nhất:
    $$\text{HR (BPM)} = f_{\text{heart}} \times 60 \quad (\text{nhịp/phút})$$
- **5.2. Đo nhịp thở (Respiration Rate - RR)**:
  - Cho $S_{\text{CHROM}}(t)$ thô qua bộ lọc Bandpass dải tần số thấp $[0.1\,\text{Hz} - 0.5\,\text{Hz}]$ (tương ứng 6 - 30 lần/phút).
  - Áp dụng biến đổi Fourier (FFT), tìm tần số $f_{\text{resp}}$ có biên độ lớn nhất:
    $$\text{RR (RPM)} = f_{\text{resp}} \times 60 \quad (\text{lần/phút})$$

---

## 2. Cấu trúc thư mục mã nguồn

```
d:\son/
├── venv/                     # Môi trường ảo Python độc lập (Virtual Environment)
├── rBBG.docx                 # Tài liệu mô tả yêu cầu gốc
├── requirements.txt          # Danh sách thư viện phụ thuộc
├── roi_extractor.py          # Module trích xuất FaceMesh & 5 tham số ROI, Affine warping
├── chrom_processor.py        # Module thuật toán CHROM, bộ lọc Butterworth, FFT đo HR & RR
├── rppg_pipeline.py          # Điều phối luồng dữ liệu trượt (Sliding Window), FPS động, EMA
├── ui_visualizer.py          # Giao diện HUD chuyên nghiệp trên OpenCV (Oscilloscope, FFT plot)
├── main.py                   # Chương trình chính (Hỗ trợ Webcam, Video file, Simulation)
├── test_pipeline.py          # Bộ kiểm thử tự động (Unit Tests) độ chính xác thuật toán
├── plot_signals.py           # Công cụ vẽ báo cáo phân tích tín hiệu chi tiết ra file ảnh PNG
└── README.md                 # Tài liệu hướng dẫn sử dụng
```

---

## 3. Hướng dẫn cài đặt & Kích hoạt Môi trường ảo (`venv`)

Môi trường ảo `venv` đã được khởi tạo sẵn tại `d:\son\venv`.

### Kích hoạt môi trường ảo trong PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```
*(Nếu gặp thông báo về Execution Policy trong PowerShell, chạy lệnh: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### Hoặc trong Command Prompt (cmd.exe):
```cmd
venv\Scripts\activate.bat
```

---

## 4. Hướng dẫn chạy chương trình & DEMO TEST

### Cách 1: Chạy DEMO TEST Video Mẫu (Khuyên dùng thử nghiệm ngay)
Chạy video mẫu `demo_video.mp4` đã được tích hợp sẵn dao động vi mạch máu (74 BPM, 16 RPM) và nắn xoay đầu Affine với giao diện HUD thời gian thực:
```powershell
.\venv\Scripts\python demo.py
```
*(Hoặc click đúp file **`run_demo.bat`**)*

---

### Cách 2: Chạy DEMO Tương tác Bằng Bàn phím (Interactive Demo)
Bạn có thể tự tăng giảm nhịp tim và nhịp thở bằng phím mũi tên để xem thuật toán thích ứng trực tiếp trên màn hình:
```powershell
.\venv\Scripts\python demo.py --interactive
```
*(Hoặc click đúp file **`run_interactive.bat`**)*
- Bấm **MŨI TÊN LÊN / XUỐNG**: Tăng / giảm nhịp tim mục tiêu (+/- 5 BPM).
- Bấm **MŨI TÊN PHẢI / TRÁI**: Tăng / giảm nhịp thở mục tiêu (+/- 2 RPM).

---

### Cách 3: Chạy trực tiếp qua Webcam máy tính
Đo nhịp tim và nhịp thở thực tế của chính bạn qua camera:
```powershell
.\venv\Scripts\python demo.py --webcam
```
*(Hoặc click đúp file **`run_webcam.bat`**)*

> **Phím tắt trong cửa sổ hiển thị:**
> - `Q` hoặc `ESC`: Dừng và thoát.
> - `M`: Bật/tắt lưới tọa độ 3D MediaPipe Face Mesh.
> - `R`: Reset lại bộ đệm tích lũy.
> - `S`: Lưu ảnh chụp màn hình HUD (`demo_snapshot.png`).
> - `1`, `2`, `3`: Đổi chế độ ROI (1: Cả trán và 2 má | 2: Chỉ trán | 3: Chỉ 2 má).

### Cách 4: Chạy bộ kiểm thử tự động (Unit Tests)
Kiểm chứng tính đúng đắn của các công thức toán học và độ chính xác:
```powershell
.\venv\Scripts\python test_pipeline.py
```

### Cách 5: Xuất biểu đồ báo cáo phân tích tín hiệu
```powershell
.\venv\Scripts\python plot_signals.py --csv test_simulation.csv --output rppg_analysis_report.png
```
# ssss
