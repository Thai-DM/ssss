"""
plot_signals.py
---------------
Vẽ biểu đồ phân tích chuyên sâu các bước xử lý tín hiệu theo tài liệu rBBG.docx:
1. Tín hiệu thô R(t), G(t), B(t).
2. Tín hiệu chuẩn hóa Rn(t), Gn(t), Bn(t).
3. Hai đường sắc độ X(t) và Y(t).
4. Tín hiệu BVP thô và sau lọc Bandpass [0.75 - 2.5 Hz].
5. Tín hiệu hô hấp sau lọc Bandpass [0.1 - 0.5 Hz].
6. Phổ tần số FFT nhịp tim & nhịp thở với các đỉnh peak.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import matplotlib.pyplot as plt
import numpy as np

from chrom_processor import CHROMProcessor


def plot_chrom_pipeline_analysis(csv_file: str, output_image: str = "rppg_analysis_report.png"):
    """Đọc dữ liệu CSV và xuất biểu đồ phân tích các bước thuật toán CHROM."""
    data = np.genfromtxt(csv_file, delimiter=",", names=True)
    t = data["time_s"]
    R = data["R"]
    G = data["G"]
    B = data["B"]

    N = len(t)
    fs = float(N / (t[-1] - t[0]))

    chrom = CHROMProcessor(default_fs=fs)
    res = chrom.process(R, G, B, fs=fs)

    if not res["success"]:
        print("LỖI:", res.get("error"))
        return

    S_chrom, X, Y, alpha = chrom.extract_bvp_chrom(R, G, B)
    bvp_cardiac = res["bvp_cardiac"]
    bvp_resp = res["bvp_respiratory"]
    hr_bpm = res["hr_bpm"]
    rr_rpm = res["rr_rpm"]
    f_heart = res["f_heart"]
    f_resp = res["f_resp"]

    # Thiết lập giao diện biểu đồ
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(4, 2, figsize=(15, 11))
    fig.suptitle(f"PHÂN TÍCH TÍN HIỆU rPPG / rBBG THEO THUẬT TOÁN CHROM\n(HR: {hr_bpm:.1f} BPM | RR: {rr_rpm:.1f} RPM | alpha: {alpha:.2f})", fontsize=14, fontweight="bold")

    # 1. Kênh màu thô
    ax = axes[0, 0]
    ax.plot(t, R, color="red", label="R(t)", alpha=0.8)
    ax.plot(t, G, color="green", label="G(t)", alpha=0.8)
    ax.plot(t, B, color="blue", label="B(t)", alpha=0.8)
    ax.set_title("1. Tín hiệu màu thô từ ROI: R(t), G(t), B(t)")
    ax.set_ylabel("Cường độ sáng I")
    ax.legend(loc="upper right")

    # 2. Chuẩn hóa trung bình động
    Rf = np.convolve(R, np.ones(5)/5, mode="same")
    Gf = np.convolve(G, np.ones(5)/5, mode="same")
    Bf = np.convolve(B, np.ones(5)/5, mode="same")
    Rn = Rf / np.mean(Rf)
    Gn = Gf / np.mean(Gf)
    Bn = Bf / np.mean(Bf)

    ax = axes[0, 1]
    ax.plot(t, Rn, color="firebrick", label="Rn(t)")
    ax.plot(t, Gn, color="forestgreen", label="Gn(t)")
    ax.plot(t, Bn, color="royalblue", label="Bn(t)")
    ax.set_title("2. Chuẩn hóa theo giá trị trung bình động (dao động quanh 1.0)")
    ax.set_ylabel("Giá trị chuẩn hóa")
    ax.legend(loc="upper right")

    # 3. Tín hiệu sắc độ X(t) và Y(t)
    ax = axes[1, 0]
    ax.plot(t, X, color="darkorange", label="X(t) = 3*Rn - 2*Gn")
    ax.plot(t, Y, color="purple", label="Y(t) = 1.5*Rn + 1.5*Gn - 3*Bn")
    ax.set_title("3. Hai đường sắc độ X(t) và Y(t)")
    ax.set_ylabel("Biên độ")
    ax.legend(loc="upper right")

    # 4. Sóng BVP thô S_CHROM = X - alpha * Y
    ax = axes[1, 1]
    ax.plot(t, S_chrom, color="black", label=f"S_CHROM (alpha={alpha:.2f})")
    ax.set_title("4. Tín hiệu BVP thô: S_CHROM(t) = X(t) - alpha * Y(t)")
    ax.set_ylabel("Biên độ")
    ax.legend(loc="upper right")

    # 5. Sóng mạch tim BVP lọc Bandpass [0.75 - 2.5 Hz]
    ax = axes[2, 0]
    ax.plot(t, bvp_cardiac, color="crimson", lw=1.5, label="BVP Bandpassed [0.75 - 2.5 Hz]")
    ax.set_title(f"5.1. Sóng nhịp tim lọc Butterworth Bandpass (HR: {hr_bpm:.1f} BPM)")
    ax.set_ylabel("Biên độ")
    ax.legend(loc="upper right")

    # 6. Sóng nhịp thở lọc Bandpass [0.1 - 0.5 Hz]
    ax = axes[2, 1]
    ax.plot(t, bvp_resp, color="teal", lw=1.5, label="Resp Bandpassed [0.1 - 0.5 Hz]")
    ax.set_title(f"5.2. Sóng nhịp thở lọc Bandpass tần số thấp (RR: {rr_rpm:.1f} RPM)")
    ax.set_ylabel("Biên độ")
    ax.legend(loc="upper right")

    # 7. Phổ FFT nhịp tim
    freqs = res["freqs"]
    fft_hr = res["fft_hr"]
    ax = axes[3, 0]
    mask_hr = (freqs >= 0.5) & (freqs <= 3.0)
    ax.plot(freqs[mask_hr], fft_hr[mask_hr], color="crimson")
    ax.axvline(f_heart, color="black", linestyle="--", label=f"f_heart = {f_heart:.2f} Hz ({hr_bpm:.1f} BPM)")
    ax.axvspan(0.75, 2.5, color="pink", alpha=0.2, label="Dải nhịp tim [0.75 - 2.5 Hz]")
    ax.set_title("Phổ tần số FFT Nhịp tim (Cardiac Spectrum)")
    ax.set_xlabel("Tần số (Hz)")
    ax.set_ylabel("Biên độ phổ")
    ax.legend(loc="upper right")

    # 8. Phổ FFT nhịp thở
    fft_rr = res["fft_rr"]
    ax = axes[3, 1]
    mask_rr = (freqs >= 0.05) & (freqs <= 1.0)
    ax.plot(freqs[mask_rr], fft_rr[mask_rr], color="teal")
    ax.axvline(f_resp, color="black", linestyle="--", label=f"f_resp = {f_resp:.2f} Hz ({rr_rpm:.1f} RPM)")
    ax.axvspan(0.1, 0.5, color="cyan", alpha=0.2, label="Dải nhịp thở [0.1 - 0.5 Hz]")
    ax.set_title("Phổ tần số FFT Nhịp thở (Respiratory Spectrum)")
    ax.set_xlabel("Tần số (Hz)")
    ax.set_ylabel("Biên độ phổ")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_image, dpi=200)
    plt.close()
    print(f"Đã lưu biểu đồ phân tích chi tiết ra: {output_image}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ phân tích tín hiệu CHROM rPPG từ file CSV.")
    parser.add_argument("--csv", type=str, default="test_simulation.csv", help="Đường dẫn file CSV dữ liệu.")
    parser.add_argument("--output", type=str, default="rppg_analysis_report.png", help="Đường dẫn file ảnh đầu ra.")
    args = parser.parse_args()

    plot_chrom_pipeline_analysis(args.csv, args.output)
