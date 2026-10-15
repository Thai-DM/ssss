"""
test_pipeline.py
----------------
Bộ kiểm thử tự động (Unit Tests & Algorithmic Benchmarks)
để kiểm chứng tính chính xác của các thuật toán trong tài liệu rBBG.docx.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import unittest
import numpy as np
import cv2

from roi_extractor import FaceROIExtractor
from chrom_processor import CHROMProcessor
from rppg_pipeline import RPPGPipeline


class TestRPBGPipeline(unittest.TestCase):

    def setUp(self):
        self.chrom = CHROMProcessor(default_fs=30.0)

    def test_min_area_rect_and_affine_warp(self):
        """Kiểm thử tính toán 5 tham số (xc, yc, w, h, theta) và ma trận Affine M."""
        extractor = FaceROIExtractor()
        
        # Tạo tập điểm chữ nhật nghiêng
        pts = np.array([[100, 100], [200, 120], [190, 220], [90, 200]], dtype=np.float32)
        (xc, yc), (rw, rh), angle, box_points = extractor._get_oriented_box(pts)
        
        self.assertAlmostEqual(xc, 145.0, delta=2.0)
        self.assertAlmostEqual(yc, 160.0, delta=2.0)
        self.assertGreater(rw, 50.0)
        self.assertGreater(rh, 50.0)
        
        # Test Affine warp
        dummy_frame = np.ones((400, 400, 3), dtype=np.uint8) * 128
        warped, M = extractor._affine_warp_roi(dummy_frame, (xc, yc), (rw, rh), angle)
        
        self.assertEqual(warped.shape[0], int(round(rh)))
        self.assertEqual(warped.shape[1], int(round(rw)))
        extractor.close()

    def test_chrom_mathematics(self):
        """Kiểm chứng các công thức toán học của CHROM: Rn, Gn, Bn, X, Y, alpha, S_chrom."""
        N = 300
        fs = 30.0
        t = np.arange(N) / fs
        
        # Tạo tín hiệu kiểm thử
        R = 170.0 + 1.2 * np.sin(2 * np.pi * 1.2 * t)
        G = 130.0 - 2.0 * np.sin(2 * np.pi * 1.2 * t)
        B = 100.0 - 0.4 * np.sin(2 * np.pi * 1.2 * t)
        
        S_chrom, X, Y, alpha = self.chrom.extract_bvp_chrom(R, G, B)
        
        # S_chrom phải có cùng độ dài N
        self.assertEqual(len(S_chrom), N)
        self.assertGreater(alpha, 0.0)
        
        # Kiểm tra X và Y
        self.assertEqual(len(X), N)
        self.assertEqual(len(Y), N)

    def test_heart_rate_estimation_accuracy(self):
        """Kiểm tra độ chính xác đo nhịp tim (HR BPM) ở các tần số khác nhau: 60, 75, 90 BPM."""
        test_bpms = [60.0, 75.0, 90.0]
        fs = 30.0
        duration = 12.0
        t = np.arange(0, duration, 1.0 / fs)
        N = len(t)

        for bpm in test_bpms:
            f_heart = bpm / 60.0
            pulse = 2.0 * np.sin(2 * np.pi * f_heart * t)
            
            # Mô phỏng hấp thụ mao mạch da
            R = 180.0 + 0.3 * pulse
            G = 135.0 - 1.8 * pulse
            B = 110.0 - 0.2 * pulse
            
            res = self.chrom.process(R, G, B, fs=fs)
            self.assertTrue(res["success"])
            est_bpm = res["hr_bpm"]
            
            # Sai số cho phép < 1.0 BPM
            self.assertAlmostEqual(est_bpm, bpm, delta=1.0, msg=f"Kỳ vọng {bpm} BPM, đo được {est_bpm:.2f} BPM")

    def test_respiration_rate_estimation_accuracy(self):
        """Kiểm tra độ chính xác đo nhịp thở (RR RPM) ở các tần số 12, 18, 24 RPM."""
        test_rpms = [12.0, 18.0, 24.0]
        fs = 30.0
        duration = 20.0  # Cần thời gian đủ dài cho tần số thấp
        t = np.arange(0, duration, 1.0 / fs)

        for rpm in test_rpms:
            f_resp = rpm / 60.0
            # Sóng mạch tim 72 bpm
            pulse = 1.5 * np.sin(2 * np.pi * 1.2 * t)
            # Sóng hô hấp điều biến
            resp = 0.8 * np.sin(2 * np.pi * f_resp * t)
            
            R = 175.0 + 0.2 * pulse + 0.4 * resp
            G = 130.0 - 1.2 * pulse + 0.4 * resp
            B = 105.0 - 0.2 * pulse + 0.4 * resp
            
            res = self.chrom.process(R, G, B, fs=fs)
            self.assertTrue(res["success"])
            est_rpm = res["rr_rpm"]
            
            # Tần số thấp với độ phân giải thời gian thực cho phép sai số < 1.5 RPM
            self.assertAlmostEqual(est_rpm, rpm, delta=1.5, msg=f"Kỳ vọng {rpm} RPM, đo được {est_rpm:.2f} RPM")


if __name__ == "__main__":
    unittest.main()
