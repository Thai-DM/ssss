@echo off
chcp 65001 > nul
echo ========================================================
echo   KHOI DONG LIVE WEBCAM rPPG / rBBG (Thuat toan CHROM)
echo ========================================================
echo Dang ket noi Webcam 0...
echo Nhan phim [Q] de thoat, [M] de bat/tat Face Mesh.
.\venv\Scripts\python demo.py --webcam
pause
