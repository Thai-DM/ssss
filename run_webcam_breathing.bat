@echo off
chcp 65001 > nul
title YOLO Chest Breathing Worker - Laptop Webcam
echo =====================================================================
echo    KHỞI CHẠY AI WORKER ĐO NHỊP THỞ CỬ ĐỘNG NGỰC (WEBCAM LAPTOP)
echo =====================================================================
echo.
cd /d "%~dp0"
venv\Scripts\python.exe chest_breathing\webcam_worker.py --api http://127.0.0.1:5000
pause
