@echo off
chcp 65001 > nul
title YOLO Chest Breathing Worker - ESP32-CAM
echo =====================================================================
echo    KHỞI CHẠY AI WORKER ĐO NHỊP THỞ CỬ ĐỘNG NGỰC (ESP32-CAM)
echo =====================================================================
echo.
cd /d "%~dp0"
venv\Scripts\python.exe chest_breathing\breathing_worker.py --api http://127.0.0.1:5000
pause
