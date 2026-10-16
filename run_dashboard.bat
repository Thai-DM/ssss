@echo off
chcp 65001 > nul
title VitalsHub - Contactless Health Monitoring Dashboard
echo =====================================================================
echo    KHỞI ĐỘNG HỆ THỐNG VITALSHUB DASHBOARD (rPPG + CHEST POSE)
echo =====================================================================
echo.
echo [1/2] Đang kiểm tra môi trường ảo Python...
if not exist "venv\Scripts\python.exe" (
    echo [LỖI] Không tìm thấy venv\Scripts\python.exe!
    pause
    exit /b 1
)

echo [2/2] Đang khởi động Backend Flask & Máy chủ Dashboard...
start "" http://localhost:5000
venv\Scripts\python.exe backend\app.py
pause
