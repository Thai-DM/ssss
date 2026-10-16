import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime

from config import *

def send_breathing_alert(bpm, image_path):

    msg = MIMEMultipart()

    msg["Subject"] = "🚨 CẢNH BÁO NHỊP THỞ BẤT THƯỜNG"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    html = f"""
    <html>
    <body style="font-family:Arial">

    <h2 style="color:red">
    🚨 AI Breathing Monitor
    </h2>

    <p>Hệ thống phát hiện nhịp thở bất thường.</p>

    <table border="1" cellpadding="8">

    <tr>
        <td><b>Nhịp thở</b></td>
        <td>{bpm:.1f} BPM</td>
    </tr>

    <tr>
        <td><b>Thời gian</b></td>
        <td>{datetime.now()}</td>
    </tr>

    </table>

    <br>

    <p>Ảnh tại thời điểm phát hiện:</p>

    <img src="cid:image1" width="500">

    <br><br>

    Đây là email tự động.

    </body>
    </html>
    """

    msg.attach(MIMEText(html, "html"))

    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            img = MIMEImage(f.read())

        img.add_header("Content-ID", "<image1>")
        img.add_header(
            "Content-Disposition",
            "inline",
            filename=os.path.basename(image_path),
        )

        msg.attach(img)

    smtp = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)

    smtp.starttls()

    smtp.login(SENDER_EMAIL, APP_PASSWORD)

    start = datetime.now()

    print(f"[Email] Đang gửi Gmail lúc: {start:%Y-%m-%d %H:%M:%S}")
    smtp.send_message(msg)
    finish = datetime.now()

    print(f"[Email] Gửi thành công lúc: {finish:%Y-%m-%d %H:%M:%S}")
    print(f"[Email] Thời gian gửi: {(finish - start).total_seconds():.2f} giây")

    smtp.quit()