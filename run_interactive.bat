@echo off
chcp 65001 > nul
echo ========================================================
echo   KHOI DONG INTERACTIVE DEMO (Dung mui ten dieu chinh)
echo ========================================================
echo Huong dan:
echo   - Bam MUI TEN LEN / XUONG de tang/giam nhip tim mo phong.
echo   - Bam MUI TEN PHAI / TRAI de tang/giam nhip tho mo phong.
echo   - Bam [Q] de thoat.
.\venv\Scripts\python demo.py --interactive
pause
