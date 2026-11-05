@echo off
cd /d "%~dp0"
call conda activate drowsiness_detection
python my_drowsiness_detection.py
pause
