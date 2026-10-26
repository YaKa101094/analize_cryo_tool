@echo off
rem Startet Schichtdicke.py mit der lokalen .venv.
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo Keine Python-Umgebung gefunden. Bitte zuerst setup.bat ausfuehren.
    pause
    exit /b 1
)

.venv\Scripts\python.exe Schichtdicke.py
if errorlevel 1 pause
