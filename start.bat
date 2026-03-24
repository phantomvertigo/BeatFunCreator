@echo off
cd /d "%~dp0"

:: Check that Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in your PATH.
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Check required packages
python -c "import librosa, numpy, scipy, sounddevice, matplotlib" 2>nul
if %errorlevel% neq 0 (
    echo Missing dependencies detected.
    echo Please run install.bat first to install the required packages.
    pause
    exit /b 1
)

start "" pythonw beatfuncreator.py %*
