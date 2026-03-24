@echo off
cd /d "%~dp0"
echo Installing dependencies...
echo.
pip install -r requirements.txt
echo.
if %errorlevel% neq 0 (
    echo Installation failed. Make sure Python and pip are installed and in your PATH.
    pause
    exit /b 1
)
echo All dependencies installed successfully.
echo You can now run start.bat to launch the application.
pause
