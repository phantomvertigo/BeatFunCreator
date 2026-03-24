@echo off
cd /d "%~dp0"
echo This will reset all BeatFunCreator settings to their defaults.
echo (Your project files will NOT be affected.)
echo.
set /p confirm="Are you sure? (Y/N): "
if /i "%confirm%" neq "Y" (
    echo Cancelled.
    pause
    exit /b 0
)
if exist ".bfc_config.json" (
    del ".bfc_config.json"
    echo Configuration reset to defaults.
) else (
    echo No configuration file found. Already using defaults.
)
pause
