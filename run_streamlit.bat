@echo off
echo ===============================================
echo  MinerU2PPT Streamlit GUI Launcher
echo ===============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found, using system Python
)

REM Run the Python launcher
python run_streamlit.py

REM Keep window open on error
if errorlevel 1 (
    echo.
    echo Press any key to exit...
    pause >nul
)