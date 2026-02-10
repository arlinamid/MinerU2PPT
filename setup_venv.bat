@echo off
REM MinerU2PPT Virtual Environment Setup Script for Windows
REM This script creates a virtual environment and installs all required packages

echo === MinerU2PPT Setup Script ===
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH
    echo Please install Python 3.10+ and add to PATH
    pause
    exit /b 1
)

echo [INFO] Python found, checking version...
python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"
if %errorlevel% neq 0 (
    echo [WARNING] Python 3.10+ recommended for best compatibility
)

REM Remove old venv if it exists
if exist "venv" (
    echo [INFO] Removing existing virtual environment...
    rmdir /s /q venv
)

echo [INFO] Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel

echo [INFO] Checking for deprecated google-generativeai package...
pip show google-generativeai >nul 2>&1
if %errorlevel% equ 0 (
    echo [WARNING] Found deprecated google-generativeai, removing...
    pip uninstall -y google-generativeai
)

echo [INFO] Installing latest packages...
pip install --upgrade python-pptx>=0.6.23 PyMuPDF>=1.24.0 opencv-python>=4.9.0 numpy>=1.26.0 Pillow>=10.2.0 tkinterdnd2>=0.3.0 scikit-image>=0.22.0 pyinstaller>=6.3.0

echo [INFO] Installing AI packages...
pip install --upgrade openai>=1.8.0 google-genai>=0.3.0 anthropic>=0.8.0 groq>=0.4.2 httpx>=0.26.0

echo [INFO] Verifying installation...
python -c "import google.genai; print('[OK] Google GenAI')" 2>nul || echo "[ERROR] Google GenAI import failed"
python -c "import openai; print('[OK] OpenAI')" 2>nul || echo "[ERROR] OpenAI import failed"
python -c "import anthropic; print('[OK] Anthropic')" 2>nul || echo "[ERROR] Anthropic import failed"
python -c "import groq; print('[OK] Groq')" 2>nul || echo "[ERROR] Groq import failed"

echo [INFO] Generating requirements file...
pip freeze > requirements_installed.txt

echo.
echo === Setup Complete! ===
echo.
echo To use MinerU2PPT:
echo 1. Activate the environment: venv\Scripts\activate.bat
echo 2. Run the GUI: python gui.py
echo 3. Or run CLI: python main.py --help
echo.
echo To deactivate when done: deactivate
echo.
echo For package maintenance, run: python check_packages.py
echo.

pause