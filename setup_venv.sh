#!/bin/bash
# MinerU2PPT Virtual Environment Setup Script for Linux/macOS
# This script creates a virtual environment and installs all required packages

echo "=== MinerU2PPT Setup Script ==="
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found"
    echo "Please install Python 3.10+ first"
    exit 1
fi

echo "[INFO] Python found, checking version..."
python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"
if [ $? -ne 0 ]; then
    echo "[WARNING] Python 3.10+ recommended for best compatibility"
fi

# Remove old venv if it exists
if [ -d "venv" ]; then
    echo "[INFO] Removing existing virtual environment..."
    rm -rf venv
fi

echo "[INFO] Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to create virtual environment"
    exit 1
fi

echo "[INFO] Activating virtual environment..."
source venv/bin/activate

echo "[INFO] Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "[INFO] Checking for deprecated google-generativeai package..."
pip show google-generativeai > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "[WARNING] Found deprecated google-generativeai, removing..."
    pip uninstall -y google-generativeai
fi

echo "[INFO] Installing latest packages..."
pip install --upgrade \
    "python-pptx>=0.6.23" \
    "PyMuPDF>=1.24.0" \
    "opencv-python>=4.9.0" \
    "numpy>=1.26.0" \
    "Pillow>=10.2.0" \
    "tkinterdnd2>=0.3.0" \
    "scikit-image>=0.22.0" \
    "pyinstaller>=6.3.0"

echo "[INFO] Installing AI packages..."
pip install --upgrade \
    "openai>=1.8.0" \
    "google-genai>=0.3.0" \
    "anthropic>=0.8.0" \
    "groq>=0.4.2" \
    "httpx>=0.26.0"

echo "[INFO] Verifying installation..."
python -c "import google.genai; print('[OK] Google GenAI')" 2>/dev/null || echo "[ERROR] Google GenAI import failed"
python -c "import openai; print('[OK] OpenAI')" 2>/dev/null || echo "[ERROR] OpenAI import failed"
python -c "import anthropic; print('[OK] Anthropic')" 2>/dev/null || echo "[ERROR] Anthropic import failed"
python -c "import groq; print('[OK] Groq')" 2>/dev/null || echo "[ERROR] Groq import failed"

echo "[INFO] Generating requirements file..."
pip freeze > requirements_installed.txt

echo
echo "=== Setup Complete! ==="
echo
echo "To use MinerU2PPT:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Run the GUI: python gui.py"
echo "3. Or run CLI: python main.py --help"
echo
echo "To deactivate when done: deactivate"
echo
echo "For package maintenance, run: python check_packages.py"
echo