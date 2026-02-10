#!/bin/bash
# MinerU2PPTX Executable Builder for Linux/macOS
# This script builds a standalone executable from gui.py

echo "========================================"
echo "MinerU2PPTX Executable Builder"
echo "========================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    echo "Please install Python and try again"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "Using Python: $($PYTHON_CMD --version)"

# Check if gui.py exists
if [ ! -f "gui.py" ]; then
    echo "Error: gui.py not found!"
    echo "Please run this script from the MinerU2PPT root directory"
    exit 1
fi

# Check if PyInstaller is installed
echo "Checking PyInstaller installation..."
if ! $PYTHON_CMD -c "import PyInstaller" 2>/dev/null; then
    echo "PyInstaller not found. Installing..."
    $PYTHON_CMD -m pip install pyinstaller>=6.3.0
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install PyInstaller"
        exit 1
    fi
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist *.spec

# Method 1: Try using the Python build script
echo
echo "Building executable using build script..."
$PYTHON_CMD build_exe.py
if [ $? -eq 0 ]; then
    echo "Build completed successfully!"
    success=true
else
    echo "Build script failed, trying direct PyInstaller..."
    success=false
fi

# Method 2: Direct PyInstaller command (if Method 1 failed)
if [ "$success" = false ]; then
    echo "Building executable with PyInstaller directly..."
    $PYTHON_CMD -m PyInstaller --windowed --onefile --name MinerU2PPTX \
        --add-data "translations:translations" \
        --hidden-import tkinterdnd2 \
        --hidden-import pptx \
        --hidden-import PIL \
        --hidden-import cv2 \
        --hidden-import fitz \
        --hidden-import numpy \
        --hidden-import skimage \
        --hidden-import translations \
        --hidden-import converter.generator \
        --hidden-import converter.ai_services \
        --hidden-import converter.config \
        --hidden-import converter.cache_manager \
        gui.py

    if [ $? -ne 0 ]; then
        echo "Build failed!"
        echo "Please check the error messages above"
        exit 1
    fi
fi

echo
echo "========================================"
echo "Build completed successfully!"
echo "========================================"
echo

# Determine executable name based on platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    EXE_NAME="MinerU2PPTX"  # macOS
else
    EXE_NAME="MinerU2PPTX"  # Linux
fi

echo "Executable location: dist/$EXE_NAME"
echo

if [ -f "dist/$EXE_NAME" ]; then
    # Make executable
    chmod +x "dist/$EXE_NAME"
    
    # Show file size
    size=$(du -h "dist/$EXE_NAME" | cut -f1)
    echo "Executable size: $size"
    echo
    
    echo "The executable includes all dependencies and can be"
    echo "distributed as a standalone application."
    echo
    echo "To test the executable:"
    echo "  cd dist"
    echo "  ./$EXE_NAME"
else
    echo "Warning: Executable not found at expected location"
fi

echo
echo "Build process completed."