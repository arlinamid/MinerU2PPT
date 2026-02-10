# Building MinerU2PPTX Executable

This document provides instructions for creating a standalone executable from the MinerU2PPTX GUI application.

## Quick Start

### **Windows (Recommended)**
1. **Double-click** `build.bat` for automatic build
2. **Or run manually**: `python build_exe.py`
3. **Or use PyInstaller directly**: See manual instructions below

### **Linux/macOS**
1. **Run**: `chmod +x build.sh && ./build.sh`
2. **Or run manually**: `python build_exe.py` 
3. **Or use PyInstaller directly**: See manual instructions below

## Prerequisites

### **Required Python Packages**
All packages should be installed via `requirements.txt`:
```bash
pip install -r requirements.txt
```

Key packages for building:
- `pyinstaller>=6.3.0` - Executable builder
- `tkinterdnd2` - Drag & drop support
- `python-pptx` - PowerPoint generation
- `PyMuPDF` (fitz) - PDF handling
- `opencv-python` - Image processing
- `Pillow` - Image operations
- `numpy` - Numerical operations
- `scikit-image` - Advanced image processing

### **Project Structure Required**
```
MinerU2PPT/
├── gui.py                    # Main GUI script
├── converter/
│   ├── generator.py          # Core conversion logic
│   ├── ai_services.py        # AI text correction
│   ├── config.py             # Configuration
│   └── cache_manager.py      # Cache management
├── translations/
│   ├── en.json               # English translations
│   ├── hu.json               # Hungarian translations
│   ├── zh.json               # Chinese translations
│   └── translator.py         # Translation system
└── requirements.txt          # Dependencies
```

## Build Methods

### **Method 1: Automated Build Script (Recommended)**

**Windows:**
```cmd
build.bat
```

**Linux/macOS:**
```bash
./build.sh
```

**Python (All Platforms):**
```bash
python build_exe.py
```

### **Method 2: PyInstaller Direct Command**

**Basic build:**
```bash
pyinstaller --windowed --onefile --name MinerU2PPTX gui.py
```

**With translations and hidden imports:**
```bash
pyinstaller --windowed --onefile --name MinerU2PPTX \
    --add-data "translations;translations" \
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
```

**Note:** On Linux/macOS, use `:` instead of `;` in `--add-data`:
```bash
--add-data "translations:translations"
```

### **Method 3: Custom Spec File**

The build script creates `MinerU2PPTX.spec` with optimized settings:
```bash
pyinstaller MinerU2PPTX.spec
```

## Build Output

### **Successful Build**
- **Executable location**: `dist/MinerU2PPTX.exe` (Windows) or `dist/MinerU2PPTX` (Linux/macOS)
- **Size**: Typically 80-150 MB (includes all dependencies)
- **Standalone**: No Python installation required on target machines

### **Build Artifacts**
- `build/` - Temporary build files (can be deleted)
- `dist/` - Final executable location
- `MinerU2PPTX.spec` - PyInstaller specification file
- `version_info.txt` - Windows executable metadata

## Testing the Executable

### **Basic Test**
```bash
# Windows
dist\MinerU2PPTX.exe

# Linux/macOS  
./dist/MinerU2PPTX
```

### **Automated Test**
```bash
python test_build.py
```

This script verifies:
- ✅ All required packages are installed
- ✅ Project structure is complete
- ✅ GUI can be imported and created
- ✅ Translation system works

## Troubleshooting

### **Common Issues**

#### **"ModuleNotFoundError" during execution**
- **Cause**: Missing hidden import
- **Solution**: Add `--hidden-import module_name` to PyInstaller command

#### **"FileNotFoundError" for translation files**
- **Cause**: Translation files not included
- **Solution**: Ensure `--add-data "translations;translations"` is used

#### **Large executable size**
- **Cause**: PyInstaller includes many dependencies
- **Solution**: Normal behavior, optimize with `--exclude-module` if needed

#### **GUI doesn't appear**
- **Cause**: Missing `--windowed` flag or GUI initialization error
- **Solution**: Use `--windowed` and check console output without it

### **Build Environment Issues**

#### **PyInstaller not found**
```bash
pip install pyinstaller>=6.3.0
```

#### **Permission errors (Linux/macOS)**
```bash
chmod +x build.sh
chmod +x dist/MinerU2PPTX
```

#### **Python environment issues**
- Ensure you're in the correct virtual environment
- Verify Python version compatibility (3.8+)
- Check that all dependencies are installed

## Distribution

### **Single File Distribution**
The `--onefile` option creates a single executable:
- **Pros**: Easy distribution, single file
- **Cons**: Slower startup (extracts to temp directory)

### **Directory Distribution**
Remove `--onefile` for directory distribution:
- **Pros**: Faster startup
- **Cons**: Multiple files to distribute

### **Recommended Distribution**
1. **Test the executable** on the target platform
2. **Include any additional resources** (config files, icons) 
3. **Create an installer** (optional) using tools like NSIS or Inno Setup
4. **Document system requirements** (Windows 10+, specific Linux distributions)

## Advanced Configuration

### **Custom Icon**
```bash
pyinstaller --icon=icon.ico --windowed --onefile gui.py
```

### **Version Information (Windows)**
The build script creates `version_info.txt` automatically for:
- File description
- Company name  
- Version numbers
- Copyright information

### **Optimization Options**
- `--upx-dir=UPX_DIR` - Use UPX compression (if installed)
- `--strip` - Strip debug symbols (Linux/macOS)
- `--exclude-module` - Exclude unnecessary modules

## Build Script Features

The included `build_exe.py` script provides:

- ✅ **Dependency checking** before build
- ✅ **Automatic cleanup** of previous builds  
- ✅ **Spec file generation** with optimized settings
- ✅ **Version information** creation
- ✅ **Build verification** and testing
- ✅ **Cross-platform support** (Windows/Linux/macOS)
- ✅ **Error handling** and user-friendly messages

Run with: `python build_exe.py`