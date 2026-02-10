#!/usr/bin/env python3
"""
Build script for creating MinerU2PPTX executable from gui.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    required_packages = ['pyinstaller', 'tkinterdnd2', 'pptx']
    missing = []
    
    for package in required_packages:
        try:
            if package == 'pyinstaller':
                import PyInstaller
            else:
                __import__(package.replace('-', '_'))
            print(f"[OK] {package} is installed")
        except ImportError:
            missing.append(package)
            print(f"[MISS] {package} is missing")
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def clean_build_directories():
    """Clean previous build artifacts."""
    directories_to_clean = ['build', 'dist', '__pycache__']
    files_to_clean = ['*.spec']
    
    print("Cleaning previous build artifacts...")
    
    for dir_name in directories_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"  Removed {dir_name}/")
    
    # Remove spec files
    for spec_file in Path('.').glob('*.spec'):
        spec_file.unlink()
        print(f"  Removed {spec_file}")

def create_spec_file():
    """Create a PyInstaller spec file for better control."""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect translation files
datas = [
    ('translations/*.json', 'translations'),
    ('translations/__init__.py', 'translations'),
    ('translations/translator.py', 'translations'),
]

# Collect hidden imports
hiddenimports = [
    'tkinterdnd2',
    'pptx',
    'pptx.util',
    'pptx.dml.color',
    'pptx.enum.text',
    'PIL',
    'PIL.Image',
    'cv2',
    'numpy',
    'fitz',  # PyMuPDF
    'skimage',
    'translations',
    'translations.translator',
    'converter.generator',
    'converter.ai_services',
    'converter.config',
    'converter.cache_manager',
]

# Add converter modules
hiddenimports.extend([
    'converter',
    'converter.generator',
    'converter.ai_services', 
    'converter.config',
    'converter.cache_manager',
])

a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MinerU2PPTX',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='version_info.txt',
    icon=None,  # Add icon file path if available
)
'''
    
    with open('MinerU2PPTX.spec', 'w', encoding='utf-8') as f:
        f.write(spec_content)
    
    print("Created MinerU2PPTX.spec file")

def create_version_info():
    """Create version information file for Windows executable."""
    version_content = '''# UTF-8
#
# For more details about fixed file info 'ffi' see:
# http://msdn.microsoft.com/en-us/library/ms646997.aspx
VSVersionInfo(
  ffi=FixedFileInfo(
# filevers and prodvers should be always a tuple with four items: (1, 2, 3, 4)
# Set not needed items to zero 0.
filevers=(1,0,0,0),
prodvers=(1,0,0,0),
# Contains a bitmask that specifies the valid bits 'flags'r
mask=0x3f,
# Contains a bitmask that specifies the Boolean attributes of the file.
flags=0x0,
# The operating system for which this file was designed.
# 0x4 - NT and there is no need to change it.
OS=0x4,
# The general type of file.
# 0x1 - the file is an application.
fileType=0x1,
# The function of the file.
# 0x0 - the function is not defined for this fileType
subtype=0x0,
# Creation date and time stamp.
date=(0, 0)
),
  kids=[
StringFileInfo(
  [
  StringTable(
    u'040904B0',
    [StringStruct(u'CompanyName', u'MinerU2PPTX'),
    StringStruct(u'FileDescription', u'MinerU to PPTX Converter'),
    StringStruct(u'FileVersion', u'1.0.0.0'),
    StringStruct(u'InternalName', u'MinerU2PPTX'),
    StringStruct(u'LegalCopyright', u'Open Source'),
    StringStruct(u'OriginalFilename', u'MinerU2PPTX.exe'),
    StringStruct(u'ProductName', u'MinerU2PPTX Converter'),
    StringStruct(u'ProductVersion', u'1.0.0.0')])
  ]), 
VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
'''
    
    with open('version_info.txt', 'w', encoding='utf-8') as f:
        f.write(version_content)
    
    print("Created version_info.txt file")

def build_executable():
    """Build the executable using PyInstaller."""
    print("Building executable with PyInstaller...")
    
    try:
        # Build using spec file
        result = subprocess.run([
            sys.executable, '-m', 'PyInstaller',
            '--clean',
            'MinerU2PPTX.spec'
        ], check=True, capture_output=True, text=True)
        
        print("[OK] Build completed successfully!")
        print(f"Executable location: dist/MinerU2PPTX.exe")
        
        # Check if file was created
        exe_path = Path('dist/MinerU2PPTX.exe')
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"Executable size: {size_mb:.1f} MB")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print("[ERROR] Build failed!")
        print(f"Error output: {e.stderr}")
        return False

def create_simple_build_script():
    """Create a simple build script for quick builds."""
    script_content = '''@echo off
echo Building MinerU2PPTX Executable...
echo.

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.spec del *.spec

REM Run the build script
python build_exe.py

pause
'''
    
    with open('build.bat', 'w') as f:
        f.write(script_content)
    
    print("Created build.bat for Windows")

def main():
    """Main build process."""
    print("=== MinerU2PPTX Executable Builder ===")
    print()
    
    # Check current directory
    if not os.path.exists('gui.py'):
        print("Error: gui.py not found in current directory!")
        print("Please run this script from the MinerU2PPT root directory.")
        return False
    
    # Check requirements
    print("1. Checking requirements...")
    if not check_requirements():
        return False
    
    # Clean build directories
    print("\n2. Cleaning build artifacts...")
    clean_build_directories()
    
    # Create spec file
    print("\n3. Creating build configuration...")
    create_spec_file()
    create_version_info()
    create_simple_build_script()
    
    # Build executable
    print("\n4. Building executable...")
    success = build_executable()
    
    if success:
        print("\n[SUCCESS] Build completed successfully!")
        print("\nNext steps:")
        print("1. Test the executable: dist/MinerU2PPTX.exe")
        print("2. The executable includes all dependencies")
        print("3. Distribute the entire dist/ folder or just the .exe file")
        
        # Test basic import
        print("\n5. Testing executable...")
        try:
            result = subprocess.run(['dist/MinerU2PPTX.exe', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("[OK] Executable test passed")
            else:
                print("[WARN] Executable may have issues (exit code: {})".format(result.returncode))
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("[WARN] Could not test executable (this is normal)")
    
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)