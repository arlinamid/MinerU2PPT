#!/usr/bin/env python3
"""
Build script for creating MinerU2PPTX executables.

Builds two executables:
  - cli.exe       (console-mode CLI)
  - MinerU2PPTX.exe  (windowed GUI)

Author: Arlinamid (Rózsavölgyi János)
Version: 2.0.1
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

VERSION = "2.0.1"


def check_requirements():
    """Check if required packages are installed."""
    required = ['pyinstaller', 'pptx', 'cv2', 'fitz', 'numpy', 'PIL']
    missing = []
    for pkg in required:
        try:
            if pkg == 'pyinstaller':
                import PyInstaller
            elif pkg == 'PIL':
                import PIL
            elif pkg == 'cv2':
                import cv2
            elif pkg == 'fitz':
                import fitz
            else:
                __import__(pkg.replace('-', '_'))
            print(f"  [OK]   {pkg}")
        except ImportError:
            missing.append(pkg)
            print(f"  [MISS] {pkg}")
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    return True


def clean_build():
    """Remove previous build artifacts."""
    for d in ('build', 'dist', '__pycache__'):
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"  Removed {d}/")


def build_spec(spec_file: str, label: str):
    """Run PyInstaller on a .spec file."""
    print(f"\n  Building {label} from {spec_file} ...")
    result = subprocess.run(
        [sys.executable, '-m', 'PyInstaller', '--clean', spec_file],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  [ERROR] {label} build failed!")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        return False
    print(f"  [OK]   {label} built successfully.")
    return True


def show_results():
    """Print resulting file sizes."""
    print("\nBuild artifacts:")
    for name in ('dist/cli.exe', 'dist/MinerU2PPTX.exe'):
        p = Path(name)
        if p.exists():
            mb = p.stat().st_size / (1024 * 1024)
            print(f"  {name:30s}  {mb:6.1f} MB")
        else:
            print(f"  {name:30s}  NOT FOUND")


def main():
    print(f"=== MinerU2PPTX Executable Builder  v{VERSION} ===\n")

    if not os.path.exists('gui.py') or not os.path.exists('main.py'):
        print("Error: Run this script from the MinerU2PPT root directory.")
        return False

    print("1. Checking requirements ...")
    if not check_requirements():
        return False

    print("\n2. Cleaning previous build ...")
    clean_build()

    print("\n3. Building executables ...")
    ok_cli = build_spec('cli.spec', 'cli.exe')
    ok_gui = build_spec('MinerU2PPTX.spec', 'MinerU2PPTX.exe')

    print("\n4. Results")
    show_results()

    if ok_cli and ok_gui:
        print("\n[SUCCESS] Both executables built.")
    elif ok_gui:
        print("\n[PARTIAL] Only MinerU2PPTX.exe built; cli.exe failed.")
    elif ok_cli:
        print("\n[PARTIAL] Only cli.exe built; MinerU2PPTX.exe failed.")
    else:
        print("\n[FAILURE] Both builds failed.")

    return ok_cli and ok_gui


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
