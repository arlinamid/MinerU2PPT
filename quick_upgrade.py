#!/usr/bin/env python3
"""
Quick upgrade script for MinerU2PPT
Performs complete migration to latest packages with virtual environment
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and report results"""
    print(f"\n[INFO] {description}")
    print(f"[CMD] {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {e}")
        if e.stderr:
            print(f"[STDERR] {e.stderr.strip()}")
        return False

def check_python():
    """Check Python version"""
    version = sys.version_info
    print(f"[INFO] Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 10:
        print("[OK] Python version is suitable")
        return True
    else:
        print("[ERROR] Python 3.10+ required")
        return False

def check_venv():
    """Check if we're in a virtual environment"""
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if in_venv:
        print("[WARNING] Already in virtual environment")
        print("This script will create a new venv. Deactivate first if needed.")
        return True
    else:
        print("[INFO] Not in virtual environment (good)")
        return True

def main():
    """Perform complete upgrade"""
    print("=== MinerU2PPT Quick Upgrade ===")
    print("This script will:")
    print("1. Create a fresh virtual environment")
    print("2. Remove deprecated google-generativeai")  
    print("3. Install latest packages")
    print("4. Verify installation")
    print()
    
    # Check prerequisites
    if not check_python():
        return False
    
    check_venv()
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    # Step 1: Remove old venv if exists
    if Path("venv").exists():
        print("\n[INFO] Removing existing virtual environment...")
        if os.name == 'nt':  # Windows
            run_command("rmdir /s /q venv", "Remove old venv (Windows)")
        else:  # Linux/macOS
            run_command("rm -rf venv", "Remove old venv (Unix)")
    
    # Step 2: Create new virtual environment
    if not run_command("python -m venv venv", "Create virtual environment"):
        return False
    
    # Step 3: Determine activation command
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate && "
        pip_cmd = activate_cmd + "python -m pip"
    else:  # Linux/macOS  
        activate_cmd = "source venv/bin/activate && "
        pip_cmd = activate_cmd + "python -m pip"
    
    # Step 4: Upgrade pip
    if not run_command(f"{pip_cmd} install --upgrade pip setuptools wheel", 
                      "Upgrade pip and tools"):
        return False
    
    # Step 5: Remove deprecated package if present
    print("\n[INFO] Removing deprecated google-generativeai if present...")
    run_command(f"{pip_cmd} uninstall -y google-generativeai", 
                "Remove deprecated package (may not exist)")
    
    # Step 6: Install latest packages
    packages_core = [
        "python-pptx>=0.6.23",
        "PyMuPDF>=1.24.0", 
        "opencv-python>=4.9.0",
        "numpy>=1.26.0",
        "Pillow>=10.2.0",
        "tkinterdnd2>=0.3.0",
        "scikit-image>=0.22.0",
        "pyinstaller>=6.3.0"
    ]
    
    packages_ai = [
        "openai>=1.8.0",
        "google-genai>=0.3.0",  # New package
        "anthropic>=0.8.0",
        "groq>=0.4.2", 
        "httpx>=0.26.0"
    ]
    
    all_packages = packages_core + packages_ai
    
    if not run_command(f"{pip_cmd} install --upgrade {' '.join(all_packages)}", 
                      "Install latest packages"):
        return False
    
    # Step 7: Verify installation
    print("\n[INFO] Verifying installation...")
    
    test_imports = [
        ("google.genai", "Google GenAI"),
        ("openai", "OpenAI"),
        ("anthropic", "Anthropic"), 
        ("groq", "Groq"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow")
    ]
    
    all_good = True
    for module, name in test_imports:
        cmd = f"{activate_cmd}python -c \"import {module}; print('[OK] {name}')\""
        if not run_command(cmd, f"Test {name} import"):
            print(f"[WARNING] {name} import failed")
            all_good = False
    
    # Step 8: Generate requirements
    run_command(f"{pip_cmd} freeze > requirements_installed.txt", 
                "Generate installed requirements")
    
    # Step 9: Final status
    print("\n" + "="*50)
    if all_good:
        print("✅ UPGRADE COMPLETE!")
        print("\nTo use MinerU2PPT:")
        if os.name == 'nt':
            print("1. Activate: venv\\Scripts\\activate")
        else:
            print("1. Activate: source venv/bin/activate")
        print("2. Run GUI: python gui.py")
        print("3. Or CLI: python main.py --help")
        
        print(f"\nPackage summary:")
        print(f"- Virtual environment: Created")
        print(f"- Deprecated packages: Removed")  
        print(f"- Latest packages: Installed")
        print(f"- Google GenAI: New package")
        print(f"- All imports: {'Working' if all_good else 'Some issues'}")
        
    else:
        print("⚠️  UPGRADE COMPLETED WITH WARNINGS")
        print("Some package imports failed. Check error messages above.")
        print("You may need to install additional system dependencies.")
    
    print(f"\nMaintenance:")
    print(f"- Check packages: python check_packages.py") 
    print(f"- Test setup: python test_gui_fixes.py")
    
    return all_good

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[INFO] Upgrade cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)