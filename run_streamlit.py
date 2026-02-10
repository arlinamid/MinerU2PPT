#!/usr/bin/env python3
"""
Runner script for MinerU2PPT Streamlit GUI
"""

import sys
import subprocess
import os
from pathlib import Path

def check_streamlit_installed():
    """Check if Streamlit is installed"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_requirements():
    """Install requirements if needed"""
    print("Installing/updating requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def run_streamlit():
    """Run the Streamlit application"""
    script_path = Path(__file__).parent / "streamlit_gui.py"
    
    if not script_path.exists():
        print(f"❌ Streamlit GUI script not found: {script_path}")
        return False
    
    print("🚀 Starting MinerU2PPT Streamlit GUI...")
    print("📱 The web interface will open automatically in your browser")
    print("🔗 If it doesn't open, go to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(script_path),
            "--theme.base", "light",
            "--theme.primaryColor", "#FF6B6B",
            "--theme.backgroundColor", "#FFFFFF",
            "--theme.secondaryBackgroundColor", "#F0F2F6",
            "--theme.textColor", "#262730"
        ])
        
    except KeyboardInterrupt:
        print("\n⏹️  Streamlit server stopped.")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("=" * 60)
    print("🎯 MinerU2PPT Streamlit GUI Launcher")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("converter").exists():
        print("❌ Please run this script from the MinerU2PPT root directory")
        print("   The 'converter' folder should be in the current directory.")
        sys.exit(1)
    
    # Check Streamlit installation
    if not check_streamlit_installed():
        print("⚠️  Streamlit not found. Installing requirements...")
        if not install_requirements():
            sys.exit(1)
    
    # Check if AI dependencies are available
    try:
        from converter.ai_services import ai_manager
        from converter.config import ai_config
        print("✅ AI services are available")
    except ImportError as e:
        print(f"⚠️  AI services may not be fully functional: {e}")
        print("   Install AI dependencies: pip install -r requirements.txt")
    
    # Run the Streamlit app
    success = run_streamlit()
    
    if not success:
        print("\n❌ Failed to start Streamlit GUI")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you're in the MinerU2PPT directory")
        print("2. Install requirements: pip install -r requirements.txt")
        print("3. Try running manually: streamlit run streamlit_gui.py")
        sys.exit(1)

if __name__ == "__main__":
    main()