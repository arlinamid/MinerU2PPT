@echo off
echo Building MinerU2PPTX Executable...
echo.

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.spec del *.spec

REM Run the build script
python build_exe.py

pause
