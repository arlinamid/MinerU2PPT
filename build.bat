@echo off
echo ============================================
echo   MinerU2PPTX Build Script  v2.0.1
echo   Author: Arlinamid (Rozsavolgyi Janos)
echo ============================================
echo.

REM Clean previous builds
echo [1/4] Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
echo       Done.
echo.

REM Build CLI executable
echo [2/4] Building cli.exe (console) ...
pyinstaller --clean cli.spec
if errorlevel 1 (
    echo [ERROR] cli.exe build failed!
    pause
    exit /b 1
)
echo       cli.exe built successfully.
echo.

REM Build GUI executable
echo [3/4] Building MinerU2PPTX.exe (GUI) ...
pyinstaller --clean MinerU2PPTX.spec
if errorlevel 1 (
    echo [ERROR] MinerU2PPTX.exe build failed!
    pause
    exit /b 1
)
echo       MinerU2PPTX.exe built successfully.
echo.

REM Show results
echo [4/4] Build complete!
echo.
echo   Output files:
if exist dist\cli.exe (
    echo     dist\cli.exe
    for %%A in (dist\cli.exe) do echo       Size: %%~zA bytes
)
if exist dist\MinerU2PPTX.exe (
    echo     dist\MinerU2PPTX.exe
    for %%A in (dist\MinerU2PPTX.exe) do echo       Size: %%~zA bytes
)
echo.
echo   Usage:
echo     cli.exe --version
echo     cli.exe convert --json data.json --input file.pdf --output out.pptx
echo     cli.exe extract-images --json data.json --input file.pdf --output imgs/
echo     MinerU2PPTX.exe   (launches GUI)
echo.
pause
