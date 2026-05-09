@echo off
echo ========================================
echo   Training with CUDA (GPU)
echo ========================================
echo.
echo Setting up Visual Studio environment...
set CUDA_COMPUTE_CAP=89
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Could not find Visual Studio 2022 Community
    echo Please update the path in this batch file to match your VS installation:
    echo   - Community:   C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat
    echo   - Professional: C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat
    echo   - Enterprise:  C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat
    echo.
    pause
    exit /b 1
)
echo.
echo Starting training...
cargo run --release --features cuda --bin train
echo.
echo ========================================
echo   Training Complete!
echo ========================================
pause
