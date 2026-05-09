@echo off
REM Tiny Multiscreen LM - CUDA Inference Runner
REM This script sets up the Visual Studio environment and runs inference with CUDA

echo Setting up CUDA environment...
set CUDA_COMPUTE_CAP=89

echo Loading Visual Studio environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo.
echo ============================================
echo Running Tiny Multiscreen LM - CUDA Inference
echo ============================================
echo.

if "%~1"=="" (
    echo No arguments provided, starting interactive mode...
    echo.
    cargo run --release --features cuda --bin infer -- --interactive
) else (
    cargo run --release --features cuda --bin infer -- %*
)

echo.
echo ============================================
echo Inference Complete!
echo ============================================
echo.
pause
