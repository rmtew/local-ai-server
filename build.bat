@echo off
REM Build local-ai-server.exe -- OpenAI-compatible local inference server
REM Can be run from any terminal and any directory
REM
REM Usage: build.bat [target]
REM   (no target) - build server (default)
REM   bench       - build vocoder-bench.exe

setlocal EnableDelayedExpansion
cd /d "%~dp0"

REM Parse target (default: server)
set TARGET=server
if /I "%~1"=="bench" set TARGET=bench

REM Auto-setup MSVC environment if not already configured
where cl.exe >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Setting up MSVC environment...
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if not exist "!VSWHERE!" (
        echo ERROR: vswhere.exe not found. Install Visual Studio with C++ workload.
        exit /b 1
    )
    for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -property installationPath`) do set "VSINSTALL=%%i"
    if not defined VSINSTALL (
        echo ERROR: Could not find Visual Studio installation
        exit /b 1
    )
    call "!VSINSTALL!\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
    where cl.exe >nul 2>&1
    if !ERRORLEVEL! NEQ 0 (
        echo ERROR: Failed to initialize MSVC environment
        exit /b 1
    )
)

REM Paths relative to repo root
set QWEN_ASR_DIR=qwen-asr
set BUILD_DIR=build
set BIN_DIR=bin

REM Resolve shared deps (DEPS_ROOT required for OpenBLAS)
if not defined DEPS_ROOT (
    echo ERROR: DEPS_ROOT environment variable is not set.
    echo Set it to the shared deps directory, e.g.: set DEPS_ROOT=C:\Data\R\git\claude-repos\deps
    exit /b 1
)
set "OPENBLAS_DIR=%DEPS_ROOT%\openblas"

REM Check prerequisites
if not exist "%QWEN_ASR_DIR%\qwen_asr.c" (
    echo ERROR: qwen-asr source not found at %QWEN_ASR_DIR%
    echo Did you run: git submodule update --init
    exit /b 1
)

REM Detect OpenBLAS (strongly recommended for performance)
set BLAS_CFLAGS=
set BLAS_LIBS=
if exist "%OPENBLAS_DIR%\openblas_msvc.lib" (
    echo OpenBLAS found -- enabling BLAS acceleration
    set BLAS_CFLAGS=/DUSE_BLAS /I"%OPENBLAS_DIR%\include"
    set BLAS_LIBS="%OPENBLAS_DIR%\openblas_msvc.lib"
) else (
    echo WARNING: OpenBLAS not found at %OPENBLAS_DIR% -- will be slow
)

REM Create output directories
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if not exist "%BIN_DIR%" mkdir "%BIN_DIR%"

REM Detect CUDA toolkit (optional -- enables cuBLAS GPU acceleration)
set CUDA_CFLAGS=
set CUDA_LIBS=
set "CUDA_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if exist "%CUDA_DIR%\lib\x64\cublas.lib" (
    echo CUDA found -- enabling cuBLAS GPU acceleration
    set CUDA_CFLAGS=/DUSE_CUBLAS /I"%CUDA_DIR%\include"
    set CUDA_LIBS="%CUDA_DIR%\lib\x64\cublas.lib" "%CUDA_DIR%\lib\x64\cudart.lib" "%CUDA_DIR%\lib\x64\cuda.lib"

    REM Try to compile CUDA kernels to CUBIN (requires nvcc)
    if exist "%CUDA_DIR%\bin\nvcc.exe" (
        echo Compiling CUDA kernels to CUBIN...
        "%CUDA_DIR%\bin\nvcc.exe" -cubin -arch=sm_86 -o "%BUILD_DIR%\qwen_asr_kernels.cubin" "%QWEN_ASR_DIR%\qwen_asr_kernels.cu" 2>&1
        if !ERRORLEVEL! EQU 0 (
            echo Generating CUBIN header...
            py "%QWEN_ASR_DIR%\tools\cubin_to_header.py" "%BUILD_DIR%\qwen_asr_kernels.cubin" "%QWEN_ASR_DIR%\qwen_asr_kernels_cubin.h"
            if !ERRORLEVEL! EQU 0 (
                set CUDA_CFLAGS=!CUDA_CFLAGS! /DUSE_CUDA_KERNELS
                echo CUDA kernels enabled
            ) else (
                echo WARNING: CUBIN header generation failed -- using cuBLAS only
            )
        ) else (
            echo WARNING: nvcc compilation failed -- using cuBLAS only
        )
    ) else (
        echo nvcc not found -- using cuBLAS only ^(no custom kernels^)
    )
) else (
    echo CUDA not found -- GPU acceleration disabled
)

REM Detect ONNX Runtime (optional -- enables TTS via Qwen3-TTS ONNX models)
set ORT_CFLAGS=
set ORT_LIBS=
set "ORT_CPU_DIR=%DEPS_ROOT%\onnxruntime\1.23.2"
if exist "%ORT_CPU_DIR%\lib\onnxruntime.lib" (
    echo ONNX Runtime found -- enabling TTS
    set ORT_CFLAGS=/DUSE_ORT /I"%ORT_CPU_DIR%\include"
    set ORT_LIBS="%ORT_CPU_DIR%\lib\onnxruntime.lib"
    set "ORT_DIR=%ORT_CPU_DIR%"
) else (
    echo ONNX Runtime not found -- TTS disabled
)

REM Copy OpenBLAS DLL if not present
if not exist "%BIN_DIR%\libopenblas.dll" (
    if exist "%OPENBLAS_DIR%\bin\libopenblas.dll" (
        echo Copying libopenblas.dll...
        copy "%OPENBLAS_DIR%\bin\libopenblas.dll" "%BIN_DIR%" >nul
    ) else if exist "%OPENBLAS_DIR%\libopenblas.dll" (
        echo Copying libopenblas.dll...
        copy "%OPENBLAS_DIR%\libopenblas.dll" "%BIN_DIR%" >nul
    )
)

REM Copy ONNX Runtime DLL (always update to match linked version)
if defined ORT_DIR (
    if exist "%ORT_DIR%\lib\onnxruntime.dll" (
        echo Copying onnxruntime.dll...
        copy /Y "%ORT_DIR%\lib\onnxruntime.dll" "%BIN_DIR%" >nul
    )
)

REM Qwen-ASR source files (excluding main.c -- server has its own entry point)
set QWEN_SOURCES="%QWEN_ASR_DIR%\qwen_asr.c" "%QWEN_ASR_DIR%\qwen_asr_audio.c" "%QWEN_ASR_DIR%\qwen_asr_decoder.c" "%QWEN_ASR_DIR%\qwen_asr_encoder.c" "%QWEN_ASR_DIR%\qwen_asr_kernels.c" "%QWEN_ASR_DIR%\qwen_asr_kernels_avx.c" "%QWEN_ASR_DIR%\qwen_asr_kernels_generic.c" "%QWEN_ASR_DIR%\qwen_asr_safetensors.c" "%QWEN_ASR_DIR%\qwen_asr_tokenizer.c" "%QWEN_ASR_DIR%\qwen_asr_gpu.c"

REM Compile qwen-asr sources with full optimization (inference is CPU-bound)
echo Compiling qwen-asr (optimized)...
cl /nologo /W3 /O2 /arch:AVX2 /fp:fast /DNDEBUG !BLAS_CFLAGS! !CUDA_CFLAGS! /I"%QWEN_ASR_DIR%" /c %QWEN_SOURCES% /Fo:"%BUILD_DIR%\\"
if %ERRORLEVEL% NEQ 0 (
    echo qwen-asr compilation failed.
    exit /b 1
)

REM Collect qwen-asr object files (shared by all targets)
set QWEN_OBJS="%BUILD_DIR%\qwen_asr.obj" "%BUILD_DIR%\qwen_asr_audio.obj" "%BUILD_DIR%\qwen_asr_decoder.obj" "%BUILD_DIR%\qwen_asr_encoder.obj" "%BUILD_DIR%\qwen_asr_kernels.obj" "%BUILD_DIR%\qwen_asr_kernels_avx.obj" "%BUILD_DIR%\qwen_asr_kernels_generic.obj" "%BUILD_DIR%\qwen_asr_safetensors.obj" "%BUILD_DIR%\qwen_asr_tokenizer.obj" "%BUILD_DIR%\qwen_asr_gpu.obj"

REM Vocoder source files (compiled with optimization -- inference-critical, same as qwen-asr)
set VOC_SOURCES=src\tts_vocoder.c src\tts_vocoder_ops.c src\tts_vocoder_xfmr.c src\tts_mel.c src\tts_speaker_enc.c

echo Compiling vocoder (optimized)...
cl /nologo /W3 /O2 /arch:AVX2 /fp:fast /DNDEBUG !BLAS_CFLAGS! !CUDA_CFLAGS! !ORT_CFLAGS! /I"%QWEN_ASR_DIR%" /Isrc /c %VOC_SOURCES% /Fo:"%BUILD_DIR%\\"
if %ERRORLEVEL% NEQ 0 (
    echo Vocoder compilation failed.
    exit /b 1
)

set VOC_OBJS="%BUILD_DIR%\tts_vocoder.obj" "%BUILD_DIR%\tts_vocoder_ops.obj" "%BUILD_DIR%\tts_vocoder_xfmr.obj" "%BUILD_DIR%\tts_mel.obj" "%BUILD_DIR%\tts_speaker_enc.obj"

REM ---- Target: bench ----
if /I "%TARGET%"=="bench" goto :build_bench

REM ---- Target: server (default) ----

REM Server source files (excluding vocoder -- compiled separately with optimization)
set SRV_SOURCES=src\main.c src\http.c src\multipart.c src\handler_asr.c src\json.c src\json_reader.c src\handler_tts.c src\tts_ort.c src\tts_pipeline.c src\tts_sampling.c src\tts_native.c src\tts_voice_presets.c

REM Compile server sources with debug info
echo Compiling server (debug)...
cl /nologo /W3 /Od /Zi /DDEBUG !BLAS_CFLAGS! !CUDA_CFLAGS! !ORT_CFLAGS! /I"%QWEN_ASR_DIR%" /Isrc /c %SRV_SOURCES% /Fo:"%BUILD_DIR%\\"
if %ERRORLEVEL% NEQ 0 (
    echo Server compilation failed.
    exit /b 1
)

REM Collect server object files
set SRV_OBJS="%BUILD_DIR%\main.obj" "%BUILD_DIR%\http.obj" "%BUILD_DIR%\multipart.obj" "%BUILD_DIR%\handler_asr.obj" "%BUILD_DIR%\json.obj" "%BUILD_DIR%\json_reader.obj" "%BUILD_DIR%\handler_tts.obj" "%BUILD_DIR%\tts_ort.obj" "%BUILD_DIR%\tts_pipeline.obj" "%BUILD_DIR%\tts_sampling.obj" "%BUILD_DIR%\tts_native.obj" "%BUILD_DIR%\tts_voice_presets.obj"

REM Link everything together
echo Linking server...
link /nologo /DEBUG /SUBSYSTEM:CONSOLE /OUT:"%BIN_DIR%\local-ai-server.exe" %SRV_OBJS% %VOC_OBJS% %QWEN_OBJS% !BLAS_LIBS! !CUDA_LIBS! !ORT_LIBS! ws2_32.lib advapi32.lib psapi.lib

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Build complete: %BIN_DIR%\local-ai-server.exe
) else (
    echo.
    echo Build failed.
    exit /b 1
)
goto :eof

REM ---- Target: bench ----
:build_bench

REM Bench uses shared TTS sources (no server main/http/handlers) + its own main
REM Vocoder objs are already compiled above with optimization
set BENCH_SOURCES=tools\vocoder_bench.c src\tts_ort.c src\tts_pipeline.c src\tts_sampling.c src\tts_native.c

echo Compiling vocoder-bench...
cl /nologo /W3 /Od /Zi /DDEBUG !BLAS_CFLAGS! !CUDA_CFLAGS! !ORT_CFLAGS! /I"%QWEN_ASR_DIR%" /Isrc /c %BENCH_SOURCES% /Fo:"%BUILD_DIR%\\"
if %ERRORLEVEL% NEQ 0 (
    echo Bench compilation failed.
    exit /b 1
)

set BENCH_OBJS="%BUILD_DIR%\vocoder_bench.obj" "%BUILD_DIR%\tts_ort.obj" "%BUILD_DIR%\tts_pipeline.obj" "%BUILD_DIR%\tts_sampling.obj" "%BUILD_DIR%\tts_native.obj"

echo Linking vocoder-bench...
link /nologo /DEBUG /SUBSYSTEM:CONSOLE /OUT:"%BIN_DIR%\vocoder-bench.exe" %BENCH_OBJS% %VOC_OBJS% %QWEN_OBJS% !BLAS_LIBS! !CUDA_LIBS! !ORT_LIBS! ws2_32.lib advapi32.lib psapi.lib

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Build complete: %BIN_DIR%\vocoder-bench.exe
) else (
    echo.
    echo Build failed.
    exit /b 1
)
goto :eof
