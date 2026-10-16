@echo off
:: Vega 64 launcher with torch 2.7 + comfy-kitchen (venv-t24), setup in VEGA64.md.
:: fp16-unet: Lumina 2 produces NaNs in bf16/fp32 under ZLUDA, YuE2 sounds the same in fp16.

set "PATH=C:\Program Files\AMD\ROCm\5.7\bin;%PATH%"
set "PYTHON=%~dp0venv-t24\Scripts\python.exe"
set "COMMANDLINE_ARGS=--use-quad-cross-attention --reserve-vram 2.2 --disable-async-offload --disable-pinned-memory --disable-mmap --disable-smart-memory --fp16-unet"
set "ZLUDA_COMGR_LOG_LEVEL=1"

copy comfy\customzluda\zluda-default.py comfy\zluda.py /y >NUL

echo [INFO] torch 2.7 + comfy-kitchen (venv-t24)
echo [INFO] The first run of a new model compiles GPU kernels via ZLUDA, this can take 10-30 minutes.
echo.
.\zluda\zluda.exe -- %PYTHON% main.py %COMMANDLINE_ARGS%
pause
