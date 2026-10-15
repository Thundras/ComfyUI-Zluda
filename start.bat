@echo off
:: Vega 64 launcher menu, see VEGA64.md
cd /d "%~dp0"

echo.
echo  ComfyUI-Zluda (Radeon RX Vega 64)
echo  ---------------------------------
echo  [1] Torch 2.7 + comfy-kitchen  (comfyui-t24.bat)  - YuE2, Lumina, SD
echo  [2] Torch 2.2                  (comfyui-user.bat) - fallback, git pull
echo  [Q] Quit
echo.
choice /c 12Q /n /m "Selection: "

if errorlevel 3 exit /b
if errorlevel 2 call comfyui-user.bat & exit /b
if errorlevel 1 call comfyui-t24.bat & exit /b
