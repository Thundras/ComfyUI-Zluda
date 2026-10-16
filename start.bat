@echo off
:: Vega 64 launcher menu, see VEGA64.md
cd /d "%~dp0"
:: Only fetches patientx, nothing is merged. Offline the check is skipped.
set "GIT_TERMINAL_PROMPT=0"
git fetch -q zluda master 2>nul
set "BEHIND="
for /f %%n in ('git rev-list --count HEAD..zluda/master 2^>nul') do set "BEHIND=%%n"
for /f %%m in ('git merge-base HEAD zluda/master 2^>nul') do set "BASE=%%m"
for /f %%d in ('git show -s --format^=%%cd --date^=short %BASE% 2^>nul') do set "SINCE=%%d"
if defined BEHIND if not "%BEHIND%"=="0" (
    echo.
    echo  [UPDATE] %BEHIND% commits behind patientx, your patientx state is from %SINCE%.
    echo           Ask Claude: "zieh patientx nach"
)

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
