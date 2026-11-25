@echo off
rem 双击启动训练台。整个 lstm_workbench 文件夹复制到别的电脑就能用，不用联网、不用装任何东西：
rem 运行环境就在本文件夹的 runtime\ 里（Python、PyTorch、Ultralytics 等全都在里面）。
chcp 65001 >nul
setlocal
set "WB=%~dp0"
if "%WB:~-1%"=="\" set "WB=%WB:~0,-1%"
for %%I in ("%WB%\..") do set "PARENT=%%~fI"
for %%I in ("%WB%") do set "PKG=%%~nxI"

rem 依次找：本文件夹 runtime\（离线便携包）→ 本文件夹 .venv\ → 上一级 .venv\（开发机）。
set "PY="
if exist "%WB%\runtime\python.exe" set "PY=%WB%\runtime\python.exe"
if not defined PY if exist "%WB%\.venv\Scripts\python.exe" set "PY=%WB%\.venv\Scripts\python.exe"
if not defined PY if exist "%PARENT%\.venv\Scripts\python.exe" set "PY=%PARENT%\.venv\Scripts\python.exe"
if not defined PY (
  echo 找不到运行环境：本文件夹里没有 runtime\python.exe。
  echo 请在开发机上运行 make_offline_runtime.py 生成 runtime，再整个文件夹复制过来。
  pause
  exit /b 1
)

rem 不让本机装的其他 Python 干扰。
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "YOLO_OFFLINE=1"
rem 以模块方式运行：工作目录切到上一级，Python 才能按包名找到工作台。
cd /d "%PARENT%"
"%PY%" -m %PKG%
if errorlevel 1 pause
endlocal
