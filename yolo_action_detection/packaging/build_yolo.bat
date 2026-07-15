@echo off
setlocal

pushd "%~dp0.."
set "APP_DIR=%CD%"
popd
pushd "%APP_DIR%\.."
set "REPO_DIR=%CD%"
popd

set "PYINSTALLER=%REPO_DIR%\.venv\Scripts\pyinstaller.exe"
set "PYTHON=%REPO_DIR%\.venv\Scripts\python.exe"
set "DIST_DIR=%APP_DIR%\dist\YOLOActionDetection"

if not exist "%PYINSTALLER%" (
  echo [ERROR] PyInstaller not found: %PYINSTALLER%
  exit /b 1
)

if not exist "%APP_DIR%\config\config.json" (
  echo [ERROR] Missing config: %APP_DIR%\config\config.json
  exit /b 1
)

if not exist "%APP_DIR%\config\best.onnx" (
  echo [ERROR] Missing model: %APP_DIR%\config\best.onnx
  exit /b 1
)

if not exist "%APP_DIR%\assets\sounds\Pass.wav" (
  echo [ERROR] Missing PASS sound: %APP_DIR%\assets\sounds\Pass.wav
  exit /b 1
)

if not exist "%APP_DIR%\assets\sounds\Fail.wav" (
  echo [ERROR] Missing FAIL sound: %APP_DIR%\assets\sounds\Fail.wav
  exit /b 1
)

if not exist "%APP_DIR%\python_demo\mvsdk.py" (
  echo [ERROR] Missing MvSDK wrapper: %APP_DIR%\python_demo\mvsdk.py
  exit /b 1
)

pushd "%APP_DIR%"
"%PYINSTALLER%" packaging\yolo_action_detection.spec --noconfirm --clean
if errorlevel 1 (
  popd
  exit /b 1
)

if not exist "%DIST_DIR%\config" mkdir "%DIST_DIR%\config"
if not exist "%DIST_DIR%\python_demo" mkdir "%DIST_DIR%\python_demo"
if not exist "%DIST_DIR%\assets\sounds" mkdir "%DIST_DIR%\assets\sounds"

copy /Y "%APP_DIR%\config\*.onnx" "%DIST_DIR%\config\" >nul
"%PYTHON%" "%APP_DIR%\packaging\prepare_portable_config.py" "%APP_DIR%\config\config.json" "%DIST_DIR%\config.json" "%DIST_DIR%\config"
if errorlevel 1 (
  popd
  exit /b 1
)
copy /Y "%APP_DIR%\assets\sounds\Pass.wav" "%DIST_DIR%\assets\sounds\Pass.wav" >nul
copy /Y "%APP_DIR%\assets\sounds\Fail.wav" "%DIST_DIR%\assets\sounds\Fail.wav" >nul
copy /Y "%APP_DIR%\python_demo\mvsdk.py" "%DIST_DIR%\python_demo\mvsdk.py" >nul

(
  echo @echo off
  echo cd /d "%%~dp0"
  echo start "" "%%~dp0YOLOActionDetection.exe"
) > "%DIST_DIR%\start.bat"

echo.
echo [OK] Portable package created:
echo %DIST_DIR%
echo.
echo Double-click start.bat or YOLOActionDetection.exe on the target computer.
popd
endlocal
