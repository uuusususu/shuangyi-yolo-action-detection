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
set "PRESERVED_CONFIG=%APP_DIR%\build\preserved_user_config.json"
set "PRESERVED_TEMPLATE=%APP_DIR%\build\preserved_locator_template.png"
set "PRESERVED_MODEL=%APP_DIR%\build\preserved_locator_model.npz"
set "PRESERVED_LOGS=%APP_DIR%\build\preserved_logs"
set "PRESERVED_OUTPUTS=%APP_DIR%\build\preserved_outputs"
set "HAS_PRESERVED_CONFIG="
set "HAS_PRESERVED_TEMPLATE="
set "HAS_PRESERVED_MODEL="

if not exist "%PYINSTALLER%" (
  echo [ERROR] PyInstaller not found: %PYINSTALLER%
  exit /b 1
)

if not exist "%APP_DIR%\config\config.json" (
  echo [ERROR] Missing config: %APP_DIR%\config\config.json
  exit /b 1
)

if not exist "%APP_DIR%\config\*.onnx" (
  echo [ERROR] Missing ONNX model in: %APP_DIR%\config
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

if exist "%DIST_DIR%\config.json" (
  if not exist "%APP_DIR%\build" mkdir "%APP_DIR%\build"
  copy /Y "%DIST_DIR%\config.json" "%PRESERVED_CONFIG%" >nul
  if errorlevel 1 (
    echo [ERROR] Failed to preserve existing user config: %DIST_DIR%\config.json
    exit /b 1
  )
  set "HAS_PRESERVED_CONFIG=1"
  echo [OK] Existing user config preserved before rebuild.
)

if exist "%DIST_DIR%\locator_template.png" (
  if not exist "%APP_DIR%\build" mkdir "%APP_DIR%\build"
  copy /Y "%DIST_DIR%\locator_template.png" "%PRESERVED_TEMPLATE%" >nul
  if errorlevel 1 (
    echo [ERROR] Failed to preserve existing user template: %DIST_DIR%\locator_template.png
    exit /b 1
  )
  set "HAS_PRESERVED_TEMPLATE=1"
  echo [OK] Existing user template preserved before rebuild.
)

if exist "%DIST_DIR%\locator_model.npz" (
  if not exist "%APP_DIR%\build" mkdir "%APP_DIR%\build"
  copy /Y "%DIST_DIR%\locator_model.npz" "%PRESERVED_MODEL%" >nul
  if errorlevel 1 (
    echo [ERROR] Failed to preserve existing shape model: %DIST_DIR%\locator_model.npz
    exit /b 1
  )
  set "HAS_PRESERVED_MODEL=1"
  echo [OK] Existing shape model preserved before rebuild.
)

rem PyInstaller 的 COLLECT 会清空整个 dist 目录，诊断日志、生产记录和失败证据
rem 都会被一起删掉。这里先搬出去，构建完再搬回来。
if exist "%DIST_DIR%\logs" (
  if exist "%PRESERVED_LOGS%" rmdir /s /q "%PRESERVED_LOGS%"
  xcopy "%DIST_DIR%\logs" "%PRESERVED_LOGS%\" /E /I /Q /Y >nul
  if errorlevel 1 (
    echo [ERROR] Failed to preserve logs: %DIST_DIR%\logs
    exit /b 1
  )
  echo [OK] Existing logs preserved before rebuild.
)

if exist "%DIST_DIR%\outputs" (
  if exist "%PRESERVED_OUTPUTS%" rmdir /s /q "%PRESERVED_OUTPUTS%"
  xcopy "%DIST_DIR%\outputs" "%PRESERVED_OUTPUTS%\" /E /I /Q /Y >nul
  if errorlevel 1 (
    echo [ERROR] Failed to preserve outputs: %DIST_DIR%\outputs
    exit /b 1
  )
  echo [OK] Existing outputs preserved before rebuild.
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
if defined HAS_PRESERVED_CONFIG (
  copy /Y "%PRESERVED_CONFIG%" "%DIST_DIR%\config.json" >nul
  if errorlevel 1 (
    echo [ERROR] Failed to restore preserved user config: %PRESERVED_CONFIG%
    popd
    exit /b 1
  )
  echo [OK] Existing user config restored after rebuild.
) else (
  "%PYTHON%" "%APP_DIR%\packaging\prepare_portable_config.py" "%APP_DIR%\config\config.json" "%DIST_DIR%\config.json" "%DIST_DIR%\config"
  if errorlevel 1 (
    popd
    exit /b 1
  )
)

if defined HAS_PRESERVED_TEMPLATE (
  copy /Y "%PRESERVED_TEMPLATE%" "%DIST_DIR%\locator_template.png" >nul
  if errorlevel 1 (
    echo [ERROR] Failed to restore preserved user template: %PRESERVED_TEMPLATE%
    popd
    exit /b 1
  )
  echo [OK] Existing user template restored after rebuild.
)

if defined HAS_PRESERVED_MODEL (
  copy /Y "%PRESERVED_MODEL%" "%DIST_DIR%\locator_model.npz" >nul
  if errorlevel 1 (
    echo [ERROR] Failed to restore preserved shape model: %PRESERVED_MODEL%
    popd
    exit /b 1
  )
  echo [OK] Existing shape model restored after rebuild.
)

if exist "%PRESERVED_LOGS%" (
  xcopy "%PRESERVED_LOGS%" "%DIST_DIR%\logs\" /E /I /Q /Y >nul
  if errorlevel 1 (
    echo [ERROR] Failed to restore preserved logs: %PRESERVED_LOGS%
    popd
    exit /b 1
  )
  echo [OK] Logs restored after rebuild.
)

if exist "%PRESERVED_OUTPUTS%" (
  xcopy "%PRESERVED_OUTPUTS%" "%DIST_DIR%\outputs\" /E /I /Q /Y >nul
  if errorlevel 1 (
    echo [ERROR] Failed to restore preserved outputs: %PRESERVED_OUTPUTS%
    popd
    exit /b 1
  )
  echo [OK] Outputs restored after rebuild.
)

if not exist "%APP_DIR%\docs\*.docx" (
  echo [ERROR] Missing DOCX user guide in: %APP_DIR%\docs
  exit /b 1
)
copy /Y "%APP_DIR%\assets\sounds\Pass.wav" "%DIST_DIR%\assets\sounds\Pass.wav" >nul
copy /Y "%APP_DIR%\assets\sounds\Fail.wav" "%DIST_DIR%\assets\sounds\Fail.wav" >nul
copy /Y "%APP_DIR%\docs\*.docx" "%DIST_DIR%\" >nul
if errorlevel 1 (
  echo [ERROR] Failed to copy DOCX user guide.
  popd
  exit /b 1
)
copy /Y "%APP_DIR%\python_demo\mvsdk.py" "%DIST_DIR%\python_demo\mvsdk.py" >nul

(
  echo @echo off
  echo cd /d "%%~dp0"
  echo start "" "%%~dp0YOLOActionDetection.exe"
) > "%DIST_DIR%\start.bat"

(
  echo @echo off
  echo cd /d "%%~dp0"
  echo "%%~dp0YOLOActionDetection.exe" --portable-smoke-test
  echo set "DIAG_EXIT=%%ERRORLEVEL%%"
  echo echo.
  echo echo Diagnostics exit code: %%DIAG_EXIT%%
  echo echo Copy these files back for analysis:
  echo echo   %%~dp0logs\startup_diagnostics.log
  echo echo   %%~dp0logs\portable_smoke_test.log
  echo pause
  echo exit /b %%DIAG_EXIT%%
) > "%DIST_DIR%\diagnose.bat"

echo.
echo [OK] Portable package created:
echo %DIST_DIR%
echo.
echo Double-click start.bat to run, or diagnose.bat to create diagnostic logs.
popd
endlocal
