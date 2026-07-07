@echo off
echo ========================================
echo 动作检测应用 - 打包脚本
echo ========================================

echo.
echo [1/3] 检查虚拟环境...
set "VENV_ACTIVATE=..\.venv\Scripts\activate.bat"
if not exist "%VENV_ACTIVATE%" (
    if exist "venv\Scripts\activate.bat" (
        set "VENV_ACTIVATE=venv\Scripts\activate.bat"
    ) else (
        echo 错误: 未找到可用虚拟环境，请先准备 `.venv` 或 `opencv\venv`
        exit /b 1
    )
)

echo.
echo [2/3] 激活虚拟环境...
call "%VENV_ACTIVATE%"

echo.
echo [3/3] 开始打包...
pyinstaller build.spec --clean --noconfirm

echo.
if exist "dist\动作检测\动作检测.exe" (
    echo ========================================
    echo 打包成功!
    echo 输出文件: dist\动作检测\动作检测.exe
    echo ========================================
) else (
    echo ========================================
    echo 打包失败，请检查错误信息。
    echo ========================================
    exit /b 1
)

pause
