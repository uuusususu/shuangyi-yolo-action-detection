# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置，面向当前 PySide6 + MediaPipe task 运行时。"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

project_dir = Path.cwd()
src_dir = project_dir / "src"

datas = [
    (str(project_dir / "config.json"), "."),
    (str(project_dir / "config"), "config"),
    (str(project_dir / "Pass.wav"), "."),
    (str(project_dir / "python_demo"), "python_demo"),
    (str(project_dir.parent / "mediapipe_quick_task" / "exports"), "mediapipe_quick_task/exports"),
]

binaries = []
hiddenimports = [
    "config",
    "state",
    "models",
    "mediapipe_frame_processor",
    "mediapipe_gesture_runtime",
    "camera_worker",
    "gesture_trigger_cycle",
    "mvsdk_camera",
    "mv_mvsdk",
    "offline_validation",
    "ui",
    "ui.main_window",
    "ui.config_page",
    "ui.control_panel",
]

for package_name in ("PySide6", "cv2", "numpy", "mediapipe"):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package_name)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

a = Analysis(
    [str(src_dir / "main.py")],
    pathex=[str(src_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "_pytest", "py", "hypothesis", "IPython", "tkinter", "ultralytics", "torch", "torchvision", "onnxruntime"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="动作检测",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="动作检测",
)
