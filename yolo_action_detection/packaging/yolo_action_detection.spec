# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller config for the portable YOLO action detection desktop app."""
from pathlib import Path

project_dir = Path(SPECPATH).resolve().parent
src_dir = project_dir / "src"
config_dir = project_dir / "config"
assets_dir = project_dir / "assets"
python_demo_dir = project_dir / "python_demo"

datas = [
    (str(config_dir / "config.json"), "."),
    (str(config_dir / "config.example.json"), "."),
    (str(config_dir / "jiaodian3.onnx"), "config"),
    (str(assets_dir / "sounds" / "Pass.wav"), "assets/sounds"),
    (str(assets_dir / "sounds" / "Fail.wav"), "assets/sounds"),
    (str(python_demo_dir / "mvsdk.py"), "python_demo"),
]

a = Analysis(
    [str(src_dir / "main.py")],
    pathex=[str(src_dir)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "cv2",
        "numpy",
        "onnxruntime",
        # 形状匹配与诊断是按需导入的，显式声明避免便携包里缺模块。
        "template_matching.shape_model",
        "template_matching.template_store",
        "template_matching.match_diagnostics",
        "qdarktheme",
        "darkdetect",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(project_dir / "packaging" / "windowed_stdio.py")],
    excludes=[
        "matplotlib",
        "pytest",
        "tkinter",
        "torch",
        "torchvision",
        "ultralytics",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="YOLOActionDetection",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="YOLOActionDetection",
)
