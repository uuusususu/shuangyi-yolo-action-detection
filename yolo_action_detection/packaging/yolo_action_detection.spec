# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller config for the portable YOLO action detection desktop app."""
from pathlib import Path

project_dir = Path(SPECPATH).resolve().parent
repo_dir = project_dir.parent
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

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# 资源收集失败应终止构建，不能交付运行时才发现没有手部计算图的程序。
datas += collect_data_files("mediapipe")
binaries = collect_dynamic_libs("mediapipe")
# 不递归导入 MediaPipe 的任务、训练与测试命名空间。Analysis 跟踪真正使用的
# Solutions 接口，原生扩展与 protobuf 模块交给 PyInstaller 官方 hooks 收集。
mediapipe_imports = [
    "mediapipe.python._framework_bindings",
    "mediapipe.python.solutions.hands",
    "mediapipe.python.solutions.hands_connections",
    "mediapipe.python.solution_base",
]
# 工作台若实际使用本地模型库，便携程序保留同一资源来源与指纹。
hands_library = repo_dir / "lstm_workbench" / "models" / "hands"
if hands_library.is_dir():
    for resource in hands_library.rglob("*"):
        if resource.is_file() and resource.suffix in {".tflite", ".binarypb", ".txt"}:
            target = Path("lstm_workbench/models/hands") / resource.parent.relative_to(hands_library)
            datas.append((str(resource), str(target)))

a = Analysis(
    [str(src_dir / "main.py")],
    pathex=[str(src_dir), str(repo_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=mediapipe_imports + [
        "cv2",
        "numpy",
        "onnxruntime",
        # 形状匹配与诊断是按需导入的，显式声明避免便携包里缺模块。
        "template_matching.shape_model",
        "template_matching.template_store",
        "template_matching.match_diagnostics",
        "hand_tracking.mediapipe_hands",
        "lstm_action.engine",
        "lstm_action.bundle",
        "lstm_action.stream",
        "lstm_workbench.features",
        "lstm_workbench.model",
        "lstm_workbench.model_registry",
        "torch",
        "mediapipe",
        "mediapipe.python.solutions.hands",
        "mediapipe.python.solutions.hands_connections",
        "mediapipe.python.solution_base",
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
        # matplotlib is stubbed in hand_tracking because mediapipe drawing_utils imports it.
        "matplotlib",
        "pytest",
        "tkinter",
        "torchvision",
        "ultralytics",
    ],
    noarchive=False,
)

# 产线只用 CPU。显卡版 PyTorch 会把 CUDA、cuDNN、cuBLAS 一并收进来，运行时用不到。
_CUDA_LIBRARY_MARKERS = (
    "cuda", "cudnn", "cublas", "cufft", "cusparse", "cusolver", "curand",
    "nvrtc", "nvjitlink", "nvperf", "cupti", "cudart",
)


def _cpu_only_binary(entry) -> bool:
    name = Path(entry[0]).name.lower()
    return not any(marker in name for marker in _CUDA_LIBRARY_MARKERS)


a.binaries = [entry for entry in a.binaries if _cpu_only_binary(entry)]

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
