"""把运行环境整份放进 lstm_workbench/runtime/，做成离线便携包。

在开发机（能跑训练台的那台）上运行一次：

    .venv\\Scripts\\python.exe lstm_workbench\\make_offline_runtime.py

做出来的 runtime\\ 是一个独立的 Python 3.11：解释器 + 标准库 + 训练台用到的全部第三方包
（PyTorch CUDA 版、MediaPipe、OpenCV、ONNX Runtime、Ultralytics……）。
不是 venv：venv 里写死了本机 Python 的安装路径，复制到别的电脑就失效；这个复制到哪都能用。

之后把整个 lstm_workbench 文件夹复制到不联网的电脑，双击 启动训练台.bat 即可，不用装任何东西。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "runtime"

# 开发、测试和桌面工位用的包，训练台运行用不到，打包时不带（PySide6 一项就近 1 GB）。
SKIP_PACKAGES = {
    "pyside6", "pyside6_addons", "pyside6_essentials", "shiboken6", "qdarktheme", "pyqtdarktheme", "darkdetect",
    "pytest", "_pytest", "pytest_cov", "pytestqt", "pytest_qt", "py", "pluggy", "iniconfig", "coverage",
    "_hypothesis_pytestplugin", "hypothesis",
    "lightning", "lightning_fabric", "pytorch_lightning", "torchmetrics", "lightning_utilities",
    "pyinstaller", "pyinstaller_hooks_contrib", "altgraph", "pefile",
    "pip", "setuptools", "wheel", "_distutils_hack",
    "ultralytics",  # 由下面的 --ultralytics 源码目录替换成 8.4.102
}
SKIP_PTH = {"a1_coverage.pth", "distutils-precedence.pth"}
# 标准库里训练台用不到、体积又大的部分。
SKIP_STDLIB = {"test", "idlelib", "tkinter", "turtledemo", "lib2to3", "ensurepip", "venv", "site-packages", "__pycache__"}
SKIP_DLLS = {"_tkinter.pyd", "tcl86t.dll", "tk86t.dll"}


def _norm(name: str) -> str:
    """包目录名 / dist-info 名 → 规范名，用来和 SKIP_PACKAGES 比。"""
    base = name.split("-")[0] if name.endswith((".dist-info", ".egg-info")) else name
    base = base.removesuffix(".py").removesuffix(".pth")
    return base.lower().replace("-", "_")


def _copytree(src: Path, dst: Path, skip: set[str] = frozenset()) -> None:
    shutil.copytree(
        src, dst, dirs_exist_ok=True,
        ignore=lambda folder, names: [n for n in names if n in skip or n == "__pycache__" or n.endswith(".pyc")],
    )


def build(ultralytics_src: Path, keep_torch_cuda: bool = True) -> None:
    base = Path(sys.base_prefix)
    site = Path(sysconfig.get_paths()["purelib"])
    if not (base / "python.exe").is_file():
        raise SystemExit(f"找不到基础 Python：{base}")
    if not (ultralytics_src / "__init__.py").is_file():
        raise SystemExit(f"找不到 ultralytics 源码目录：{ultralytics_src}")

    staging = TARGET.with_name("runtime.building")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    print(f"基础 Python：{base}")
    for item in base.iterdir():
        if item.is_file() and item.suffix.lower() in {".exe", ".dll"}:
            shutil.copy2(item, staging / item.name)
    _copytree(base / "DLLs", staging / "DLLs", SKIP_DLLS)
    _copytree(base / "Lib", staging / "Lib", SKIP_STDLIB)

    packages = staging / "Lib" / "site-packages"
    packages.mkdir(parents=True, exist_ok=True)
    print(f"第三方包：{site}")
    for item in sorted(site.iterdir()):
        name = item.name
        if name == "__pycache__" or name in SKIP_PTH or _norm(name) in SKIP_PACKAGES:
            continue
        if name.startswith("__editable__") or name.endswith(".egg-link"):
            continue
        if item.is_dir():
            _copytree(item, packages / name)
        else:
            shutil.copy2(item, packages / name)
    print(f"Ultralytics 源码：{ultralytics_src}")
    _copytree(ultralytics_src, packages / "ultralytics")

    # 独立 Python 的模块搜索路径。有这个文件时 Python 不读注册表、不读 PYTHONPATH，
    # 现场电脑上装没装别的 Python 都不会串。"import site" 让 site-packages 和 .pth 生效。
    # "..\\.." 是 lstm_workbench 的上一级，这样能 import lstm_workbench。
    (staging / "python311._pth").write_text(
        "python311.zip\nDLLs\nLib\nLib\\site-packages\n..\\..\nimport site\n", encoding="utf-8"
    )
    # 用户目录下的 site-packages（%APPDATA%\\Python）不读：现场电脑上别人装的包不能混进来。
    (packages / "zz_no_user_site.pth").write_text(
        "import sys, site; site.ENABLE_USER_SITE = False; "
        "sys.path[:] = [p for p in sys.path if 'AppData' not in p or 'Roaming' not in p]\n",
        encoding="utf-8",
    )

    if TARGET.exists():
        shutil.rmtree(TARGET)
    os.replace(staging, TARGET)
    print(f"完成：{TARGET}")


def verify() -> None:
    python = TARGET / "python.exe"
    script = (
        "import sys, json\n"
        "import torch, cv2, numpy, mediapipe, onnxruntime, ultralytics\n"
        "from ultralytics.cfg import entrypoint\n"
        "import lstm_workbench.app, lstm_workbench.trainer\n"
        "print(json.dumps({'python': sys.version.split()[0], 'exe': sys.executable, 'torch': torch.__version__,"
        " 'cuda': torch.cuda.is_available(), 'ultralytics': ultralytics.__version__, 'path': sys.path[:3]}))\n"
    )
    # 刻意清空 PYTHONPATH / PYTHONHOME，模拟一台什么都没装的电脑。
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("PYTHON")}
    done = subprocess.run([str(python), "-c", script], capture_output=True, text=True, env=env,
                          encoding="utf-8", errors="replace", cwd=str(ROOT.parent))
    print(done.stdout.strip() or done.stderr.strip()[-2000:])
    if done.returncode != 0:
        raise SystemExit("自检没通过")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_src = ROOT.parent.parent / "ultralytics-main" / "ultralytics"
    parser.add_argument("--ultralytics", type=Path, default=default_src, help="Ultralytics 源码目录（含 __init__.py）")
    parser.add_argument("--verify-only", action="store_true", help="只检查已有的 runtime")
    args = parser.parse_args()
    if not args.verify_only:
        build(args.ultralytics)
    verify()


if __name__ == "__main__":
    main()
