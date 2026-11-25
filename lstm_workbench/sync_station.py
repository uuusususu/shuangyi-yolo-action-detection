"""把工作台用到的工位代码复制进 lstm_workbench/station/。

工作台抽特征用的手部追踪和 YOLO 推理必须和工位完全一致（特征才能对上），
代码只维护在 yolo_action_detection/src 一处。改了那边以后在仓库根目录运行：

    .venv\Scripts\python.exe -m lstm_workbench.sync_station

再把整个 lstm_workbench 文件夹复制走，别的电脑上就不需要 yolo_action_detection 了。
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "yolo_action_detection" / "src"
TARGET = ROOT / "station"

# 工作台实际导入的模块，及它们的依赖。
MODULES = [
    "runtime_diagnostics.py",
    "hand_tracking/__init__.py",
    "hand_tracking/mediapipe_hands.py",
    "yolo_runtime/__init__.py",
    "yolo_runtime/onnx_obb_processor.py",
    "yolo_runtime/yolo_result_models.py",
]


def main() -> None:
    if not SOURCE.is_dir():
        raise SystemExit(f"找不到工位源码：{SOURCE}")
    if TARGET.exists():
        shutil.rmtree(TARGET)
    for relative in MODULES:
        source = SOURCE / relative
        target = TARGET / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (TARGET / "README.txt").write_text(
        "这里是 yolo_action_detection/src 的副本，由 sync_station.py 生成，不要手改。\n",
        encoding="utf-8",
    )
    print(f"已同步 {len(MODULES)} 个文件到 {TARGET}")


if __name__ == "__main__":
    main()
