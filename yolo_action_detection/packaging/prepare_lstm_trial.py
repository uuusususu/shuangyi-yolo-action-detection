"""给隔离构建目录准备真实模型和试用配置；拒绝覆盖已有现场配置。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(base: Path, source_config: Path, bundle: Path, yolo: Path) -> None:
    if not (base / "YOLOActionDetection.exe").is_file():
        raise ValueError("目录中没有构建后的可执行程序")
    if (base / "config.json").exists() or (base / "models" / "action").exists():
        raise ValueError("试用目录已有配置或动作模型，不覆盖")
    contract = json.loads((bundle / "contract.json").read_text(encoding="utf-8"))
    manifest = json.loads((bundle / "runtime_manifest.json").read_text(encoding="utf-8"))
    if sha256(yolo) != contract["yolo_sha256"]:
        raise ValueError("试用 YOLO 与动作契约指纹不一致")
    if sha256(bundle / "contract.json") != manifest["contract_sha256"] or sha256(bundle / "best_lstm.pt") != manifest["weights_sha256"]:
        raise ValueError("部署模型包指纹不一致")
    names = [item["name"] for item in manifest["action_classes"] if item["id"] != "idle"]
    payload = json.loads(source_config.read_text(encoding="utf-8"))
    payload.update({
        "yolo_model_path": f"config/{yolo.name}",
        "yolo_conf_threshold": contract["yolo_conf_threshold"],
        "category_names": names + [""] * max(0, 6 - len(names)),
        "category_counts": [1] * max(6, len(names)),
        "roi_inspection_enabled": False, "roi_template_enabled": False,
        "action_order_constraint_enabled": True, "lstm_bundle_path": "models/action",
        "fail_evidence_enabled": True,
    })
    (base / "config").mkdir(exist_ok=True)
    shutil.copy2(yolo, base / "config" / yolo.name)
    (base / "models").mkdir(exist_ok=True)
    shutil.copytree(bundle, base / "models" / "action")
    # 所有候选文件已复制后，最后写新的试用配置；从不写源配置。
    (base / "config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (base / "start.bat").write_text('@echo off\r\ncd /d "%~dp0"\r\nstart "" "%~dp0YOLOActionDetection.exe"\r\n', encoding="ascii")
    (base / "diagnose.bat").write_text('@echo off\r\ncd /d "%~dp0"\r\n"%~dp0YOLOActionDetection.exe" --portable-smoke-test\r\npause\r\n', encoding="ascii")
    print(f"已准备独立试用目录：{base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.base, args.config, args.bundle, args.yolo)
