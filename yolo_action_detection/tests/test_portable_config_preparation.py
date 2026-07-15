from __future__ import annotations

import json
import sys
from pathlib import Path


PACKAGING_DIR = Path(__file__).resolve().parents[1] / "packaging"
if str(PACKAGING_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGING_DIR))

from prepare_portable_config import prepare_portable_config


def test_portable_config_uses_bundled_model_and_preserves_camera_sn(tmp_path):
    model_dir = tmp_path / "config"
    model_dir.mkdir()
    (model_dir / "sanrepian2.onnx").write_bytes(b"model")
    source = tmp_path / "source.json"
    destination = tmp_path / "dist" / "config.json"
    source.write_text(
        json.dumps(
            {
                "yolo_model_path": "E:/source/config/sanrepian2.onnx",
                "mvsdk_camera_sn": "SN-SELECTED",
                "category_names": ["parent", "child"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    prepare_portable_config(source, destination, model_dir)

    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved["yolo_model_path"] == "config/sanrepian2.onnx"
    assert saved["mvsdk_camera_sn"] == "SN-SELECTED"
    assert saved["category_names"] == ["parent", "child"]
