from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


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
                "config_schema_version": 4,
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


def test_portable_config_preserves_v4_roi_and_timing_fields(tmp_path):
    model_dir = tmp_path / "config"
    model_dir.mkdir()
    (model_dir / "site.onnx").write_bytes(b"model")
    source = tmp_path / "source.json"
    destination = tmp_path / "dist" / "config.json"
    payload = {
        "config_schema_version": 4,
        "yolo_model_path": "E:/models/site.onnx",
        "roi_inspection_rect": {"x": 0.1, "y": 0.2, "width": 0.8, "height": 0.7},
        "roi_template_rect": {"x": 0.2, "y": 0.3, "width": 0.2, "height": 0.2},
        "roi_template_search_rect": {"x": 0.1, "y": 0.2, "width": 0.6, "height": 0.6},
        "roi_template_angle_step": 2.0,
        "action_pass_stable_frames": 3,
        "action_ng_stable_frames": 12,
        "round_cooldown_seconds": 4.0,
    }
    source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    prepare_portable_config(source, destination, model_dir)

    saved = json.loads(destination.read_text(encoding="utf-8"))
    for key, value in payload.items():
        if key != "yolo_model_path":
            assert saved[key] == value


def test_portable_config_rejects_legacy_schema_instead_of_migrating(tmp_path):
    model_dir = tmp_path / "config"
    model_dir.mkdir()
    (model_dir / "best.onnx").write_bytes(b"model")
    source = tmp_path / "legacy.json"
    source.write_text(
        json.dumps(
            {
                "yolo_model_path": "best.onnx",
                "first_category_region_check_enabled": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="config_schema_version"):
        prepare_portable_config(source, tmp_path / "dist" / "config.json", model_dir)


def test_build_script_preserves_existing_portable_user_config():
    script = (PACKAGING_DIR / "build_yolo.bat").read_text(encoding="utf-8")
    backup = 'copy /Y "%DIST_DIR%\\config.json" "%PRESERVED_CONFIG%"'
    build = '"%PYINSTALLER%" packaging\\yolo_action_detection.spec --noconfirm --clean'
    restore = 'copy /Y "%PRESERVED_CONFIG%" "%DIST_DIR%\\config.json"'

    assert 'set "PRESERVED_CONFIG=' in script
    assert 'if defined HAS_PRESERVED_CONFIG' in script
    assert backup in script
    assert restore in script
    assert script.index(backup) < script.index(build) < script.index(restore)


def test_build_script_preserves_existing_user_template():
    script = (PACKAGING_DIR / "build_yolo.bat").read_text(encoding="utf-8")
    backup = 'copy /Y "%DIST_DIR%\\locator_template.png" "%PRESERVED_TEMPLATE%"'
    build = '"%PYINSTALLER%" packaging\\yolo_action_detection.spec --noconfirm --clean'
    restore = 'copy /Y "%PRESERVED_TEMPLATE%" "%DIST_DIR%\\locator_template.png"'

    assert 'set "PRESERVED_TEMPLATE=' in script
    assert 'if defined HAS_PRESERVED_TEMPLATE' in script
    assert backup in script
    assert restore in script
    assert script.index(backup) < script.index(build) < script.index(restore)
