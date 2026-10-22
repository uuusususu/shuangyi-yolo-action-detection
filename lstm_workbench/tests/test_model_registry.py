"""模型指纹和加载检查。"""
from __future__ import annotations

from pathlib import Path

import pytest

from lstm_workbench import model_registry


def test_file_sha256_is_stable_and_matches_a_known_value(tmp_path):
    target = tmp_path / "m.bin"
    target.write_bytes(b"abc")
    digest = model_registry.file_sha256(target)
    assert digest == model_registry.file_sha256(target)
    assert digest == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_yolo_providers_prefer_cuda_when_the_runtime_has_it(monkeypatch):
    import onnxruntime as ort

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider", "CUDAExecutionProvider"])
    assert model_registry.yolo_providers() == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_yolo_providers_use_directml_when_cuda_is_absent(monkeypatch):
    import onnxruntime as ort

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider", "DmlExecutionProvider"])
    assert model_registry.yolo_providers() == ["DmlExecutionProvider", "CPUExecutionProvider"]
    blank = model_registry.probe_yolo("", 0.8)
    assert blank["ok"] is False
    assert "路径" in blank["error"]

    missing = model_registry.probe_yolo(str(Path("nowhere") / "n.onnx"), 0.8)
    assert missing["ok"] is False
    assert "不存在" in missing["error"]


def test_hands_fingerprint_covers_the_files_it_actually_loads():
    got = model_registry.hands_fingerprint()
    assert got["ok"] is True, got.get("error")
    assert len(got["sha256"]) == 64
    assert got["model_complexity"] == model_registry.HANDS_MODEL_COMPLEXITY
    assert set(got["files"]) == {"palm_detection_lite.tflite", "hand_landmark_lite.tflite"}, "工位用 lite，指纹也要按 lite"


def test_hands_probe_can_actually_construct_the_tracker():
    """site-packages 路径含中文时 MediaPipe 打不开，工位那套接入已经处理过。"""
    got = model_registry.probe_hands()
    assert got["ok"] is True, got.get("error")
    assert model_registry.hands_tracker().available


def test_live_backends_refuses_without_a_yolo():
    with pytest.raises(FileNotFoundError, match="模型"):
        model_registry.live_backends({"yolo_path": ""})
