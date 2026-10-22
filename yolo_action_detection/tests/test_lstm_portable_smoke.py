"""动作模式的便携诊断必须真正加载模型并前向，失败不能报告成功。"""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

for entry in (Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[1] / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from config import ConfigManager
import main
from lstm_action import bundle as bundles
from lstm_action import stream
from lstm_workbench import model_registry


def _environment(tmp_path, *, action=True):
    model = tmp_path / "config" / "parts.onnx"
    model.parent.mkdir()
    model.write_bytes(b"parts")
    for sound in ("Pass", "Fail"):
        target = tmp_path / "assets" / "sounds" / f"{sound}.wav"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"sound")
    config = ConfigManager()
    config.yolo_model_path = "config/parts.onnx"
    config.yolo_conf_threshold = 0.6
    config.category_names = ["包装"] if action else ["零件"]
    config.category_counts = [1]
    config.action_order_constraint_enabled = action
    config.lstm_bundle_path = "models/action"
    path = tmp_path / "config.json"
    config.save(path)
    processor = SimpleNamespace(get_class_names=lambda: {0: "零件"})
    return path, processor


def test_action_smoke_loads_actual_candidate_and_runs_forward(tmp_path, monkeypatch):
    path, processor = _environment(tmp_path)
    calls = []

    def load(directory, **kwargs):
        calls.append((directory, kwargs))
        return SimpleNamespace(weights_sha256="weights", contract={"hands_sha256": "hands"}, time_base_fps=10)

    class Session:
        def __init__(self, bundle, confirm_seconds=None):
            calls.append(bundle.weights_sha256)

        def observe_frame(self, **kwargs):
            calls.append(kwargs)
            return {"fault": "", "phase": "待就绪"}

    monkeypatch.setattr(bundles, "load_runtime_bundle", load)
    monkeypatch.setattr(stream, "TimelineSession", Session)
    monkeypatch.setattr(model_registry, "create_hands_tracker", lambda: SimpleNamespace(
        available=True, error="", process=lambda _frame: [], close=lambda: None,
    ))
    monkeypatch.setattr(main, "require_theme_module", lambda: None)
    assert main.run_portable_smoke_check(path, tmp_path, processor_loader=lambda *_: processor) == 0
    assert calls[0][0] == tmp_path / "models" / "action"
    assert calls[0][1]["yolo_path"] == str(tmp_path / "config" / "parts.onnx")
    assert calls[0][1]["yolo_conf"] == 0.6
    assert calls[0][1]["step_counts"] == [1]
    assert "hands_sha" not in calls[0][1]
    assert calls[2]["source_frame"] == 0
    assert "torch_forward=ok" in (tmp_path / "logs" / "portable_smoke_test.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("failure", ["缺少模型包", "手部指纹不一致", "真实前向失败"])
def test_action_smoke_load_failure_is_nonzero(tmp_path, monkeypatch, failure):
    path, processor = _environment(tmp_path)

    def fail(*_args, **_kwargs):
        raise RuntimeError(failure)

    monkeypatch.setattr(bundles, "load_runtime_bundle", fail)
    assert main.run_portable_smoke_check(path, tmp_path, processor_loader=lambda *_: processor) == 7
    text = (tmp_path / "logs" / "portable_smoke_test.log").read_text(encoding="utf-8")
    assert failure in text
    assert "[portable-smoke] ok" not in text


def test_yolo_smoke_does_not_require_action_bundle(tmp_path, monkeypatch):
    path, processor = _environment(tmp_path, action=False)
    monkeypatch.setattr(bundles, "load_runtime_bundle", lambda *_args, **_kwargs: pytest.fail("纯 YOLO 不应加载动作模型"))
    monkeypatch.setattr(main, "require_theme_module", lambda: None)
    assert main.run_portable_smoke_check(path, tmp_path, processor_loader=lambda *_: processor) == 0
