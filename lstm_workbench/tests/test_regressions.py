"""评审里发现的三个问题 + 启动加载手部模型。每条都对应一个真实复现过的故障。

1. 时间基覆盖成和片子帧率不同的值后，已确认的验证集划分在训练时永远对不上。
2. 门禁把满分的训练判成「验证损失抬头」—— 损失降到 0.00x 后浮点抖动就够触发。
3. settings 里的 window_seconds 被接受、校验、存取，却从没参与过窗口计算。
4. 训练台启动时不加载手部模型，页面只能等人手动点「检查并保存」。
"""
from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pytest

from lstm_workbench import model_registry, service, settings as settings_module
from lstm_workbench.app import serve
from lstm_workbench.features import feature_size
from lstm_workbench.metrics import (
    LOSS_RISE_PATIENCE,
    gate_passed,
    val_loss_rose,
)
from lstm_workbench.project_store import ProjectStore
from lstm_workbench.timeline import window_frames
from lstm_workbench.trainer import train_lstm


# ----------------------------------------------------------------------
# 公共夹具
# ----------------------------------------------------------------------
def _annotations(native_fps: float) -> dict:
    """两条片子，每条两类各两段。c2 整条留作验证。"""
    clips = []
    for clip_id in ("c1", "c2"):
        segments, cursor, serial = [], 0, 0
        for class_id in ("a", "b"):
            for _ in range(2):
                segments.append({"id": f"{clip_id}-{serial}", "class_id": class_id, "start": cursor, "end": cursor + 300})
                cursor += 400
                serial += 1
        clips.append({
            "id": clip_id,
            "frame_count": cursor + 100,
            "fps": native_fps,
            "width": 640,
            "height": 480,
            "segments": segments,
        })
    return {"classes": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}], "clips": clips}


def _project(tmp_path: Path, *, native_fps: float, patch: dict | None = None):
    store = ProjectStore(tmp_path)
    store.create("p", settings_module.clean({"epochs": 2, **(patch or {})}))
    payload = _annotations(native_fps)
    store.save_annotations("p", payload)
    width = payload["clips"][0]["frame_count"]
    state = np.random.RandomState(0)
    features = {
        clip["id"]: state.rand(width, feature_size(2)).astype(np.float32)
        for clip in payload["clips"]
    }
    return store, payload, features


def _confirm_like_the_endpoint(store: ProjectStore, payload: dict) -> dict:
    """整条留出 c2。新训练不再按段确认划分。"""
    del payload
    settings = settings_module.clean({"split_mode": "clip", "val_clip_ids": ["c2"]}, store.load_settings("p"))
    store.save_settings("p", settings)
    return settings


# ----------------------------------------------------------------------
# 1. 时间基覆盖 + 已确认划分
# ----------------------------------------------------------------------
@pytest.mark.parametrize("native_fps, override", [(30.0, 0.0), (30.0, 25.0)])
def test_confirmed_split_survives_time_base_override(tmp_path, native_fps, override):
    """时间基不比片子快时，整条留出的划分可以直接开训。"""
    store, payload, features = _project(tmp_path, native_fps=native_fps, patch={"time_base_fps": override})
    settings = _confirm_like_the_endpoint(store, payload)

    overview = service.split_overview(store, "p", settings)
    assert overview["ok"] and overview["confirmed"] is True

    result = service.start_training(store, "p", features)
    assert Path(result["run"]).is_dir()


def test_a_faster_time_base_is_refused(tmp_path):
    store, payload, features = _project(tmp_path, native_fps=25.0, patch={"time_base_fps": 30.0})
    _confirm_like_the_endpoint(store, payload)
    with pytest.raises(ValueError, match="不能用重复帧补齐"):
        service.start_training(store, "p", features)


def test_resampled_window_is_computed_on_the_time_base(tmp_path):
    """窗口是时间基上的帧数；换算后的片子必须和它同口径，而不是原始帧数。"""
    store, payload, _features = _project(tmp_path, native_fps=30.0, patch={"time_base_fps": 25.0})
    prepared = service.prepare_clips(payload, settings_module.clean({}, store.load_settings("p")))

    assert prepared.base == 25.0
    assert prepared.window == window_frames(25.0) == 48
    original = prepared.originals[0]
    scaled = prepared.clips[0]
    assert scaled.fps == 25.0
    assert scaled.frame_count == len(prepared.indices["c1"]) < original.frame_count
    # 签名取自原始片子：回答「素材变没变」，和换算无关
    assert prepared.signature == service.clip_signature(prepared.originals)


def test_removing_the_validation_clip_blocks_training(tmp_path):
    """验证片子不在了，就不能再开训。"""
    store, payload, _features = _project(tmp_path, native_fps=30.0, patch={"time_base_fps": 25.0})
    settings = _confirm_like_the_endpoint(store, payload)
    assert service.split_overview(store, "p", settings)["confirmed"] is True

    payload["clips"] = [clip for clip in payload["clips"] if clip["id"] != "c2"]
    store.save_annotations("p", payload)
    assert service.split_overview(store, "p", settings)["ok"] is False


@contextmanager
def _running(tmp_path, **kwargs):
    server = serve(tmp_path, port=0, open_browser=False, **kwargs)
    _host, port = server.server_address
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def call(method, path, body=None):
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"http://127.0.0.1:{port}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    try:
        yield call
    finally:
        server.shutdown()


def test_split_confirm_endpoint_then_train_with_override(tmp_path):
    """端到端：走真实的 /split 接口确认，再训练。这是用户实际走的路径。"""
    store, payload, features = _project(tmp_path, native_fps=30.0, patch={"time_base_fps": 25.0})
    with _running(tmp_path) as call:
        status, body = call("POST", "/api/projects/p/split", {
            "action": "participate", "split_mode": "clip", "val_clip_ids": ["c2"], "train_excluded": [],
        })
        assert status == 200 and body["split"]["ok"]
        assert body["split"]["confirmed"] is True

        status, project = call("GET", "/api/projects/p")
        assert project["split"]["confirmed"] is True
        assert project["blockers"] == []

    result = service.start_training(store, "p", features)
    assert Path(result["run"]).is_dir()


# ----------------------------------------------------------------------
# 2. 门禁：容差 + 连续抬头
# ----------------------------------------------------------------------
def test_float_jitter_at_near_zero_loss_is_not_a_rise():
    assert val_loss_rose(0.0040, 0.0030, 0.0040, 0.0042) is False
    assert val_loss_rose(1e-4, 5e-5, 1e-4, 1.1e-4) is False


def test_a_real_rise_is_still_a_rise():
    assert val_loss_rose(0.50, 0.30, 0.45, 0.60) is True


def test_a_small_blip_on_a_realistic_loss_is_tolerated():
    assert val_loss_rose(0.50, 0.30, 0.45, 0.46) is False


def test_no_previous_epoch_is_never_a_rise():
    assert val_loss_rose(None, 0.3, None, 0.9) is False


def test_train_loss_not_falling_is_not_overfitting():
    assert val_loss_rose(0.30, 0.31, 0.40, 0.90) is False


_GOOD = {"macro_f1": 0.95, "min_recall": 0.9, "idle_false_alarm": 0.0}


def _gate(**kwargs):
    defaults = dict(
        train_loss=0.2, previous_train_loss=0.4, val_loss=0.3, previous_val_loss=0.5,
        missing_val_class=False,
    )
    return gate_passed(_GOOD, **{**defaults, **kwargs})


def test_gate_does_not_reject_a_saved_checkpoint():
    """0.90 / 0.85 / 0.10 和验证损失抬头都不再决定这一轮能不能用。"""
    assert _gate(rising_epochs=0) == (True, "已保存")
    assert _gate(rising_epochs=10) == (True, "已保存")
    assert _gate(train_loss=0.3, previous_train_loss=0.5, val_loss=0.6, previous_val_loss=0.45)[0] is True
    low = dict(_GOOD)
    low["macro_f1"] = 0.2
    low["min_recall"] = 0.0
    low["idle_false_alarm"] = 1.0
    passed, reason = gate_passed(
        low, train_loss=0.01, previous_train_loss=0.2, val_loss=3.0, previous_val_loss=1.0,
        missing_val_class=True, rising_epochs=10,
    )
    assert passed is True and reason == "已保存"


def _synthetic(noise: float, epochs: int, seed: int = 1) -> dict:
    window, features, classes = 12, 40, ["idle", "a", "b"]
    rng = np.random.RandomState(seed)

    def make(label, count):
        rows = []
        for _ in range(count):
            x = rng.normal(0, noise, (window, features)).astype(np.float32)
            x[:, label * 8:(label + 1) * 8] += 0.6
            rows.append(x)
        return rows

    train_x, train_y = [], []
    for label in range(3):
        rows = make(label, 60)
        train_x += rows
        train_y += [label] * len(rows)
    val_groups = []
    for label, class_id in enumerate(classes):
        for k in range(3):
            rows = make(label, 8)
            val_groups.append((class_id, rows, [(i * 8, i * 8 + window) for i in range(len(rows))], f"c{k}"))
    import tempfile

    return train_lstm(
        train_x, train_y, val_groups, classes, Path(tempfile.mkdtemp()),
        epochs=epochs, window=window, stride=8,
    )


def test_a_perfect_run_is_not_rejected_for_loss_jitter():
    """修之前：F1 = 1.000 的 150 轮训练被判「验证损失在训练损失下降时抬头」。"""
    report = _synthetic(noise=0.8, epochs=150)
    assert report["metrics"]["macro_f1"] == pytest.approx(1.0)
    assert report["usable"] is True, report["reason"]


# ----------------------------------------------------------------------
# 3. window_seconds 真的生效
# ----------------------------------------------------------------------
def test_window_frames_honours_the_given_seconds():
    assert window_frames(25.0) == 48
    assert window_frames(25.0, 2.0) == 50
    assert window_frames(10.0, 3.0) == 30
    with pytest.raises(ValueError):
        window_frames(25.0, 0.0)


def test_window_seconds_setting_drives_split_and_training_contract(tmp_path):
    store, payload, features = _project(tmp_path, native_fps=10.0, patch={"window_seconds": 3.0})
    settings = _confirm_like_the_endpoint(store, payload)

    overview = service.split_overview(store, "p", settings)
    assert overview["window_seconds"] == 3.0
    assert overview["window"] == 30, "10fps × 3 秒；以前永远是 1.92 秒 = 19 帧"

    result = service.start_training(store, "p", features)
    contract = json.loads((Path(result["run"]) / "contract.json").read_text(encoding="utf-8"))
    assert contract["window_seconds"] == 3.0
    assert contract["window"] == 30


def test_default_window_is_unchanged_when_the_setting_is_untouched(tmp_path):
    store, payload, _features = _project(tmp_path, native_fps=25.0)
    overview = service.split_overview(store, "p", settings_module.clean({}, store.load_settings("p")))
    assert overview["window_seconds"] == 1.92
    assert overview["window"] == 48


def test_changing_the_window_discards_a_confirmed_split():
    """窗口一变，哪些段够长、哪些是空闲段都跟着变，旧的确认不能沿用。"""
    confirmed = settings_module.clean({}, {
        **settings_module.DEFAULT_SETTINGS,
        "split_train": [["c1", "s0"]], "split_val": [["c1", "s1"]], "split_for": "c1:100:30",
    })
    assert confirmed["split_for"] == "c1:100:30"

    same = settings_module.clean({"window_seconds": 1.92}, confirmed)
    assert same["split_for"] == "c1:100:30", "值没变就不该作废"

    changed = settings_module.clean({"window_seconds": 3.0}, confirmed)
    assert changed["split_train"] == [] and changed["split_val"] == [] and changed["split_for"] == ""


# ----------------------------------------------------------------------
# 4. 启动加载手部模型
# ----------------------------------------------------------------------
def _wait_for_hands(timeout: float = 90.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = model_registry.hands_status()
        if status.get("state") not in {"loading", "idle"}:
            return status
        time.sleep(0.2)
    raise AssertionError(f"手部模型预热超时：{model_registry.hands_status()}")


@pytest.fixture()
def clean_warmup():
    model_registry.reset_hands_warmup()
    yield
    model_registry.reset_hands_warmup()


def test_warmup_loads_the_hand_model_and_reports_what_it_loaded(clean_warmup):
    assert model_registry.hands_status()["state"] == "idle"
    model_registry.start_hands_warmup()
    status = _wait_for_hands()

    assert status["state"] == "ready", status.get("error")
    assert status["ok"] is True
    assert len(status["sha256"]) == 64
    assert status["source"] == "library", "库齐就必须从库里读，而不是悄悄退回内置"
    assert any(item["name"].endswith(".tflite") for item in status["paths"])
    assert status["seconds"] >= 0.0
    assert model_registry.hands_tracker().available


def test_warmup_is_idempotent(clean_warmup):
    model_registry.start_hands_warmup()
    first = _wait_for_hands()
    model_registry.start_hands_warmup()
    assert model_registry.hands_status() == first, "已经加载好就不该重来"


def test_warmup_failure_is_reported_not_raised(clean_warmup, monkeypatch):
    def boom():
        raise RuntimeError("tflite 打不开")

    monkeypatch.setattr(model_registry, "probe_hands", boom)
    model_registry.start_hands_warmup()          # 不能抛
    status = _wait_for_hands()
    assert status["state"] == "failed"
    assert "tflite 打不开" in status["error"]


def test_manual_check_updates_the_cached_status(clean_warmup, monkeypatch):
    """预热失败过一次，之后手动检查通过了，页面不能还显示失败。"""
    monkeypatch.setattr(model_registry, "probe_hands", lambda: {"ok": False, "error": "第一次失败"})
    model_registry.start_hands_warmup()
    assert _wait_for_hands()["state"] == "failed"

    model_registry.record_hands_probe({"ok": True, "sha256": "x" * 64, "source": "library"})
    assert model_registry.hands_status()["state"] == "ready"


def test_models_endpoint_exposes_the_hand_load_state(tmp_path, clean_warmup):
    with _running(tmp_path, warm_models=True) as call:
        status = {}
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            _code, body = call("GET", "/api/models")
            status = body["hands"]["load"]
            if status["state"] not in {"loading", "idle"}:
                break
            time.sleep(0.3)
        assert status["state"] == "ready", status
        assert len(status["sha256"]) == 64


def test_server_does_not_warm_models_unless_asked(tmp_path, clean_warmup):
    """测试里反复起服务不该每次都去构造 MediaPipe 图。"""
    with _running(tmp_path) as call:
        _code, body = call("GET", "/api/models")
    assert body["hands"]["load"]["state"] == "idle"


def test_cuda_probe_failure_uses_cpu(monkeypatch):
    import torch
    from lstm_workbench.trainer import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fail(*_args, **_kwargs):
        raise RuntimeError("驱动没装好")

    monkeypatch.setattr(torch, "zeros", fail)
    device, info = resolve_device()
    assert device.type == "cpu"
    assert info["device"] == "CPU"
    assert info["device_kind"] == "cpu"
    assert "驱动没装好" in info["device_note"]


def test_training_records_the_settings_it_actually_used(tmp_path, monkeypatch):
    """页面要能核对这一轮读进去的五项。权重必须是 CPU 张量，工位没有显卡也能加载。"""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    window, features = 4, 8
    train_x = [np.zeros((window, features), np.float32) for _ in range(4)]
    train_x[2][:] = 1
    train_x[3][:] = 1
    train_y = [0, 0, 1, 1]
    val_groups = [
        ("idle", [train_x[0]], [(0, window)], "c"),
        ("a", [train_x[2]], [(0, window)], "c"),
    ]
    report = train_lstm(
        train_x, train_y, val_groups, ["idle", "a"], tmp_path,
        epochs=1, learning_rate=0.02, dropout=0.1, window=window, stride=2, window_seconds=1.5,
    )
    recipe = report["recipe"]
    assert recipe["epochs"] == 1
    assert recipe["learning_rate"] == 0.02
    assert recipe["dropout"] == 0.1
    assert recipe["window_seconds"] == 1.5
    assert recipe["window"] == 4
    assert recipe["stride"] == 2
    assert recipe["device"] == "CPU"
    assert recipe["device_note"] == "没有可用显卡"
    state = torch.load(tmp_path / "best_lstm.pt", map_location="cpu", weights_only=True)
    assert all(tensor.device.type == "cpu" for tensor in state.values())
    contract = json.loads((tmp_path / "contract.json").read_text(encoding="utf-8"))
    assert contract["learning_rate"] == 0.02
    assert contract["epochs"] == 1
    assert contract["device_kind"] == "cpu"
