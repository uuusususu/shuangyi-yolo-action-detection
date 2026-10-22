"""手部配置改变使旧缓存过期，视频追踪状态不互相污染。"""
import numpy as np
import pytest

from lstm_workbench import feature_store, model_registry, check_clip


def test_cache_uses_current_hand_identity_and_preprocessing(monkeypatch):
    fingerprint = {"ok": True, "sha256": "first", "model_complexity": 0, "version": "test"}
    monkeypatch.setattr(model_registry, "hands_fingerprint", lambda: fingerprint.copy())
    first = feature_store.cache_key({"hands_in_training": True}, "video", 2)
    assert first["hands_sha256"] == "first"
    assert first["hands_model_complexity"] == 0
    assert first["hands_min_detection_confidence"] == 0.5
    assert first["hands_min_tracking_confidence"] == 0.5
    assert first["hands_preprocess_version"]
    fingerprint["sha256"] = "second"
    assert feature_store.cache_key({"hands_in_training": True}, "video", 2) != first


@pytest.mark.parametrize("field,value", [("model_complexity", 1), ("version", "new"), ("graph_sha256", "newgraph")])
def test_cache_invalidates_for_other_actual_hand_resources(monkeypatch, field, value):
    fingerprint = {"ok": True, "sha256": "actual", "model_complexity": 0, "version": "v1", "graph_sha256": "oldgraph"}
    monkeypatch.setattr(model_registry, "hands_fingerprint", lambda: fingerprint.copy())
    first = feature_store.cache_key({"hands_in_training": True}, "video", 2)
    fingerprint[field] = value
    assert feature_store.cache_key({"hands_in_training": True}, "video", 2) != first


@pytest.mark.parametrize("constant", ["HANDS_MIN_DETECTION_CONFIDENCE", "HANDS_MIN_TRACKING_CONFIDENCE", "HANDS_PREPROCESS_VERSION"])
def test_cache_invalidates_when_tracking_parameters_change(monkeypatch, constant):
    monkeypatch.setattr(model_registry, "hands_fingerprint", lambda: {"ok": True, "sha256": "actual", "model_complexity": 0})
    first = feature_store.cache_key({"hands_in_training": True}, "video", 2)
    monkeypatch.setattr(model_registry, constant, "changed" if constant.endswith("VERSION") else 0.3)
    assert feature_store.cache_key({"hands_in_training": True}, "video", 2) != first


@pytest.mark.parametrize("fail", [False, True])
def test_each_extracted_clip_closes_its_session(tmp_path, monkeypatch, fail):
    media = tmp_path / "media"
    media.mkdir()
    for name in ("a", "b"):
        (media / f"{name}.mp4").write_bytes(name.encode())
    sessions = []
    def backends(_settings):
        count = {"frames": 0, "closed": False}
        sessions.append(count)
        def hands(_f):
            count["frames"] += 1
            if fail:
                raise RuntimeError("graph failure")
            return []
        hands.close = lambda: count.update(closed=True)
        return lambda _f: [], hands
    monkeypatch.setattr(feature_store, "live_backends", backends)
    monkeypatch.setattr(feature_store, "iter_frames", lambda _p: iter([np.zeros((4, 4, 3), np.uint8)] * 2))
    monkeypatch.setattr(model_registry, "hands_fingerprint", lambda: {"ok": True, "sha256": "actual", "model_complexity": 0})
    annotation = {"clips": [{"id": name, "frame_count": 2} for name in ("a", "b")]}
    if fail:
        with pytest.raises(RuntimeError, match="graph failure"):
            feature_store.extract_project(tmp_path, annotation, {"yolo_classes": ["part"], "hands_in_training": True})
        assert sessions == [{"frames": 1, "closed": True}]
        assert not (tmp_path / "features" / "a.npy").exists()
    else:
        feature_store.extract_project(tmp_path, annotation, {"yolo_classes": ["part"], "hands_in_training": True})
        assert sessions == [{"frames": 2, "closed": True}] * 2


def test_check_clip_releases_tracker_when_stopped(monkeypatch, tmp_path):
    closed = []
    def hands(_f):
        return []
    hands.close = lambda: closed.append(True)
    monkeypatch.setattr(model_registry, "live_backends", lambda _s: (lambda _f: [], hands))
    monkeypatch.setattr(check_clip, "_iter_frames", lambda _p: iter([np.zeros((4, 4, 3), np.uint8)]))
    with pytest.raises(check_clip.Stopped):
        check_clip._extract_rows(tmp_path / "x.mp4", {"hands_in_training": True}, ["part"], 0.5, None, lambda: True)
    assert closed == [True]
