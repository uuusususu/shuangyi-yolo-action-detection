"""特征缓存：键没变就复用，键变了就重抽。"""
from __future__ import annotations

import numpy as np

from lstm_workbench import feature_store
from lstm_workbench.project_store import ProjectStore


def _settings(**over):
    base = {
        "yolo_conf_threshold": 0.8,
        "yolo_path": "x.onnx",
        "yolo_sha256": "aaa",
        "yolo_classes": ["part"],
        "epochs": 3,
        "hands_in_training": True,
    }
    return {**base, **over}


def _project(tmp_path, settings):
    store = ProjectStore(tmp_path)
    store.create("job", settings)
    (tmp_path / "job" / "media" / "a.mp4").write_bytes(b"video")
    annotations = {
        "classes": [],
        "clips": [{"id": "a", "frame_count": 2, "fps": 10, "width": 64, "height": 64, "segments": []}],
    }
    store.save_annotations("job", annotations)
    return store, tmp_path / "job", annotations


def _fake(monkeypatch, counter, detections=True, hand_points=True):
    frame = np.zeros((8, 16, 3), dtype=np.uint8)

    def detect(_frame):
        counter.append(1)
        if not detections:
            return []
        return [{"label": "part", "conf": 0.9, "box": (0, 0, 8, 8)}]

    def hands(_frame):
        if not hand_points:
            return []
        return [[(4, 2, 0.1)] + [(0, 0, 0)] * 20]

    monkeypatch.setattr(feature_store, "live_backends", lambda settings: (detect, hands))
    monkeypatch.setattr(feature_store, "iter_frames", lambda path: iter([frame, frame]))


def test_second_run_reuses_and_changing_yolo_reextracts(tmp_path, monkeypatch):
    settings = _settings()
    _store, folder, annotations = _project(tmp_path, settings)
    counter = []
    _fake(monkeypatch, counter)

    _arrays, first = feature_store.extract_project(folder, annotations, settings)
    assert first["extracted"] == 1
    assert first["reused"] == 0
    assert first["clips"][0]["feature_length"] == 131
    assert first["no_hand_ratio"] == 0.0
    assert first["no_part_ratio"] == 0.0
    runs = len(counter)

    _arrays, second = feature_store.extract_project(folder, annotations, settings)
    assert second["reused"] == 1
    assert second["extracted"] == 0
    assert len(counter) == runs, "键没变不该再跑一遍检测"

    _arrays, third = feature_store.extract_project(folder, annotations, _settings(yolo_sha256="bbb"))
    assert third["extracted"] == 1, "换了 YOLO 就要重抽"
    assert len(counter) > runs


def test_confidence_and_class_names_and_media_all_invalidate(tmp_path, monkeypatch):
    settings = _settings()
    _store, folder, annotations = _project(tmp_path, settings)
    counter = []
    _fake(monkeypatch, counter)
    feature_store.extract_project(folder, annotations, settings)

    for changed in (
        _settings(yolo_conf_threshold=0.5),
        _settings(yolo_classes=["part", "screw"]),
    ):
        _arrays, report = feature_store.extract_project(folder, annotations, changed)
        assert report["extracted"] == 1, f"这些变了要重抽：{changed}"

    (folder / "media" / "a.mp4").write_bytes(b"video-changed")
    _arrays, report = feature_store.extract_project(folder, annotations, settings)
    assert report["extracted"] == 1, "素材换了要重抽"


def test_missing_hands_and_parts_are_counted(tmp_path, monkeypatch):
    settings = _settings()
    _store, folder, annotations = _project(tmp_path, settings)
    _fake(monkeypatch, [], detections=False, hand_points=False)

    _arrays, report = feature_store.extract_project(folder, annotations, settings)
    assert report["no_hand_ratio"] == 1.0
    assert report["no_part_ratio"] == 1.0
    assert report["frames"] == 2


def test_summary_marks_fresh_stale_and_missing(tmp_path, monkeypatch):
    settings = _settings()
    store, folder, annotations = _project(tmp_path, settings)
    _fake(monkeypatch, [])

    before = feature_store.summary(folder, annotations, settings)
    assert before["stale"] == 1 and before["fresh"] == 0

    feature_store.extract_project(folder, annotations, settings)
    after = feature_store.summary(folder, annotations, settings)
    assert after["fresh"] == 1 and after["stale"] == 0

    store.save_annotations("job", {"classes": [], "clips": [
        *annotations["clips"],
        {"id": "gone", "frame_count": 2, "fps": 10, "width": 64, "height": 64, "segments": []},
    ]})
    gone = feature_store.summary(folder, store.load_annotations("job"), settings)
    assert gone["missing"] == 1
    assert "缺少素材" in gone["clips"][1]["note"]


def test_mixed_feature_lengths_are_refused(tmp_path, monkeypatch):
    """零件类名改了会让 F 变。两条片子长度不一致就不能堆成一个训练集。"""
    settings = _settings()
    _store, folder, annotations = _project(tmp_path, settings)
    annotations["clips"].append(
        {"id": "b", "frame_count": 2, "fps": 10, "width": 64, "height": 64, "segments": []}
    )
    (folder / "media" / "b.mp4").write_bytes(b"video-b")
    frame = np.zeros((8, 16, 3), dtype=np.uint8)
    lengths = [131, 136]

    def fake_rows(frames, detect, hands, names, conf, on_frame=None, frame_count=None, include_hands=True):
        frames = list(frames)
        length = lengths.pop(0)
        return (
            np.zeros((len(frames), length), dtype=np.float32),
            {"frames": len(frames), "feature_length": length, "no_hand_frames": 0,
             "no_hand_ratio": 0.0, "no_part_frames": 0, "no_part_ratio": 0.0},
        )

    monkeypatch.setattr(feature_store, "live_backends", lambda settings: (lambda _f: [], lambda _f: []))
    monkeypatch.setattr(feature_store, "iter_frames", lambda path: iter([frame, frame]))
    monkeypatch.setattr(feature_store, "rows_and_stats_from_frames", fake_rows)

    try:
        feature_store.extract_project(folder, annotations, settings)
    except ValueError as exc:
        assert "特征长度不一致" in str(exc)
    else:
        raise AssertionError("长度不一致应该拒绝，不能堆成训练集")


def test_extract_only_the_checked_clip_and_records_the_file(tmp_path, monkeypatch):
    settings = _settings()
    _store, folder, annotations = _project(tmp_path, settings)
    annotations["clips"].append(
        {"id": "b", "frame_count": 2, "fps": 10, "width": 64, "height": 64, "segments": []}
    )
    (folder / "media" / "b.mp4").write_bytes(b"video-b")
    counter = []
    _fake(monkeypatch, counter)

    arrays, report = feature_store.extract_project(folder, annotations, settings, clip_ids=["b"])
    assert list(arrays) == ["b"]
    assert report["extracted"] == 1
    assert not (folder / "features" / "a.npy").exists()
    assert (folder / "features" / "b.npy").is_file()
    assert len(counter) == 2, "只抽勾选的那一条，两条帧各检测一次"

    info = feature_store.summary(folder, annotations, settings)
    row = next(item for item in info["clips"] if item["clip_id"] == "b")
    assert row["file"] == "features/b.npy"
    assert row["bytes"] > 0
    assert row["frames"] == 2
    assert row["feature_length"] == 131
    assert row["extracted_at"]
