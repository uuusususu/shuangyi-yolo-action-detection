"""工程目录和训练写入。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lstm_workbench.project_store import ProjectStore
from lstm_workbench.service import (
    annotations_to_clips,
    clip_signature,
    prepare_clips,
    resolve_split,
    start_training,
    time_base,
)
from lstm_workbench.timeline import window_frames


def _settings():
    return {
        "yolo_conf_threshold": 0.8,
        "yolo_path": "",
        "yolo_classes": ["part"],
        "epochs": 3,
    }


def _clip(clip_id: str, step_end: int):
    return {
        "id": clip_id,
        "annotation_complete": True,
        "frame_count": 160,
        "fps": 10,
        "width": 64,
        "height": 64,
        "segments": [{"id": clip_id + "-s", "class_id": "step1", "start": 0, "end": step_end}],
    }


def test_an_unchecked_clip_does_not_block_the_time_base():
    from lstm_workbench.service import prepare_clips

    payload = {
        "classes": [],
        "clips": [
            {"id": "fast", "frame_count": 100, "fps": 10.277, "segments": []},
            {"id": "slow", "frame_count": 80, "fps": 6.25, "segments": []},
        ],
    }
    prepared = prepare_clips(payload, {"train_excluded": ["slow"], "time_base_fps": 0.0, "window_seconds": 1.92})
    assert [clip.clip_id for clip in prepared.originals] == ["fast"]
    assert prepared.base == pytest.approx(10.277)


def test_different_frame_rates_share_the_slower_clock():
    """快慢片子放进同一个数据集。时间基用较慢的那档，快片子按秒抽帧。"""
    from lstm_workbench.timeline import ClipAnnotation

    near = [
        ClipAnnotation("a", 10, 6.24808, 8, 8, []),
        ClipAnnotation("b", 10, 6.26564, 8, 8, []),
        ClipAnnotation("c", 10, 6.29459, 8, 8, []),
    ]
    assert time_base(near) == pytest.approx(6.26564)
    mixed = near + [ClipAnnotation("d", 10, 10.2773, 8, 8, [])]
    assert time_base(mixed) == pytest.approx(6.24808)
    with pytest.raises(ValueError, match="不能用重复帧补齐"):
        time_base(mixed, 10.2773)

    payload = {
        "classes": [],
        "clips": [
            {"id": "fast", "frame_count": 100, "fps": 10.0, "width": 8, "height": 8, "segments": [
                {"id": "s", "class_id": "pack", "start": 0, "end": 50},
            ]},
            {"id": "slow", "frame_count": 50, "fps": 5.0, "width": 8, "height": 8, "segments": [
                {"id": "t", "class_id": "pack", "start": 0, "end": 25},
            ]},
        ],
    }
    prepared = prepare_clips(payload, {"time_base_fps": 0.0, "window_seconds": 1.92})
    assert prepared.base == pytest.approx(5.0)
    assert len(prepared.indices["fast"]) == 50
    assert len(prepared.indices["slow"]) == 50
    fast = next(clip for clip in prepared.clips if clip.clip_id == "fast")
    assert fast.segments[0].end == 25


def test_training_writes_into_project_run_not_a_download(tmp_path: Path):
    store = ProjectStore(tmp_path)
    store.create("job", _settings())
    store.save_annotations(
        "job",
        {"classes": [{"id": "step1", "name": "步骤1"}], "clips": [_clip("a", 80), _clip("b", 80)]},
    )
    features = {
        "a": np.zeros((160, 131), dtype=np.float32),
        "b": np.ones((160, 131), dtype=np.float32),
    }
    settings = store.load_settings("job")
    settings["split_mode"] = "clip"
    settings["val_clip_ids"] = ["b"]
    store.save_settings("job", settings)
    result = start_training(store, "job", features)
    run = Path(result["run"])
    assert run.parent == tmp_path / "job" / "runs"
    assert (run / "best_lstm.pt").is_file()
    assert (run / "contract.json").is_file()
    assert (run / "split.json").is_file()
    assert (run / "metrics.json").is_file()
    assert "usable" in result["report"]
    progress = json.loads((tmp_path / "job" / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "done"
    assert len(progress["history"]) == 3
    assert "f1_50" in progress["history"][0]


def test_image_sequence_stays_in_one_folder_and_rejects_parent_path(tmp_path: Path):
    store = ProjectStore(tmp_path)
    store.create("job", _settings())
    store.save_upload("job", "clipa/10.jpg", b"a")
    store.save_upload("job", "clipa/2.jpg", b"b")
    names = [item.name for item in (tmp_path / "job" / "media" / "clipa").iterdir()]
    from lstm_workbench.timeline import sort_image_names
    assert sort_image_names(names) == ["2.jpg", "10.jpg"]
    try:
        store.save_upload("job", "../config.json", b"no")
    except ValueError as exc:
        assert "不合法" in str(exc)
    else:
        raise AssertionError("不允许写到工程目录外面")


def test_overlap_refuses_training(tmp_path: Path):
    store = ProjectStore(tmp_path)
    store.create("job", _settings())
    clip = _clip("a", 80)
    clip["segments"].append({"id": "overlap", "class_id": "step1", "start": 70, "end": 140})
    other = _clip("b", 80)
    store.save_annotations("job", {"classes": [{"id": "step1", "name": "步骤1"}], "clips": [clip, other]})
    try:
        start_training(store, "job", {"a": np.zeros((160, 131)), "b": np.zeros((160, 131))})
    except ValueError as exc:
        assert "交叠" in str(exc)
    else:
        raise AssertionError("交叠应该不能训练")


def test_same_filename_keeps_old_annotations_until_replace(tmp_path: Path):
    from lstm_workbench.service import adopt_clip

    store = ProjectStore(tmp_path)
    store.create("p", _settings())
    store.save_annotations("p", {"classes": [], "clips": [{
        "id": "1", "file": "1.mp4", "frame_count": 10, "fps": 10,
        "width": 8, "height": 8,
        "segments": [{"id": "s", "class_id": "a", "start": 0, "end": 5}],
    }]})
    (tmp_path / "p" / "media" / "1.mp4").write_bytes(b"old")
    probe = {"frame_count": 4, "fps": 10, "width": 8, "height": 8}
    try:
        adopt_clip(store, "p", "1.mp4", probe, replace=False)
    except FileExistsError:
        pass
    else:
        raise AssertionError("同名片子应该拒绝")
    assert store.load_annotations("p")["clips"][0]["segments"][0]["id"] == "s"
    adopted = adopt_clip(store, "p", "1.mp4", probe, replace=True)
    clips = store.load_annotations("p")["clips"]
    assert adopted["replaced"] is True
    assert len(clips) == 1
    assert clips[0]["segments"] == []
    assert clips[0]["frame_count"] == 4
    assert store.load_settings("p")["split_for"] == ""


def test_holding_out_a_whole_clip_keeps_it_out_of_training():
    payload = {
        "classes": [{"id": "step1", "name": "步骤1"}],
        "clips": [_clip("a", 80), _clip("b", 80)],
    }
    settings = {"split_mode": "clip", "val_clip_ids": ["b"], "window_seconds": 1.92, "time_base_fps": 0}
    prepared = prepare_clips(payload, settings)
    split, confirmed = resolve_split(prepared.clips, prepared.window, settings, prepared.signature)
    assert split.ok and confirmed
    assert split.same_source is False
    assert split.train_clip_ids == ["a"]
    assert split.val_clip_ids == ["b"]


def test_a_validation_clip_missing_a_class_is_refused():
    payload = {
        "classes": [{"id": "pack", "name": "包装"}, {"id": "seal", "name": "封装"}],
        "clips": [
            {"id": "a", "frame_count": 160, "fps": 10, "width": 8, "height": 8, "segments": [
                {"id": "ap", "class_id": "pack", "start": 0, "end": 80},
            ]},
            {"id": "b", "frame_count": 160, "fps": 10, "width": 8, "height": 8, "segments": [
                {"id": "bs", "class_id": "seal", "start": 0, "end": 80},
            ]},
        ],
    }
    settings = {"split_mode": "clip", "val_clip_ids": ["b"], "window_seconds": 1.92}
    prepared = prepare_clips(payload, settings)
    split, confirmed = resolve_split(prepared.clips, prepared.window, settings, prepared.signature)
    assert not split.ok
    assert not confirmed
    assert "seal" in split.blocked_reason


def test_validation_clip_does_not_add_required_parts(tmp_path, monkeypatch):
    captured = {}

    def fake_train(*_args, **kwargs):
        captured["required_parts"] = kwargs["required_parts"]
        return {
            "usable": False, "stopped": True, "reason": "停", "metrics": None,
            "mistakes": [], "history": [], "recipe": {},
        }

    monkeypatch.setattr("lstm_workbench.service.train_timeline", fake_train)
    store = ProjectStore(tmp_path)
    store.create("job", _settings())
    store.save_annotations(
        "job",
        {"classes": [{"id": "step1", "name": "步骤1"}], "clips": [_clip("a", 80), _clip("b", 80)]},
    )
    settings = store.load_settings("job")
    settings["split_mode"] = "clip"
    settings["val_clip_ids"] = ["b"]
    store.save_settings("job", settings)
    features = {
        "a": np.zeros((160, 131), dtype=np.float32),
        "b": np.zeros((160, 131), dtype=np.float32),
    }
    features["b"][:, 0] = 1
    start_training(store, "job", features)
    assert captured["required_parts"] == {"step1": []}
