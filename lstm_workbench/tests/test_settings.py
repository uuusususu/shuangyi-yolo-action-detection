"""工程设置的默认值和边界校验。"""
from __future__ import annotations

import pytest

from lstm_workbench import settings


def test_defaults_are_filled_in():
    got = settings.clean({})
    assert got["window_seconds"] == 1.92
    assert got["epochs"] == 30
    assert got["yolo_classes"] == []
    assert got["station_debounce_frames"] == 3
    assert got["time_base_fps"] == 0.0
    assert got["split_seed"] == 7
    assert got["split_val_count"] == 0


def test_reading_back_an_old_file_fills_missing_keys():
    old = {"window_seconds": 2.5, "yolo_path": "m.onnx", "junk_from_before": 1}
    got = settings.clean({}, old)
    assert got["window_seconds"] == 2.5
    assert got["yolo_path"] == "m.onnx"
    assert got["epochs"] == 30
    assert "junk_from_before" not in got


def test_station_fps_is_gone():
    """时间基改成由片子帧率推导，老文件里的 station_fps 不再被接受。"""
    with pytest.raises(ValueError, match="不接受"):
        settings.clean({"station_fps": 25})
    got = settings.clean({}, {"station_fps": 25, "epochs": 12})
    assert "station_fps" not in got
    assert got["epochs"] == 12


def test_unknown_key_is_refused():
    with pytest.raises(ValueError, match="不接受"):
        settings.clean({"nope": 1})


def test_server_owned_keys_cannot_be_set_from_the_page():
    got = settings.clean({"yolo_sha256": "假的", "hands_sha256": "假的", "yolo_classes": ["编的"]})
    assert got["yolo_sha256"] == ""
    assert got["hands_sha256"] == ""
    assert got["yolo_classes"] == []


def test_part_class_names_is_gone():
    """零件类别改成由模型决定，页面再传这个键要明确拒绝，不能悄悄存下去。"""
    with pytest.raises(ValueError, match="不接受"):
        settings.clean({"part_class_names": ["part"]})


@pytest.mark.parametrize("key", ["station_width", "station_height", "pose_matches_station"])
def test_station_alignment_keys_are_gone(key):
    """特征按每一帧自己的宽高归一化，分辨率差异早就被中和掉了；
    机位勾选也只是个人工声明。两个都不该再挡住训练，所以键一起撤掉。"""
    with pytest.raises(ValueError, match="不接受"):
        settings.clean({key: 1})
    got = settings.clean({}, {key: 1, "epochs": 12})
    assert key not in got
    assert got["epochs"] == 12


@pytest.mark.parametrize("patch", [
    {"window_seconds": 0},
    {"window_seconds": "abc"},
    {"time_base_fps": "abc"},
    {"split_seed": -1},
    {"split_train": "不是列表"},
    {"split_train": [["只有一项"]]},
    {"split_val": [["clip", "  "]]},
    {"yolo_conf_threshold": 1.5},
    {"epochs": 0},
    {"epochs": 501},
    {"station_debounce_frames": 0},
])
def test_bad_values_are_refused(patch):
    with pytest.raises(ValueError):
        settings.clean(patch)


def test_changing_split_inputs_invalidates_the_confirmation():
    """换了种子或验证窗口数就等于换一套划分，旧的确认结果必须作废。"""
    confirmed = settings.clean({
        "split_train": [["c1", "a"]],
        "split_val": [["c2", "b"]],
    })
    confirmed["split_for"] = "签名"
    for key, value in (("split_seed", 9), ("split_val_count", 3)):
        got = settings.clean({key: value}, confirmed)
        assert got["split_train"] == []
        assert got["split_val"] == []
        assert got["split_for"] == ""


def test_split_entries_are_kept():
    got = settings.clean({"split_train": [["c1", "a"]], "split_val": [["c2", "b"]]})
    assert got["split_train"] == [["c1", "a"]]
    assert got["split_val"] == [["c2", "b"]]


def test_good_values_are_kept():
    got = settings.clean({
        "epochs": 50,
        "yolo_conf_threshold": 0.55,
        "window_seconds": 2.0,
    })
    assert got["epochs"] == 50
    assert got["yolo_conf_threshold"] == 0.55
    assert got["window_seconds"] == 2.0


def test_patch_merges_onto_existing_settings():
    old = settings.clean({"epochs": 50, "yolo_path": "m.onnx"})
    got = settings.clean({"epochs": 12}, old)
    assert got["epochs"] == 12
    assert got["yolo_path"] == "m.onnx", "没提到的一项不能被冲掉"


def test_display_name_is_trimmed_and_cannot_be_a_path():
    assert settings.display_name("  包装-现场1  ") == "包装-现场1"
    assert settings.display_name("") == ""
    assert settings.display_name(None) == ""
    with pytest.raises(ValueError, match="模型名字"):
        settings.display_name("a/b")
    with pytest.raises(ValueError, match="模型名字"):
        settings.display_name("a" * 41)


def test_changing_which_clips_train_clears_a_confirmed_split():
    old = settings.clean({"split_train": [["c1", "a"]], "split_val": [["c2", "b"]], "train_excluded": []})
    old["split_for"] = "kept"
    got = settings.clean({"train_excluded": ["c2"]}, old)
    assert got["train_excluded"] == ["c2"]
    assert got["split_train"] == []
    assert got["split_val"] == []
    assert got["split_for"] == ""
    again = settings.clean({"train_excluded": ["c2"]}, got)
    again["split_train"] = [["c1", "a"]]
    kept = settings.clean({"train_excluded": ["c2"]}, again)
    assert kept["split_train"] == [["c1", "a"]]


def test_blank_yolo_path_does_not_erase_a_saved_model():
    """下拉框没读到值时会交来空路径。空路径不能把已经选定的模型清掉。"""
    old = settings.clean({"yolo_path": "m.onnx"})
    assert settings.clean({"yolo_path": ""}, old)["yolo_path"] == "m.onnx"
    assert settings.clean({"yolo_path": "  "}, old)["yolo_path"] == "m.onnx"
    assert settings.clean({"yolo_path": "other.onnx"}, old)["yolo_path"] == "other.onnx"


def test_network_and_whole_clip_holdout_are_saved():
    old = settings.clean({
        "split_train": [["c1", "a"]],
        "split_val": [["c1", "b"]],
        "split_mode": "segment",
        "network": "lstm",
    })
    old["split_for"] = "kept"
    got = settings.clean({"network": "mstcn", "split_mode": "clip", "val_clip_ids": ["c2"]}, old)
    assert got["network"] == "mstcn"
    assert got["split_mode"] == "clip"
    assert got["val_clip_ids"] == ["c2"]
    assert got["split_train"] == []
    assert got["split_for"] == ""
    kept = settings.clean({"network": "lstm"}, got)
    assert kept["split_mode"] == "clip"
    assert kept["val_clip_ids"] == ["c2"]
    with pytest.raises(ValueError, match="网络"):
        settings.clean({"network": "transformer"})
    dropped = settings.clean({"train_excluded": ["c2"]}, got)
    assert dropped["val_clip_ids"] == []
