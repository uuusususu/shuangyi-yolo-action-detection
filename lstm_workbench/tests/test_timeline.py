"""整条连续训练：逐帧标签、段级分数、往回看长度、截段。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lstm_workbench.dataset import IGNORE_LABEL, draw_crops, frame_labels
from lstm_workbench.metrics import (
    edit_score,
    pick_best,
    score_timeline,
    segmental_f1,
    segments_from_labels,
    timeline_mistakes,
)
from lstm_workbench.timeline import ClipAnnotation, Segment
from lstm_workbench.trainer import fit_feature_norm


def _clip(segments, frame_count=120, excluded=None) -> ClipAnnotation:
    return ClipAnnotation(
        "c", frame_count, 10.0, 8, 8,
        [Segment(item[0], item[1], item[2], item[3]) for item in segments],
        [Segment(item[0], "cut", item[1], item[2]) for item in (excluded or [])],
    )


def test_unpainted_frames_are_idle_and_short_gaps_stay():
    labels = frame_labels(_clip([("p", "pack", 10, 50), ("s", "seal", 60, 100)]), ["idle", "pack", "seal"])
    assert labels[0] == 0 and labels[55] == 0 and labels[110] == 0
    assert set(labels[10:50]) == {1}
    assert set(labels[60:100]) == {2}
    # 相邻两段中间只有 2 帧，也是空闲，不丢掉
    tight = frame_labels(_clip([("p", "pack", 0, 10), ("s", "seal", 12, 30)], 30), ["idle", "pack", "seal"])
    assert list(tight[10:12]) == [0, 0]


def test_adjacent_actions_switch_without_an_idle_frame():
    labels = frame_labels(_clip([("p", "pack", 0, 10), ("s", "seal", 10, 20)], 20), ["idle", "pack", "seal"])
    assert list(labels[:10]) == [1] * 10
    assert list(labels[10:]) == [2] * 10


def test_excluded_frames_are_ignored():
    labels = frame_labels(
        _clip([("p", "pack", 0, 40)], 40, excluded=[("x", 10, 15)]),
        ["idle", "pack"],
    )
    assert list(labels[10:15]) == [IGNORE_LABEL] * 5
    assert labels[9] == 1 and labels[15] == 1


def test_one_predicted_segment_against_twelve_scores_two_thirteenths():
    truth = [("seal", index * 10, index * 10 + 8) for index in range(12)]
    predicted = [("seal", 0, 8)]
    assert segmental_f1(truth, predicted, 0.5) == pytest.approx(2 / 13)


def test_over_segmentation_lowers_the_edit_score():
    truth = [("pack", 0, 30)]
    whole = [("pack", 0, 30)]
    chopped = [("pack", 0, 8), ("seal", 8, 12), ("pack", 12, 20), ("seal", 20, 24), ("pack", 24, 30)]
    assert edit_score(truth, chopped) < edit_score(truth, whole)


def test_wrong_class_and_idle_alarms_are_both_listed():
    labels = np.array([1] * 20 + [0] * 20)
    predicted = [("seal", 0, 20), ("pack", 20, 30), ("idle", 30, 40)]
    got = timeline_mistakes("c", labels, predicted, ["idle", "pack", "seal"])
    assert { (item["truth"], item["predicted"]) for item in got } == {("pack", "seal"), ("idle", "pack")}
    assert all(item["clip_id"] == "c" for item in got)


def test_a_matched_segment_is_not_a_mistake():
    labels = np.array([1] * 20)
    assert timeline_mistakes("c", labels, [("pack", 0, 20)], ["idle", "pack"]) == []


def test_score_timeline_reports_the_weaker_class_recall():
    truth = np.array([1] * 10 + [2] * 10)
    pred = np.array([1] * 10 + [0] * 10)
    scored = score_timeline(truth, pred, ["idle", "pack", "seal"])
    assert scored["min_recall"] == pytest.approx(0.0)


def test_clip_split_counts_frames_and_segments():
    from lstm_workbench.dataset import holdout_split, split_stats

    clips = [
        ClipAnnotation("a", 40, 10.0, 8, 8, [Segment("s", "pack", 0, 20)]),
        ClipAnnotation("b", 40, 10.0, 8, 8, [Segment("t", "pack", 5, 25)]),
    ]
    split = holdout_split(clips, 10, ["b"])
    stats = split_stats(clips, 10, split)
    train_pack = next(row for row in stats["train"]["class_counts"] if row["class_id"] == "pack")
    train_idle = next(row for row in stats["train"]["class_counts"] if row["class_id"] == "idle")
    val_pack = next(row for row in stats["val"]["class_counts"] if row["class_id"] == "pack")
    assert train_pack == {"class_id": "pack", "frames": 20, "segments": 1}
    assert train_idle["frames"] == 20
    assert val_pack == {"class_id": "pack", "frames": 20, "segments": 1}


def test_ignored_frames_do_not_change_the_score():
    class_ids = ["idle", "pack"]
    truth = np.array([0, 0, 1, 1, -100, -100])
    pred = np.array([0, 0, 1, 1, 0, 1])
    scored = score_timeline(truth, pred, class_ids)
    assert scored["frame_accuracy"] == pytest.approx(1.0)
    assert scored["idle_false_alarm"] == pytest.approx(0.0)


def test_best_epoch_follows_segmentation_not_frame_accuracy():
    history = [
        {"epoch": 40, "f1_50": 0.70, "edit": 80, "frame_accuracy": 0.95},
        {"epoch": 60, "f1_50": 0.88, "edit": 70, "frame_accuracy": 0.93},
    ]
    assert history[pick_best(history)]["epoch"] == 60


def test_crops_cover_every_frame_and_repeat():
    first = draw_crops([40, 25], target=10, context=4, seed=7, epoch=1)
    again = draw_crops([40, 25], target=10, context=4, seed=7, epoch=1)
    assert first == again
    covered = [set() for _ in range(2)]
    for clip_index, _input, start, end in first:
        covered[clip_index].update(range(start, end))
    assert covered[0] == set(range(40))
    assert covered[1] == set(range(25))
    assert any(item[1] == 0 and item[2] == 0 for item in first if item[0] == 0)


def test_each_epoch_shifts_the_crops_and_mixes_their_order():
    one = draw_crops([80, 50], target=10, context=4, seed=7, epoch=1)
    two = draw_crops([80, 50], target=10, context=4, seed=7, epoch=2)
    assert one != two
    for crops in (one, two):
        for clip_index, length in ((0, 80), (1, 50)):
            covered = set()
            for index, _input, start, end in crops:
                if index == clip_index:
                    covered.update(range(start, end))
            assert covered == set(range(length))
            assert any(index == clip_index and start == 0 and item_input == 0
                       for index, item_input, start, _end in crops)


def test_validation_frames_do_not_enter_feature_norm():
    train = np.zeros((4, 2), dtype=np.float32)
    train[:, 0] = 2
    val = np.ones((4, 2), dtype=np.float32) * 100
    labels = np.zeros(4, dtype=np.int64)
    mean, _std = fit_feature_norm([train], [labels])
    assert mean[0] == pytest.approx(2)
    assert mean[1] == pytest.approx(0)


def test_a_constant_feature_uses_std_one_not_a_tiny_floor():
    rows = np.zeros((6, 2), dtype=np.float32)
    rows[:, 0] = np.linspace(0, 5, 6)
    labels = np.zeros(6, dtype=np.int64)
    _mean, std = fit_feature_norm([rows], [labels])
    assert std[1] == pytest.approx(1.0)
    assert std[0] > 1e-3


def test_future_frames_do_not_change_past_outputs():
    import torch

    from lstm_workbench.model import ActionLSTM

    past = torch.zeros(1, 16, 4)
    changed = past.clone()
    changed[:, 10:] = 1
    for model in (ActionLSTM(4, 2, hidden_size=8, num_layers=1, dropout=0.0),):
        model.eval()
        with torch.no_grad():
            before = model(past, all_steps=True)
            after = model(changed, all_steps=True)
        assert torch.allclose(before[:, :10], after[:, :10])


def test_held_out_clip_is_learned_by_both_networks(tmp_path: Path):
    """合成动作很明显：每一类占一维。整条留出的片子应该被切对。"""
    from lstm_workbench.project_store import ProjectStore
    from lstm_workbench.service import start_training

    def pattern() -> tuple[np.ndarray, list[dict]]:
        labels = [0] * 8 + [1] * 16 + [0] * 3 + [2] * 16 + [1] * 12
        rows = np.zeros((len(labels), 6), dtype=np.float32)
        for index, label in enumerate(labels):
            rows[index, label] = 1.0
        segments = []
        cursor = 0
        serial = 0
        names = {1: "pack", 2: "seal"}
        while cursor < len(labels):
            end = cursor + 1
            while end < len(labels) and labels[end] == labels[cursor]:
                end += 1
            if labels[cursor] in names:
                segments.append({"id": f"s{serial}", "class_id": names[labels[cursor]], "start": cursor, "end": end})
                serial += 1
            cursor = end
        return rows, segments

    store = ProjectStore(tmp_path)
    store.create("job", {
        "epochs": 8,
        "learning_rate": 0.01,
        "dropout": 0.0,
        "context_seconds": 2,
        "target_seconds": 3,
        "batch_size": 4,
        "window_seconds": 0.5,
        "yolo_classes": [],
        "split_mode": "clip",
        "val_clip_ids": ["v"],
    })
    rows, segments = pattern()
    clips = []
    features = {}
    for clip_id in ("a", "b", "v"):
        clips.append({
            "id": clip_id, "frame_count": len(rows), "fps": 10.0,
            "width": 8, "height": 8, "segments": segments,
            "annotation_complete": True,
        })
        features[clip_id] = rows.copy()
    store.save_annotations("job", {
        "classes": [{"id": "pack", "name": "包装"}, {"id": "seal", "name": "封装"}],
        "clips": clips,
    })
    for network in ("lstm",):
        settings = store.load_settings("job")
        settings["network"] = network
        settings["val_clip_ids"] = ["v"]
        settings["split_mode"] = "clip"
        store.save_settings("job", settings)
        result = start_training(store, "job", {key: value.copy() for key, value in features.items()})
        assert result["report"]["metrics"]["f1_50"] >= 0.9, (network, result["report"]["metrics"])
        assert result["report"]["metrics"]["min_recall"] > 0


def _timeline_contract(**overrides):
    contract = {
        "train_unit": "timeline",
        "decode_version": 1,
        "network": "lstm",
        "context_frames": 8,
        "time_base_fps": 10,
        "min_action_seconds": 1.92,
        "feature_mean": [1.0],
        "feature_std": [1.0],
        "class_ids": ["idle"],
        "part_class_names": [],
        "required_parts": {},
        "input_size": 1,
        "yolo_conf_threshold": 0.6,
    }
    contract.update(overrides)
    return contract


def test_a_timeline_contract_without_std_is_refused():
    from lstm_workbench.check_clip import require_timeline_contract

    contract = _timeline_contract()
    del contract["feature_std"]
    with pytest.raises(ValueError, match="feature_std"):
        require_timeline_contract(contract)


def test_a_blank_contract_field_is_refused():
    from lstm_workbench.check_clip import require_timeline_contract

    with pytest.raises(ValueError, match="network"):
        require_timeline_contract(_timeline_contract(network=None))


def test_the_other_sides_action_is_not_counted_as_idle():
    from lstm_workbench.dataset import Split, split_stats

    clip = ClipAnnotation(
        "a", 40, 10.0, 8, 8,
        [Segment("s", "pack", 0, 20), Segment("t", "seal", 20, 40)],
    )
    stats = split_stats([clip], 10, Split([("a", "s")], [("a", "t")]))
    train_idle = next(row for row in stats["train"]["class_counts"] if row["class_id"] == "idle")
    train_pack = next(row for row in stats["train"]["class_counts"] if row["class_id"] == "pack")
    assert train_idle["frames"] == 0
    assert train_pack == {"class_id": "pack", "frames": 20, "segments": 1}
    assert "seal" not in {row["class_id"] for row in stats["train"]["class_counts"]}


def test_an_empty_feature_mean_is_named_as_mean():
    from lstm_workbench.check_clip import require_timeline_contract

    with pytest.raises(ValueError, match="feature_mean"):
        require_timeline_contract(_timeline_contract(feature_mean=[]))
