"""标注、切窗、打分。"""
from __future__ import annotations

import numpy as np

from lstm_workbench.dataset import labeled_windows, propose_split, windows_inside
from lstm_workbench.features import build_feature_row, feature_size
from lstm_workbench.metrics import gate_passed, predict_segment, score_segments
from lstm_workbench.timeline import (
    ClipAnnotation,
    Segment,
    merge_narrow_gap,
    numeric_image_key,
    paint_idle,
    timeline_issues,
    window_frames,
)

# 窗口按秒定义，测试统一按 25fps 换算，得到 48 帧。
WINDOW = window_frames(25)
assert WINDOW == 48


def test_cut_span_is_skipped_by_idle_and_windows():
    """裁掉的区间不变成空闲，也不产训练窗口。"""
    clip = ClipAnnotation(
        "c", 400, 25, 10, 10,
        [Segment("a", "step1", 0, 100)],
        [Segment("x", "cut", 150, 250)],
    )
    idle = [segment for segment in clip.trainable_segments(WINDOW) if segment.class_id == "idle"]
    assert idle
    assert all(segment.end <= 150 or segment.start >= 250 for segment in idle)
    windows = labeled_windows(clip, WINDOW)
    assert windows
    assert all(end <= 150 or start >= 250 for _class_id, start, end in windows)


def test_numeric_sort_is_not_alphabetical():
    names = ["10.jpg", "2.jpg", "1.jpg"]
    ordered = sorted(names, key=numeric_image_key)
    assert ordered == ["1.jpg", "2.jpg", "10.jpg"]


def test_overlap_blocks_and_idle_is_unpainted_span():
    segments = [Segment("a", "step1", 48, 120), Segment("b", "step2", 100, 160)]
    assert any(issue.code == "overlap" for issue in timeline_issues(200, segments, WINDOW))
    idle = paint_idle(200, [Segment("a", "step1", 0, 60)], WINDOW)
    assert idle[0].class_id == "idle"
    assert idle[0].start == 60
    assert idle[0].end == 200


def test_short_segment_has_no_window_and_narrow_gap_can_merge():
    short = Segment("s", "step1", 0, 20)
    assert windows_inside(short, WINDOW) == []
    left = Segment("l", "step1", 0, 50)
    right = Segment("r", "step1", 60, 120)
    merged = merge_narrow_gap([left, right], "l", "r", WINDOW)
    assert merged[0].end == 120


def test_window_stays_inside_segment_and_split_is_by_segment():
    clip = ClipAnnotation("c1", 200, 30, 640, 480, [Segment("a", "step1", 0, 100)])
    labels = {label for label, _start, _end in labeled_windows(clip, WINDOW)}
    assert "step1" in labels
    assert "idle" in labels
    # 一类只有一段，凑不出「每类两边都有」。
    blocked = propose_split([clip], WINDOW)
    assert not blocked.ok
    # 同一条片子里给同一类两段，就能按段拆开 —— 这正是按片子拆不开的场景。
    other = ClipAnnotation("c1", 400, 30, 640, 480, [
        Segment("a", "step1", 0, 100),
        Segment("b", "step1", 200, 300),
    ])
    split = propose_split([other], WINDOW, seed=7)
    assert split.ok
    assert split.same_source, "两段来自同一条片子，必须标成同源"
    placed = set(split.train + split.val)
    assert {("c1", "a"), ("c1", "b")} <= placed
    assert any(segment_id.startswith("idle-") for _clip, segment_id in placed)
    classes_train = {unit[1] for unit in split.train}
    classes_val = {unit[1] for unit in split.val}
    assert "a" in classes_train or "b" in classes_train
    assert "a" in classes_val or "b" in classes_val


def test_segment_score_does_not_count_each_window():
    probs = [np.array([0.1, 0.9]), np.array([0.2, 0.8])]
    assert predict_segment(probs) == 1
    metrics = score_segments(["step1", "idle"], ["step1", "step1"], ["idle", "step1"])
    assert metrics["idle_false_alarm"] == 1.0
    passed, _reason = gate_passed(
        {**metrics, "macro_f1": 0.95, "min_recall": 0.9, "idle_false_alarm": 0.0},
        train_loss=0.2,
        previous_train_loss=0.4,
        val_loss=0.3,
        previous_val_loss=0.5,
        missing_val_class=False,
    )
    assert passed


def test_feature_row_drops_low_conf_and_orders_hands_left_to_right():
    row = build_feature_row(
        [
            {"label": "part", "conf": 0.2, "box": (0, 0, 10, 10)},
            {"label": "part", "conf": 0.9, "box": (0, 0, 10, 10)},
        ],
        [
            [(80, 0, 0.1)] + [(0, 0, 0)] * 20,
            [(10, 0, 0.2)] + [(0, 0, 0)] * 20,
        ],
        100,
        50,
        ["part"],
        0.5,
    )
    assert len(row) == feature_size(1)
    assert row[0] == 0.9
    assert row[5] == 10 / 100
    parts_only = build_feature_row(
        [{"label": "part", "conf": 0.9, "box": (0, 0, 10, 10)}],
        [],
        100,
        50,
        ["part"],
        0.5,
        include_hands=False,
    )
    assert parts_only == [0.9, 0.05, 0.1, 0.1, 0.2]
    assert feature_size(1, hands=False) == 5
