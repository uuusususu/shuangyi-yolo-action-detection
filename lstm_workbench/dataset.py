"""验证集划分：单位是「段」，不是片子。

一条片子里有多个同类段，每个段独立进出训练或验证。段与段本身不重叠，
所以没有窗口级泄漏；但同一条片子的段共享背景、光照、机位与作业者，
所以只要有一条片子的段被分到两边，same_source 就是真，验证分数是乐观的。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .timeline import ClipAnnotation, Segment

IGNORE_LABEL = -100


STRIDE = 8
DEFAULT_VAL_RATIO = 0.2


@dataclass
class Split:
    """entries 是 (clip_id, segment_id)。"""

    train: list[tuple[str, str]] = field(default_factory=list)
    val: list[tuple[str, str]] = field(default_factory=list)
    blocked_reason: str = ""

    @property
    def ok(self) -> bool:
        return not self.blocked_reason

    @property
    def train_clip_ids(self) -> list[str]:
        return sorted({clip_id for clip_id, _ in self.train})

    @property
    def val_clip_ids(self) -> list[str]:
        return sorted({clip_id for clip_id, _ in self.val})

    @property
    def same_source(self) -> bool:
        """同一条片子的段有没有被分到两边。有，验证分数就只能当参考。"""
        return bool(set(self.train_clip_ids) & set(self.val_clip_ids))


def windows_inside(segment: Segment, window: int, stride: int = STRIDE) -> list[tuple[int, int]]:
    if segment.length() < window:
        return []
    spans: list[tuple[int, int]] = []
    start = segment.start
    while start + window <= segment.end:
        spans.append((start, start + window))
        start += stride
    return spans


def frame_labels(clip: ClipAnnotation, class_ids: list[str]) -> np.ndarray:
    """逐帧标签。没涂的是空闲，「不训练」是 -100。短段和短空档都保留。"""
    index = {class_id: position for position, class_id in enumerate(class_ids)}
    if "idle" not in index:
        raise ValueError("类别里没有空闲")
    labels = np.full(clip.frame_count, index["idle"], dtype=np.int64)
    for segment in sorted(clip.segments, key=lambda item: (item.start, item.end, item.id)):
        class_index = index.get(segment.class_id)
        if class_index is None:
            continue
        start = max(0, int(segment.start))
        end = min(clip.frame_count, int(segment.end))
        if end > start:
            labels[start:end] = class_index
    for cut in clip.excluded:
        start = max(0, int(cut.start))
        end = min(clip.frame_count, int(cut.end))
        if end > start:
            labels[start:end] = IGNORE_LABEL
    return labels


def unlabeled_tail_frames(clip: ClipAnnotation) -> int:
    """最后一段色块或剪掉区间之后，一直到片尾都没标的帧数。"""
    covered = 0
    for segment in list(clip.segments) + list(clip.excluded):
        covered = max(covered, int(segment.end))
    return max(0, clip.frame_count - covered)


def draw_crops(
    lengths: list[int],
    target: int,
    context: int,
    seed: int,
    epoch: int,
) -> list[tuple[int, int, int, int]]:
    """每条片子按半段步长铺开，起点每轮随机偏一点，并总有一段从第 0 帧开始。

    铺完打乱顺序。返回 (片子下标, 输入起点, 目标起点, 目标终点)。输入从 max(0, 目标起点-上文) 开始。
    """
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("没有可以截的片子")
    target = max(2, int(target))
    context = max(0, int(context))
    crops: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    rng = random.Random(seed + epoch * 10007)

    def add(clip_index: int, target_start: int) -> None:
        if (clip_index, target_start) in seen:
            return
        length = lengths[clip_index]
        target_end = min(length, target_start + min(target, length))
        if target_end <= target_start:
            return
        seen.add((clip_index, target_start))
        input_start = max(0, target_start - context)
        crops.append((clip_index, input_start, target_start, target_end))

    for clip_index, length in enumerate(lengths):
        span = min(target, length)
        step = max(1, span // 2)
        offset = rng.randrange(step)
        start = offset
        while start < length:
            add(clip_index, start)
            if start + span >= length:
                break
            start += step
        if offset:
            add(clip_index, 0)
    covered = sum(item[3] - item[2] for item in crops)
    need = sum(lengths)
    guard = 0
    while covered < need and guard < need * 4:
        guard += 1
        clip_index = rng.randrange(len(lengths))
        length = lengths[clip_index]
        span = min(target, length)
        start_max = max(0, length - span)
        target_start = rng.randint(0, start_max) if start_max else 0
        before = len(crops)
        add(clip_index, target_start)
        if len(crops) > before:
            covered += crops[-1][3] - crops[-1][2]
    rng.shuffle(crops)
    return crops


def segment_spans(
    clip: ClipAnnotation,
    window: int,
    only: set[str] | None = None,
) -> list[tuple[str, int, int]]:
    """一条标注段就是一条样本，从起始帧到结束帧，不再切成多个小窗口。

    短于一个窗口的段本来就产不出样本，仍由 trainable_segments 排除。
    """
    rows: list[tuple[str, int, int]] = []
    for segment in clip.trainable_segments(window):
        if only is not None and segment.id not in only:
            continue
        if any(segment.start < cut.end and cut.start < segment.end for cut in clip.excluded):
            continue
        rows.append((segment.class_id, segment.start, segment.end))
    return rows


def labeled_windows(
    clip: ClipAnnotation,
    window: int,
    stride: int = STRIDE,
    only: set[str] | None = None,
) -> list[tuple[str, int, int]]:
    """返回 (class_id, start, end)。只在段内滑动，所以不会有跨段窗口。

    only 给定时只取这些段 —— 划分是按段做的，取窗口要能按段筛。
    """
    rows: list[tuple[str, int, int]] = []
    for segment in clip.trainable_segments(window):
        if only is not None and segment.id not in only:
            continue
        for start, end in windows_inside(segment, window, stride):
            if any(start < cut.end and cut.start < end for cut in clip.excluded):
                continue
            rows.append((segment.class_id, start, end))
    return rows


def units_of(clips: list[ClipAnnotation], window: int) -> dict[tuple[str, str], str]:
    """(clip_id, segment_id) -> class_id，只收可训练段。"""
    out: dict[tuple[str, str], str] = {}
    for clip in clips:
        for segment in clip.trainable_segments(window):
            out[(clip.clip_id, segment.id)] = segment.class_id
    return out


def unit_windows(clips: list[ClipAnnotation], window: int) -> dict[tuple[str, str], int]:
    """每个段能产出多少窗口。按段数分配会让短段吃掉名额，所以要按窗口数分。"""
    out: dict[tuple[str, str], int] = {}
    for clip in clips:
        for segment in clip.trainable_segments(window):
            out[(clip.clip_id, segment.id)] = len(windows_inside(segment, window))
    return out


def default_val_windows(total_windows: int) -> int:
    """验证集目标窗口数。窗口少的时候比例是假精度，所以夹到 [1, total-1]。"""
    if total_windows < 2:
        return 0
    return max(1, min(total_windows - 1, int(round(total_windows * DEFAULT_VAL_RATIO))))


def _coverage(entries: list[tuple[str, str]], unit_class: dict[tuple[str, str], str]) -> set[str]:
    return {unit_class[e] for e in entries if e in unit_class}


def _pick_val(
    units: list[tuple[str, str]],
    unit_class: dict[tuple[str, str], str],
    unit_windows_: dict[tuple[str, str], int],
    classes: set[str],
    seed: int,
    target_windows: int,
) -> list[tuple[str, str]]:
    """先保证每个类都进验证集，再按种子顺序补到目标窗口数。"""
    order = list(units)
    random.Random(seed).shuffle(order)

    val: list[tuple[str, str]] = []
    covered: set[str] = set()
    got = 0
    for unit in order:
        if covered == classes:
            break
        if unit_class[unit] in covered:
            continue
        val.append(unit)
        covered.add(unit_class[unit])
        got += unit_windows_[unit]
    for unit in order:
        if got >= target_windows:
            break
        if unit in val:
            continue
        val.append(unit)
        got += unit_windows_[unit]
    return sorted(val)


def _finish(train: list, val: list, unit_class: dict, classes: set[str]) -> Split:
    if not train:
        return Split([], [], "训练集是空的")
    if not val:
        return Split([], [], "验证集是空的")
    for side, label in ((train, "训练"), (val, "验证")):
        missing = sorted(classes - _coverage(side, unit_class))
        if missing:
            return Split([], [], f"类别 {missing[0]} 没有进入{label}集，不能同时进入训练和验证")
    return Split(train, val)


def holdout_split(clips: list[ClipAnnotation], window: int, val_clip_ids: list[str]) -> Split:
    """整条片子留作验证。这条片子的段全部进验证，其余片子的段全部进训练。"""
    unit_class = units_of(clips, window)
    if not unit_class:
        return Split([], [], f"没有任何段够长（不足 {window} 帧），产不出训练窗口")
    known = {clip.clip_id for clip in clips}
    val_ids = {str(item) for item in val_clip_ids} & known
    if not val_ids:
        return Split([], [], "先把至少一条片子标成只做验证。页上的分数才会来自没参与训练的整条片子。")
    train = sorted(key for key in unit_class if key[0] not in val_ids)
    val = sorted(key for key in unit_class if key[0] in val_ids)
    if not val:
        return Split([], [], "标成只做验证的片子里没有够长的段，换一条再试")
    if not train:
        return Split([], [], "训练集是空的，至少留一条片子参加训练")
    return _finish(train, val, unit_class, set(unit_class.values()))


def propose_split(
    clips: list[ClipAnnotation],
    window: int,
    seed: int = 7,
    val_windows: int | None = None,
) -> Split:
    """按段划分。每一个类别的段都分别进训练和验证。

    val_windows 是验证集目标窗口数；不给就按 DEFAULT_VAL_RATIO 自动定。
    """
    unit_class = units_of(clips, window)
    if not unit_class:
        return Split([], [], f"没有任何段够长（不足 {window} 帧），产不出训练窗口")

    classes = set(unit_class.values())
    for name in sorted(classes):
        holders = [unit for unit, cls in unit_class.items() if cls == name]
        if len(holders) < 2:
            return Split([], [], f"类别 {name} 只有 {len(holders)} 段，不能同时进入训练和验证")

    per_unit = unit_windows(clips, window)
    units = sorted(unit_class)
    total = sum(per_unit.values())
    target = default_val_windows(total) if val_windows is None else max(1, min(total - 1, val_windows))
    val = _pick_val(units, unit_class, per_unit, classes, seed, target)
    train = [unit for unit in units if unit not in val]
    return _finish(train, val, unit_class, classes)


def check_split(clips: list[ClipAnnotation], window: int, train: list, val: list) -> Split:
    """校验人工调整过的划分。与提案走同一套约束，不放松。"""
    unit_class = units_of(clips, window)
    known = set(unit_class)
    train = [(str(a), str(b)) for a, b in train]
    val = [(str(a), str(b)) for a, b in val]
    for label, side in (("训练", train), ("验证", val)):
        unknown = [unit for unit in side if unit not in known]
        if unknown:
            raise ValueError(f"{label}集里有对不上的段：{unknown[0][0]} / {unknown[0][1]}")
    both = set(train) & set(val)
    if both:
        clip_id, segment_id = sorted(both)[0]
        raise ValueError(f"段 {segment_id}（{clip_id}）同时出现在两边 —— 同一个段的窗口不能跨边")
    classes = set(unit_class.values())
    for side, label in ((train, "训练"), (val, "验证")):
        missing = sorted(classes - _coverage(side, unit_class))
        if missing:
            return Split([], [], f"类别 {missing[0]} 没有进入{label}集，不能同时进入训练和验证")
    return _finish(train, val, unit_class, classes)


def split_stats(clips: list[ClipAnnotation], window: int, split: Split, stride: int = STRIDE) -> dict:
    """给界面看的：两边各多少段、多少窗口、各类别覆盖。"""
    by_id = {clip.clip_id: clip for clip in clips}

    def side(entries):
        per_clip: dict[str, set[str]] = {}
        for clip_id, segment_id in entries:
            per_clip.setdefault(clip_id, set()).add(segment_id)
        windows = 0
        classes: set[str] = set()
        frames_by_class: dict[str, int] = {}
        segments_by_class: dict[str, int] = {}
        idle_frames = 0
        for clip_id, segment_ids in per_clip.items():
            clip = by_id[clip_id]
            for class_id, _start, _end in labeled_windows(clip, window, stride, segment_ids):
                windows += 1
                classes.add(class_id)
            taken = bytearray(clip.frame_count)
            for segment in clip.segments:
                start = max(0, int(segment.start))
                end = min(clip.frame_count, int(segment.end))
                if end > start:
                    taken[start:end] = b"\x01" * (end - start)
                if segment.id not in segment_ids:
                    continue
                frames_by_class[segment.class_id] = frames_by_class.get(segment.class_id, 0) + max(0, end - start)
                segments_by_class[segment.class_id] = segments_by_class.get(segment.class_id, 0) + 1
                classes.add(segment.class_id)
            for cut in clip.excluded:
                start = max(0, int(cut.start))
                end = min(clip.frame_count, int(cut.end))
                if end > start:
                    taken[start:end] = b"\x01" * (end - start)
            idle_frames += taken.count(0)
        class_counts = [{"class_id": "idle", "frames": idle_frames, "segments": 0}]
        for class_id in sorted(segments_by_class):
            class_counts.append({
                "class_id": class_id,
                "frames": frames_by_class[class_id],
                "segments": segments_by_class[class_id],
            })
        return {
            "clips": sorted(per_clip),
            "segments": len(entries),
            "windows": windows,
            "classes": sorted(class_id for class_id in classes if class_id != "idle"),
            "class_counts": class_counts,
        }

    train = side(split.train)
    val = side(split.val)
    total = train["windows"] + val["windows"]
    return {
        "same_source": split.same_source,
        "train": train,
        "val": val,
        "val_share": (val["windows"] / total) if total else 0.0,
    }
