"""从标注段的特征里记下每类动作稳定出现的零件，验证时缺件则这段失败。"""
from __future__ import annotations

import numpy as np

from .dataset import segment_spans
from .timeline import ClipAnnotation

# 一段动作里，零件要占到这段帧数的两成，才算这段真的有它。一帧闪过不算。
PART_FRAME_RATIO = 0.20
# 同一动作的标注段里，至少一半达到上面这条，才列入必检。
PART_SEGMENT_SHARE = 0.50


def parts_stable_in_block(block: np.ndarray, part_names: list[str]) -> set[str]:
    """这一段里占比达到门槛的零件。空段什么都不算。"""
    found: set[str] = set()
    array = np.asarray(block)
    if array.ndim != 2 or array.shape[0] == 0:
        return found
    for index, name in enumerate(part_names):
        column = index * 5
        if column >= array.shape[1]:
            continue
        if float(np.mean(array[:, column] > 0)) >= PART_FRAME_RATIO:
            found.add(name)
    return found


def required_parts_for_split(
    clips: list[ClipAnnotation],
    feature_arrays: dict[str, np.ndarray],
    split,
    window: int,
    part_names: list[str],
) -> dict[str, list[str]]:
    """训练段和验证段里，每个动作稳定出现的零件。空闲不记。

    一段里占比不足两成的零件不算这段有它。同一类里不足一半的段达到这条，不列入。
    一类里一件都没稳定出现，对应空列表，验证时不卡。
    """
    clips_by_id = {clip.clip_id: clip for clip in clips}
    by_clip: dict[str, set[str]] = {}
    for clip_id, segment_id in list(split.train) + list(split.val):
        by_clip.setdefault(clip_id, set()).add(segment_id)
    totals: dict[str, int] = {}
    stable_counts: dict[str, dict[str, int]] = {}
    touched: list[str] = []
    for clip_id, segment_ids in by_clip.items():
        clip = clips_by_id[clip_id]
        array = np.asarray(feature_arrays[clip.clip_id])
        for class_id, start, end in segment_spans(clip, window, segment_ids):
            if class_id == "idle" or end <= start:
                continue
            if class_id not in totals:
                touched.append(class_id)
                totals[class_id] = 0
                stable_counts[class_id] = {name: 0 for name in part_names}
            totals[class_id] += 1
            for name in parts_stable_in_block(array[start:end], part_names):
                stable_counts[class_id][name] += 1
    return {
        class_id: [
            name for name in part_names
            if stable_counts[class_id][name] / totals[class_id] >= PART_SEGMENT_SHARE
        ]
        for class_id in touched
    }


def required_parts_from_labeled_rows(
    labeled_rows: list[tuple[np.ndarray, np.ndarray]],
    class_ids: list[str],
    part_names: list[str],
) -> dict[str, list[str]]:
    """连续相同标签算一段。只统计传进来的帧，验证集不要放进来。"""
    totals = {class_id: 0 for class_id in class_ids if class_id != "idle"}
    stable_counts = {class_id: {name: 0 for name in part_names} for class_id in totals}
    for rows, labels in labeled_rows:
        array = np.asarray(rows)
        marks = np.asarray(labels)
        length = min(int(array.shape[0]) if array.ndim == 2 else 0, int(marks.shape[0]))
        start = 0
        while start < length:
            value = int(marks[start])
            end = start + 1
            while end < length and int(marks[end]) == value:
                end += 1
            if 0 <= value < len(class_ids):
                class_id = class_ids[value]
                if class_id in totals:
                    totals[class_id] += 1
                    for name in parts_stable_in_block(array[start:end], part_names):
                        stable_counts[class_id][name] += 1
            start = end
    return {
        class_id: [
            name for name in part_names
            if total > 0 and stable_counts[class_id][name] / total >= PART_SEGMENT_SHARE
        ]
        for class_id, total in totals.items()
    }


def apply_part_gate(
    segments: list[dict],
    rows: np.ndarray,
    part_names: list[str],
    required_parts: dict[str, list[str]] | None,
) -> list[dict]:
    """动作段已经分好之后才查零件。没有记录的类别不写检查结果。"""
    if not required_parts:
        return segments
    index = {name: position for position, name in enumerate(part_names)}
    array = np.asarray(rows)
    for segment in segments:
        needed = required_parts.get(segment.get("class_id"))
        if not needed:
            continue
        start = int(segment.get("start_frame") or 0)
        end = int(segment.get("end_frame") or start)
        block = array[start:end] if array.ndim == 2 else array[:0]
        missing = []
        for name in needed:
            slot = index.get(name)
            column = None if slot is None else slot * 5
            if column is None or block.size == 0 or column >= block.shape[1] or float(block[:, column].max()) <= 0:
                missing.append(name)
        segment["part_check"] = "fail" if missing else "pass"
        segment["missing_parts"] = missing
    return segments
