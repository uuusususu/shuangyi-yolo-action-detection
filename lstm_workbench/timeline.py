"""时间轴：一条片子、每类一条轨道、空闲与交叠。"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# 窗口按秒定义，不按帧。帧数由时间基换算 —— 换相机时窗口的真实时长不变。
WINDOW_SECONDS = 1.92


def window_frames(time_base_fps: float, window_seconds: float = WINDOW_SECONDS) -> int:
    """窗口秒数 → 帧数。全工程只有这一处做这个换算，别在别处重算。

    window_seconds 不传就用默认的 WINDOW_SECONDS；工程设置里改过的值由调用方传进来。
    """
    if time_base_fps <= 0:
        raise ValueError("时间基帧率必须大于 0")
    if window_seconds <= 0:
        raise ValueError("窗口秒数必须大于 0")
    return max(1, int(round(window_seconds * time_base_fps)))


def window_text(time_base_fps: float, window_seconds: float = WINDOW_SECONDS) -> str:
    """给人看的窗口说明，界面和错误文案共用。"""
    return f"{window_seconds:g} 秒（{time_base_fps:g}fps 下 {window_frames(time_base_fps, window_seconds)} 帧）"


@dataclass
class ClassTrack:
    id: str
    name: str


@dataclass
class Segment:
    id: str
    class_id: str
    start: int
    end: int

    def length(self) -> int:
        return self.end - self.start


@dataclass
class TimelineIssue:
    code: str
    message: str


def numeric_image_key(name: str) -> tuple:
    numbers = [int(part) for part in re.findall(r"\d+", name)]
    return (numbers, name)


def sort_image_names(names: list[str]) -> list[str]:
    return sorted(names, key=numeric_image_key)


def overlaps(left: Segment, right: Segment) -> bool:
    return left.start < right.end and right.start < left.end


def find_overlaps(segments: list[Segment]) -> list[tuple[Segment, Segment]]:
    ordered = sorted(segments, key=lambda item: (item.start, item.end, item.id))
    found: list[tuple[Segment, Segment]] = []
    for index, segment in enumerate(ordered):
        for other in ordered[index + 1 :]:
            if other.start >= segment.end:
                break
            if overlaps(segment, other):
                found.append((segment, other))
    return found


def paint_idle(frame_count: int, segments: list[Segment], window: int) -> list[Segment]:
    """没涂到且连续达到窗口长度的区间变成空闲段。"""
    if frame_count <= 0:
        return []
    covered = sorted(segments, key=lambda item: item.start)
    idle: list[Segment] = []
    cursor = 0
    serial = 0
    for segment in covered:
        start = max(cursor, 0)
        end = min(segment.start, frame_count)
        if end - start >= window:
            idle.append(Segment(id=f"idle-{serial}", class_id="idle", start=start, end=end))
            serial += 1
        cursor = max(cursor, segment.end)
    if frame_count - cursor >= window:
        idle.append(Segment(id=f"idle-{serial}", class_id="idle", start=cursor, end=frame_count))
    return idle


def short_segments(segments: list[Segment], window: int) -> list[Segment]:
    return [segment for segment in segments if segment.length() < window]


def same_class_narrow_gaps(segments: list[Segment], window: int) -> list[tuple[Segment, Segment]]:
    gaps: list[tuple[Segment, Segment]] = []
    by_class: dict[str, list[Segment]] = {}
    for segment in segments:
        if segment.class_id == "idle":
            continue
        by_class.setdefault(segment.class_id, []).append(segment)
    for group in by_class.values():
        ordered = sorted(group, key=lambda item: item.start)
        for prev, nxt in zip(ordered, ordered[1:]):
            gap = nxt.start - prev.end
            if 0 <= gap < window:
                gaps.append((prev, nxt))
    return gaps


def merge_narrow_gap(segments: list[Segment], left_id: str, right_id: str, window: int) -> list[Segment]:
    by_id = {segment.id: segment for segment in segments}
    left = by_id[left_id]
    right = by_id[right_id]
    if left.class_id != right.class_id or left.class_id == "idle":
        raise ValueError("只能合并同一类别的相邻段")
    if not (0 <= right.start - left.end < window):
        raise ValueError("空档没有短于窗口，不能合并")
    merged = Segment(id=left.id, class_id=left.class_id, start=left.start, end=right.end)
    kept = [segment for segment in segments if segment.id not in {left.id, right.id}]
    kept.append(merged)
    return sorted(kept, key=lambda item: item.start)


def timeline_issues(
    frame_count: int,
    segments: list[Segment],
    window: int,
    window_seconds: float = WINDOW_SECONDS,
) -> list[TimelineIssue]:
    issues: list[TimelineIssue] = []
    for left, right in find_overlaps(segments):
        issues.append(TimelineIssue("overlap", f"{left.id} 与 {right.id} 交叠"))
    for segment in short_segments(segments, window):
        issues.append(TimelineIssue("short", f"{segment.id} 不足 {window_seconds:g} 秒（{window} 帧）"))
    for left, right in same_class_narrow_gaps(segments, window):
        issues.append(TimelineIssue("narrow-gap", f"{left.id} 与 {right.id} 空档不足 {window_seconds:g} 秒（{window} 帧）"))
    if frame_count < 0:
        issues.append(TimelineIssue("frames", "帧数不能为负"))
    return issues


@dataclass
class ClipAnnotation:
    clip_id: str
    frame_count: int
    fps: float
    width: int
    height: int
    segments: list[Segment] = field(default_factory=list)
    # 不参与训练的区间。不是类别，不会变成空闲，也不会产窗口。
    excluded: list[Segment] = field(default_factory=list)

    def trainable_segments(self, window: int) -> list[Segment]:
        covered = list(self.segments) + list(self.excluded)
        painted = [segment for segment in self.segments if segment.length() >= window]
        idle = paint_idle(self.frame_count, covered, window)
        return painted + idle
