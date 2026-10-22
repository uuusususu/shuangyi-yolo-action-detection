"""按帧抽出特征。检测和手部模型由调用方传入，保证同一帧。"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .features import build_feature_row
from .timeline import sort_image_names

Detect = Callable[[np.ndarray], list[dict]]
Hands = Callable[[np.ndarray], list[list[tuple[float, float, float]]]]


def rows_from_frames(
    frames: list[np.ndarray],
    detect: Detect,
    hands: Hands,
    part_class_names: list[str],
    conf_threshold: float,
) -> np.ndarray:
    return rows_and_stats_from_frames(frames, detect, hands, part_class_names, conf_threshold)[0]


def iter_frames(path: Path):
    """一帧一帧读。整段视频不再先堆进内存。"""
    import cv2

    path = Path(path)
    if path.is_dir():
        names = sort_image_names([item.name for item in path.iterdir() if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
        if not names:
            raise ValueError(f"{path} 里没有连续图")
        for name in names:
            image = cv2.imread(str(path / name))
            if image is None:
                raise ValueError(f"打不开图片 {name}")
            yield image
        return
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"打不开视频 {path.name}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame
    finally:
        capture.release()


def rows_and_stats_from_frames(
    frames: list[np.ndarray],
    detect: Detect,
    hands: Hands,
    part_class_names: list[str],
    conf_threshold: float,
    on_frame=None,
    frame_count: int | None = None,
    include_hands: bool = True,
) -> tuple[np.ndarray, dict]:
    """抽特征，并统计没检出手、没检出零件的帧数。零件判空直接看特征行，不另写一套阈值。"""
    rows = []
    no_hand = 0
    no_part = 0
    total = int(frame_count or 0)
    if total <= 0:
        try:
            total = len(frames)
        except TypeError:
            total = 0
    for frame in frames:
        height, width = frame.shape[:2]
        detections = detect(frame)
        hand_points = hands(frame) if include_hands else []
        row = build_feature_row(
            detections,
            hand_points,
            width,
            height,
            part_class_names,
            conf_threshold,
            include_hands=include_hands,
        )
        rows.append(row)
        if include_hands and not hand_points:
            no_hand += 1
        if not any(row[index * 5] for index in range(len(part_class_names))):
            no_part += 1
        if on_frame is not None and (len(rows) == total or len(rows) % 8 == 0):
            on_frame(len(rows), total or len(rows))
    if not rows:
        raise ValueError("没有帧")
    array = np.stack(rows, axis=0)
    count = len(rows)
    stats = {
        "frames": count,
        "feature_length": int(array.shape[1]),
        "no_hand_frames": no_hand,
        "no_hand_ratio": no_hand / count,
        "no_part_frames": no_part,
        "no_part_ratio": no_part / count,
    }
    return array, stats


def probe_media(path: Path) -> dict:
    import cv2

    path = Path(path)
    if path.is_dir():
        names = sort_image_names(
            [item.name for item in path.iterdir() if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        )
        if not names:
            raise ValueError(f"{path.name} 里没有连续图")
        image = cv2.imread(str(path / names[0]))
        if image is None:
            raise ValueError(f"打不开图片 {names[0]}")
        height, width = image.shape[:2]
        return {"frame_count": len(names), "fps": 0.0, "width": int(width), "height": int(height), "kind": "images"}
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"打不开图片 {path.name}")
        height, width = image.shape[:2]
        return {"frame_count": 1, "fps": 0.0, "width": int(width), "height": int(height), "kind": "image"}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"打不开视频 {path.name}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"读不到视频信息 {path.name}")
    return {"frame_count": frame_count, "fps": fps, "width": width, "height": height, "kind": "video"}


def read_frames(path: Path) -> list[np.ndarray]:
    import cv2

    path = Path(path)
    if path.is_dir():
        names = sort_image_names([item.name for item in path.iterdir() if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
        if not names:
            raise ValueError(f"{path} 里没有连续图")
        frames = []
        for name in names:
            image = cv2.imread(str(path / name))
            if image is None:
                raise ValueError(f"打不开图片 {name}")
            frames.append(image)
        return frames
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"打不开视频 {path.name}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError(f"视频没有帧 {path.name}")
    return frames
