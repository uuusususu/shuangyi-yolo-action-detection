"""特征行。零件类和手部分开，顺序写死。"""
from __future__ import annotations

import numpy as np

FEATURE_CODE_VERSION = "1"


def safe_feature_std(std) -> np.ndarray:
    """标准差小于 1e-3 的维换成 1。常数维若只抬到 1e-3，验证时一出现就会被放大上千倍。"""
    values = np.asarray(std, dtype=np.float32)
    return np.where(values < np.float32(1e-3), np.float32(1.0), values).astype(np.float32)


def feature_size(part_class_count: int, *, hands: bool = True) -> int:
    """每个零件 5 个数。手部参与时再加两只手、每只 21 点、每点 xyz。"""
    return part_class_count * 5 + (126 if hands else 0)


def hands_in_training(payload: dict | None) -> bool:
    """新设置看开关。旧契约没有这一项时，有手部指纹就仍按带手部处理。"""
    data = payload or {}
    if "hands_in_training" in data:
        return bool(data.get("hands_in_training"))
    return bool(str(data.get("hands_sha256") or ""))


def build_feature_row(
    detections: list[dict],
    hands: list[list[tuple[float, float, float]]],
    frame_w: int,
    frame_h: int,
    part_class_names: list[str],
    conf_threshold: float,
    *,
    include_hands: bool = True,
) -> list[float]:
    """detections: label, conf, box(x1,y1,x2,y2)。hands 为像素或已是任意坐标，这里按画面归一化。"""
    best: dict[str, list[float]] = {name: [0.0, 0.0, 0.0, 0.0, 0.0] for name in part_class_names}
    width = max(int(frame_w), 1)
    height = max(int(frame_h), 1)
    for detection in detections:
        name = str(detection.get("label", ""))
        if name not in best:
            continue
        conf = float(detection.get("conf", 0.0))
        if conf < conf_threshold or conf <= best[name][0]:
            continue
        x1, y1, x2, y2 = detection.get("box", (0, 0, 0, 0))
        box_w = max(0.0, float(x2) - float(x1))
        box_h = max(0.0, float(y2) - float(y1))
        cx = float(x1) + box_w / 2.0
        cy = float(y1) + box_h / 2.0
        best[name] = [conf, cx / width, cy / height, box_w / width, box_h / height]

    boxes: list[float] = []
    for name in part_class_names:
        boxes.extend(best[name])
    if not include_hands:
        return boxes

    ranked = []
    for hand in hands:
        if len(hand) < 21:
            continue
        palm_x = float(hand[0][0])
        palm_y = float(hand[0][1])
        ranked.append((palm_x, palm_y, hand))
    ranked.sort(key=lambda item: (item[0], item[1]))
    if len(ranked) > 2:
        ranked = [ranked[0], ranked[-1]]

    hand_values = [0.0] * 126
    for slot, (_x, _y, hand) in enumerate(ranked[:2]):
        for point_index, point in enumerate(hand[:21]):
            base = slot * 63 + point_index * 3
            hand_values[base] = float(point[0]) / width
            hand_values[base + 1] = float(point[1]) / height
            hand_values[base + 2] = float(point[2])
    return boxes + hand_values
