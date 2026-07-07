"""标定与距离算法。"""
from __future__ import annotations

from typing import List, Optional, Tuple

from calibration.models import (
    CalibrationTransform, HoleDefinition, HoleJudgement, HoleZone, TipPoint,
)


def distance_mm(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return (dx * dx + dy * dy) ** 0.5


def judge_hole_zone(tip_mm: Tuple[float, float], hole: HoleDefinition) -> HoleJudgement:
    dist = distance_mm(tip_mm, hole.center_mm)
    if dist <= hole.inner_radius_mm:
        return HoleJudgement(zone=HoleZone.INSIDE, distance_mm=dist, hole=hole)
    elif dist <= hole.outer_radius_mm:
        return HoleJudgement(zone=HoleZone.NEAR, distance_mm=dist, hole=hole)
    else:
        return HoleJudgement(zone=HoleZone.OUTSIDE, distance_mm=dist, hole=hole)


def find_nearest_hole(tip_mm: Tuple[float, float], holes: List[HoleDefinition]) -> Tuple[Optional[HoleDefinition], float]:
    best_hole = None
    best_dist = float("inf")
    for hole in holes:
        if not hole.enabled:
            continue
        dist = distance_mm(tip_mm, hole.center_mm)
        if dist < best_dist:
            best_dist = dist
            best_hole = hole
    return best_hole, best_dist


def extract_tip_point(detections, tool_class_name: str, calibrator: Optional[CalibrationTransform] = None, frame_id: int = 0, conf_threshold: float = 0.0) -> Tuple[Optional[TipPoint], List[str]]:
    diagnostics: List[str] = []
    tool_dets = [d for d in detections if d.label == tool_class_name]
    if not tool_dets:
        diagnostics.append("NoToolTipDetected")
        return None, diagnostics
    valid_dets = [d for d in tool_dets if d.conf >= conf_threshold]
    if not valid_dets:
        diagnostics.append(f"AllToolTipBelowThreshold({conf_threshold})")
        return None, diagnostics
    if len(valid_dets) > 1:
        diagnostics.append(f"DuplicateToolTipDetection({len(valid_dets)})")
    best = max(valid_dets, key=lambda d: d.conf)
    px, py = best.center
    mm_x, mm_y = 0.0, 0.0
    if calibrator and calibrator.is_valid:
        mm_x, mm_y = calibrator.pixel_to_mm(px, py)
    tip = TipPoint(px=px, py=py, mm_x=mm_x, mm_y=mm_y, conf=best.conf, track_id=best.track_id, frame_id=frame_id, label=best.label)
    return tip, diagnostics
