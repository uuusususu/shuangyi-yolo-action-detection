"""PCB 元器件空间归属几何计算。"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from step_sequence.geometry import polygon_iou, point_in_convex_polygon, polygon_area


def expand_polygon(polygon: List[Tuple[float, float]], ratio: float) -> List[Tuple[float, float]]:
    """以多边形中心为原点，按比例向外扩张顶点。"""
    if len(polygon) < 3 or ratio <= 0:
        return list(polygon)
    cx = sum(p[0] for p in polygon) / len(polygon)
    cy = sum(p[1] for p in polygon) / len(polygon)
    expanded = []
    for x, y in polygon:
        dx = x - cx
        dy = y - cy
        expanded.append((cx + dx * (1.0 + ratio), cy + dy * (1.0 + ratio)))
    return expanded


def overlap_ratio(component_poly: List[Tuple[float, float]],
                  pcb_poly: List[Tuple[float, float]]) -> float:
    """元器件多边形与 PCB 多边形的交叠面积 / 元器件面积。"""
    if len(component_poly) < 3 or len(pcb_poly) < 3:
        return 0.0
    comp_area = polygon_area(component_poly)
    if comp_area <= 0:
        return 0.0
    pts_a = np.array(component_poly, dtype=np.float32).reshape(-1, 1, 2)
    pts_b = np.array(pcb_poly, dtype=np.float32).reshape(-1, 1, 2)
    intersection_area, _ = cv2.intersectConvexConvex(pts_a, pts_b)
    return float(intersection_area) / comp_area


def center_in_polygon(center: Tuple[float, float],
                      polygon: List[Tuple[float, float]]) -> bool:
    """元器件中心点是否在 PCB 多边形内。"""
    return point_in_convex_polygon(center, polygon)


class OwnershipCandidate:
    """单个元器件对一块 PCB 的归属候选。"""

    __slots__ = ("pcb_track_id", "score", "overlap", "center_inside")

    def __init__(self, pcb_track_id: int, score: float, overlap: float, center_inside: bool):
        self.pcb_track_id = pcb_track_id
        self.score = score
        self.overlap = overlap
        self.center_inside = center_inside


def compute_ownership_score(
    component_center: Tuple[float, float],
    component_poly: List[Tuple[float, float]],
    pcb_poly: List[Tuple[float, float]],
    expanded_pcb_poly: List[Tuple[float, float]],
) -> Tuple[float, bool]:
    """计算元器件对一块 PCB 的归属得分和中心点是否在扩张区域内。

    得分 = max(overlap_ratio vs expanded, overlap_ratio vs original)
    中心在原始 PCB 内直接合格；否则用交叠比例判断。
    """
    center_inside_original = center_in_polygon(component_center, pcb_poly)
    if center_inside_original:
        # 中心在原始 PCB 内，得分为 1.0（最高）
        return 1.0, True

    center_inside_expanded = center_in_polygon(component_center, expanded_pcb_poly)
    overlap_orig = overlap_ratio(component_poly, pcb_poly)
    overlap_exp = overlap_ratio(component_poly, expanded_pcb_poly)
    score = max(overlap_orig, overlap_exp)

    return score, center_inside_expanded


# 归属合格阈值：交叠比例或中心点包含
OWNERSHIP_THRESHOLD = 0.15
# 安全差值：多候选时第一和第二的得分差需大于此值
OWNERSHIP_SAFE_GAP = 0.10


class ComponentOwnershipResolver:
    """把每个元器件检测归属到至多一块 PCB。

    规则：
    1. 优先判断中心点是否在 PCB 原始 OBB 内。
    2. 使用交叠比例作为归属得分。
    3. 多候选时选得分最高且与第二名差值足够大的 PCB。
    4. 歧义时不归属，且涉及的 PCB 本帧观测标记不可靠。
    """

    def __init__(
        self,
        margin_ratio: float = 0.15,
        ownership_threshold: float = OWNERSHIP_THRESHOLD,
        safe_gap: float = OWNERSHIP_SAFE_GAP,
    ) -> None:
        self._margin_ratio = margin_ratio
        self._threshold = ownership_threshold
        self._safe_gap = safe_gap

    def resolve(
        self,
        component_center: Tuple[float, float],
        component_poly: List[Tuple[float, float]],
        pcb_detections: List[dict],
    ) -> Tuple[Optional[int], List[int]]:
        """返回 (归属的 pcb_track_id, 歧义涉及的 pcb_track_id 列表)。

        pcb_detections: [{"track_id": int, "polygon": [...], "center": (x,y)}]
        """
        if not pcb_detections:
            return None, []

        candidates: List[OwnershipCandidate] = []
        for pcb in pcb_detections:
            tid = pcb.get("track_id")
            if tid is None:
                continue
            pcb_poly = pcb.get("polygon", [])
            if len(pcb_poly) < 3:
                continue
            expanded = expand_polygon(pcb_poly, self._margin_ratio)
            score, center_inside = compute_ownership_score(
                component_center, component_poly, pcb_poly, expanded
            )
            if score >= self._threshold or center_inside:
                candidates.append(OwnershipCandidate(tid, score, score, center_inside))

        if not candidates:
            return None, []

        if len(candidates) == 1:
            return candidates[0].pcb_track_id, []

        # 多候选：按得分排序
        candidates.sort(key=lambda c: c.score, reverse=True)
        best = candidates[0]
        second = candidates[1]

        if best.score - second.score >= self._safe_gap:
            return best.pcb_track_id, []

        # 歧义：返回 None，涉及的 PCB 全部标记不可靠
        ambiguous_ids = [c.pcb_track_id for c in candidates]
        return None, ambiguous_ids