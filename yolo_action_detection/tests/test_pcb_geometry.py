"""PCB 元器件空间归属测试。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from pcb_inspection.geometry import (
    ComponentOwnershipResolver,
    expand_polygon,
    overlap_ratio,
    center_in_polygon,
    compute_ownership_score,
)


def _rect_poly(x, y, w, h):
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def test_expand_polygon_grows():
    """扩张后多边形面积增大。"""
    poly = _rect_poly(100, 100, 100, 100)
    expanded = expand_polygon(poly, 0.2)
    # 扩张后顶点距离中心更远
    cx, cy = 150, 150
    orig_dist = ((poly[0][0] - cx) ** 2 + (poly[0][1] - cy) ** 2) ** 0.5
    exp_dist = ((expanded[0][0] - cx) ** 2 + (expanded[0][1] - cy) ** 2) ** 0.5
    assert exp_dist > orig_dist


def test_overlap_ratio_full_containment():
    """元器件完全在 PCB 内时交叠比例为 1.0。"""
    comp = _rect_poly(110, 110, 20, 20)
    pcb = _rect_poly(100, 100, 100, 100)
    assert overlap_ratio(comp, pcb) == pytest.approx(1.0, abs=0.01)


def test_overlap_ratio_no_intersection():
    """分离的多边形交叠为 0。"""
    comp = _rect_poly(500, 500, 20, 20)
    pcb = _rect_poly(100, 100, 100, 100)
    assert overlap_ratio(comp, pcb) == 0.0


def test_overlap_ratio_partial():
    """部分交叠时比例在 0-1 之间。"""
    comp = _rect_poly(190, 190, 20, 20)
    pcb = _rect_poly(100, 100, 100, 100)
    r = overlap_ratio(comp, pcb)
    assert 0.0 < r < 1.0


def test_center_in_polygon_inside():
    assert center_in_polygon((150, 150), _rect_poly(100, 100, 100, 100)) is True


def test_center_in_polygon_outside():
    assert center_in_polygon((500, 500), _rect_poly(100, 100, 100, 100)) is False


def test_resolver_single_pcb_center_inside():
    """元器件中心在 PCB 内 -> 直接归属。"""
    resolver = ComponentOwnershipResolver(margin_ratio=0.15)
    pcb_dets = [{"track_id": 1, "polygon": _rect_poly(100, 100, 200, 200), "center": (200, 200)}]
    owner, ambiguous = resolver.resolve(
        component_center=(150, 150),
        component_poly=_rect_poly(140, 140, 20, 20),
        pcb_detections=pcb_dets,
    )
    assert owner == 1
    assert ambiguous == []


def test_resolver_no_pcb():
    """没有 PCB 检测时不归属。"""
    resolver = ComponentOwnershipResolver()
    owner, ambiguous = resolver.resolve(
        component_center=(150, 150),
        component_poly=_rect_poly(140, 140, 20, 20),
        pcb_detections=[],
    )
    assert owner is None
    assert ambiguous == []


def test_resolver_component_outside_all_pcb():
    """元器件在所有 PCB 之外 -> 不归属。"""
    resolver = ComponentOwnershipResolver()
    pcb_dets = [{"track_id": 1, "polygon": _rect_poly(100, 100, 50, 50), "center": (125, 125)}]
    owner, ambiguous = resolver.resolve(
        component_center=(500, 500),
        component_poly=_rect_poly(490, 490, 20, 20),
        pcb_detections=pcb_dets,
    )
    assert owner is None
    assert ambiguous == []


def test_resolver_two_pcb_clear_assignment():
    """两块 PCB 明确分离，元器件归属正确的一块。"""
    resolver = ComponentOwnershipResolver(margin_ratio=0.15)
    pcb_dets = [
        {"track_id": 1, "polygon": _rect_poly(0, 0, 200, 200), "center": (100, 100)},
        {"track_id": 2, "polygon": _rect_poly(500, 0, 200, 200), "center": (600, 100)},
    ]
    # 元器件在第一块 PCB 内
    owner, ambiguous = resolver.resolve(
        component_center=(100, 100),
        component_poly=_rect_poly(90, 90, 20, 20),
        pcb_detections=pcb_dets,
    )
    assert owner == 1
    assert ambiguous == []


def test_resolver_ambiguous_returns_none():
    """元器件在两块 PCB 边界附近 -> 歧义，不归属。"""
    resolver = ComponentOwnershipResolver(margin_ratio=0.5, ownership_threshold=0.01, safe_gap=0.9)
    # 两块 PCB 紧挨着
    pcb_dets = [
        {"track_id": 1, "polygon": _rect_poly(0, 0, 100, 100), "center": (50, 50)},
        {"track_id": 2, "polygon": _rect_poly(100, 0, 100, 100), "center": (150, 50)},
    ]
    # 元器件在边界上，两块都部分覆盖
    owner, ambiguous = resolver.resolve(
        component_center=(100, 50),
        component_poly=_rect_poly(90, 40, 20, 20),
        pcb_detections=pcb_dets,
    )
    assert owner is None
    assert len(ambiguous) == 2


def test_resolver_board_edge_component():
    """板边元器件中心在 PCB 外但交叠比例足够高 -> 仍归属。"""
    resolver = ComponentOwnershipResolver(margin_ratio=0.3, ownership_threshold=0.1)
    pcb_dets = [{"track_id": 1, "polygon": _rect_poly(100, 100, 100, 100), "center": (150, 150)}]
    # 元器件中心在 PCB 右边外一点，但有较大交叠
    owner, ambiguous = resolver.resolve(
        component_center=(195, 150),
        component_poly=_rect_poly(180, 140, 30, 20),
        pcb_detections=pcb_dets,
    )
    assert owner == 1


def test_resolver_multiple_same_class_takes_best():
    """同类元器件多个检测时，由调用方取最高置信度；resolver 处理单个。"""
    resolver = ComponentOwnershipResolver()
    pcb_dets = [{"track_id": 1, "polygon": _rect_poly(100, 100, 100, 100), "center": (150, 150)}]
    owner, _ = resolver.resolve(
        component_center=(130, 130),
        component_poly=_rect_poly(120, 120, 20, 20),
        pcb_detections=pcb_dets,
    )
    assert owner == 1