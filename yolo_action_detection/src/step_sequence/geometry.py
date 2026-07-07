"""Polygon 交集计算工具。"""
from typing import List, Tuple

import cv2
import math
import numpy as np


def polygon_area(polygon: List[Tuple[float, float]]) -> float:
    """计算 polygon 面积。"""
    if len(polygon) < 3:
        return 0.0
    pts = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return float(cv2.contourArea(pts))


def polygon_intersection_ratio(
    poly_a: List[Tuple[float, float]],
    poly_b: List[Tuple[float, float]],
) -> float:
    """计算两个 convex polygon 的交集面积与较小面积的比值。

    Returns:
        交集面积 / min(area_a, area_b)，如果面积为 0 返回 0.0。
    """
    if len(poly_a) < 3 or len(poly_b) < 3:
        return 0.0

    pts_a = np.array(poly_a, dtype=np.float32).reshape(-1, 1, 2)
    pts_b = np.array(poly_b, dtype=np.float32).reshape(-1, 1, 2)

    area_a = cv2.contourArea(pts_a)
    area_b = cv2.contourArea(pts_b)
    min_area = min(area_a, area_b)

    if min_area <= 0:
        return 0.0

    intersection_area, _ = cv2.intersectConvexConvex(pts_a, pts_b)
    return float(intersection_area) / min_area


def polygon_iou(
    poly_a: List[Tuple[float, float]],
    poly_b: List[Tuple[float, float]],
) -> float:
    """计算两个 convex polygon 的 IoU。"""
    if len(poly_a) < 3 or len(poly_b) < 3:
        return 0.0

    pts_a = np.array(poly_a, dtype=np.float32).reshape(-1, 1, 2)
    pts_b = np.array(poly_b, dtype=np.float32).reshape(-1, 1, 2)

    area_a = abs(float(cv2.contourArea(pts_a)))
    area_b = abs(float(cv2.contourArea(pts_b)))
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0

    intersection_area, _ = cv2.intersectConvexConvex(pts_a, pts_b)
    union = area_a + area_b - float(intersection_area)
    return float(intersection_area) / union if union > 0.0 else 0.0


def point_in_convex_polygon(point, polygon):
    """判断点是否在凸多边形内部（叉积法）。"""
    x, y = point
    n = len(polygon)
    if n < 3:
        return False

    signs = []
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if abs(cross) < 1e-10:
            return True
        signs.append(cross > 0)

    return all(signs) or not any(signs)


def any_corner_inside(poly_a, poly_b):
    """检测 poly_a 是否有任意角点在 poly_b 内部（双向检查）。"""
    for corner in poly_a:
        if point_in_convex_polygon(corner, poly_b):
            return True
    for corner in poly_b:
        if point_in_convex_polygon(corner, poly_a):
            return True
    return False


def center_distance_px(center_a, center_b):
    """两个中心点之间的像素距离。"""
    dx = center_a[0] - center_b[0]
    dy = center_a[1] - center_b[1]
    return math.sqrt(dx * dx + dy * dy)
