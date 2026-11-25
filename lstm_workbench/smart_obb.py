"""智能标注掩码到归一化 OBB 旋转框的边界安全拟合工具（纯 OpenCV/NumPy，无 PyTorch 依赖）。"""
from __future__ import annotations

import math

import cv2
import numpy as np

_EPS = 1e-5
_MIN_AREA = 1e-6


def _order_quad_cw(points: np.ndarray) -> np.ndarray:
    """按围绕质心的顺时针顺序排列四顶点，确保凸四边形不交叉。"""
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    return pts[order]


def is_valid_obb(points: list[list[float]]) -> bool:
    """校验归一化四点是否满足 PartsStore 的 [0,1] 非退化凸四边形约束。"""
    if not isinstance(points, list) or len(points) != 4:
        return False
    for pt in points:
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            return False
        x, y = pt
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            return False
        if not (math.isfinite(x) and math.isfinite(y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return False
    poly = np.asarray(points, dtype=np.float32)
    return bool(cv2.isContourConvex(poly) and abs(float(cv2.contourArea(poly))) > _MIN_AREA)


def clamp_obb_to_unit_square(points_norm: np.ndarray) -> list[list[float]] | None:
    """将可能贴边越界的旋转框安全缩放回 [0, 1]^2 内，同时严格保持旋转矩形形状与凸性。"""
    pts = _order_quad_cw(points_norm)
    if not np.all(np.isfinite(pts)):
        return None

    # 已经在单位正方形内且合法，直接返回
    raw = [[float(p[0]), float(p[1])] for p in pts]
    if is_valid_obb(raw):
        return raw

    # 计算中心并确保中心落在内部
    cx = float(np.clip(pts[:, 0].mean(), 0.002, 0.998))
    cy = float(np.clip(pts[:, 1].mean(), 0.002, 0.998))
    center = np.array([cx, cy], dtype=np.float64)

    # 先按比例向中心收缩，使四顶点全部落入 [_EPS, 1 - _EPS]
    alpha = 1.0
    for vx, vy in pts:
        dx = vx - cx
        dy = vy - cy
        if vx < _EPS and dx < -1e-12:
            alpha = min(alpha, (cx - _EPS) / abs(dx))
        elif vx > 1.0 - _EPS and dx > 1e-12:
            alpha = min(alpha, (1.0 - _EPS - cx) / abs(dx))
        if vy < _EPS and dy < -1e-12:
            alpha = min(alpha, (cy - _EPS) / abs(dy))
        elif vy > 1.0 - _EPS and dy > 1e-12:
            alpha = min(alpha, (1.0 - _EPS - cy) / abs(dy))

    alpha = max(0.0, min(1.0, alpha * 0.999))
    if alpha > 1e-4:
        shrunk = center + (pts - center) * alpha
        shrunk = np.clip(shrunk, 0.0, 1.0)
        cand = [[float(p[0]), float(p[1])] for p in _order_quad_cw(shrunk)]
        if is_valid_obb(cand):
            return cand

    # 兜底：退回裁剪后的外接轴对齐矩形
    x1 = float(np.clip(pts[:, 0].min(), 0.0, 1.0))
    y1 = float(np.clip(pts[:, 1].min(), 0.0, 1.0))
    x2 = float(np.clip(pts[:, 0].max(), 0.0, 1.0))
    y2 = float(np.clip(pts[:, 1].max(), 0.0, 1.0))
    if x2 - x1 < 0.002:
        mid = float(np.clip((x1 + x2) * 0.5, 0.002, 0.998))
        x1, x2 = max(0.0, mid - 0.002), min(1.0, mid + 0.002)
    if y2 - y1 < 0.002:
        mid = float(np.clip((y1 + y2) * 0.5, 0.002, 0.998))
        y1, y2 = max(0.0, mid - 0.002), min(1.0, mid + 0.002)
    fallback = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return fallback if is_valid_obb(fallback) else None


def _simplify_contour(contour: np.ndarray, width: int, height: int, max_points: int = 64) -> list[list[float]]:
    """将最大轮廓简化为归一化多边形点集，用于前端轻量半透明掩码叠加预览。"""
    peri = cv2.arcLength(contour, True)
    eps = max(1.0, 0.004 * peri)
    approx = cv2.approxPolyDP(contour, eps, True)
    while len(approx) > max_points and eps < peri * 0.1:
        eps *= 1.35
        approx = cv2.approxPolyDP(contour, eps, True)
    pts = approx.reshape(-1, 2)
    return [
        [round(float(np.clip(x / width, 0.0, 1.0)), 5), round(float(np.clip(y / height, 0.0, 1.0)), 5)]
        for x, y in pts
    ]


def mask_to_obb(
    mask: np.ndarray,
    width: int,
    height: int,
    *,
    fit_mode: str = "obb",
) -> dict | None:
    """将二值/浮点分割掩码拟合为合法的归一化 4 点 OBB（及预览多边形轮廓）。

    返回：{"points": [[x1,y1],...], "polygon": [[px,py],...], "area_ratio": float} 或 None。
    """
    if mask is None or width <= 0 or height <= 0:
        return None
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2 or arr.size == 0:
        return None

    binary = (arr > 0.5).astype(np.uint8) * 255
    if binary.shape[1] != width or binary.shape[0] != height:
        binary = cv2.resize(binary, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)

    fg_pixels = int(np.count_nonzero(binary))
    if fg_pixels < 4:
        return None

    # 仅在前景面积足够时执行轻度去噪，避免误抹细小零件
    if fg_pixels > 64:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        if int(np.count_nonzero(cleaned)) >= max(4, int(fg_pixels * 0.25)):
            binary = cleaned

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    areas = [float(cv2.contourArea(c)) for c in contours]
    max_area = max(areas) if areas else 0.0
    best_idx = int(np.argmax(areas))
    main_contour = contours[best_idx]

    if max_area < 1.0:
        # 极细长 1px 目标：退化到前景像素外接框
        ys, xs = np.nonzero(binary)
        if xs.size == 0:
            return None
        pts_px = np.stack([xs, ys], axis=1).astype(np.float32)
    else:
        significant = [c for c, a in zip(contours, areas) if a >= max(4.0, max_area * 0.12)]
        pts_px = np.vstack(significant).astype(np.float32)

    if fit_mode == "hbb":
        x, y, bw, bh = cv2.boundingRect(pts_px)
        x1 = max(0.0, x / width)
        y1 = max(0.0, y / height)
        x2 = min(1.0, (x + max(bw, 2)) / width)
        y2 = min(1.0, (y + max(bh, 2)) / height)
        obb = clamp_obb_to_unit_square(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64))
    else:
        rect = cv2.minAreaRect(pts_px)
        box_px = cv2.boxPoints(rect)
        box_norm = np.column_stack([box_px[:, 0] / float(width), box_px[:, 1] / float(height)])
        obb = clamp_obb_to_unit_square(box_norm)

    if obb is None:
        return None

    polygon = _simplify_contour(main_contour, width, height)
    area_ratio = float(fg_pixels) / float(width * height)
    return {
        "points": [[round(p[0], 6), round(p[1], 6)] for p in obb],
        "polygon": polygon,
        "area_ratio": round(area_ratio, 6),
    }


def bbox_xyxy_to_obb(bbox_xyxy: list[float], width: int, height: int) -> list[list[float]] | None:
    """将像素坐标 [x1, y1, x2, y2] 转为合法归一化 4 点 OBB。"""
    if not bbox_xyxy or len(bbox_xyxy) != 4 or width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    pts = np.array(
        [
            [x1 / width, y1 / height],
            [x2 / width, y1 / height],
            [x2 / width, y2 / height],
            [x1 / width, y2 / height],
        ],
        dtype=np.float64,
    )
    return clamp_obb_to_unit_square(pts)
