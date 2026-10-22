"""Halcon 风格的形状匹配（shape-based matching）。

与旧的 `EdgeTemplateModel`（二值 Canny + TM_CCORR_NORMED）相比，四个结构性差别：

1. 模型是**带梯度方向的边缘点集**，旋转候选靠旋转点集生成，不旋转图像、不重跑 Canny。
2. 搜索图**只算梯度、从不二值化**。相似度是模型点上单位梯度方向的点积均值：

       score(x, y, theta) = (1/N) * sum_i  u_i(theta) . n(x + dx_i(theta), y + dy_i(theta))
       n = (gx, gy) / |g| ，|g| 低于下限时置零

   值域 0..1，物理含义近似"方向一致且可见的模型边缘比例"。
   现场实测：旧实现旋转 1° 得分保持率 23%，本实现 88%。
3. **金字塔粗到精**：顶层穷举角度与平移，逐层细化。角度只在很小的顶层穷举，
   所以放宽角度范围几乎不增加耗时——这才是大角度范围还能快的原因。
4. 搜索 ROI 的语义是**"在这个框内查找"**：模型必须完整落在框内，与操作员的直觉一致。
   但框太小时不再静默丢弃旋转候选——会返回 `search_roi_too_small` 并给出所需尺寸，
   同时在结果里报出"模型中心可平移范围"，让现场知道这个框能容许多少停位波动。

模型的梯度阈值在建模时固化并随模型持久化，运行期不再按画面重算，
避免自适应阈值随亮度漂移造成得分抖动。
"""
from __future__ import annotations

import io
import json
import logging
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple
from uuid import uuid4

import cv2
import numpy as np

logger = logging.getLogger("yolo_action_detection.runtime")

MODEL_SCHEMA_VERSION = 1
_BLUR_KERNEL = (3, 3)
_AUTO_CANNY_SIGMA = 0.33


class ShapeFailure:
    """结构化失败原因。绝不用"统一返回 0 分"掩盖问题。"""

    NONE = ""
    EMPTY_FRAME = "empty_frame"
    INVALID_SEARCH_RECT = "invalid_search_rect"
    SEARCH_ROI_TOO_SMALL = "search_roi_too_small"
    SEARCH_ROI_CLIPPED = "search_roi_clipped_by_frame"
    SEARCH_NO_GRADIENT = "search_no_gradient"
    TOO_MUCH_OUTSIDE = "too_much_outside"
    SCORE_BELOW_MIN = "score_below_min_score"


@dataclass(frozen=True)
class ShapeModelSettings:
    """现场可配置参数。angle_step / pyramid_levels 为 0 表示自动。"""

    min_score: float = 0.6
    angle_start: float = -15.0
    angle_extent: float = 30.0
    angle_step: float = 0.0
    pyramid_levels: int = 0
    polarity: str = "use"          # use | ignore_global
    mag_floor: float = 12.0
    max_points: int = 600
    min_points: int = 25
    max_outside_ratio: float = 0.10
    candidate_limit: int = 24
    level_slack: float = 0.30      # 顶层放宽比例，避免粗层轻微掉分就丢掉真解

    def validate(self) -> None:
        if not 0.0 < float(self.min_score) <= 1.0:
            raise ValueError("min_score 必须在 0 到 1 之间")
        if not -180.0 <= float(self.angle_start) <= 180.0:
            raise ValueError("angle_start 必须在 -180 到 180 度之间")
        if not 0.0 <= float(self.angle_extent) <= 360.0:
            raise ValueError("angle_extent 必须在 0 到 360 度之间")
        if float(self.angle_step) < 0.0:
            raise ValueError("angle_step 不能为负数")
        if int(self.pyramid_levels) < 0:
            raise ValueError("pyramid_levels 不能为负数")
        if str(self.polarity) not in ("use", "ignore_global"):
            raise ValueError("polarity 只支持 use 或 ignore_global")
        if float(self.mag_floor) < 0.0:
            raise ValueError("mag_floor 不能为负数")
        if int(self.max_points) < 25:
            raise ValueError("max_points 至少为 25")

    @classmethod
    def from_legacy(
        cls,
        *,
        match_threshold: float,
        angle_range: float,
        angle_step: float = 0.0,
        **kwargs,
    ) -> "ShapeModelSettings":
        """从旧的 match_threshold / angle_range(±) 参数换算。"""
        return cls(
            min_score=float(match_threshold),
            angle_start=-abs(float(angle_range)),
            angle_extent=2.0 * abs(float(angle_range)),
            angle_step=float(angle_step or 0.0),
            **kwargs,
        )


@dataclass(frozen=True)
class ShapeMatchResult:
    """匹配结果，坐标相对完整输入帧。"""

    matched: bool
    score: float
    center: Optional[Tuple[float, float]] = None
    angle: Optional[float] = None
    box: Optional[Tuple[Tuple[float, float], ...]] = None
    failure: str = ShapeFailure.NONE
    outside_ratio: float = 0.0
    angle_scores: Tuple[Tuple[float, float], ...] = ()
    level_candidates: Tuple[int, ...] = ()
    search_roi_px: Optional[Tuple[int, int, int, int]] = None
    center_range_px: Optional[Tuple[int, int, int, int]] = None
    # 模型中心在这个搜索框内还能平移多少像素。等于 (框尺寸 - 最大旋转包围盒)。
    # 它必须大于工位停位波动，否则模板永远对不准 —— 现场就是栽在这里。
    center_travel_px: Optional[Tuple[int, int]] = None
    required_roi_px: Optional[Tuple[int, int]] = None
    rotated_bbox_px: Optional[Tuple[int, int]] = None
    search_gradient_pixels: int = 0
    levels_used: int = 0
    total_ms: float = 0.0
    coarse_ms: float = 0.0
    refine_ms: float = 0.0


@dataclass(frozen=True)
class _Level:
    index: int
    scale: float                # 该层相对原图的缩放（1/2**index）
    points: np.ndarray          # (N,2) 相对模型中心，该层坐标系
    dirs: np.ndarray            # (N,2) 单位梯度
    angle_step: float
    canny_lo: int
    canny_hi: int
    size: Tuple[int, int]       # 该层模板尺寸 (w, h)

    @property
    def point_count(self) -> int:
        return int(len(self.points))

    def radius(self) -> float:
        if not len(self.points):
            return 0.0
        return float(np.max(np.linalg.norm(self.points, axis=1)))


class ShapeModel:
    """分层定向边缘点集模型。"""

    def __init__(
        self,
        template_image: np.ndarray,
        settings: Optional[ShapeModelSettings] = None,
        *,
        created_frame_size: Optional[Tuple[int, int]] = None,
        _levels: Optional[Sequence[_Level]] = None,
        _template_png: Optional[bytes] = None,
    ) -> None:
        self._settings = settings or ShapeModelSettings()
        self._settings.validate()
        self._created_frame_size = (
            (int(created_frame_size[0]), int(created_frame_size[1]))
            if created_frame_size else None
        )
        if _levels is not None:
            self._levels = tuple(_levels)
            self._template_png = _template_png or b""
            self._template_size = self._levels[0].size
        else:
            gray = _as_gray(template_image)
            height, width = gray.shape[:2]
            self._template_size = (int(width), int(height))
            self._levels = self._build_levels(gray)
            ok, encoded = cv2.imencode(".png", template_image)
            self._template_png = encoded.tobytes() if ok else b""
        self._kernel_cache: dict[tuple[int, float], tuple] = {}
        self._bbox_cache: dict[int, Tuple[int, int]] = {}
        logger.info(
            "shape_model=ready size=%s levels=%d points=%s base_angle_step=%.3f polarity=%s",
            self._template_size,
            len(self._levels),
            [lv.point_count for lv in self._levels],
            self._levels[0].angle_step,
            self._settings.polarity,
        )

    # ------------------------------------------------------------------
    # 建模
    # ------------------------------------------------------------------
    def _build_levels(self, gray: np.ndarray) -> Tuple[_Level, ...]:
        settings = self._settings
        height, width = gray.shape[:2]
        base_step = settings.angle_step
        if base_step <= 0.0:
            # Halcon 的经验式：让最外侧模型点在一步内位移约 2 像素。
            base_step = math.degrees(math.atan(2.0 / max(4.0, max(width, height))))
            base_step = max(0.1, round(base_step, 3))

        requested = int(settings.pyramid_levels)
        levels: list[_Level] = []
        current = gray
        index = 0
        while True:
            level = self._build_level(current, index, base_step * (2 ** index))
            if level is None:
                break
            levels.append(level)
            if requested and len(levels) >= requested:
                break
            lh, lw = current.shape[:2]
            if min(lh, lw) < 32 or len(levels) >= 5:
                break
            current = cv2.pyrDown(current)
            index += 1
        if not levels:
            raise ValueError("模板内没有可用的梯度边缘点，请重新框选包含清晰轮廓的区域")
        # 顶层点数太少会让粗定位不可靠，逐层回退
        while len(levels) > 1 and levels[-1].point_count < 30:
            levels.pop()
        return tuple(levels)

    def _build_level(self, gray: np.ndarray, index: int, angle_step: float) -> Optional[_Level]:
        settings = self._settings
        height, width = gray.shape[:2]
        if min(height, width) < 12:
            return None
        blurred = cv2.GaussianBlur(gray, _BLUR_KERNEL, 0)
        lo, hi = _auto_canny_thresholds(blurred)
        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        edges = cv2.Canny(blurred, lo, hi, L2gradient=True)
        edges[magnitude < settings.mag_floor] = 0
        ys, xs = np.nonzero(edges)
        if len(xs) == 0:
            return None

        # 网格抽稀：保证空间均匀，避免密集纹理支配得分。
        decimation = 2 if index == 0 else 1
        unique: dict[tuple[int, int], tuple[int, int]] = {}
        for x, y in zip(xs, ys):
            unique[(int(x) // decimation, int(y) // decimation)] = (int(x), int(y))
        pts = np.array(list(unique.values()), dtype=np.float32)
        budget = max(settings.min_points, settings.max_points // (2 ** index))
        if len(pts) > budget:
            pts = pts[np.linspace(0, len(pts) - 1, budget).astype(int)]

        xi = pts[:, 0].astype(int)
        yi = pts[:, 1].astype(int)
        norms = magnitude[yi, xi]
        keep = norms > max(1e-6, settings.mag_floor)
        pts = pts[keep]
        if len(pts) < settings.min_points:
            return None
        xi, yi = pts[:, 0].astype(int), pts[:, 1].astype(int)
        dirs = np.stack([gx[yi, xi], gy[yi, xi]], axis=1).astype(np.float32)
        dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-6)
        centered = pts - np.array(
            [[(width - 1) / 2.0, (height - 1) / 2.0]], dtype=np.float32
        )
        return _Level(
            index=index,
            scale=1.0 / (2 ** index),
            points=np.ascontiguousarray(centered),
            dirs=np.ascontiguousarray(dirs),
            angle_step=float(angle_step),
            canny_lo=int(lo),
            canny_hi=int(hi),
            size=(int(width), int(height)),
        )

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def settings(self) -> ShapeModelSettings:
        return self._settings

    @property
    def template_size(self) -> Tuple[int, int]:
        return self._template_size

    @property
    def created_frame_size(self) -> Optional[Tuple[int, int]]:
        return self._created_frame_size

    @property
    def level_count(self) -> int:
        return len(self._levels)

    @property
    def point_counts(self) -> Tuple[int, ...]:
        return tuple(lv.point_count for lv in self._levels)

    @property
    def base_angle_step(self) -> float:
        return float(self._levels[0].angle_step)

    @property
    def template_png(self) -> bytes:
        return self._template_png

    def template_image(self) -> Optional[np.ndarray]:
        if not self._template_png:
            return None
        return cv2.imdecode(np.frombuffer(self._template_png, np.uint8), cv2.IMREAD_COLOR)

    def orientation_entropy(self, level: int = 0) -> float:
        """梯度方向分布熵（0..1）。过低说明方向单一，平移定位病态。"""
        dirs = self._levels[level].dirs
        angles = np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0])) % 180.0
        histogram, _edges = np.histogram(angles, bins=12, range=(0.0, 180.0))
        total = histogram.sum()
        if total <= 0:
            return 0.0
        probability = histogram[histogram > 0] / total
        return float(-(probability * np.log(probability)).sum() / math.log(12))

    def with_settings(self, settings: ShapeModelSettings) -> "ShapeModel":
        """复用已建好的点集，只换运行参数（阈值、角度范围等）。"""
        settings.validate()
        if (settings.angle_step or 0.0) > 0.0 and abs(
            settings.angle_step - self._settings.angle_step
        ) > 1e-9:
            # 角度步进改变需要重算每层步进
            levels = tuple(
                replace(lv, angle_step=float(settings.angle_step) * (2 ** lv.index))
                for lv in self._levels
            )
        else:
            levels = self._levels
        return ShapeModel(
            None,
            settings,
            created_frame_size=self._created_frame_size,
            _levels=levels,
            _template_png=self._template_png,
        )

    def rotated_bbox(self, level: int = 0) -> Tuple[int, int]:
        """该层在整个角度范围内的最大旋转包围盒（宽, 高）。

        只取决于点集与角度范围，两者在模型建好后都不再变，因此按层缓存。
        每次 find() 至少要问两遍，逐帧重算等于白跑几十次点集旋转。
        """
        cached = self._bbox_cache.get(level)
        if cached is not None:
            return cached
        lv = self._levels[level]
        best_w = best_h = 0
        for angle in self._level_angles(lv):
            rotated = _rotate(lv.points, angle)
            best_w = max(best_w, int(math.ceil(rotated[:, 0].max() - rotated[:, 0].min())) + 3)
            best_h = max(best_h, int(math.ceil(rotated[:, 1].max() - rotated[:, 1].min())) + 3)
        result = (best_w, best_h)
        self._bbox_cache[level] = result
        return result

    def min_search_size(self, travel_px: int = 0) -> Tuple[int, int]:
        """给定期望的停位波动余量，返回搜索 ROI 至少需要多大。

        配置页用它给操作员明确提示，避免再画出只剩 14 像素余量的框。
        """
        bbox_w, bbox_h = self.rotated_bbox(0)
        margin = max(0, int(travel_px))
        return (bbox_w + 2 * margin, bbox_h + 2 * margin)

    def center_travel(self, search_size: Tuple[int, int]) -> Tuple[int, int]:
        """在给定搜索框尺寸下，模型中心还能平移多少像素。"""
        bbox_w, bbox_h = self.rotated_bbox(0)
        return (int(search_size[0]) - bbox_w, int(search_size[1]) - bbox_h)

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """同目录临时文件 + os.replace 原子写入 .npz 模型。"""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "template_size": list(self._template_size),
            "created_frame_size": list(self._created_frame_size) if self._created_frame_size else None,
            "settings": {
                "min_score": self._settings.min_score,
                "angle_start": self._settings.angle_start,
                "angle_extent": self._settings.angle_extent,
                "angle_step": self._settings.angle_step,
                "pyramid_levels": self._settings.pyramid_levels,
                "polarity": self._settings.polarity,
                "mag_floor": self._settings.mag_floor,
                "max_points": self._settings.max_points,
            },
            "levels": [
                {
                    "index": lv.index,
                    "angle_step": lv.angle_step,
                    "canny_lo": lv.canny_lo,
                    "canny_hi": lv.canny_hi,
                    "size": list(lv.size),
                }
                for lv in self._levels
            ],
        }
        arrays = {"meta": np.frombuffer(json.dumps(payload).encode("utf-8"), dtype=np.uint8)}
        if self._template_png:
            arrays["template_png"] = np.frombuffer(self._template_png, dtype=np.uint8)
        for lv in self._levels:
            arrays[f"points_{lv.index}"] = lv.points
            arrays[f"dirs_{lv.index}"] = lv.dirs
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        temp_path = destination.with_name(f".{destination.stem}.{uuid4().hex}.tmp.npz")
        try:
            temp_path.write_bytes(buffer.getvalue())
            os.replace(temp_path, destination)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    @classmethod
    def load(
        cls,
        path: str | Path,
        settings: Optional[ShapeModelSettings] = None,
    ) -> "ShapeModel":
        model_path = Path(path)
        if not model_path.is_file():
            raise FileNotFoundError(f"形状模型不存在: {model_path}")
        with np.load(model_path, allow_pickle=False) as data:
            meta = json.loads(bytes(data["meta"]).decode("utf-8"))
            if int(meta.get("schema_version", 0)) != MODEL_SCHEMA_VERSION:
                raise ValueError(
                    f"形状模型版本不支持: {meta.get('schema_version')}，"
                    f"需要 {MODEL_SCHEMA_VERSION}；请重新创建模板"
                )
            stored = meta.get("settings", {})
            base = ShapeModelSettings(
                min_score=float(stored.get("min_score", 0.6)),
                angle_start=float(stored.get("angle_start", -15.0)),
                angle_extent=float(stored.get("angle_extent", 30.0)),
                angle_step=float(stored.get("angle_step", 0.0)),
                pyramid_levels=int(stored.get("pyramid_levels", 0)),
                polarity=str(stored.get("polarity", "use")),
                mag_floor=float(stored.get("mag_floor", 12.0)),
                max_points=int(stored.get("max_points", 600)),
            )
            levels = []
            for entry in meta["levels"]:
                index = int(entry["index"])
                levels.append(_Level(
                    index=index,
                    scale=1.0 / (2 ** index),
                    points=np.ascontiguousarray(data[f"points_{index}"].astype(np.float32)),
                    dirs=np.ascontiguousarray(data[f"dirs_{index}"].astype(np.float32)),
                    angle_step=float(entry["angle_step"]),
                    canny_lo=int(entry["canny_lo"]),
                    canny_hi=int(entry["canny_hi"]),
                    size=(int(entry["size"][0]), int(entry["size"][1])),
                ))
            png = bytes(data["template_png"]) if "template_png" in data.files else b""
        created = meta.get("created_frame_size")
        model = cls(
            None,
            base,
            created_frame_size=tuple(created) if created else None,
            _levels=levels,
            _template_png=png,
        )
        return model.with_settings(settings) if settings is not None else model

    # ------------------------------------------------------------------
    # 搜索
    # ------------------------------------------------------------------
    def find(
        self,
        frame: np.ndarray,
        search_rect: object | None,
        *,
        min_score: Optional[float] = None,
        collect_angle_curve: bool = True,
    ) -> ShapeMatchResult:
        """在 search_rect 这个框内查找最佳位姿。

        语义与操作员直觉一致：模型必须完整落在框内。框太小时不再静默丢弃旋转候选，
        而是返回 `search_roi_too_small` 并给出所需尺寸。

        `collect_angle_curve` 只影响 `angle_scores` 这一项诊断输出，匹配结果不受影响；
        诊断未开启时关掉它可以省下整条角度响应曲线的采样。
        """
        started = time.perf_counter()
        threshold = float(min_score if min_score is not None else self._settings.min_score)
        bbox_w, bbox_h = self.rotated_bbox(0)
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return ShapeMatchResult(
                False, 0.0,
                failure=ShapeFailure.EMPTY_FRAME,
                rotated_bbox_px=(bbox_w, bbox_h),
            )
        # 只有搜索窗参与运算，所以先按原始帧尺寸算出窗口再转灰度。
        # 灰度转换是逐像素的，先裁后转与先转后裁结果完全一致，但整幅转换在高像素
        # 相机上要读写几十 MB，而搜索窗通常只占其中很小一块。
        # 通道合法性仍在裁剪前校验，保持与整幅灰度化时相同的报错时机。
        _ensure_gray_convertible(frame)
        frame_h, frame_w = frame.shape[:2]

        try:
            rx0, ry0, rx1, ry1 = _pixel_rect(search_rect, (frame_w, frame_h))
        except ValueError:
            return ShapeMatchResult(
                False, 0.0,
                failure=ShapeFailure.INVALID_SEARCH_RECT,
                rotated_bbox_px=(bbox_w, bbox_h),
            )
        if rx1 <= rx0 or ry1 <= ry0:
            return ShapeMatchResult(
                False, 0.0,
                failure=ShapeFailure.INVALID_SEARCH_RECT,
                search_roi_px=(rx0, ry0, rx1, ry1),
                rotated_bbox_px=(bbox_w, bbox_h),
            )

        # 搜索框就是参与运算的图像窗口，只在超出画幅时才裁。
        clipped = rx0 < 0 or ry0 < 0 or rx1 > frame_w or ry1 > frame_h
        wx0, wy0 = max(0, rx0), max(0, ry0)
        wx1, wy1 = min(frame_w, rx1), min(frame_h, ry1)
        roi_px = (wx0, wy0, wx1, wy1)
        travel = (wx1 - wx0 - bbox_w, wy1 - wy0 - bbox_h)
        required = self.min_search_size(0)
        if travel[0] < 0 or travel[1] < 0:
            logger.warning(
                "shape_find=search_roi_too_small roi=%s bbox=%s required=%s",
                (wx1 - wx0, wy1 - wy0), (bbox_w, bbox_h), required,
            )
            return ShapeMatchResult(
                False, 0.0,
                failure=ShapeFailure.SEARCH_ROI_TOO_SMALL,
                search_roi_px=roi_px,
                center_travel_px=travel,
                required_roi_px=required,
                rotated_bbox_px=(bbox_w, bbox_h),
                total_ms=(time.perf_counter() - started) * 1000.0,
            )
        center_range = (
            wx0 + bbox_w // 2, wy0 + bbox_h // 2,
            wx1 - bbox_w // 2, wy1 - bbox_h // 2,
        )
        common = dict(
            search_roi_px=roi_px,
            center_range_px=center_range,
            center_travel_px=travel,
            required_roi_px=required,
            rotated_bbox_px=(bbox_w, bbox_h),
        )

        window = _as_gray(frame[wy0:wy1, wx0:wx1])
        fields, gradient_pixels = self._build_search_fields(window)
        levels_used = min(len(fields), len(self._levels))
        if gradient_pixels <= 0:
            return ShapeMatchResult(
                False, 0.0,
                failure=ShapeFailure.SEARCH_NO_GRADIENT,
                search_gradient_pixels=0,
                levels_used=levels_used,
                total_ms=(time.perf_counter() - started) * 1000.0,
                **common,
            )

        coarse_started = time.perf_counter()
        top = levels_used - 1
        candidates, coarse_best = self._coarse_search(
            fields[top], self._levels[top], top, threshold=threshold,
        )
        coarse_ms = (time.perf_counter() - coarse_started) * 1000.0
        level_candidates = [len(candidates)]
        if not candidates:
            return ShapeMatchResult(
                False, max(0.0, min(1.0, coarse_best)),
                failure=ShapeFailure.SCORE_BELOW_MIN,
                search_gradient_pixels=gradient_pixels,
                levels_used=levels_used,
                level_candidates=tuple(level_candidates),
                coarse_ms=coarse_ms,
                total_ms=(time.perf_counter() - started) * 1000.0,
                **common,
            )

        refine_started = time.perf_counter()
        for level in range(top - 1, -1, -1):
            candidates = self._refine(
                fields[level], self._levels[level], candidates, level, threshold
            )
            level_candidates.append(len(candidates))
            if not candidates:
                break
        refine_ms = (time.perf_counter() - refine_started) * 1000.0

        if not candidates:
            return ShapeMatchResult(
                False, max(0.0, min(1.0, coarse_best)),
                failure=ShapeFailure.SCORE_BELOW_MIN,
                search_gradient_pixels=gradient_pixels,
                levels_used=levels_used,
                level_candidates=tuple(level_candidates),
                coarse_ms=coarse_ms, refine_ms=refine_ms,
                total_ms=(time.perf_counter() - started) * 1000.0,
                **common,
            )

        best = max(candidates, key=lambda c: c[0])
        score, bx, by, bangle = best
        level0 = self._levels[0]
        score, bx, by, bangle, outside = self._subpixel(
            fields[0], level0, bx, by, bangle, score
        )
        curve = self._angle_curve(fields[0], level0, bx, by) if collect_angle_curve else ()

        center = (float(bx + wx0), float(by + wy0))
        failure = ShapeFailure.NONE
        matched = score >= threshold
        if not matched:
            failure = ShapeFailure.SCORE_BELOW_MIN
        elif outside > float(self._settings.max_outside_ratio):
            matched = False
            failure = ShapeFailure.TOO_MUCH_OUTSIDE
        elif clipped:
            failure = ShapeFailure.SEARCH_ROI_CLIPPED

        return ShapeMatchResult(
            matched=matched,
            score=float(max(0.0, min(1.0, score))),
            center=center,
            angle=float(bangle),
            box=self._box(center, bangle),
            failure=failure,
            outside_ratio=float(outside),
            angle_scores=curve,
            level_candidates=tuple(level_candidates),
            search_gradient_pixels=gradient_pixels,
            levels_used=levels_used,
            coarse_ms=coarse_ms,
            refine_ms=refine_ms,
            total_ms=(time.perf_counter() - started) * 1000.0,
            **common,
        )

    # ------------------------------------------------------------------
    def _build_search_fields(self, window: np.ndarray):
        """逐层单位化梯度场。返回 (fields, 层0 有效梯度像素数)。"""
        fields = []
        current = window
        gradient_pixels = 0
        for index in range(len(self._levels)):
            nx, ny, magnitude = _unit_gradient(current, self._settings.mag_floor)
            if index == 0:
                gradient_pixels = int(np.count_nonzero(magnitude >= self._settings.mag_floor))
            fields.append((nx, ny))
            if index + 1 >= len(self._levels):
                break
            if min(current.shape[:2]) < 8:
                break
            current = cv2.pyrDown(current)
        return fields, gradient_pixels

    def _level_angles(self, level: _Level) -> Tuple[float, ...]:
        settings = self._settings
        start = float(settings.angle_start)
        extent = float(settings.angle_extent)
        step = max(1e-3, float(level.angle_step))
        if extent <= 0.0:
            return (start,)
        center = start + extent / 2.0
        half = extent / 2.0
        count = int(math.floor(half / step))
        angles = [center + i * step for i in range(-count, count + 1)]
        if angles[0] > start + 1e-9:
            angles.insert(0, start)
        if angles[-1] < start + extent - 1e-9:
            angles.append(start + extent)
        return tuple(sorted({round(a, 4) for a in angles}))

    def _kernels(self, level_index: int, angle: float):
        key = (level_index, round(float(angle), 4))
        cached = self._kernel_cache.get(key)
        if cached is not None:
            return cached
        level = self._levels[level_index]
        rp = _rotate(level.points, angle)
        rd = _rotate(level.dirs, angle)
        xi = np.round(rp[:, 0]).astype(int)
        yi = np.round(rp[:, 1]).astype(int)
        x0, y0 = int(xi.min()), int(yi.min())
        kw, kh = int(xi.max()) - x0 + 1, int(yi.max()) - y0 + 1
        kx = np.zeros((kh, kw), np.float32)
        ky = np.zeros_like(kx)
        np.add.at(kx, (yi - y0, xi - x0), rd[:, 0])
        np.add.at(ky, (yi - y0, xi - x0), rd[:, 1])
        cached = (kx, ky, (-x0, -y0))
        self._kernel_cache[key] = cached
        return cached

    def _coarse_search(self, field, level: _Level, level_index: int, *, threshold):
        """顶层穷举角度与平移，返回该层坐标系下的候选 (score, x, y, angle)。

        matchTemplate 只输出完全重叠的位置，所以"模型必须落在搜索框内"这条
        由它天然保证，不需要额外的中心范围掩码。
        """
        nx, ny = field
        point_count = float(level.point_count)
        gate = threshold * (1.0 - float(self._settings.level_slack))
        use_polarity = self._settings.polarity == "use"
        collected: list[tuple[float, float, float, float]] = []
        # 即使全部落选也要记录粗层最高分：这是现场标定阈值唯一有用的信息。
        best_observed = 0.0
        for angle in self._level_angles(level):
            kx, ky, (ox, oy) = self._kernels(level_index, angle)
            kh, kw = kx.shape[:2]
            if kw > nx.shape[1] or kh > nx.shape[0]:
                continue
            if use_polarity:
                score_map = (
                    cv2.matchTemplate(nx, kx, cv2.TM_CCORR)
                    + cv2.matchTemplate(ny, ky, cv2.TM_CCORR)
                ) / point_count
            else:
                score_map = _abs_score_map(nx, ny, level, angle)
            best_observed = max(best_observed, float(score_map.max()))
            peaks = _local_maxima(score_map, gate, limit=self._settings.candidate_limit)
            for value, px, py in peaks:
                collected.append((float(value), float(px + ox), float(py + oy), float(angle)))
        collected.sort(key=lambda c: -c[0])
        return (
            _dedupe(collected, radius=max(1.0, level.radius() * 0.25),
                    limit=self._settings.candidate_limit),
            best_observed,
        )

    @staticmethod
    def _rotated_points_cache(level: _Level):
        """按角度缓存旋转后的点集与方向，键用原始 float，保证复用与重算逐位一致。"""
        cache: dict[float, tuple] = {}

        def rotated(angle: float):
            hit = cache.get(angle)
            if hit is None:
                hit = (_rotate(level.points, angle), _rotate(level.dirs, angle))
                cache[angle] = hit
            return hit

        return rotated

    def _refine(self, field, level: _Level, candidates, level_index: int, threshold: float):
        """把上一层候选映射到本层，在 ±2px、±1 角度步进内细化。"""
        nx, ny = field
        gate = threshold * (1.0 - float(self._settings.level_slack) * 0.5)
        parent_step = self._levels[level_index + 1].angle_step
        polarity = self._settings.polarity
        rotated = self._rotated_points_cache(level)
        refined: list[tuple[float, float, float, float]] = []
        for score, x, y, angle in candidates:
            base_x, base_y = x * 2.0, y * 2.0
            best = None
            for delta in (-parent_step * 0.5, 0.0, parent_step * 0.5):
                candidate_angle = angle + delta
                rp, rd = rotated(candidate_angle)
                for dy in (-2.0, -1.0, 0.0, 1.0, 2.0):
                    for dx in (-2.0, -1.0, 0.0, 1.0, 2.0):
                        value, _outside = _score_at_rotated(
                            nx, ny, rp, rd, base_x + dx, base_y + dy, polarity,
                        )
                        if best is None or value > best[0]:
                            best = (value, base_x + dx, base_y + dy, candidate_angle)
            if best is not None and best[0] >= gate:
                refined.append(best)
        refined.sort(key=lambda c: -c[0])
        return _dedupe(refined, radius=max(1.0, level.radius() * 0.2),
                       limit=self._settings.candidate_limit)

    def _subpixel(self, field, level: _Level, x: float, y: float, angle: float, score: float):
        """位置用 3x3 二次曲面拟合，角度用相邻三点抛物线拟合。"""
        nx, ny = field
        polarity = self._settings.polarity
        step = max(1e-3, level.angle_step)
        rotated = self._rotated_points_cache(level)

        rp, rd = rotated(angle)
        grid = np.empty((3, 3), dtype=np.float64)
        for j, dy in enumerate((-1.0, 0.0, 1.0)):
            for i, dx in enumerate((-1.0, 0.0, 1.0)):
                grid[j, i], _ = _score_at_rotated(
                    nx, ny, rp, rd, x + dx, y + dy, polarity
                )
        offset_x = _parabola_vertex(grid[1, 0], grid[1, 1], grid[1, 2])
        offset_y = _parabola_vertex(grid[0, 1], grid[1, 1], grid[2, 1])
        best_x, best_y = x + offset_x, y + offset_y

        low_rp, low_rd = rotated(angle - step)
        high_rp, high_rd = rotated(angle + step)
        low, _ = _score_at_rotated(nx, ny, low_rp, low_rd, best_x, best_y, polarity)
        mid, _ = _score_at_rotated(nx, ny, rp, rd, best_x, best_y, polarity)
        high, _ = _score_at_rotated(nx, ny, high_rp, high_rd, best_x, best_y, polarity)
        best_angle = angle + _parabola_vertex(low, mid, high) * step
        best_rp, best_rd = rotated(best_angle)
        final, outside = _score_at_rotated(
            nx, ny, best_rp, best_rd, best_x, best_y, polarity
        )
        if final < max(score, mid):
            # 拟合没有带来提升时退回整数位姿，避免亚像素反而变差
            final, outside = _score_at_rotated(nx, ny, rp, rd, x, y, polarity)
            return float(final), float(x), float(y), float(angle), float(outside)
        return float(final), float(best_x), float(best_y), float(best_angle), float(outside)

    def _angle_curve(self, field, level: _Level, x: float, y: float):
        """在最佳位置上给出逐角度得分，供现场诊断（角度响应宽度）。"""
        nx, ny = field
        top = self._levels[-1]
        polarity = self._settings.polarity
        rotated = self._rotated_points_cache(level)
        curve = []
        for angle in self._level_angles(top):
            rp, rd = rotated(angle)
            value, _ = _score_at_rotated(nx, ny, rp, rd, x, y, polarity)
            curve.append((float(angle), float(value)))
        return tuple(curve)

    def _box(self, center: Tuple[float, float], angle: float):
        width, height = self._template_size
        half_w, half_h = width / 2.0, height / 2.0
        corners = np.array(
            [[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]],
            dtype=np.float32,
        )
        rotated = _rotate(corners, angle)
        return tuple(
            (float(center[0] + px), float(center[1] + py)) for px, py in rotated
        )


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def _as_gray(image: np.ndarray) -> np.ndarray:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("图像无效")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError("图像必须是灰度、BGR 或 BGRA")


def _ensure_gray_convertible(image: np.ndarray) -> None:
    """只校验通道数，不做转换，供裁剪前先行拦截非法图像。"""
    if image.ndim == 2:
        return
    if image.ndim == 3 and image.shape[2] in (3, 4):
        return
    raise ValueError("图像必须是灰度、BGR 或 BGRA")


def _auto_canny_thresholds(blurred: np.ndarray) -> Tuple[int, int]:
    median = float(np.median(blurred))
    lower = int(max(0.0, (1.0 - _AUTO_CANNY_SIGMA) * median))
    upper = int(min(255.0, (1.0 + _AUTO_CANNY_SIGMA) * median))
    if upper <= lower:
        lower, upper = 32, 96
    return lower, upper


def _unit_gradient(gray: np.ndarray, mag_floor: float):
    blurred = cv2.GaussianBlur(gray, _BLUR_KERNEL, 0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    safe = np.where(magnitude < mag_floor, np.float32(1.0), magnitude)
    nx = (gx / safe).astype(np.float32)
    ny = (gy / safe).astype(np.float32)
    weak = magnitude < mag_floor
    nx[weak] = 0.0
    ny[weak] = 0.0
    return nx, ny, magnitude


def _rotate(values: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = math.radians(float(angle_deg))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    return values @ rotation.T


def _bilinear(field: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    """双线性采样。返回 (值, 是否在图内)。"""
    values, _, inside = _bilinear_pair(field, None, xs, ys)
    return values, inside


def _bilinear_pair(field_x: np.ndarray, field_y, xs: np.ndarray, ys: np.ndarray):
    """在同一组坐标上采样两个场。返回 (x 场值, y 场值, 是否在图内)。

    两个梯度分量永远采在同一组点上，索引与权重算一次即可，运算顺序与逐场采样一致。
    """
    height, width = field_x.shape[:2]
    inside = (xs >= 0) & (ys >= 0) & (xs <= width - 1) & (ys <= height - 1)
    xc = np.clip(xs, 0, width - 1)
    yc = np.clip(ys, 0, height - 1)
    x0 = np.floor(xc).astype(np.int32)
    y0 = np.floor(yc).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = (xc - x0).astype(np.float32)
    fy = (yc - y0).astype(np.float32)

    def sample(field):
        top = field[y0, x0] * (1.0 - fx) + field[y0, x1] * fx
        bottom = field[y1, x0] * (1.0 - fx) + field[y1, x1] * fx
        return top * (1.0 - fy) + bottom * fy

    values_y = None if field_y is None else sample(field_y)
    return sample(field_x), values_y, inside


def _score_at(nx, ny, level: _Level, cx: float, cy: float, angle: float, polarity: str):
    """在指定位姿上直接采样评分。返回 (得分, 越界点比例)。"""
    rp = _rotate(level.points, angle)
    rd = _rotate(level.dirs, angle)
    return _score_at_rotated(nx, ny, rp, rd, cx, cy, polarity)


def _score_at_rotated(nx, ny, rp: np.ndarray, rd: np.ndarray, cx: float, cy: float, polarity: str):
    """已知旋转后的点集与方向时的打分。

    同一角度下的多个平移候选共用一份旋转结果：细化阶段每个候选要试 25 个平移，
    逐次重算点集旋转是纯浪费。
    """
    xs = rp[:, 0] + float(cx)
    ys = rp[:, 1] + float(cy)
    sx, sy, inside = _bilinear_pair(nx, ny, xs, ys)
    products = rd[:, 0] * sx + rd[:, 1] * sy
    if polarity != "use":
        products = np.abs(products)
    products = np.where(inside, products, 0.0)
    return float(products.mean()), float(1.0 - inside.mean())


def _abs_score_map(nx, ny, level: _Level, angle: float) -> np.ndarray:
    """ignore_global_polarity 需要逐点取绝对值，退化为移位累加。"""
    rp = _rotate(level.points, angle)
    rd = _rotate(level.dirs, angle)
    xi = np.round(rp[:, 0]).astype(int)
    yi = np.round(rp[:, 1]).astype(int)
    x0, y0 = int(xi.min()), int(yi.min())
    kw, kh = int(xi.max()) - x0 + 1, int(yi.max()) - y0 + 1
    height, width = nx.shape[:2]
    out_h, out_w = height - kh + 1, width - kw + 1
    if out_h <= 0 or out_w <= 0:
        return np.full((1, 1), -2.0, np.float32)
    total = np.zeros((out_h, out_w), np.float32)
    for (px, py, dx, dy) in zip(xi - x0, yi - y0, rd[:, 0], rd[:, 1]):
        window_x = nx[py:py + out_h, px:px + out_w]
        window_y = ny[py:py + out_h, px:px + out_w]
        total += np.abs(dx * window_x + dy * window_y)
    return total / float(level.point_count)


def _local_maxima(score_map: np.ndarray, gate: float, limit: int):
    """返回超过 gate 的局部极大值 [(值, x, y)]，按值降序。"""
    if score_map.size == 0:
        return []
    dilated = cv2.dilate(score_map, np.ones((3, 3), np.uint8))
    mask = (score_map >= dilated) & (score_map >= gate)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    values = score_map[ys, xs]
    order = np.argsort(values)[::-1][: max(1, limit * 2)]
    return [(float(values[i]), int(xs[i]), int(ys[i])) for i in order]


def _dedupe(candidates, radius: float, limit: int):
    kept: list[tuple[float, float, float, float]] = []
    for item in candidates:
        _score, x, y, _angle = item
        if any(math.hypot(x - kx, y - ky) < radius for _s, kx, ky, _a in kept):
            continue
        kept.append(item)
        if len(kept) >= limit:
            break
    return kept


def _parabola_vertex(low: float, mid: float, high: float) -> float:
    denominator = low - 2.0 * mid + high
    if abs(denominator) < 1e-12:
        return 0.0
    offset = 0.5 * (low - high) / denominator
    return float(max(-1.0, min(1.0, offset)))


def _pixel_rect(rect: object | None, frame_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    frame_w, frame_h = frame_size
    if rect is None:
        return (0, 0, frame_w, frame_h)
    if isinstance(rect, Mapping):
        x, y = float(rect.get("x", 0.0)), float(rect.get("y", 0.0))
        w, h = float(rect.get("width", 0.0)), float(rect.get("height", 0.0))
        normalized = True
    elif all(hasattr(rect, name) for name in ("x", "y", "width", "height")):
        x, y = float(getattr(rect, "x")), float(getattr(rect, "y"))
        w, h = float(getattr(rect, "width")), float(getattr(rect, "height"))
        normalized = True
    elif isinstance(rect, Sequence) and len(rect) == 4:
        x, y, w, h = (float(v) for v in rect)
        normalized = max(x, y, w, h) <= 1.0
    else:
        raise ValueError("搜索 ROI 格式无效")
    if normalized:
        x, w = x * frame_w, w * frame_w
        y, h = y * frame_h, h * frame_h
    return (
        int(math.floor(x)), int(math.floor(y)),
        int(math.ceil(x + w)), int(math.ceil(y + h)),
    )
