"""形状匹配（shape-based matching）的验收测试。

这些断言锁住的正是现场暴露的三个失效模式：
  1. 旋转 1° 得分保持率 —— 旧实现 22%，本实现必须 > 85%
  2. 噪声与光照鲁棒性 —— 旧实现 sigma=2 就跌破阈值
  3. 得分的物理含义 —— 遮挡比例应近似等于得分损失，操作员才能据此设阈值
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from template_matching.shape_model import (  # noqa: E402
    ShapeFailure,
    ShapeModel,
    ShapeModelSettings,
)


def make_scene(width=640, height=480, cx=320.0, cy=240.0, angle=0.0,
               noise=0.0, gain=1.0, bias=0.0, occlude=0.0, seed=0):
    """合成一个带方向多样的刚体目标：矩形外框 + 两个圆孔 + 一条斜边。"""
    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), 40, np.uint8)
    box = cv2.boxPoints(((cx, cy), (170, 110), angle)).astype(np.int32)
    cv2.fillPoly(image, [box], (200, 200, 200))
    cv2.polylines(image, [box], True, (255, 255, 255), 2)
    rad = math.radians(angle)

    def local(dx, dy):
        return (int(cx + dx * math.cos(rad) - dy * math.sin(rad)),
                int(cy + dx * math.sin(rad) + dy * math.cos(rad)))

    cv2.circle(image, local(-45, 0), 13, (30, 30, 30), -1)
    cv2.circle(image, local(45, 0), 13, (30, 30, 30), -1)
    cv2.line(image, local(-70, -40), local(20, 40), (60, 60, 60), 3)
    if occlude > 0.0:
        w = int(170 * occlude)
        pts = cv2.boxPoints(((cx - 85 + w / 2, cy), (max(1, w), 130), angle)).astype(np.int32)
        cv2.fillPoly(image, [pts], (40, 40, 40))
    if gain != 1.0 or bias != 0.0:
        image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    if noise > 0.0:
        image = np.clip(image + rng.normal(0, noise, image.shape), 0, 255).astype(np.uint8)
    return image


@pytest.fixture(scope="module")
def model_and_scene():
    scene = make_scene()
    template = scene[175:305, 225:415].copy()
    model = ShapeModel(template, ShapeModelSettings.from_legacy(
        match_threshold=0.6, angle_range=15.0))
    return model, scene, template


def search_rect(cx, cy, half_w, half_h, width, height):
    """搜索框 = 在这个框内查找（模型必须完整落在框内）。"""
    return {"x": (cx - half_w) / width, "y": (cy - half_h) / height,
            "width": 2 * half_w / width, "height": 2 * half_h / height}


def roi_around(model, cx, cy, travel, width, height):
    """围绕 (cx, cy) 构造一个留出 travel 像素平移余量的搜索框。"""
    need_w, need_h = model.min_search_size(travel)
    return search_rect(cx, cy, need_w / 2, need_h / 2, width, height)


# ----------------------------------------------------------------------
# 建模
# ----------------------------------------------------------------------
def test_model_builds_pyramid_with_auto_settings(model_and_scene):
    model, _scene, template = model_and_scene
    assert model.level_count >= 3
    assert model.point_counts[0] >= 100
    assert model.point_counts[-1] >= 30
    # 每上一层点数递减，角度步进翻倍
    assert list(model.point_counts) == sorted(model.point_counts, reverse=True)
    assert 0.05 < model.base_angle_step < 5.0
    assert model.orientation_entropy() > 0.5
    assert model.template_size == (template.shape[1], template.shape[0])


def test_model_rejects_featureless_template():
    blank = np.full((120, 160, 3), 128, np.uint8)
    with pytest.raises(ValueError):
        ShapeModel(blank)


def test_settings_validation():
    with pytest.raises(ValueError):
        ShapeModelSettings(min_score=0.0).validate()
    with pytest.raises(ValueError):
        ShapeModelSettings(polarity="weird").validate()
    with pytest.raises(ValueError):
        ShapeModelSettings(angle_step=-1.0).validate()


# ----------------------------------------------------------------------
# 角度响应宽度：现场失效模式 1
# ----------------------------------------------------------------------
def test_angle_response_width_is_wider_than_one_degree(model_and_scene):
    """现场实测旧实现偏 1° 保持率 22%，本实现必须 > 85%。"""
    from template_matching.shape_model import _score_at, _unit_gradient

    model, scene, _template = model_and_scene
    gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    nx, ny, _mag = _unit_gradient(gray, model.settings.mag_floor)
    level = model._levels[0]
    base, _ = _score_at(nx, ny, level, 320.0, 240.0, 0.0, "use")
    assert base > 0.9, base

    retention = {}
    for deviation in (1.0, 2.0, 3.0):
        values = [
            _score_at(nx, ny, level, 320.0, 240.0, sign * deviation, "use")[0]
            for sign in (1.0, -1.0)
        ]
        retention[deviation] = (sum(values) / len(values)) / base
    assert retention[1.0] > 0.85, retention
    assert retention[2.0] > 0.60, retention
    # 仍然要有角度选择性，否则角度估计无意义
    assert retention[3.0] < retention[1.0]


@pytest.mark.parametrize("rotation", [0.0, 0.5, 1.0, 2.0, 5.0, 8.0, 12.0, -3.0, -9.0])
def test_finds_rotated_part_with_stable_score(model_and_scene, rotation):
    model, _scene, _template = model_and_scene
    scene = make_scene(angle=rotation)
    height, width = scene.shape[:2]
    result = model.find(scene, roi_around(model, 320, 240, 60, width, height))
    assert result.matched, (rotation, result.score, result.failure)
    assert result.score > 0.85, (rotation, result.score)
    assert result.center is not None
    assert math.hypot(result.center[0] - 320.0, result.center[1] - 240.0) < 2.0
    # 角度符号与 OpenCV RotatedRect / cv2.boxPoints 一致（旧实现是反的）
    assert abs(result.angle - rotation) < 1.5, (rotation, result.angle)


def test_subpixel_and_subangle_accuracy(model_and_scene):
    model, _scene, _template = model_and_scene
    scene = make_scene(cx=320.4, cy=239.6, angle=1.4)
    height, width = scene.shape[:2]
    result = model.find(scene, roi_around(model, 320, 240, 50, width, height))
    assert result.matched
    assert abs(result.center[0] - 320.4) < 1.2, result.center
    assert abs(result.center[1] - 239.6) < 1.2, result.center
    assert abs(result.angle - 1.4) < 0.8, result.angle


# ----------------------------------------------------------------------
# 噪声与光照：现场失效模式 2
# ----------------------------------------------------------------------
@pytest.mark.parametrize("noise", [0.0, 2.0, 4.0, 8.0])
def test_noise_robustness(model_and_scene, noise):
    model, _scene, _template = model_and_scene
    scene = make_scene(noise=noise, seed=11)
    height, width = scene.shape[:2]
    result = model.find(scene, roi_around(model, 320, 240, 50, width, height))
    assert result.matched, (noise, result.score, result.failure)
    assert result.score > 0.80, (noise, result.score)


@pytest.mark.parametrize("gain,bias", [(0.6, 0.0), (1.4, 0.0), (1.0, 40.0), (0.8, -20.0)])
def test_illumination_robustness(model_and_scene, gain, bias):
    model, _scene, _template = model_and_scene
    scene = make_scene(gain=gain, bias=bias)
    height, width = scene.shape[:2]
    result = model.find(scene, roi_around(model, 320, 240, 50, width, height))
    assert result.matched, (gain, bias, result.score, result.failure)
    assert result.score > 0.80, (gain, bias, result.score)


# ----------------------------------------------------------------------
# 得分的物理含义：现场失效模式 3（阈值可解释）
# ----------------------------------------------------------------------
def test_score_tracks_visible_edge_ratio(model_and_scene):
    """遮挡 x% 边缘，得分应大致下降 x% —— 操作员才能据此设阈值。"""
    model, _scene, _template = model_and_scene
    height, width = 480, 640
    scores = {}
    for occlusion in (0.0, 0.2, 0.4):
        scene = make_scene(occlude=occlusion)
        result = model.find(scene, roi_around(model, 320, 240, 50, width, height),
                            min_score=0.05)
        scores[occlusion] = result.score
    assert scores[0.0] > 0.9
    assert scores[0.0] > scores[0.2] > scores[0.4]
    for occlusion in (0.2, 0.4):
        expected = scores[0.0] * (1.0 - occlusion)
        assert abs(scores[occlusion] - expected) < 0.18, scores


def test_rejects_clutter_and_blank(model_and_scene):
    model, _scene, _template = model_and_scene
    rng = np.random.default_rng(5)
    clutter = np.clip(np.full((480, 640, 3), 40) + rng.normal(0, 35, (480, 640, 3)),
                      0, 255).astype(np.uint8)
    for _ in range(14):
        cv2.rectangle(
            clutter,
            (int(rng.integers(0, 560)), int(rng.integers(0, 400))),
            (int(rng.integers(60, 640)), int(rng.integers(60, 480))),
            (int(rng.integers(60, 255)),) * 3, int(rng.integers(1, 4)),
        )
    result = model.find(clutter, roi_around(model, 320, 240, 50, 640, 480))
    assert not result.matched
    # 关键不是绝对值，而是与真阳（约 0.99）之间要有足够间隔供阈值落脚
    clutter_score = result.score
    true_positive = model.find(make_scene(), roi_around(model, 320, 240, 50, 640, 480)).score
    assert true_positive - clutter_score > 0.35, (true_positive, clutter_score)

    blank = np.full((480, 640, 3), 128, np.uint8)
    result = model.find(blank, roi_around(model, 320, 240, 50, 640, 480))
    assert not result.matched
    assert result.failure == ShapeFailure.SEARCH_NO_GRADIENT


# ----------------------------------------------------------------------
# 搜索窗语义与结构化失败原因
# ----------------------------------------------------------------------
def test_too_small_roi_is_rejected_with_required_size(model_and_scene):
    """框太小必须显式报错并给出所需尺寸，不能像旧实现那样静默丢弃旋转候选。

    现场就是栽在这里：294x170 的框对 200x156 的模板只剩 14px 纵向余量，
    31 个角度候选里 22 个被悄悄跳过，表现为“随机不触发”。
    """
    model, scene, _template = model_and_scene
    height, width = scene.shape[:2]
    bbox_w, bbox_h = model.rotated_bbox(0)

    tiny = search_rect(320, 240, bbox_w / 2 - 20, bbox_h / 2 - 20, width, height)
    result = model.find(scene, tiny)
    assert not result.matched
    assert result.failure == ShapeFailure.SEARCH_ROI_TOO_SMALL
    assert result.required_roi_px == (bbox_w, bbox_h)
    assert result.center_travel_px is not None
    assert min(result.center_travel_px) < 0
    assert result.rotated_bbox_px == (bbox_w, bbox_h)


def test_reports_center_travel_headroom(model_and_scene):
    """成功匹配时也要报出余量，供现场判断这个框够不够抗停位波动。"""
    model, scene, _template = model_and_scene
    height, width = scene.shape[:2]
    result = model.find(scene, roi_around(model, 320, 240, 50, width, height))
    assert result.matched
    travel = result.center_travel_px
    assert travel is not None
    assert travel[0] >= 90 and travel[1] >= 90        # 留了 ±50px
    roi = result.search_roi_px
    bbox_w, bbox_h = model.rotated_bbox(0)
    assert roi[2] - roi[0] - bbox_w == travel[0]
    assert roi[3] - roi[1] - bbox_h == travel[1]


def test_min_search_size_and_center_travel_are_consistent(model_and_scene):
    model, _scene, _template = model_and_scene
    bbox_w, bbox_h = model.rotated_bbox(0)
    assert model.min_search_size(0) == (bbox_w, bbox_h)
    assert model.min_search_size(40) == (bbox_w + 80, bbox_h + 80)
    assert model.center_travel((bbox_w + 80, bbox_h + 80)) == (80, 80)


def test_reports_clipped_roi_at_frame_border(model_and_scene):
    model, _scene, _template = model_and_scene
    scene = make_scene(cx=100.0, cy=80.0)
    height, width = scene.shape[:2]
    need_w, need_h = model.min_search_size(60)
    rect = {"x": (100 - need_w / 2) / width, "y": (80 - need_h / 2) / height,
            "width": need_w / width, "height": need_h / height}
    result = model.find(scene, rect)
    # 框被画幅裁掉时必须显式报出来，而不是静默按裁剪后的范围搜索
    assert result.failure in (ShapeFailure.SEARCH_ROI_CLIPPED,
                              ShapeFailure.SEARCH_ROI_TOO_SMALL,
                              ShapeFailure.SCORE_BELOW_MIN)
    if result.failure == ShapeFailure.SEARCH_ROI_CLIPPED:
        assert result.matched


def test_invalid_and_empty_inputs(model_and_scene):
    model, scene, _template = model_and_scene
    assert model.find(None, None).failure == ShapeFailure.EMPTY_FRAME
    assert model.find(scene, "bad").failure == ShapeFailure.INVALID_SEARCH_RECT
    bad = model.find(scene, {"x": 0.5, "y": 0.5, "width": 0.0, "height": 0.0})
    assert bad.failure == ShapeFailure.INVALID_SEARCH_RECT


def test_reports_best_score_even_when_below_threshold(model_and_scene):
    """落选时也要给出最高可达得分，否则现场无法标定阈值。"""
    model, scene, _template = model_and_scene
    height, width = scene.shape[:2]
    result = model.find(scene, roi_around(model, 320, 240, 50, width, height), min_score=0.999)
    assert not result.matched
    assert result.failure == ShapeFailure.SCORE_BELOW_MIN
    assert result.score > 0.5, result.score


def test_angle_curve_is_reported_for_diagnostics(model_and_scene):
    model, scene, _template = model_and_scene
    height, width = scene.shape[:2]
    result = model.find(scene, roi_around(model, 320, 240, 50, width, height))
    assert len(result.angle_scores) >= 5
    angles = [a for a, _s in result.angle_scores]
    assert min(angles) <= -10.0 and max(angles) >= 10.0
    peak = max(result.angle_scores, key=lambda item: item[1])
    assert abs(peak[0]) < 3.0, result.angle_scores


# ----------------------------------------------------------------------
# 持久化与性能
# ----------------------------------------------------------------------
def test_save_and_load_roundtrip(tmp_path, model_and_scene):
    model, scene, _template = model_and_scene
    path = tmp_path / "locator_model.npz"
    model.save(path)
    assert path.is_file()
    assert not list(tmp_path.glob(".*tmp*"))       # 临时文件已清理

    loaded = ShapeModel.load(path)
    assert loaded.level_count == model.level_count
    assert loaded.point_counts == model.point_counts
    assert loaded.template_size == model.template_size
    assert loaded.template_image() is not None

    height, width = scene.shape[:2]
    rect = roi_around(model, 320, 240, 50, width, height)
    original = model.find(scene, rect)
    restored = loaded.find(scene, rect)
    assert abs(original.score - restored.score) < 1e-6
    assert restored.matched


def test_with_settings_reuses_points(model_and_scene):
    model, scene, _template = model_and_scene
    tightened = model.with_settings(
        ShapeModelSettings.from_legacy(match_threshold=0.999, angle_range=15.0))
    assert tightened.point_counts == model.point_counts
    height, width = scene.shape[:2]
    result = tightened.find(scene, roi_around(model, 320, 240, 50, width, height))
    assert not result.matched          # 阈值 0.999 过不了
    assert result.score > 0.85         # 但得分照样报出来


def test_performance_budget(model_and_scene):
    """典型现场尺寸下必须远快于旧实现（现场实测 660ms）。"""
    model, _scene, _template = model_and_scene
    scene = make_scene(angle=3.0)
    height, width = scene.shape[:2]
    rect = roi_around(model, 320, 240, 70, width, height)
    model.find(scene, rect)            # 预热核缓存
    started = time.perf_counter()
    runs = 5
    for _ in range(runs):
        model.find(scene, rect)
    elapsed_ms = (time.perf_counter() - started) / runs * 1000.0
    assert elapsed_ms < 120.0, elapsed_ms


def test_ignore_polarity_mode_also_matches(model_and_scene):
    """对比反转的工件（暗底亮件 <-> 亮底暗件）需要 ignore_global。"""
    model, scene, template = model_and_scene
    inverted_scene = 255 - scene
    height, width = scene.shape[:2]
    rect = roi_around(model, 320, 240, 50, width, height)

    signed = model.find(inverted_scene, rect)
    assert not signed.matched          # use_polarity 下极性反转应当匹配不上

    tolerant = ShapeModel(template, ShapeModelSettings.from_legacy(
        match_threshold=0.6, angle_range=6.0).__class__(
            min_score=0.6, angle_start=-6.0, angle_extent=12.0,
            polarity="ignore_global"))
    result = tolerant.find(inverted_scene, rect)
    assert result.matched, (result.score, result.failure)
    assert result.score > 0.8
