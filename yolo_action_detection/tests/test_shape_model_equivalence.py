"""模板匹配优化的等价性安全网。

这些用例不判断匹配质量好坏（那是 test_shape_model.py 的职责），只锁住一件事：
性能优化不允许改变 `ShapeModel.find()` 的任何输出数值。基线由优化前的实现生成，
优化后必须逐位复现，否则现场的定位结果与阈值标定就全部失效。

重新生成基线（仅当匹配算法本身有意变更时）：
    python -c "import sys; sys.path.insert(0, 'tests'); \
import test_shape_model_equivalence as m; m.write_baseline()"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from template_matching.shape_model import (  # noqa: E402
    ShapeModel,
    ShapeModelSettings,
)

from test_shape_model import make_scene, search_rect  # noqa: E402

BASELINE_PATH = Path(__file__).resolve().parent / "data" / "shape_model_baseline.json"

# OpenCV 的并行归约让同一份输入在重复调用间就有微小抖动（实测得分 ~6e-8、
# 角度 ~2e-6 度、角点 ~3e-5 像素）。容差取得比该抖动高一个量级、又远低于任何
# 真实算法改变的幅度：搜索窗偏一个像素就会让 center 差 1.0，必然被抓住。
FIELD_TOLERANCE = {
    "score": 1e-6,
    "outside_ratio": 1e-6,
    "angle_scores": 1e-6,
    "angle": 1e-3,
    "center": 1e-2,
    "box": 1e-2,
}
EXACT_FIELDS = {
    "matched",
    "failure",
    "level_candidates",
    "search_gradient_pixels",
    "levels_used",
    "search_roi_px",
    "center_range_px",
    "center_travel_px",
    "required_roi_px",
    "rotated_bbox_px",
}


def _build_models():
    scene = make_scene()
    template = scene[175:305, 225:415].copy()
    strict = ShapeModel(
        template,
        ShapeModelSettings.from_legacy(match_threshold=0.6, angle_range=15.0),
    )
    tolerant = ShapeModel(
        template,
        ShapeModelSettings(
            min_score=0.6,
            angle_start=-6.0,
            angle_extent=12.0,
            polarity="ignore_global",
        ),
    )
    return strict, tolerant


def _roi_around(model, cx, cy, travel, width, height):
    need_w, need_h = model.min_search_size(travel)
    return search_rect(cx, cy, need_w / 2, need_h / 2, width, height)


def _cases(strict: ShapeModel, tolerant: ShapeModel):
    """(名称, 模型, 场景, 搜索框, min_score) 列表，覆盖成功与各类失败分支。"""
    width, height = 640, 480
    cases = []

    for name, kwargs, travel in (
        ("aligned", {}, 60),
        ("rotated_pos2", {"angle": 2.0}, 60),
        ("rotated_neg9", {"angle": -9.0}, 60),
        ("subpixel_shift", {"cx": 320.4, "cy": 239.6, "angle": 1.4}, 50),
        ("noisy", {"noise": 8.0, "seed": 11}, 50),
        ("dim", {"gain": 0.6}, 50),
        ("bright_biased", {"gain": 1.0, "bias": 40.0}, 50),
    ):
        cases.append(
            (
                name,
                strict,
                make_scene(**kwargs),
                _roi_around(strict, 320, 240, travel, width, height),
                None,
            )
        )

    cases.append(
        (
            "occluded",
            strict,
            make_scene(occlude=0.4),
            _roi_around(strict, 320, 240, 50, width, height),
            0.05,
        )
    )
    cases.append(
        (
            "score_below_threshold",
            strict,
            make_scene(),
            _roi_around(strict, 320, 240, 50, width, height),
            0.999,
        )
    )

    bbox_w, bbox_h = strict.rotated_bbox(0)
    cases.append(
        (
            "roi_too_small",
            strict,
            make_scene(),
            search_rect(320, 240, bbox_w / 2 - 20, bbox_h / 2 - 20, width, height),
            None,
        )
    )

    need_w, need_h = strict.min_search_size(60)
    cases.append(
        (
            "roi_clipped_at_border",
            strict,
            make_scene(cx=100.0, cy=80.0),
            {
                "x": (100 - need_w / 2) / width,
                "y": (80 - need_h / 2) / height,
                "width": need_w / width,
                "height": need_h / height,
            },
            None,
        )
    )

    cases.append(
        (
            "blank_no_gradient",
            strict,
            np.full((height, width, 3), 128, np.uint8),
            _roi_around(strict, 320, 240, 50, width, height),
            None,
        )
    )

    cases.append(
        (
            "inverted_ignore_polarity",
            tolerant,
            255 - make_scene(),
            _roi_around(strict, 320, 240, 50, width, height),
            None,
        )
    )

    return cases


def _serialize(result) -> dict:
    """只留下与匹配语义有关的字段，耗时字段天然不稳定，排除在外。"""
    failure = result.failure
    return {
        "matched": bool(result.matched),
        "score": float(result.score),
        "center": list(result.center) if result.center else None,
        "angle": float(result.angle) if result.angle is not None else None,
        "box": [list(point) for point in result.box] if result.box else None,
        "failure": getattr(failure, "value", str(failure)),
        "outside_ratio": float(result.outside_ratio),
        "angle_scores": [[float(a), float(s)] for a, s in result.angle_scores],
        "level_candidates": [int(v) for v in result.level_candidates],
        "search_gradient_pixels": int(result.search_gradient_pixels),
        "levels_used": int(result.levels_used),
        "search_roi_px": list(result.search_roi_px) if result.search_roi_px else None,
        "center_range_px": list(result.center_range_px) if result.center_range_px else None,
        "center_travel_px": list(result.center_travel_px) if result.center_travel_px else None,
        "required_roi_px": list(result.required_roi_px) if result.required_roi_px else None,
        "rotated_bbox_px": list(result.rotated_bbox_px) if result.rotated_bbox_px else None,
    }


def collect_results() -> dict:
    strict, tolerant = _build_models()
    collected = {}
    for name, model, scene, rect, min_score in _cases(strict, tolerant):
        if min_score is None:
            result = model.find(scene, rect)
        else:
            result = model.find(scene, rect, min_score=min_score)
        collected[name] = _serialize(result)
    return collected


def write_baseline() -> Path:
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(
        json.dumps(collect_results(), indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return BASELINE_PATH


def _assert_equivalent(actual, expected, path: str, tolerance: float) -> None:
    if isinstance(expected, list):
        assert isinstance(actual, list), path
        assert len(actual) == len(expected), f"{path}: 长度 {len(actual)} != {len(expected)}"
        for index, (a, e) in enumerate(zip(actual, expected)):
            _assert_equivalent(a, e, f"{path}[{index}]", tolerance)
    elif isinstance(expected, bool) or expected is None:
        assert actual == expected, f"{path}: {actual} != {expected}"
    elif isinstance(expected, (int, float)):
        if tolerance <= 0.0:
            assert actual == expected, f"{path}: {actual} != {expected}"
        else:
            assert actual == pytest.approx(expected, abs=tolerance, rel=0.0), (
                f"{path}: {actual} != {expected}"
            )
    else:
        assert actual == expected, f"{path}: {actual} != {expected}"


@pytest.fixture(scope="module")
def baseline() -> dict:
    if not BASELINE_PATH.is_file():
        pytest.fail(
            f"形状匹配基线缺失: {BASELINE_PATH}。"
            "请在未优化的实现上运行 write_baseline() 生成后再提交。"
        )
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def actual() -> dict:
    return collect_results()


def test_baseline_covers_success_and_failure_branches(baseline):
    failures = {entry["failure"] for entry in baseline.values()}
    assert len(baseline) >= 12
    assert len(failures) >= 4


def test_find_output_matches_baseline(baseline, actual):
    assert sorted(actual) == sorted(baseline)
    for name in sorted(baseline):
        for field in sorted(baseline[name]):
            tolerance = 0.0 if field in EXACT_FIELDS else FIELD_TOLERANCE[field]
            _assert_equivalent(
                actual[name][field], baseline[name][field], f"{name}.{field}", tolerance
            )


def test_repeated_runs_stay_within_the_same_tolerance(actual):
    """把既有的浮点抖动量级也锁住：容差是为它留的，不是给算法改动留的。"""
    second = collect_results()
    for name in sorted(actual):
        for field in sorted(actual[name]):
            tolerance = 0.0 if field in EXACT_FIELDS else FIELD_TOLERANCE[field]
            _assert_equivalent(
                second[name][field], actual[name][field], f"{name}.{field}", tolerance
            )
