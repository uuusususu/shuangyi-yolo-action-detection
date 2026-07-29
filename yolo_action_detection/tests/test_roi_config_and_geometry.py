"""ROI 定位检测配置与几何测试。"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import CURRENT_CONFIG_SCHEMA_VERSION, ConfigManager
from roi_inspection.geometry import NormalizedRect


def _valid_roi_config() -> ConfigManager:
    """有效 ROI 配置：识别 ROI + 待检类别，模板定位关闭。"""
    config = ConfigManager()
    config.config_schema_version = CURRENT_CONFIG_SCHEMA_VERSION
    config.roi_inspection_enabled = True
    config.roi_template_enabled = False
    config.roi_inspection_rect = {"x": 0.0, "y": 0.0, "width": 0.5, "height": 1.0}
    config.category_names = ["散热片", "螺丝", "", "", "", ""]
    config.category_counts = [6, 2, 1, 1, 1, 1]
    return config


def test_old_configuration_loads_for_editing_but_is_rejected_for_detection(tmp_path):
    assert CURRENT_CONFIG_SCHEMA_VERSION == 4

    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "model_task": "ultralytics_obb",
                "category_names": ["pcb", "散热片"],
                "category_counts": [1, 6],
                "first_category_region_check_enabled": True,
                "pcb_assignment_margin_ratio": 0.15,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    config = ConfigManager()
    config.load(path)

    assert config.category_names == ["pcb", "散热片"]
    assert config.config_schema_version == 0
    errors = config.inspection_validation_errors(["pcb", "散热片"])
    assert any("配置版本" in error for error in errors)
    assert any("4" in error for error in errors)
    # 旧字段已删除。
    assert not hasattr(config, "first_category_region_check_enabled")
    assert not hasattr(config, "pcb_assignment_margin_ratio")
    assert not hasattr(config, "roi_locator_class_name")

    saved_path = tmp_path / "legacy-resaved.json"
    config.save(saved_path)
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved["config_schema_version"] == 0
    assert "roi_template_rect" in saved and saved["roi_template_rect"] is None
    assert "roi_locator_class_name" not in saved
    assert "roi_locator_rect" not in saved
    assert "roi_template_enabled" in saved


def test_v3_configuration_migrates_to_v4_without_losing_site_parameters(tmp_path):
    path = tmp_path / "v3.json"
    payload = {
        "config_schema_version": 3,
        "model_task": "ultralytics_obb",
        "yolo_model_path": "config/site.onnx",
        "yolo_conf_threshold": 0.63,
        "category_names": ["散热片", "螺丝", "", "", "", ""],
        "category_counts": [6, 2, 1, 1, 1, 1],
        "mvsdk_camera_sn": "SITE-CAMERA",
        "camera_manual_exposure_us": 45678,
        "roi_inspection_enabled": True,
        "roi_inspection_rect": {"x": 0.1, "y": 0.2, "width": 0.7, "height": 0.6},
        "roi_template_enabled": True,
        "roi_template_path": "locator_template.png",
        "roi_template_match_threshold": 0.72,
        "roi_template_angle_range": 12.0,
        "roi_template_min_matches": 9,
        "roi_template_rect": {"x": 0.2, "y": 0.3, "width": 0.2, "height": 0.2},
        "action_pass_stable_frames": 2,
        "action_ng_stable_frames": 15,
        "round_cooldown_seconds": 3.5,
        "pass_sound_enabled": True,
        "fail_sound_enabled": False,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    config = ConfigManager()
    config.load(path)

    assert config.config_schema_version == 4
    assert config.roi_template_search_rect is None
    assert config.roi_template_angle_step == 1.0
    for key in (
        "yolo_model_path",
        "yolo_conf_threshold",
        "category_names",
        "category_counts",
        "mvsdk_camera_sn",
        "camera_manual_exposure_us",
        "roi_inspection_rect",
        "roi_template_path",
        "roi_template_match_threshold",
        "roi_template_angle_range",
        "roi_template_rect",
        "action_pass_stable_frames",
        "action_ng_stable_frames",
        "round_cooldown_seconds",
        "pass_sound_enabled",
        "fail_sound_enabled",
    ):
        assert getattr(config, key) == payload[key]

    saved_path = tmp_path / "v4.json"
    config.save(saved_path)
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved["config_schema_version"] == 4
    assert saved["roi_template_search_rect"] is None
    assert saved["roi_template_angle_step"] == 1.0
    assert "roi_template_min_matches" not in saved


def test_valid_roi_configuration_has_no_errors():
    config = _valid_roi_config()

    assert config.inspection_validation_errors(["散热片", "螺丝"]) == []


def test_draft_validation_allows_only_missing_roi_rects():
    config = _valid_roi_config()
    config.roi_inspection_rect = None

    strict_errors = config.inspection_validation_errors(["散热片", "螺丝"])
    draft_errors = config.inspection_validation_errors(
        ["散热片", "螺丝"],
        require_roi_rects=False,
    )

    assert "识别 ROI 未配置" in strict_errors
    assert draft_errors == []

    # 草稿模式仅豁免缺失的 ROI；其他真实错误仍然报告。
    config.category_counts[0] = 0
    assert "待检数量无效：类别 '散热片' 必须为正整数" in config.inspection_validation_errors(
        ["散热片", "螺丝"],
        require_roi_rects=False,
    )


def test_schema_and_quantities_must_be_actual_integers():
    config = _valid_roi_config()
    config.config_schema_version = 3.0
    config.category_counts[0] = 6.5

    errors = config.inspection_validation_errors(["散热片", "螺丝"])

    assert any("配置版本" in error for error in errors)
    assert any("待检数量" in error and "散热片" in error for error in errors)


def test_disabled_roi_mode_keeps_normal_action_mode_valid():
    config = ConfigManager()
    config.roi_inspection_enabled = False
    config.category_names = ["动作A", "", "", "", "", ""]

    assert config.inspection_validation_errors(["动作A"]) == []


def test_normalized_rect_maps_to_source_and_letterboxed_display():
    rect = NormalizedRect(x=0.1, y=0.2, width=0.5, height=0.4)

    assert rect.to_pixel_rect((1000, 500)) == pytest.approx((100, 100, 500, 200))
    assert rect.to_display_rect((1000, 500), (400, 400)) == pytest.approx((40, 140, 200, 80))


def test_normalized_rect_center_containment_includes_boundary():
    rect = NormalizedRect(x=0.25, y=0.25, width=0.5, height=0.5)

    assert rect.contains_pixel_point((250, 250), (1000, 1000))
    assert rect.contains_pixel_point((750, 750), (1000, 1000))
    assert rect.contains_pixel_point((500, 500), (1000, 1000))
    assert not rect.contains_pixel_point((249.9, 500), (1000, 1000))
    assert not rect.contains_pixel_point((500, 750.1), (1000, 1000))


@pytest.mark.parametrize(
    "mapping",
    [
        {"x": -0.1, "y": 0.0, "width": 0.5, "height": 0.5},
        {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.5},
        {"x": 0.8, "y": 0.0, "width": 0.3, "height": 0.5},
        {"x": 0.0, "y": 0.8, "width": 0.5, "height": 0.3},
    ],
)
def test_normalized_rect_rejects_invalid_or_zero_area(mapping):
    with pytest.raises(ValueError):
        NormalizedRect.from_mapping(mapping)


def test_template_enabled_but_threshold_out_of_range_reports_error():
    """模板启用但匹配阈值越界（>1）时报错。"""
    config = _valid_roi_config()
    config.roi_template_enabled = True
    config.roi_template_rect = {"x": 0.2, "y": 0.2, "width": 0.2, "height": 0.2}
    config.roi_template_search_rect = {"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5}
    config.roi_template_match_threshold = 1.5

    errors = config.inspection_validation_errors(["散热片", "螺丝"])

    assert any("匹配分数阈值" in error for error in errors)


def test_template_enabled_but_angle_negative_reports_error():
    """角度范围允许为 0，但不能为负数。"""
    config = _valid_roi_config()
    config.roi_template_enabled = True
    config.roi_template_rect = {"x": 0.2, "y": 0.2, "width": 0.2, "height": 0.2}
    config.roi_template_search_rect = {"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5}
    config.roi_template_angle_range = -1.0

    errors = config.inspection_validation_errors(["散热片", "螺丝"])

    assert any("旋转角度范围" in error for error in errors)


def test_template_search_roi_is_required_but_need_not_contain_template_roi():
    config = _valid_roi_config()
    config.roi_template_enabled = True
    config.roi_template_rect = {"x": 0.2, "y": 0.2, "width": 0.4, "height": 0.4}

    assert "搜索 ROI 未配置" in config.inspection_validation_errors(["散热片", "螺丝"])

    config.roi_template_search_rect = {"x": 0.7, "y": 0.7, "width": 0.2, "height": 0.2}
    errors = config.inspection_validation_errors(["散热片", "螺丝"])
    assert "搜索 ROI 未配置" not in errors
    assert not any("包含模板 ROI" in error for error in errors)


def test_template_angle_step_zero_means_auto():
    config = _valid_roi_config()
    config.roi_template_enabled = True
    config.roi_template_rect = {"x": 0.2, "y": 0.2, "width": 0.2, "height": 0.2}
    config.roi_template_search_rect = {"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5}
    config.roi_template_angle_step = 0.0
    assert not any(
        "角度步进" in error
        for error in config.inspection_validation_errors(["散热片", "螺丝"])
    )
    config.roi_template_angle_step = -1.0
    assert any(
        "角度步进" in error
        for error in config.inspection_validation_errors(["散热片", "螺丝"])
    )
    config.roi_template_angle_step = 200.0
    assert any(
        "角度步进" in error
        for error in config.inspection_validation_errors(["散热片", "螺丝"])
    )
