"""PCB 检查模式配置加载、保存和校验测试。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import pytest
from config import ConfigManager


def test_pcb_config_defaults():
    """PCB 检查模式默认关闭，字段有默认值。"""
    c = ConfigManager()
    assert c.pcb_inspection_enabled is False
    assert c.pcb_class_name == "pcb"
    assert c.pcb_component_class_names == ["", "", "", ""]
    assert c.pcb_fail_stable_frames == 3
    assert c.pcb_round_interval_seconds == 0.0
    assert c.pcb_assignment_margin_ratio == 0.15


def test_pcb_config_save_and_load(tmp_path):
    """PCB 配置可序列化和反序列化。"""
    c = ConfigManager()
    c.pcb_inspection_enabled = True
    c.pcb_class_name = "board"
    c.pcb_component_class_names = ["R1", "C2", "U3", "Q4"]
    c.pcb_fail_stable_frames = 5
    c.pcb_round_interval_seconds = 2.0
    c.pcb_assignment_margin_ratio = 0.2

    path = tmp_path / "pcb_config.json"
    c.save(path)

    c2 = ConfigManager()
    c2.load(path)
    assert c2.pcb_inspection_enabled is True
    assert c2.pcb_class_name == "board"
    assert c2.pcb_component_class_names == ["R1", "C2", "U3", "Q4"]
    assert c2.pcb_fail_stable_frames == 5
    assert c2.pcb_round_interval_seconds == 2.0
    assert c2.pcb_assignment_margin_ratio == 0.2


def test_pcb_config_validate_rejects_empty_class_name():
    """启用 PCB 模式但 PCB 类别为空时拒绝。"""
    c = ConfigManager()
    c.pcb_inspection_enabled = True
    c.pcb_class_name = ""
    c.pcb_component_class_names = ["A", "B", "C", "D"]
    with pytest.raises(ValueError, match="PCB 类别名称为空"):
        c.validate()


def test_pcb_config_validate_rejects_insufficient_components():
    """元器件类别不足 4 个时拒绝。"""
    c = ConfigManager()
    c.pcb_inspection_enabled = True
    c.pcb_class_name = "pcb"
    c.pcb_component_class_names = ["A", "B", "", ""]
    with pytest.raises(ValueError, match="4 个非空元器件类别"):
        c.validate()


def test_pcb_config_validate_rejects_duplicate_components():
    """元器件类别重复时拒绝。"""
    c = ConfigManager()
    c.pcb_inspection_enabled = True
    c.pcb_class_name = "pcb"
    c.pcb_component_class_names = ["A", "B", "A", "D"]
    with pytest.raises(ValueError, match="存在重复"):
        c.validate()


def test_pcb_config_validate_rejects_pcb_class_collision():
    """PCB 类别与元器件类别相同时拒绝。"""
    c = ConfigManager()
    c.pcb_inspection_enabled = True
    c.pcb_class_name = "pcb"
    c.pcb_component_class_names = ["pcb", "B", "C", "D"]
    with pytest.raises(ValueError, match="不能与元器件类别相同"):
        c.validate()


def test_pcb_config_disabled_skips_validation():
    """未启用 PCB 模式时跳过 PCB 校验。"""
    c = ConfigManager()
    c.pcb_inspection_enabled = False
    c.pcb_class_name = ""
    c.pcb_component_class_names = []
    c.validate()  # 不应抛异常


def test_pcb_fail_stable_frames_clamped():
    """FAIL 帧数最小为 1。"""
    c = ConfigManager()
    c.pcb_fail_stable_frames = 0
    c.validate()
    assert c.pcb_fail_stable_frames == 1


def test_pcb_round_interval_clamped():
    """轮次间隔不得为负。"""
    c = ConfigManager()
    c.pcb_round_interval_seconds = -1.0
    c.validate()
    assert c.pcb_round_interval_seconds == 0.0


def test_pcb_assignment_margin_clamped():
    """归属容差在 0-1 之间。"""
    c = ConfigManager()
    c.pcb_assignment_margin_ratio = 2.0
    c.validate()
    assert c.pcb_assignment_margin_ratio == 1.0