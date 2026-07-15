"""PASS/FAIL 独立提示音配置与运行时行为测试。"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import ConfigManager


def test_result_sound_defaults_and_boolean_normalization():
    config = ConfigManager()

    assert config.pass_sound_enabled is False
    assert config.fail_sound_enabled is True

    config.pass_sound_enabled = "on"
    config.fail_sound_enabled = "false"
    config.validate()

    assert config.pass_sound_enabled is True
    assert config.fail_sound_enabled is False
    assert config.sound_feedback_enabled is False


@pytest.mark.parametrize(
    ("legacy_value", "expected_fail"),
    [(True, True), (False, False), ("false", False)],
)
def test_legacy_sound_setting_migrates_only_to_fail(tmp_path, legacy_value, expected_fail):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"sound_feedback_enabled": legacy_value}),
        encoding="utf-8",
    )

    config = ConfigManager()
    config.load(path)

    assert config.pass_sound_enabled is False
    assert config.fail_sound_enabled is expected_fail
    assert config.sound_feedback_enabled is expected_fail


def test_new_sound_settings_take_priority_and_round_trip(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "pass_sound_enabled": True,
                "sound_feedback_enabled": False,
            }
        ),
        encoding="utf-8",
    )

    config = ConfigManager()
    config.load(path)

    assert config.pass_sound_enabled is True
    assert config.fail_sound_enabled is True

    config.fail_sound_enabled = False
    config.save(path)
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert saved["pass_sound_enabled"] is True
    assert saved["fail_sound_enabled"] is False
    assert saved["sound_feedback_enabled"] is False

    reloaded = ConfigManager()
    reloaded.load(path)
    assert reloaded.pass_sound_enabled is True
    assert reloaded.fail_sound_enabled is False
