"""PASS/FAIL 声音反馈。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

PASS_SOUND_NAME = "Pass.wav"
FAIL_SOUND_NAME = "Fail.wav"


def default_resource_dir() -> Path:
    """返回源码运行时的资源根目录。"""
    return Path(__file__).resolve().parents[2]


def resolve_sound_file(kind: str, resource_dir: str | Path | None = None) -> Path:
    """解析 PASS/FAIL 声音文件路径。"""
    filename = PASS_SOUND_NAME if kind.lower() == "pass" else FAIL_SOUND_NAME
    base = Path(resource_dir) if resource_dir is not None else default_resource_dir()
    return base / "assets" / "sounds" / filename


class SoundFeedback:
    """轻量声音播放器，失败时安全跳过。"""

    def __init__(
        self,
        *,
        enabled: bool = True,
        resource_dir: str | Path | None = None,
        player: Optional[Callable[[str], None]] = None,
        platform_name: str | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.resource_dir = Path(resource_dir) if resource_dir is not None else default_resource_dir()
        self._player = player
        self._platform_name = platform_name or sys.platform

    def play_pass(self) -> bool:
        return self._play("pass")

    def play_fail(self) -> bool:
        return self._play("fail")

    def _play(self, kind: str) -> bool:
        if not self.enabled:
            return False
        if not self._platform_name.startswith("win"):
            return False

        path = resolve_sound_file(kind, self.resource_dir)
        if not path.exists():
            return False

        try:
            if self._player is not None:
                self._player(str(path))
                return True

            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            return True
        except Exception:
            return False
