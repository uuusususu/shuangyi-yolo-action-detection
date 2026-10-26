"""工程目录。结果写入 runs，不下载，不碰工位配置。"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path


def valid_project_id(project_id: str) -> bool:
    """一层目录名：非空，不含路径分隔符、盘符，也不是 . / .. 。中文名照常允许。"""
    text = str(project_id or "")
    if not text or text != text.strip() or text in {".", ".."}:
        return False
    return not any(ch in text for ch in '/\\:\0')


class ProjectStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def project_dir(self, project_id: str) -> Path:
        """工程名只能是 root 下的一层目录名。URL 里任何位置拿到的 id 都经过这里。"""
        if not valid_project_id(project_id):
            raise ValueError("工程名不能带路径")
        return self.root / project_id

    def create(self, project_id: str, settings: dict) -> Path:
        folder = self.project_dir(project_id)
        if folder.exists():
            raise FileExistsError(project_id)
        (folder / "media").mkdir(parents=True)
        (folder / "features").mkdir()
        (folder / "runs").mkdir()
        self.write_json(folder / "settings.json", settings)
        self.write_json(folder / "annotations.json", {"classes": [], "clips": []})
        return folder

    def open_dir(self, project_id: str) -> Path:
        folder = self.project_dir(project_id)
        if not (folder / "annotations.json").is_file():
            raise FileNotFoundError(project_id)
        return folder

    def load_settings(self, project_id: str) -> dict:
        folder = self.open_dir(project_id)
        return json.loads((folder / "settings.json").read_text(encoding="utf-8"))

    def save_settings(self, project_id: str, settings: dict) -> None:
        folder = self.open_dir(project_id)
        self.write_json(folder / "settings.json", settings)

    def list_dir(self, project_id: str, name: str) -> list[str]:
        folder = self.open_dir(project_id)
        root = folder / name
        if not root.is_dir():
            return []
        return sorted(item.name for item in root.iterdir() if item.is_dir())

    def save_annotations(self, project_id: str, payload: dict) -> None:
        folder = self.open_dir(project_id)
        self.write_json(folder / "annotations.json", payload)

    def load_annotations(self, project_id: str) -> dict:
        folder = self.open_dir(project_id)
        return json.loads((folder / "annotations.json").read_text(encoding="utf-8"))

    def _upload_target(self, project_id: str, relative_name: str) -> Path:
        folder = self.open_dir(project_id)
        relative = Path(relative_name)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("文件名不合法")
        if len(relative.parts) > 2:
            raise ValueError("文件名不合法")
        target = folder / "media" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def save_upload(self, project_id: str, relative_name: str, content: bytes) -> Path:
        target = self._upload_target(project_id, relative_name)
        target.write_bytes(content)
        return target

    def move_upload(self, project_id: str, relative_name: str, staged: Path) -> Path:
        """把已落盘的暂存文件改名到素材目录，不再整段读回内存重写。"""
        target = self._upload_target(project_id, relative_name)
        os.replace(staged, target)
        return target

    def media_missing(self, project_id: str, relative_names: list[str]) -> list[str]:
        folder = self.open_dir(project_id)
        missing = []
        for name in relative_names:
            if not (folder / "media" / name).is_file():
                missing.append(name)
        return missing

    def create_run(self, project_id: str, when: datetime | None = None) -> Path:
        folder = self.open_dir(project_id)
        stamp = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
        run = folder / "runs" / stamp
        suffix = 2
        while run.exists():
            run = folder / "runs" / f"{stamp}-{suffix}"
            suffix += 1
        run.mkdir(parents=True)
        return run

    def write_run_files(self, run: Path, files: dict[str, object]) -> None:
        for name, payload in files.items():
            path = run / name
            if isinstance(payload, (dict, list)):
                self.write_json(path, payload)
            elif isinstance(payload, bytes):
                path.write_bytes(payload)
            else:
                path.write_text(str(payload), encoding="utf-8")

    @staticmethod
    def write_json(path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
