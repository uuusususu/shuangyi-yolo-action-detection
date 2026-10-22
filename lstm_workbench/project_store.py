"""工程目录。结果写入 runs，不下载，不碰工位配置。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class ProjectStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def project_dir(self, project_id: str) -> Path:
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

    def save_upload(self, project_id: str, relative_name: str, content: bytes) -> Path:
        folder = self.open_dir(project_id)
        relative = Path(relative_name)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("文件名不合法")
        if len(relative.parts) > 2:
            raise ValueError("文件名不合法")
        target = folder / "media" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
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
