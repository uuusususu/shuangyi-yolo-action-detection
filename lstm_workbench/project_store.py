"""工程目录。结果写入 runs，不下载，不碰工位配置。"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path


# Windows 不能用作文件名的字符，以及设备保留名（大小写、带不带扩展名都算）。
_ILLEGAL_CHARS = set('<>:"/\\|?*') | {chr(code) for code in range(32)}
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}


def project_id_problem(project_id: str) -> str:
    """工程名不能用的原因，空串表示可以用。工程名就是磁盘上的目录名，所以按 Windows 目录名的规则查。"""
    text = str(project_id or "")
    if not text.strip():
        return "需要工程名"
    if text != text.strip():
        return "工程名前后不能有空格"
    if text in {".", ".."}:
        return "工程名不合法"
    bad = sorted({ch for ch in text if ch in _ILLEGAL_CHARS})
    if bad:
        shown = "".join(ch if ch.isprintable() else "?" for ch in bad)
        return f"工程名不能包含这些字符：{shown}"
    if text.endswith("."):
        # Windows 会悄悄去掉末尾的点："abc." 建出来的目录叫 "abc"，之后用原名再也打不开。
        return "工程名不能以 . 结尾"
    if text.split(".", 1)[0].upper() in _RESERVED_NAMES:
        return f"工程名不能用系统保留名：{text}"
    if len(text.encode("utf-8")) > 200:
        return "工程名太长"
    return ""


def valid_project_id(project_id: str) -> bool:
    """一层目录名，且在 Windows 上能原样建出来。中文名照常允许。"""
    return not project_id_problem(project_id)


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
        """先在临时目录里建好全套，再一次改名到位。

        以前直接在目标位置逐个建，中途失败会留下一个没有 annotations.json 的半成品：
        它出现在工程列表里、点开是"工程不存在"、再建同名又报"已存在"。
        """
        folder = self.project_dir(project_id)
        if folder.exists() or self._exists_ignoring_case(project_id):
            raise FileExistsError(project_id)
        staging = self.root / f".creating-{uuid.uuid4().hex[:8]}"
        try:
            (staging / "media").mkdir(parents=True)
            (staging / "features").mkdir()
            (staging / "runs").mkdir()
            self.write_json(staging / "settings.json", settings)
            self.write_json(staging / "annotations.json", {"classes": [], "clips": []})
            os.rename(staging, folder)
        except FileExistsError:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        except OSError:
            shutil.rmtree(staging, ignore_errors=True)
            if folder.exists():
                raise FileExistsError(project_id) from None
            raise
        return folder

    def _exists_ignoring_case(self, project_id: str) -> bool:
        # Windows 目录名不分大小写：已有 Demo 时再建 demo 实际是同一个目录。
        lowered = project_id.casefold()
        return any(item.name.casefold() == lowered for item in self.root.iterdir() if item.is_dir())

    def list_projects(self) -> list[str]:
        """真正的工程：目录名合法且有 annotations.json。半成品、临时目录、随手放的文件夹不列。"""
        return sorted(
            item.name
            for item in self.root.iterdir()
            if item.is_dir() and valid_project_id(item.name) and (item / "annotations.json").is_file()
        )

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
        # "c:evil.mp4" 这种带盘符的相对路径不算绝对路径，拼上去却会落到 C 盘当前目录下。
        if relative.is_absolute() or relative.drive or ":" in str(relative_name):
            raise ValueError("文件名不合法")
        if any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("文件名不合法")
        if len(relative.parts) > 2:
            raise ValueError("文件名不合法")
        media = (folder / "media").resolve()
        target = folder / "media" / relative
        if media not in target.resolve().parents:
            raise ValueError("文件名不合法")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def save_upload(self, project_id: str, relative_name: str, content: bytes) -> Path:
        target = self._upload_target(project_id, relative_name)
        target.write_bytes(content)
        return target

    def park_media(self, project_id: str, relative_name: str) -> Path | None:
        """替换素材前把旧文件改名挪开，返回挪到的位置；原来没有这个文件返回 None。"""
        target = self._upload_target(project_id, relative_name)
        if not target.exists():
            return None
        parked = target.with_name(f".replaced-{os.getpid()}-{target.name}")
        os.replace(target, parked)
        return parked

    def unpark_media(self, project_id: str, relative_name: str, parked: Path) -> None:
        """放新文件失败时把旧文件挪回原处。"""
        target = self._upload_target(project_id, relative_name)
        os.replace(parked, target)

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
        """先写临时文件再改名：写到一半断电或报错时，原文件保持完整，不会留下半截 JSON 让工程打不开。

        Windows 上别的线程正读着目标文件（页面轮询工程、刷新训练状态）时，改名会报拒绝访问；
        读取只占用一瞬间，等一下再试，不把它当成保存失败。"""
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            temporary.write_text(text, encoding="utf-8")
            for attempt in range(40):
                try:
                    os.replace(temporary, path)
                    break
                except PermissionError:
                    if attempt == 39:
                        raise
                    time.sleep(0.05)
        finally:
            temporary.unlink(missing_ok=True)
