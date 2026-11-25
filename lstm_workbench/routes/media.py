"""媒体文件流式读取（支持 Range 拖动）、分块上传与连续图收编路由。"""
from __future__ import annotations

import mimetypes
from pathlib import Path
from urllib.parse import quote, unquote
import uuid

from ..http_utils import UPLOAD_CHUNK, _parse_range
from ..service import adopt_clip, clip_named


class MediaRoutesMixin:
    def _send_project_file(self, project_id: str, relative: str) -> None:
        try:
            folder = self.workbench.store.open_dir(project_id)
        except (FileNotFoundError, ValueError):
            self._send(404, {"error": "工程不存在"})
            return
        target = (folder / "media" / relative).resolve()
        media_root = (folder / "media").resolve()
        if media_root not in target.parents and target != media_root:
            self._send(400, {"error": "文件名不合法"})
            return
        if not target.is_file():
            self._send(404, {"error": "素材不存在"})
            return
        self._stream_file(target, cache=True)

    def _send_bounded_file(self, root: Path, relative: str) -> None:
        """只发送 root 里面的文件，并按 Range 读这一段，播放拖动时不必整文件进内存。"""
        relative = relative.replace("\\", "/").lstrip("/")
        target = (root / relative).resolve()
        base = root.resolve()
        if base not in target.parents and target != base:
            self._send(400, {"error": "文件名不合法"})
            return
        if not target.is_file():
            self._send(404, {"error": "文件不存在"})
            return
        self._stream_file(target, cache=False)

    def _stream_file(self, target: Path, *, cache: bool, attachment_name: str | None = None) -> None:
        """按 Range 分块发送文件：视频拖动只读请求的那一段，不把整文件读进内存。"""
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        file_size = target.stat().st_size
        start, end = 0, max(file_size - 1, 0)
        status = 200
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes=") and file_size:
            parsed = _parse_range(range_header, file_size)
            if parsed is None:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start, end = parsed
            status = 206
        length = end - start + 1 if file_size else 0
        self.send_response(status)
        self.send_header("Content-Type", mime)
        if attachment_name:
            encoded_name = quote(attachment_name)
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{encoded_name}")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with target.open("rb") as handle:
            handle.seek(start)
            remaining = length
            try:
                while remaining > 0:
                    chunk = handle.read(min(UPLOAD_CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                self.close_connection = True

    def _drain_body(self) -> None:
        """丢掉还没读的请求体，避免提前回错误时连接里残留数据。"""
        remaining = int(self.headers.get("Content-Length", "0") or 0)
        while remaining > 0:
            chunk = self.rfile.read(min(UPLOAD_CHUNK, remaining))
            if not chunk:
                break
            remaining -= len(chunk)

    def _receive_to(self, path: Path) -> None:
        """把请求体分块写到 path，上传大视频时不整段放进内存。"""
        remaining = int(self.headers.get("Content-Length", "0") or 0)
        with path.open("wb") as handle:
            while remaining > 0:
                chunk = self.rfile.read(min(UPLOAD_CHUNK, remaining))
                if not chunk:
                    break
                handle.write(chunk)
                remaining -= len(chunk)

    def _upload_name(self) -> str:
        """X-Filename 里的文件名（先经前端 encodeURIComponent 处理中文）。"""
        return unquote(self.headers.get("X-Filename") or "")

    def _adopt_directory(self, project_id: str) -> None:
        """连续图整组传完之后，收成一条片子。"""
        folder = self._open(project_id)
        if folder is None:
            return
        body = self._read_json()
        relative = str(body.get("file") or "").strip().replace("\\", "/")
        media_root = (folder / "media").resolve()
        target = (media_root / relative).resolve()
        if not relative or media_root not in target.parents:
            self._send(400, {"error": "文件名不合法"})
            return
        if not target.is_dir():
            self._send(400, {"error": "这组连续图还没放进素材目录"})
            return
        try:
            from ..extract import probe_media

            probe = probe_media(target)
            with self.workbench.data_lock(project_id):
                adopted = adopt_clip(
                    self.workbench.store, project_id, relative, probe, replace=bool(body.get("replace"))
                )
        except FileExistsError as exc:
            self._send(409, {"error": f"已经有这条片子：{exc}。要换掉它，先确认替换。"})
            return
        except (ValueError, KeyError) as exc:
            self._send(400, {"error": f"读不到这组连续图的信息：{exc}"})
            return
        self._send(201, {"clip": adopted})

    def _media(self, project_id: str) -> None:
        from .. import app as app_module

        workbench = self.workbench
        name = (self._upload_name() or "upload.bin").replace("\\", "/")
        try:
            folder = workbench.store.open_dir(project_id)
        except (FileNotFoundError, ValueError):
            self._drain_body()
            self._send(404, {"error": "工程不存在"})
            return
        single = "/" not in name
        replace = self.headers.get("X-Replace") == "1"
        if single:
            try:
                collision = clip_named(workbench.store.load_annotations(project_id), name)
            except ValueError as exc:
                self._drain_body()
                self._send(400, {"error": str(exc)})
                return
            if collision and not replace:
                self._drain_body()
                self._send(409, {"error": f"已经有这条片子：{collision.get('file') or name}。要换掉它，先确认替换。"})
                return
        staging = folder / "media" / f".incoming-{uuid.uuid4().hex[:8]}-{Path(name).name}"
        staging.parent.mkdir(parents=True, exist_ok=True)
        self._receive_to(staging)
        probe = app_module._probe_or_empty(staging)
        if "frame_count" not in probe:
            staging.unlink(missing_ok=True)
            self._send(400, {"error": probe.get("probe_error") or "读不到片子信息"})
            return
        with workbench.data_lock(project_id):
            collision = None
            if single:
                collision = clip_named(workbench.store.load_annotations(project_id), name)
                if collision is not None and not replace:
                    staging.unlink(missing_ok=True)
                    self._send(409, {"error": f"已经有这条片子：{collision.get('file') or name}。要换掉它，先确认替换。"})
                    return
            parked = None
            if collision is not None:
                try:
                    parked = workbench.store.park_media(project_id, name)
                except OSError as exc:
                    staging.unlink(missing_ok=True)
                    self._send(409, {"error": f"原来的片子正被占用，换不了：{exc}"})
                    return
            try:
                target = workbench.store.move_upload(project_id, name, staging)
            except (ValueError, OSError) as exc:
                staging.unlink(missing_ok=True)
                if parked is not None:
                    workbench.store.unpark_media(project_id, name, parked)
                code = 400 if isinstance(exc, ValueError) else 409
                message = str(exc) if code == 400 else f"写不进素材目录，文件可能正被占用：{exc}"
                self._send(code, {"error": message})
                return
            if parked is not None:
                parked.unlink(missing_ok=True)
            adopted = None
            if single:
                try:
                    adopted = adopt_clip(workbench.store, project_id, name, probe, replace=collision is not None)
                except FileExistsError as exc:
                    self._send(409, {"error": f"已经有这条片子：{exc}。要换掉它，先确认替换。"})
                    return
        self._send(201, {
            "saved": str(target.relative_to(folder / "media")).replace("\\", "/"),
            "clip": adopted,
            **probe,
        })
