"""本机训练台。只监听 127.0.0.1。"""
from __future__ import annotations

import json
import mimetypes
import os
import subprocess
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import check_clip, feature_store, model_registry, service, settings as settings_module
from .dataset import check_split
from .project_store import ProjectStore, valid_project_id
from .service import (
    adopt_clip,
    annotation_blockers,
    annotation_warnings,
    clip_named,
    forget_clip,
    start_training,
    write_progress,
)

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
HOST = "127.0.0.1"
PORT = 8771

JOB_ERRORS = (ValueError, FileNotFoundError, KeyError, RuntimeError, OSError)
LOOPBACK_NAMES = {"127.0.0.1", "localhost"}
UPLOAD_CHUNK = 1024 * 1024


def _host_name(netloc: str) -> str:
    """Host 头或 Origin 的 netloc 去掉端口，只留主机名。"""
    netloc = netloc.strip().lower()
    if netloc.startswith("["):
        return netloc[1:].split("]", 1)[0]
    return netloc.rsplit(":", 1)[0] if ":" in netloc else netloc


def _parse_range(header: str, file_size: int) -> tuple[int, int] | None:
    """解析单段 bytes=start-end。格式不对或越界返回 None，由调用方回 416。"""
    try:
        start_text, end_text = header.removeprefix("bytes=").split("-", 1)
        if not start_text:
            # bytes=-N：最后 N 个字节。
            suffix = int(end_text)
            if suffix <= 0:
                return None
            return max(0, file_size - suffix), file_size - 1
        start = int(start_text)
        end = int(end_text) if end_text else file_size - 1
    except ValueError:
        return None
    end = min(end, file_size - 1)
    if start < 0 or start > end:
        return None
    return start, end


class Workbench:
    def __init__(self, projects_root: Path):
        self.store = ProjectStore(projects_root)
        self._locks: dict[str, threading.Lock] = {}
        self._stops: dict[str, threading.Event] = {}
        self._guards = threading.Lock()

    def lock_for(self, project_id: str) -> threading.Lock:
        with self._guards:
            return self._locks.setdefault(project_id, threading.Lock())

    def stop_event(self, project_id: str) -> threading.Event:
        """停止不走工程锁，训练还在跑的时候也要能收到。"""
        with self._guards:
            return self._stops.setdefault(project_id, threading.Event())


def _read_settings(store: ProjectStore, project_id: str) -> dict:
    """文件里可能缺新加的项，读出来一律补全。"""
    return settings_module.clean({}, store.load_settings(project_id))


def make_handler(workbench: Workbench):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def _send(self, code: int, payload: object, content_type: str = "application/json", cache: bool = True) -> None:
            body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if not cache:
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _request_rejection(self, method: str) -> str:
            """只放行本机页面发来的请求，返回拒绝原因，空串表示放行。

            只监听 127.0.0.1 挡不住浏览器里别的网站：它们能发不带预检的跨站 POST，
            或者用 DNS 重绑定把自己的域名指到 127.0.0.1。所以：
            - Host 必须是 127.0.0.1/localhost（挡 DNS 重绑定）；
            - 带 Origin 时必须是本机源；
            - 写操作必须是 JSON 或带 X-Filename 的上传：这两种都会触发 CORS 预检，
              别的网站发不出来（本服务不回 CORS 头，预检必然失败）。
            """
            host = (self.headers.get("Host") or "").strip().lower()
            if _host_name(host) not in LOOPBACK_NAMES:
                return "只接受本机访问"
            origin = (self.headers.get("Origin") or "").strip().lower()
            if origin and origin != "null" and _host_name(urlparse(origin).netloc) not in LOOPBACK_NAMES:
                return "不接受其他网站发来的请求"
            # 写操作的要求依赖"别的网站发不出预检"这一点：本服务刻意不实现 OPTIONS、
            # 不回任何 CORS 头。以后别为了方便加 CORS，否则这层防护就失效了。
            if method in {"POST", "PUT", "DELETE"}:
                content_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
                try:
                    has_body = int(self.headers.get("Content-Length", "0") or 0) > 0
                except ValueError:
                    return "请求长度不合法"
                if self.headers.get("X-Filename") is None and (has_body or method != "DELETE"):
                    if content_type != "application/json":
                        return "请求格式不对：需要 application/json"
            return ""

        def _guard(self, method: str) -> bool:
            reason = self._request_rejection(method)
            if reason:
                self._send(403, {"error": reason})
                return False
            return True

        def _open(self, project_id: str) -> Path | None:
            try:
                return workbench.store.open_dir(project_id)
            except (FileNotFoundError, ValueError):
                # ValueError：工程名带路径，按不存在处理，不暴露目录结构。
                self._send(404, {"error": "工程不存在"})
                return None

        def _send_project_file(self, path: str) -> None:
            pieces = path.split("/")
            project_id = pieces[3]
            relative = "/".join(pieces[5:])
            try:
                folder = workbench.store.open_dir(project_id)
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

        def _stream_file(self, target: Path, *, cache: bool) -> None:
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
                    # 拖动进度条时浏览器会主动断开上一段请求，属正常情况。
                    self.close_connection = True

        def _run_summaries(self, project_id: str, folder: Path) -> tuple[list[str], list[dict]]:
            runs = workbench.store.list_dir(project_id, "runs")
            summaries = []
            for name in runs:
                run = folder / "runs" / name
                metrics_path = run / "metrics.json"
                contract_path = run / "contract.json"
                if not metrics_path.is_file() or not contract_path.is_file():
                    continue
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                contract = json.loads(contract_path.read_text(encoding="utf-8"))
                summaries.append({
                    "name": name,
                    "usable": metrics.get("usable"),
                    "stopped": metrics.get("stopped"),
                    "reason": metrics.get("reason"),
                    "macro_f1": (metrics.get("metrics") or {}).get("macro_f1"),
                    "class_ids": contract.get("class_ids"),
                    "input_size": contract.get("input_size"),
                    "window_seconds": contract.get("window_seconds"),
                    "time_base_fps": contract.get("time_base_fps"),
                    "same_source": contract.get("same_source_validation"),
                    "mistakes": metrics.get("mistakes") or [],
                })
            return runs, summaries

        def _project_payload(self, project_id: str, folder: Path) -> dict:
            settings = _read_settings(workbench.store, project_id)
            annotations = workbench.store.load_annotations(project_id)
            runs, summaries = self._run_summaries(project_id, folder)
            comparable = summaries[-3:]
            if comparable:
                key = (tuple(comparable[-1]["class_ids"] or []), comparable[-1]["input_size"])
                comparable = [item for item in comparable if (tuple(item["class_ids"] or []), item["input_size"]) == key]
            return {
                "id": project_id,
                "settings": settings,
                "annotations": annotations,
                "runs": runs,
                "comparable_runs": comparable,
                "features": feature_store.summary(folder, annotations, settings),
                "split": service.split_overview(workbench.store, project_id, settings),
                "blockers": annotation_blockers(annotations, settings),
                "warnings": annotation_warnings(annotations, settings),
            }

        def do_GET(self) -> None:  # noqa: N802
            if not self._guard("GET"):
                return
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8", cache=False)
                return
            if path == "/api/projects":
                names = sorted(item.name for item in workbench.store.root.iterdir() if item.is_dir())
                self._send(200, {"projects": names})
                return
            if path == "/api/models":
                self._send(200, model_registry.library_info())
                return
            if "/file/" in path and "/check/file/" not in path:
                self._send_project_file(path)
                return
            parts = path.strip("/").split("/")
            if len(parts) >= 3 and parts[0] == "api" and parts[1] == "projects":
                project_id = parts[2]
                if len(parts) == 4 and parts[3] == "progress":
                    folder = self._open(project_id)
                    if folder is None:
                        return
                    progress_path = folder / "progress.json"
                    if not progress_path.is_file():
                        self._send(200, {"status": "idle", "history": []})
                        return
                    payload = json.loads(progress_path.read_text(encoding="utf-8"))
                    payload["updated_at"] = progress_path.stat().st_mtime
                    self._send(200, payload)
                    return
                if len(parts) >= 6 and parts[0] == "api" and parts[1] == "projects" and parts[3] == "runs" and parts[5] == "check":
                    self._check_get(parts)
                    return
                if len(parts) == 5 and parts[3] == "runs":
                    folder = self._open(project_id)
                    if folder is None:
                        return
                    run = folder / "runs" / parts[4]
                    if not run.is_dir():
                        self._send(404, {"error": "没有这一轮"})
                        return
                    payload = {"name": parts[4], "files": sorted(item.name for item in run.iterdir() if item.is_file())}
                    for key, name in (("report", "metrics.json"), ("contract", "contract.json"), ("split", "split.json")):
                        if (run / name).is_file():
                            payload[key] = json.loads((run / name).read_text(encoding="utf-8"))
                    self._send(200, payload)
                    return
                if len(parts) == 3:
                    folder = self._open(project_id)
                    if folder is None:
                        return
                    self._send(200, self._project_payload(project_id, folder))
                    return
            file_path = STATIC / path.lstrip("/")
            if file_path.is_file():
                mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
                self._send(200, file_path.read_bytes(), mime, cache=False)
                return
            self._send(404, {"error": "没有这个页面"})

        def _start_job(self, project_id: str) -> threading.Event | None:
            """同一工程同一时刻只跑一个任务。拿不到锁就回 409。"""
            lock = workbench.lock_for(project_id)
            if not lock.acquire(blocking=False):
                self._send(409, {"error": "正在跑任务，再点无效。要重来先停止"})
                return None
            self._job_lock = lock
            stop = workbench.stop_event(project_id)
            stop.clear()
            return stop

        def _end_job(self) -> None:
            lock = getattr(self, "_job_lock", None)
            if lock is not None:
                lock.release()
                self._job_lock = None

        def _fail_job(self, folder: Path, exc: Exception) -> None:
            try:
                write_progress(folder, {"status": "error", "error": str(exc), "history": []})
            except OSError:
                pass
            self._send(400, {"error": str(exc)})

        def do_POST(self) -> None:  # noqa: N802
            if not self._guard("POST"):
                return
            path = urlparse(self.path).path
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[0] == "api" and parts[3] == "media":
                self._media(parts[2])
                return
            if path == "/api/projects":
                body = self._read_json()
                project_id = str(body.get("id") or "").strip()
                if not project_id:
                    self._send(400, {"error": "需要工程名"})
                    return
                if not valid_project_id(project_id):
                    self._send(400, {"error": "工程名不能带路径"})
                    return
                try:
                    workbench.store.create(project_id, settings_module.clean(body.get("settings") or {}))
                except FileExistsError:
                    self._send(409, {"error": "工程已存在"})
                    return
                except ValueError as exc:
                    self._send(400, {"error": str(exc)})
                    return
                self._send(201, {"id": project_id})
                return
            if len(parts) == 5 and parts[0] == "api" and parts[1] == "projects" and parts[3:] == ["features", "reveal"]:
                self._reveal_feature(parts[2])
                return
            if len(parts) >= 6 and parts[0] == "api" and parts[1] == "projects" and parts[3] == "runs" and parts[5] == "check":
                self._check_post(parts)
                return
            if len(parts) == 6 and parts[0] == "api" and parts[1] == "projects" and parts[3] == "runs" and parts[5] == "prelabel":
                self._prelabel(parts[2], parts[4])
                return
            if len(parts) != 4 or parts[0] != "api" or parts[1] != "projects":
                self._send(404, {"error": "没有这个接口"})
                return
            project_id, action = parts[2], parts[3]
            if action == "annotations":
                folder = self._open(project_id)
                if folder is None:
                    return
                payload = self._read_json()
                if not isinstance(payload.get("clips"), list) or not isinstance(payload.get("classes"), list):
                    self._send(400, {"error": "标注要带 classes 和 clips 两个列表"})
                    return
                workbench.store.save_annotations(project_id, payload)
                self._send(200, {"saved": True})
                return
            if action == "clips":
                self._adopt_directory(project_id)
                return
            if action == "model":
                self._model(project_id)
                return
            if action == "split":
                self._split(project_id)
                return
            if action == "features":
                self._features(project_id)
                return
            if action == "train":
                self._train(project_id)
                return
            if action == "stop":
                workbench.stop_event(project_id).set()
                self._send(200, {"stopping": True})
                return
            self._send(404, {"error": "没有这个接口"})

        def do_DELETE(self) -> None:  # noqa: N802
            if not self._guard("DELETE"):
                return
            parts = urlparse(self.path).path.strip("/").split("/")
            if len(parts) == 6 and parts[0] == "api" and parts[1] == "projects" and parts[3] == "runs" and parts[5] == "check":
                run = self._run_dir(parts[2], parts[4])
                if run is None:
                    return
                try:
                    check_clip.clear_check(run)
                except OSError as exc:
                    self._send(400, {"error": f"删不掉：{exc}"})
                    return
                self._send(200, {"cleared": True})
                return
            if len(parts) == 5 and parts[0] == "api" and parts[1] == "projects" and parts[3] == "clips":
                project_id, clip_id = parts[2], parts[4]
                if self._open(project_id) is None:
                    return
                try:
                    removed = forget_clip(workbench.store, project_id, clip_id)
                except FileNotFoundError:
                    self._send(404, {"error": "没有这条片子"})
                    return
                except OSError as exc:
                    self._send(400, {"error": f"删不掉文件：{exc}"})
                    return
                self._send(200, {"deleted": removed})
                return
            self._send(404, {"error": "没有这个接口"})

        def _adopt_directory(self, project_id: str) -> None:
            """连续图整组传完之后，收成一条片子。"""
            folder = self._open(project_id)
            if folder is None:
                return
            body = self._read_json()
            relative = str(body.get("file") or "").strip().replace("\\", "/")
            target = folder / "media" / relative
            if not target.is_dir():
                self._send(400, {"error": "这组连续图还没放进素材目录"})
                return
            try:
                from .extract import probe_media

                probe = probe_media(target)
                adopted = adopt_clip(
                    workbench.store, project_id, relative, probe, replace=bool(body.get("replace"))
                )
            except FileExistsError as exc:
                self._send(409, {"error": f"已经有这条片子：{exc}。要换掉它，先确认替换。"})
                return
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            self._send(201, {"clip": adopted})

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

        def _media(self, project_id: str) -> None:
            name = (self.headers.get("X-Filename") or "upload.bin").replace("\\", "/")
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
            # 加随机后缀：同名文件并发上传时暂存文件不会互相覆盖。
            staging = folder / "media" / f".incoming-{uuid.uuid4().hex[:8]}-{Path(name).name}"
            staging.parent.mkdir(parents=True, exist_ok=True)
            self._receive_to(staging)
            probe = _probe_or_empty(staging)
            if "frame_count" not in probe:
                staging.unlink(missing_ok=True)
                self._send(400, {"error": probe.get("probe_error") or "读不到片子信息"})
                return
            if single and replace:
                collision = clip_named(workbench.store.load_annotations(project_id), name)
                if collision:
                    try:
                        forget_clip(workbench.store, project_id, str(collision["id"]))
                    except OSError as exc:
                        staging.unlink(missing_ok=True)
                        self._send(400, {"error": f"删不掉原来的片子：{exc}"})
                        return
            try:
                target = workbench.store.move_upload(project_id, name, staging)
            except ValueError as exc:
                staging.unlink(missing_ok=True)
                self._send(400, {"error": str(exc)})
                return
            except OSError as exc:
                # Windows 上目标正被播放器占用时改名会失败。
                staging.unlink(missing_ok=True)
                self._send(409, {"error": f"写不进素材目录，文件可能正被占用：{exc}"})
                return
            adopted = None
            if single:
                adopted = adopt_clip(workbench.store, project_id, name, probe, replace=False)
            self._send(201, {
                "saved": str(target.relative_to(folder / "media")).replace("\\", "/"),
                "clip": adopted,
                **probe,
            })

        def _model(self, project_id: str) -> None:
            """选定 YOLO 和手部模型，检查能不能加载，记下指纹。"""
            folder = self._open(project_id)
            if folder is None:
                return
            try:
                merged = settings_module.clean(self._read_json(), workbench.store.load_settings(project_id))
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            yolo = model_registry.probe_yolo(merged["yolo_path"], merged["yolo_conf_threshold"])
            if not merged["yolo_path"]:
                self._send(400, {"error": "还没有选定 YOLO。先在下拉框里选一个 ONNX，再点检查并保存"})
                return
            if not yolo.get("ok"):
                self._send(400, {"yolo": yolo, "hands": {}, "error": yolo.get("error") or "YOLO 加载失败"})
                return
            if merged.get("hands_in_training"):
                hands = model_registry.probe_hands()
                model_registry.record_hands_probe(hands)
                if not hands.get("ok"):
                    self._send(400, {"yolo": yolo, "hands": hands, "error": hands.get("error") or "手部模型加载失败"})
                    return
                merged["hands_sha256"] = hands["sha256"]
            else:
                hands = {"ok": True, "skipped": True}
                merged["hands_sha256"] = ""
            if yolo.get("ok"):
                merged["yolo_sha256"] = yolo["sha256"]
                merged["yolo_classes"] = yolo["classes"]
            workbench.store.save_settings(project_id, merged)
            self._send(200, {"settings": merged, "yolo": yolo, "hands": hands})

        def _split(self, project_id: str) -> None:
            """提案或确认验证集划分。确认结果落盘，片子增删后自动作废。"""
            folder = self._open(project_id)
            if folder is None:
                return
            body = self._read_json()
            action = str(body.get("action") or "propose")
            if action not in {"propose", "confirm", "participate"}:
                self._send(400, {"error": f"不认识的动作：{action}"})
                return
            if action == "participate":
                try:
                    patch = {}
                    for key in ("train_excluded", "val_clip_ids", "split_mode", "network"):
                        if key in body:
                            patch[key] = body.get(key) or ([] if key in {"train_excluded", "val_clip_ids"} else body.get(key))
                    if "train_excluded" not in patch:
                        patch["train_excluded"] = []
                    current = settings_module.clean(patch, workbench.store.load_settings(project_id))
                except ValueError as exc:
                    self._send(400, {"error": str(exc)})
                    return
                workbench.store.save_settings(project_id, current)
                self._send(200, {"split": service.split_overview(workbench.store, project_id, current)})
                return
            try:
                current = settings_module.clean(
                    {key: body[key] for key in ("split_seed", "split_val_count") if key in body},
                    workbench.store.load_settings(project_id),
                )
                if action == "confirm":
                    current = settings_module.clean(
                        {"split_train": body.get("train") or [], "split_val": body.get("val") or []},
                        current,
                    )
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            if action == "propose":
                workbench.store.save_settings(project_id, current)
                self._send(200, {"split": service.split_overview(workbench.store, project_id, current)})
                return
            payload = workbench.store.load_annotations(project_id)
            try:
                # 确认和训练必须在同一口径上校验：换算到时间基之后的片子 + 原始片子的签名。
                prepared = service.prepare_clips(payload, current)
                checked = check_split(
                    prepared.clips, prepared.window, current["split_train"], current["split_val"]
                )
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            if not checked.ok:
                self._send(400, {"error": checked.blocked_reason})
                return
            current["split_for"] = prepared.signature
            workbench.store.save_settings(project_id, current)
            self._send(200, {"split": service.split_overview(workbench.store, project_id, current)})

        def _run_dir(self, project_id: str, run_name: str) -> Path | None:
            run_name = unquote(run_name)
            if not run_name or run_name in {".", ".."} or "/" in run_name or "\\" in run_name:
                self._send(400, {"error": "轮次名不合法"})
                return None
            folder = self._open(project_id)
            if folder is None:
                return None
            run = folder / "runs" / run_name
            if not run.is_dir() or not (run / "contract.json").is_file():
                self._send(404, {"error": "没有这一轮"})
                return None
            return run

        def _send_bounded_file(self, root: Path, relative: str) -> None:
            """只发送 root 里面的文件，并按 Range 读这一段，播放拖动时不必整文件进内存。"""
            relative = unquote(relative).replace("\\", "/").lstrip("/")
            target = (root / relative).resolve()
            base = root.resolve()
            if base not in target.parents and target != base:
                self._send(400, {"error": "文件名不合法"})
                return
            if not target.is_file():
                self._send(404, {"error": "文件不存在"})
                return
            self._stream_file(target, cache=False)

        def _check_get(self, parts: list[str]) -> None:
            run = self._run_dir(parts[2], parts[4])
            if run is None:
                return
            if len(parts) >= 8 and parts[6] == "file":
                self._send_bounded_file(check_clip.check_dir(run) / "media", "/".join(parts[7:]))
                return
            if len(parts) == 7 and parts[6] == "overlay":
                path = check_clip.overlay_path(run)
                if not path.is_file():
                    self._send(404, {"error": "还没有零件框和手部点"})
                    return
                self._send(200, path.read_bytes(), cache=False)
                return
            if len(parts) != 6:
                self._send(404, {"error": "没有这个接口"})
                return
            self._send(200, check_clip.check_status(run))

        def _check_post(self, parts: list[str]) -> None:
            project_id = parts[2]
            run = self._run_dir(project_id, parts[4])
            if run is None:
                return
            if len(parts) == 7 and parts[6] == "media":
                self._check_media(run)
                return
            if len(parts) == 7 and parts[6] == "overlay":
                self._check_overlay(project_id, run)
                return
            if len(parts) == 7 and parts[6] == "clip":
                self._check_project_clip(project_id, run)
                return
            if len(parts) != 6:
                self._send(404, {"error": "没有这个接口"})
                return
            self._check_run(project_id, run)

        def _prelabel(self, project_id: str, run_name: str) -> None:
            run = self._run_dir(project_id, run_name)
            if run is None:
                return
            body = self._read_json()
            clip_id = str(body.get("clip_id") or "").strip()
            if not clip_id:
                self._send(400, {"error": "需要指定片子"})
                return
            folder = run.parent.parent
            stop = self._start_job(project_id)
            if stop is None:
                return
            try:
                write_progress(folder, {"status": "extracting", "history": [], "features": {"done": 0, "total": 1, "clips": []}})

                def publish(done: int, total: int) -> None:
                    write_progress(folder, {
                        "status": "extracting",
                        "history": [],
                        "features": {"done": done, "total": max(total, 1), "clips": []},
                    })

                result = check_clip.prelabel_clip(
                    folder,
                    clip_id,
                    run,
                    _read_settings(workbench.store, project_id),
                    workbench.store.load_annotations(project_id),
                    on_progress=publish,
                    should_stop=stop.is_set,
                )
            except (check_clip.Stopped, feature_store.Stopped) as exc:
                write_progress(folder, {"status": "stopped", "error": str(exc), "history": []})
                self._send(200, {"stopped": True, "error": str(exc)})
                return
            except JOB_ERRORS as exc:
                self._fail_job(folder, exc)
                return
            finally:
                self._end_job()
            write_progress(folder, {"status": "idle", "history": [], "features": {"done": 1, "total": 1, "clips": []}})
            self._send(200, result)

        def _check_overlay(self, project_id: str, run: Path) -> None:
            self._read_json()
            folder = run.parent.parent
            stop = self._start_job(project_id)
            if stop is None:
                return
            try:
                write_progress(folder, {"status": "checking", "history": [], "check": {"done": 0, "total": 1}})

                def publish(done: int, total: int) -> None:
                    write_progress(folder, {"status": "checking", "history": [], "check": {"done": done, "total": total}})

                check_clip.fill_overlay(
                    run,
                    _read_settings(workbench.store, project_id),
                    on_progress=publish,
                    should_stop=stop.is_set,
                )
            except check_clip.Stopped as exc:
                write_progress(folder, {"status": "stopped", "error": str(exc), "history": []})
                self._send(200, {"stopped": True, "error": str(exc)})
                return
            except JOB_ERRORS as exc:
                self._fail_job(folder, exc)
                return
            finally:
                self._end_job()
            write_progress(folder, {"status": "check-done", "history": [], "check": {"done": 1, "total": 1}})
            self._send(200, {"overlay": True})

        def _check_media(self, run: Path) -> None:
            name = self.headers.get("X-Filename") or ""
            length = int(self.headers.get("Content-Length", "0"))
            data = self.rfile.read(length) if length else b""
            try:
                target = check_clip.save_check_video(run, name, data, replace=self.headers.get("X-Check-Reset") == "1")
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            self._send(201, {"saved": target.name})

        def _check_project_clip(self, project_id: str, run: Path) -> None:
            body = self._read_json()
            clip_id = str(body.get("clip_id") or "").strip()
            if not clip_id:
                self._send(400, {"error": "需要指定片子"})
                return
            folder = run.parent.parent
            stop = self._start_job(project_id)
            if stop is None:
                return
            try:
                write_progress(folder, {"status": "checking", "history": [], "check": {"done": 0, "total": 1}})

                def publish(done: int, total: int) -> None:
                    write_progress(folder, {"status": "checking", "history": [], "check": {"done": done, "total": max(total, 1)}})

                result = check_clip.check_project_clip(
                    folder,
                    clip_id,
                    run,
                    _read_settings(workbench.store, project_id),
                    workbench.store.load_annotations(project_id),
                    on_progress=publish,
                    should_stop=stop.is_set,
                )
            except (check_clip.Stopped, feature_store.Stopped) as exc:
                write_progress(folder, {"status": "stopped", "error": str(exc), "history": []})
                self._send(200, {"stopped": True, "error": str(exc)})
                return
            except JOB_ERRORS as exc:
                self._fail_job(folder, exc)
                return
            finally:
                self._end_job()
            write_progress(folder, {"status": "check-done", "history": [], "check": {"done": 1, "total": 1}})
            self._send(200, result)

        def _check_run(self, project_id: str, run: Path) -> None:
            self._read_json()
            folder = run.parent.parent
            stop = self._start_job(project_id)
            if stop is None:
                return
            try:
                write_progress(folder, {"status": "checking", "history": [], "check": {"done": 0, "total": 1}})

                def publish(done: int, total: int) -> None:
                    write_progress(folder, {"status": "checking", "history": [], "check": {"done": done, "total": total}})

                result = check_clip.run_check(
                    run,
                    _read_settings(workbench.store, project_id),
                    workbench.store.load_annotations(project_id),
                    on_progress=publish,
                    should_stop=stop.is_set,
                )
            except check_clip.Stopped as exc:
                write_progress(folder, {"status": "stopped", "error": str(exc), "history": []})
                self._send(200, {"stopped": True, "error": str(exc)})
                return
            except JOB_ERRORS as exc:
                self._fail_job(folder, exc)
                return
            except Exception as exc:
                self._fail_job(folder, exc)
                return
            finally:
                self._end_job()
            write_progress(folder, {"status": "check-done", "history": [], "check": {"done": 1, "total": 1}})
            self._send(200, {"check": {"ready": True, "segments": len(result.get("segments") or [])}})

        def _reveal_feature(self, project_id: str) -> None:
            """在资源管理器里选中这条片子的 npy。页面在本机，才能打开文件夹。"""
            folder = self._open(project_id)
            if folder is None:
                return
            body = self._read_json()
            clip_id = str(body.get("clip_id") or "").strip()
            if not clip_id or clip_id in {".", ".."} or "/" in clip_id or "\\" in clip_id:
                self._send(400, {"error": "片子名不合法"})
                return
            npy = (folder / "features" / f"{clip_id}.npy").resolve()
            root = (folder / "features").resolve()
            if root not in npy.parents:
                self._send(400, {"error": "文件名不合法"})
                return
            if not npy.is_file():
                self._send(404, {"error": "还没有这个特征文件"})
                return
            subprocess.Popen(["explorer", "/select,", str(npy)])
            self._send(200, {"opened": str(npy)})

        def _features(self, project_id: str) -> None:
            folder = self._open(project_id)
            if folder is None:
                return
            body = self._read_json()
            selected = body.get("clips") if isinstance(body, dict) else None
            clip_ids = [str(item) for item in selected] if isinstance(selected, list) else None
            stop = self._start_job(project_id)
            if stop is None:
                return
            try:
                annotations = workbench.store.load_annotations(project_id)
                chosen = clip_ids if clip_ids is not None else [str(clip["id"]) for clip in annotations.get("clips") or []]
                _arrays, report = _extract_tracked(
                    folder, annotations, _read_settings(workbench.store, project_id), chosen, stop.is_set
                )
            except feature_store.Stopped as exc:
                write_progress(folder, {"status": "stopped", "error": str(exc), "history": []})
                self._send(200, {"stopped": True, "error": str(exc)})
                return
            except JOB_ERRORS as exc:
                self._fail_job(folder, exc)
                return
            except Exception as exc:
                self._fail_job(folder, exc)
                return
            finally:
                self._end_job()
            write_progress(folder, {"status": "features-done", "history": [], "features": report})
            self._send(200, {"features": report})

        def _train(self, project_id: str) -> None:
            folder = self._open(project_id)
            if folder is None:
                return
            # 训练页按钮下面填的轮数、学习率等，开训前先落盘。训练读的就是这份。
            body = self._read_json()
            display_name = ""
            if isinstance(body, dict) and "display_name" in body:
                try:
                    display_name = settings_module.display_name(body.pop("display_name"))
                except ValueError as exc:
                    self._send(400, {"error": str(exc)})
                    return
            try:
                if body:
                    merged = settings_module.clean(body, workbench.store.load_settings(project_id))
                    workbench.store.save_settings(project_id, merged)
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            stop = self._start_job(project_id)
            if stop is None:
                return
            try:
                annotations = workbench.store.load_annotations(project_id)
                settings = _read_settings(workbench.store, project_id)
                chosen = service.clips_for_training(annotations, settings)
                clip_ids = [str(clip["id"]) for clip in chosen.get("clips") or [] if clip.get("id")]
                if not clip_ids:
                    raise ValueError("还没有勾选参加训练的片子")
                features, _report = _extract_tracked(folder, annotations, settings, clip_ids, stop.is_set)
                result = start_training(
                    workbench.store, project_id, features, should_stop=stop.is_set, display_name=display_name
                )
            except JOB_ERRORS as exc:
                self._fail_job(folder, exc)
                return
            except Exception as exc:
                self._fail_job(folder, exc)
                return
            finally:
                self._end_job()
            self._send(200, result)

    return Handler


def _probe_or_empty(target: Path) -> dict:
    from .extract import probe_media

    if target.is_dir() or target.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".jpg", ".jpeg", ".png", ".bmp"}:
        try:
            return probe_media(target)
        except ValueError as exc:
            return {"probe_error": str(exc)}
    return {}


def _extract_tracked(folder: Path, annotations: dict, settings: dict, clip_ids: list[str], should_stop) -> tuple[dict, dict]:
    """先抽参加训练的片子。已有最新特征就直接用，进度按帧写。"""
    by_id = {str(clip.get("id")): clip for clip in annotations.get("clips") or []}
    ordered = [by_id[item] for item in clip_ids if item in by_id]
    totals = {str(clip.get("id")): int(clip.get("frame_count") or 0) for clip in ordered}
    grand = sum(totals.values()) or 1
    finished = {"n": 0}
    done_rows: list[dict] = []

    def publish_frames(clip_id: str, done: int, total: int) -> None:
        clip_total = total or totals.get(clip_id) or 0
        write_progress(folder, {
            "status": "extracting",
            "history": [],
            "features": {
                "clips": done_rows,
                "done": finished["n"] + int(done),
                "total": grand,
                "clip_id": clip_id,
                "clip_done": int(done),
                "clip_total": clip_total,
            },
        })

    def publish(rows: list[dict]) -> None:
        done_rows[:] = rows
        finished["n"] = sum(totals.get(str(row.get("clip_id")), 0) for row in rows)
        current = str(rows[-1].get("clip_id")) if rows else ""
        write_progress(folder, {
            "status": "extracting",
            "history": [],
            "features": {
                "clips": rows,
                "done": finished["n"],
                "total": grand,
                "clip_id": current,
                "clip_done": totals.get(current, 0),
                "clip_total": totals.get(current, 0),
            },
        })

    write_progress(folder, {
        "status": "extracting",
        "history": [],
        "features": {"clips": [], "done": 0, "total": grand, "clip_id": "", "clip_done": 0, "clip_total": 0},
    })
    arrays, report = feature_store.extract_project(
        folder,
        annotations,
        settings,
        on_progress=publish,
        on_clip_frame=publish_frames,
        should_stop=should_stop,
        clip_ids=clip_ids,
    )
    return arrays, report


def _load_features(folder: Path, annotations: dict, settings: dict, clip_ids: list[str] | None = None) -> dict:
    """管线里要用特征时统一走这里，命中缓存就不重抽。只抽参加训练的片子。"""
    arrays, _report = feature_store.extract_project(folder, annotations, settings, clip_ids=clip_ids)
    return arrays


def serve(
    projects_root: Path,
    port: int = PORT,
    open_browser: bool = False,
    warm_models: bool = False,
) -> ThreadingHTTPServer:
    """warm_models 为真时，启动就在后台把手部模型加载一遍。

    默认关：测试里会反复起服务，每次都去构造一个 MediaPipe 图又慢又没意义。
    真正运行的入口 main() 会打开它。
    """
    workbench = Workbench(projects_root)
    server = ThreadingHTTPServer((HOST, port), make_handler(workbench))
    if server.server_address[0] != HOST:
        raise RuntimeError("训练台只能监听本机")
    if warm_models:
        model_registry.start_hands_warmup()
    if open_browser:
        webbrowser.open(f"http://{HOST}:{port}/")
    return server


def _port_from_env() -> int:
    """LSTM_WORKBENCH_PORT 可以改监听端口（默认 8771）。8771 被占时不用改代码。"""
    raw = os.environ.get("LSTM_WORKBENCH_PORT", "").strip()
    if not raw:
        return PORT
    try:
        port = int(raw)
    except ValueError:
        raise SystemExit(f"LSTM_WORKBENCH_PORT 不是整数：{raw}") from None
    if not 1 <= port <= 65535:
        raise SystemExit(f"LSTM_WORKBENCH_PORT 超出范围：{port}")
    return port


def main() -> None:
    port = _port_from_env()
    server = serve(ROOT / "projects", port=port, open_browser=True, warm_models=True)
    print(f"训练台 http://{HOST}:{port}/")
    print("手部模型：后台加载中，页面「模型与工位」会显示结果")
    server.serve_forever()


if __name__ == "__main__":
    main()
