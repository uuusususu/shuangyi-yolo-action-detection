"""训练台 HTTP 服务入口与路由装配。

架构分层：
- http_utils.py       Host/Origin 校验、Range 解析、局域网 IP 探测
- lan_auth.py         内网轻量访问口令（PIN）、HMAC 会话 Cookie 与只读模式
- workbench.py        单机 GPU 互斥锁、工程数据锁、后台准备与远程传输协调
- routes/media.py     视频/图片流式传输、分块上传、连续图收编
- routes/parts.py     零件素材、原生 OBB 标注、审核、质检与冻结发布
- routes/yolo.py      YOLO 训练/续训/官方图表/ONNX 导出/验证与预标注
- routes/action.py    LSTM 模型探测、划分、特征抽取、动作训练与视频复盘
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import sys
import threading
from urllib.parse import parse_qs, urlparse
import webbrowser

from . import (
    check_clip,
    dataset_transfer,
    feature_store,
    model_registry,
    service,
    settings as settings_module,
    yolo_jobs,
    yolo_validation,
)
from .http_utils import (
    BadRequest,
    LOOPBACK_NAMES,
    UPLOAD_CHUNK,
    _host_name,
    _is_allowed_host,
    _parse_range,
    _path_parts,
    _valid_name,
    detect_lan_ips,
    is_loopback_client,
    read_settings as _read_settings,
)
from .lan_auth import LanAuthManager
from .project_store import ProjectStore, project_id_problem
from .routes import ActionRoutesMixin, MediaRoutesMixin, PartsRoutesMixin, SmartLabelRoutesMixin, YoloRoutesMixin
from .service import forget_clip, start_training, write_progress
from .workbench import Workbench

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
HOST = "127.0.0.1"
PORT = 8771

_PUBLIC_LAN_PATHS = {"/api/server-info", "/api/auth/status", "/api/auth/unlock", "/api/auth/lock"}
_RUNNING_STATES = {"extracting", "training", "checking", "waiting-features", "loading-features"}


def make_handler(workbench: Workbench):
    class Handler(MediaRoutesMixin, PartsRoutesMixin, SmartLabelRoutesMixin, YoloRoutesMixin, ActionRoutesMixin, BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        @property
        def workbench(self) -> Workbench:
            return workbench

        def _client_ip(self) -> str:
            return str(self.client_address[0]) if self.client_address else ""

        def _is_local_client(self) -> bool:
            return is_loopback_client(self._client_ip())

        def send_response(self, code, message=None) -> None:  # noqa: D401
            self._response_started = True
            super().send_response(code, message)

        def _send(
            self,
            code: int,
            payload: object,
            content_type: str = "application/json",
            cache: bool = True,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if not cache:
                self.send_header("Cache-Control", "no-store")
            for key, value in (extra_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise BadRequest(f"请求内容不是合法的 JSON：{exc}") from None
            if not isinstance(body, dict):
                raise BadRequest("请求内容必须是 JSON 对象")
            return body

        def _dispatch(self, method: str, handler) -> None:
            if not self._guard(method):
                return
            try:
                handler()
            except BadRequest as exc:
                self._send_error_once(400, str(exc))
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                self.close_connection = True
            except Exception as exc:
                import traceback

                traceback.print_exc()
                self._send_error_once(500, f"训练台内部错误：{type(exc).__name__}: {exc}")

        def _send_error_once(self, code: int, message: str) -> None:
            if getattr(self, "_headers_buffer", None) or getattr(self, "_response_started", False):
                self.close_connection = True
                return
            try:
                self._send(code, {"error": message})
            except OSError:
                self.close_connection = True

        def _request_rejection(self, method: str) -> tuple[int, dict] | None:
            """三层请求守卫：Host/Origin 防跨站 → 写格式防简单表单 → 内网口令与只读策略。"""
            host = (self.headers.get("Host") or "").strip().lower()
            if not _is_allowed_host(_host_name(host), workbench.lan_enabled):
                msg = "只接受本机或内网访问" if workbench.lan_enabled else "只接受本机访问"
                return 403, {"error": msg}
            origin = (self.headers.get("Origin") or "").strip().lower()
            if origin and origin != "null" and not _is_allowed_host(_host_name(urlparse(origin).netloc), workbench.lan_enabled):
                return 403, {"error": "不接受其他网站发来的请求"}
            if method in {"POST", "PUT", "DELETE"}:
                content_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
                try:
                    has_body = int(self.headers.get("Content-Length", "0") or 0) > 0
                except ValueError:
                    return 403, {"error": "请求长度不合法"}
                if self.headers.get("X-Filename") is None and (has_body or method != "DELETE"):
                    if content_type != "application/json":
                        return 403, {"error": "请求格式不对：需要 application/json"}

            # 局域网远端设备访问时的轻量保护（本机 127.0.0.1 始终免检）
            if not self._is_local_client():
                req_path = urlparse(self.path).path
                if req_path.startswith("/api/") and req_path not in _PUBLIC_LAN_PATHS:
                    if workbench.lan_auth.pin_enabled and not workbench.lan_auth.is_request_unlocked(self.headers, self._client_ip()):
                        return 401, {"error": "需要输入内网访问口令（PIN）", "auth_required": True}
                    if workbench.lan_auth.readonly and method in {"POST", "PUT", "DELETE"}:
                        return 403, {"error": "当前处于内网只读模式，仅允许本机执行修改或训练操作", "readonly": True}
            return None

        def _guard(self, method: str) -> bool:
            rejection = self._request_rejection(method)
            if rejection is not None:
                code, payload = rejection
                self._discard_body()
                self._send(code, payload, cache=False)
                return False
            return True

        def _discard_body(self, limit: int = 1 << 20) -> None:
            try:
                remaining = int(self.headers.get("Content-Length", "0") or 0)
            except ValueError:
                remaining = 0
            if remaining > limit:
                self.close_connection = True
                return
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 65536))
                if not chunk:
                    break
                remaining -= len(chunk)

        def _open(self, project_id: str) -> Path | None:
            try:
                return workbench.store.open_dir(project_id)
            except (FileNotFoundError, ValueError):
                self._send(404, {"error": "工程不存在"})
                return None

        def _start_job(self, project_id: str, *, nonblocking: bool = False) -> threading.Event | None:
            busy = workbench.yolo_training()
            if busy:
                self._send(409, {"error": f"零件模型正在训练（{busy}），先等它结束或停止"})
                return None
            if not nonblocking and workbench._background_project is not None:
                folder = workbench.store.open_dir(project_id)
                write_progress(folder, {"status": "waiting-features", "history": []})
            admitted = workbench.try_begin_job(project_id) if nonblocking else workbench.begin_job(project_id)
            if not admitted:
                self._send(409, {"error": "正在跑任务，再点无效。要重来先停止"})
                return None
            self._job_project = project_id
            return workbench.stop_event(project_id)

        def _end_job(self) -> None:
            project_id = getattr(self, "_job_project", None)
            if project_id is not None:
                workbench.end_job(project_id)
                self._job_project = None

        def _fail_job(self, folder: Path, exc: Exception) -> None:
            try:
                write_progress(folder, {"status": "error", "error": str(exc), "history": []})
            except OSError:
                pass
            self._send(400, {"error": str(exc)})

        # ---------- GET ----------

        def do_GET(self) -> None:  # noqa: N802
            self._response_started = False
            self._dispatch("GET", self._handle_get)

        def _handle_get(self) -> None:
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8", cache=False)
                return
            if path == "/api/server-info":
                is_local = self._is_local_client()
                lan_ips = detect_lan_ips() if workbench.lan_enabled else []
                port_num = workbench.listen_port
                auth_state = workbench.lan_auth.status(is_local=is_local, headers=self.headers)
                self._send(200, {
                    "lan_enabled": workbench.lan_enabled,
                    "port": port_num,
                    "local_url": f"http://127.0.0.1:{port_num}/",
                    "lan_urls": [f"http://{ip}:{port_num}/" for ip in lan_ips],
                    "client_ip": self._client_ip(),
                    "is_local_client": is_local,
                    **auth_state,
                }, cache=False)
                return
            if path == "/api/auth/status":
                self._send(200, workbench.lan_auth.status(is_local=self._is_local_client(), headers=self.headers), cache=False)
                return
            if path == "/api/ultralytics/credentials":
                self._send(200, dataset_transfer.get_credentials_info(), cache=False)
                return
            if path == "/api/projects":
                self._send(200, {"projects": workbench.store.list_projects()})
                return
            if path == "/api/models":
                self._send(200, self._library_with_usage())
                return
            if path == "/api/yolo/env":
                refresh = "refresh=1" in (urlparse(self.path).query or "")
                self._send(200, yolo_jobs.environment(refresh=refresh))
                return

            parts = _path_parts(self.path)
            if len(parts) >= 2 and parts[:2] == ["api", "smart-models"]:
                self._smart_models_get(parts)
                return
            if len(parts) >= 5 and parts[:2] == ["api", "projects"] and parts[3:5] == ["parts", "smart"]:
                self._parts_smart_get(parts)
                return
            if len(parts) >= 4 and parts[:2] == ["api", "projects"] and parts[3] == "parts":
                self._parts_get(parts)
                return
            if len(parts) >= 4 and parts[:2] == ["api", "projects"] and parts[3] == "yolo":
                self._yolo_get(parts)
                return
            if len(parts) >= 5 and parts[:2] == ["api", "projects"] and parts[3] == "file":
                self._send_project_file(parts[2], "/".join(parts[4:]))
                return
            if len(parts) == 5 and parts[:2] == ["api", "projects"] and parts[3] == "features" and parts[4].endswith(".npy"):
                folder = self._open(parts[2])
                if folder is None:
                    return
                fname = parts[4]
                if not _valid_name(fname):
                    self._send(400, {"error": "文件名不合法"})
                    return
                npy = (folder / "features" / fname).resolve()
                root_feat = (folder / "features").resolve()
                if root_feat not in npy.parents or not npy.is_file():
                    self._send(404, {"error": "还没有这个特征文件"})
                    return
                self._stream_file(npy, cache=False, attachment_name=fname)
                return
            if len(parts) >= 3 and parts[:2] == ["api", "projects"]:
                project_id = parts[2]
                if len(parts) == 4 and parts[3] == "transfer":
                    folder = self._open(project_id)
                    if folder is None:
                        return
                    status_payload = workbench.transfer.status(project_id)
                    status_payload["archives"] = dataset_transfer.list_downloads(folder)
                    self._send(200, status_payload, cache=False)
                    return
                if len(parts) == 4 and parts[3] == "preparation":
                    if self._open(project_id) is not None:
                        self._send(200, workbench.preparation.status(project_id))
                    return
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
                if len(parts) >= 6 and parts[3] == "runs" and parts[5] == "check":
                    self._check_get(parts)
                    return
                if len(parts) == 5 and parts[3] == "runs":
                    if not _valid_name(parts[4]):
                        self._send(400, {"error": "轮次名不合法"})
                        return
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

            file_path = (STATIC / "/".join(parts)).resolve()
            static_root = STATIC.resolve()
            if static_root not in file_path.parents or not file_path.is_file():
                self._send(404, {"error": "没有这个页面"})
                return
            mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
            self._send(200, file_path.read_bytes(), mime, cache=False)

        # ---------- POST ----------

        def do_POST(self) -> None:  # noqa: N802
            self._response_started = False
            self._dispatch("POST", self._handle_post)

        def _handle_auth_post(self, action: str) -> None:
            body = self._read_json()
            if action == "unlock":
                pin = str(body.get("pin") or "")
                try:
                    ok = workbench.lan_auth.verify_pin(pin, client_ip=self._client_ip())
                except ValueError as exc:
                    self._send(429, {"error": str(exc)}, cache=False)
                    return
                if not ok:
                    self._send(401, {"error": "内网访问口令（PIN）不正确", "auth_required": True}, cache=False)
                    return
                token = workbench.lan_auth.issue_token()
                self._send(
                    200,
                    {"unlocked": True, **workbench.lan_auth.status(is_local=self._is_local_client())},
                    cache=False,
                    extra_headers={"Set-Cookie": workbench.lan_auth.cookie_header(token)},
                )
                return
            if action == "lock":
                self._send(
                    200,
                    {"unlocked": False},
                    cache=False,
                    extra_headers={"Set-Cookie": workbench.lan_auth.cookie_header(None)},
                )
                return
            if action == "config":
                if not self._is_local_client():
                    self._send(403, {"error": "仅允许在运行训练台的本机修改内网安全设置"})
                    return
                try:
                    state = workbench.lan_auth.configure(
                        pin=body.get("pin") if "pin" in body else None,
                        clear_pin=bool(body.get("clear_pin")),
                        readonly=body.get("readonly") if "readonly" in body else None,
                    )
                except ValueError as exc:
                    self._send(400, {"error": str(exc)})
                    return
                self._send(200, state, cache=False)
                return
            self._send(404, {"error": "没有这个接口"})

        def _handle_post(self) -> None:
            parts = _path_parts(self.path)
            if len(parts) == 3 and parts[:2] == ["api", "auth"]:
                self._handle_auth_post(parts[2])
                return
            if parts == ["api", "ultralytics", "credentials"]:
                body = self._read_json()
                try:
                    info = dataset_transfer.save_api_key(str(body.get("api_key") or ""))
                except ValueError as exc:
                    self._send(400, {"error": str(exc)})
                    return
                self._send(200, info)
                return
            if parts == ["api", "ultralytics", "parse"]:
                body = self._read_json()
                parsed = dataset_transfer.parse_ultralytics_input(str(body.get("text") or ""))
                parsed["has_api_key"] = bool(parsed.pop("api_key", ""))
                self._send(200, parsed)
                return
            if len(parts) >= 2 and parts[:2] == ["api", "smart-models"]:
                self._smart_models_post(parts)
                return
            if len(parts) >= 5 and parts[:2] == ["api", "projects"] and parts[3:5] == ["parts", "smart"]:
                self._parts_smart_post(parts)
                return
            if len(parts) >= 4 and parts[:2] == ["api", "projects"] and parts[3] == "parts":
                self._parts_post(parts)
                return
            if len(parts) == 4 and parts[0] == "api" and parts[3] == "media":
                self._media(parts[2])
                return
            if len(parts) >= 5 and parts[:2] == ["api", "projects"] and parts[3] == "yolo":
                self._yolo_post(parts)
                return
            if parts == ["api", "projects"]:
                body = self._read_json()
                project_id = str(body.get("id") or "").strip()
                problem = project_id_problem(project_id)
                if problem:
                    self._send(400, {"error": problem})
                    return
                try:
                    workbench.store.create(project_id, settings_module.clean(body.get("settings") or {}))
                except FileExistsError:
                    self._send(409, {"error": "工程已存在（工程名不分大小写）"})
                    return
                except ValueError as exc:
                    self._send(400, {"error": str(exc)})
                    return
                except OSError as exc:
                    self._send(400, {"error": f"建不了这个工程目录：{exc}"})
                    return
                self._send(201, {"id": project_id})
                return
            if len(parts) == 5 and parts[:2] == ["api", "projects"] and parts[3:] == ["features", "reveal"]:
                self._reveal_feature(parts[2])
                return
            if len(parts) >= 6 and parts[:2] == ["api", "projects"] and parts[3] == "runs" and parts[5] == "check":
                self._check_post(parts)
                return
            if len(parts) == 6 and parts[:2] == ["api", "projects"] and parts[3] == "runs" and parts[5] == "prelabel":
                self._prelabel(parts[2], parts[4])
                return
            if len(parts) != 4 or parts[:2] != ["api", "projects"]:
                self._send(404, {"error": "没有这个接口"})
                return

            project_id, action = parts[2], parts[3]
            if action == "transfer":
                if self._open(project_id) is None:
                    return
                body = self._read_json()
                if str(body.get("action") or "") == "cancel":
                    self._send(200, workbench.transfer.cancel(project_id))
                    return
                try:
                    job = workbench.transfer.start_remote_or_local_transfer(project_id, body)
                except RuntimeError as exc:
                    self._send(409, {"error": str(exc)})
                    return
                except (ValueError, FileNotFoundError, OSError) as exc:
                    self._send(400, {"error": str(exc)})
                    return
                self._send(202, job)
                return
            if action == "annotations":
                if self._open(project_id) is None:
                    return
                payload = self._read_json()
                problem = service.validate_annotations(payload)
                if problem:
                    self._send(400, {"error": problem})
                    return
                clip_id = parse_qs(urlparse(self.path).query).get("clip_id", [None])[0]
                known = {str(c["id"]) for c in payload.get("clips", [])}
                if clip_id is not None and clip_id not in known:
                    self._send(400, {"error": "没有这条标注视频"})
                    return
                with workbench.data_lock(project_id):
                    previous = workbench.store.load_annotations(project_id)
                    payload = service.apply_annotation_completion(previous, payload)
                    workbench.store.save_annotations(project_id, payload)
                chosen = [clip_id] if clip_id in known else [str(c["id"]) for c in payload.get("clips", []) if c.get("segments")]
                try:
                    preparation = workbench.preparation.request(project_id, chosen)
                    self._send(200, {"saved": True, "preparation": preparation})
                except (OSError, ValueError) as exc:
                    self._send(200, {"saved": True, "preparation_error": str(exc)})
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
            if action == "preparation":
                if self._open(project_id) is None:
                    return
                body = self._read_json()
                op = body.get("action", "prepare")
                try:
                    if op == "pause":
                        result = workbench.preparation.pause(project_id)
                    elif op == "prepare":
                        result = workbench.preparation.request(project_id, body.get("clips"), resume=True)
                    else:
                        raise ValueError("不认识的特征准备动作")
                except (OSError, ValueError) as exc:
                    self._send(400, {"error": str(exc)})
                    return
                self._send(200, result)
                return
            if action == "train":
                self._train(project_id)
                return
            if action == "stop":
                workbench.stop_event(project_id).set()
                workbench.transfer.cancel(project_id)
                workbench.smart_label.stop_batch(project_id)
                folder = self._open(project_id)
                if folder is None:
                    return
                run = yolo_jobs.active_run(folder)
                if run is not None:
                    yolo_jobs.stop_training(run)
                self._send(200, {"stopping": True})
                return
            self._send(404, {"error": "没有这个接口"})

        # ---------- DELETE ----------

        def do_DELETE(self) -> None:  # noqa: N802
            self._response_started = False
            self._dispatch("DELETE", self._handle_delete)

        def _handle_delete(self) -> None:
            parts = _path_parts(self.path)
            if len(parts) == 6 and parts[:2] == ["api", "projects"] and parts[3] == "runs" and parts[5] == "check":
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
            if len(parts) == 5 and parts[:2] == ["api", "projects"] and parts[3] == "clips":
                project_id, clip_id = parts[2], parts[4]
                if self._open(project_id) is None:
                    return
                if workbench.lock_for(project_id).locked():
                    self._send(409, {"error": "正在跑任务，先停止再删片子"})
                    return
                try:
                    with workbench.data_lock(project_id):
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
    fresh = {row["clip_id"] for row in feature_store.summary(folder, annotations, settings)["clips"] if row["fresh"]}
    phase = "loading-features" if all(clip_id in fresh for clip_id in clip_ids) else "extracting"
    finished = {"n": 0}
    done_rows: list[dict] = []

    def publish_frames(clip_id: str, done: int, total: int) -> None:
        clip_total = total or totals.get(clip_id) or 0
        write_progress(folder, {
            "status": phase,
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
            "status": phase,
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
        "status": phase,
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


class WorkbenchServer(ThreadingHTTPServer):
    def shutdown(self):
        if getattr(self, "workbench", None) is not None:
            self.workbench.preparation.close()
            self.workbench.close_async_jobs()
        super().shutdown()

    def server_close(self):
        if getattr(self, "workbench", None) is not None:
            self.workbench.preparation.close()
            self.workbench.close_async_jobs()
        super().server_close()


def serve(
    projects_root: Path,
    port: int = PORT,
    open_browser: bool = False,
    warm_models: bool = False,
    host: str = HOST,
    lan_pin: str = "",
    lan_readonly: bool | None = None,
    auth_config_path: Path | None = None,
) -> ThreadingHTTPServer:
    """启动训练台 HTTP 服务。默认 host='127.0.0.1'（测试仅绑本机）；main() 默认 '0.0.0.0' 开启内网。"""
    bind_host = (host or HOST).strip()
    lan_auth = LanAuthManager(config_path=auth_config_path, initial_pin=lan_pin, initial_readonly=lan_readonly)
    workbench = Workbench(projects_root, lan_auth=lan_auth)
    server = WorkbenchServer((bind_host, port), make_handler(workbench))
    actual_host, actual_port = server.server_address[:2]
    workbench.listen_port = int(actual_port)
    workbench.lan_enabled = bind_host != HOST
    server.workbench = workbench
    _mark_interrupted_jobs(workbench.store)
    workbench.preparation.start()
    if not workbench.lan_enabled and actual_host != HOST:
        raise RuntimeError("训练台本机模式只能监听 127.0.0.1")
    if warm_models:
        model_registry.start_hands_warmup()
    if open_browser:
        webbrowser.open(f"http://{HOST}:{actual_port}/")
    return server


def _mark_interrupted_jobs(store: ProjectStore) -> None:
    """启动时把上次没跑完的任务标成已中断。"""
    for project_id in store.list_projects():
        yolo_validation.mark_interrupted(store.root / project_id)
        path = store.root / project_id / "progress.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("status") in _RUNNING_STATES:
            try:
                write_progress(store.root / project_id, {
                    "status": "error",
                    "error": "上次的任务没有跑完（训练台被关闭或异常退出），请重新开始",
                    "history": payload.get("history") or [],
                })
            except OSError:
                pass


def _port_from_env() -> int:
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


def _host_from_args_env(args: list[str]) -> str:
    if "--local" in args or "--local-only" in args:
        return HOST
    if "--lan" in args:
        return "0.0.0.0"
    env_host = os.environ.get("LSTM_WORKBENCH_HOST", "").strip()
    return env_host if env_host else "0.0.0.0"


def _arg_value(args: list[str], flag: str) -> str:
    for idx, item in enumerate(args):
        if item == flag and idx + 1 < len(args):
            return args[idx + 1].strip()
        if item.startswith(flag + "="):
            return item.split("=", 1)[1].strip()
    return ""


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(encoding="utf-8", errors="replace")
            except Exception:
                pass
    args = sys.argv[1:] if argv is None else argv
    open_browser = "--no-browser" not in args
    port = _port_from_env()
    bind_host = _host_from_args_env(args)
    lan_pin = _arg_value(args, "--lan-pin")
    lan_readonly = True if "--lan-readonly" in args else None
    local_url = f"http://{HOST}:{port}/"
    try:
        server = serve(
            model_registry.HOME / "projects",
            port=port,
            open_browser=open_browser,
            warm_models=True,
            host=bind_host,
            lan_pin=lan_pin,
            lan_readonly=lan_readonly,
        )
    except OSError:
        print(f"端口 {port} 已被占用，训练台可能已经在运行：{local_url}", flush=True)
        if open_browser:
            webbrowser.open(local_url)
        return

    actual_port = server.workbench.listen_port
    local_url = f"http://{HOST}:{actual_port}/"
    lan_ips = detect_lan_ips() if server.workbench.lan_enabled else []
    auth = server.workbench.lan_auth
    banner_lines = [
        "=" * 62,
        "双翼科技 · 零件与动作统一训练工作台已启动",
        f"  本机访问：{local_url}",
    ]
    if server.workbench.lan_enabled:
        if lan_ips:
            for ip in lan_ips:
                banner_lines.append(f"  内网访问：http://{ip}:{actual_port}/   （网线直连/局域网电脑可打开并推送数据）")
        else:
            banner_lines.append(f"  内网监听：已开启 (0.0.0.0:{actual_port})，插上网线连接局域网后即可用本机 IP 访问")
        sec_tags = []
        if auth.pin_enabled:
            sec_tags.append("已开启 PIN 访问口令")
        if auth.readonly:
            sec_tags.append("内网只读模式")
        banner_lines.append(f"  内网保护：{' · '.join(sec_tags) if sec_tags else '未设口令（可在本机页面右上角一键开启 PIN 或只读模式）'}")
        banner_lines.append("  提示：若内网电脑无法打开页面，请双击运行一次「放行内网端口.bat」放行防火墙")
    banner_lines.extend([
        f"  工程与模型目录：{model_registry.HOME}",
        "  手部模型：后台加载中，页面「模型与工位」会显示结果",
        "  关闭这个窗口就会停止训练台。",
        "=" * 62,
    ])
    banner = "\n".join(banner_lines)
    print(banner, flush=True)
    try:
        log_dir = ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "last_startup.log").write_text(banner + "\n", encoding="utf-8")
    except OSError:
        pass
    server.serve_forever()


if __name__ == "__main__":
    main()
