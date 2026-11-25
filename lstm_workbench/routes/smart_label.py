"""智能标注路由：Meta SAM 3 / SAM 2.1 / MobileSAM / YOLOE 状态、激活、卸载、交互点选、下载、上传与批量自动标。"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse
import uuid

from .. import model_registry
from ..http_utils import _valid_name
from ..smart_label import DEFAULT_SMART_MODEL, TrainingLockedError


class SmartLabelRoutesMixin:
    def _smart_error(self, exc: Exception) -> None:
        if isinstance(exc, TrainingLockedError):
            self._send(409, {"error": str(exc), "training_locked": True}, cache=False)
            return
        if isinstance(exc, FileNotFoundError):
            self._send(404, {"error": str(exc)}, cache=False)
            return
        if isinstance(exc, RuntimeError) and ("正在运行" in str(exc) or "抢占" in str(exc)):
            self._send(409, {"error": str(exc)}, cache=False)
            return
        self._send(400, {"error": str(exc)}, cache=False)

    def _smart_models_get(self, parts: list[str]) -> None:
        refresh = "refresh=1" in (urlparse(self.path).query or "")
        if len(parts) == 2:
            self._send(200, self.workbench.smart_label.status(refresh_env=refresh), cache=False)
            return
        self._send(404, {"error": "没有这个接口"})

    def _smart_models_upload(self) -> None:
        """将本地 .pt / .pth 模型文件流式上传并保存到 models/sam/ 目录。"""
        raw_name = self._upload_name()
        leaf = Path(raw_name).name if raw_name else ""
        try:
            folder = model_registry.ensure_sam_library()
            target = model_registry.resolve_sam_path(leaf)
        except (ValueError, OSError) as exc:
            self._drain_body()
            self._send(400, {"error": str(exc)}, cache=False)
            return

        tmp_path = folder / f".upload-{uuid.uuid4().hex[:10]}.part"
        try:
            self._receive_to(tmp_path)
            if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
                raise ValueError("上传的模型文件为空")
            # 若覆盖的是当前正在运行的同名模型，先停止释放句柄（避免 Windows 文件占用）
            active_now = (self.workbench.smart_label.status().get("active_model") or "").lower()
            if active_now and active_now == target.name.lower():
                self.workbench.smart_label.unload("上传新权重替换原运行模型，已自动停止旧实例")
            os.replace(tmp_path, target)
            payload = self.workbench.smart_label.status()
            payload["uploaded_model"] = target.name
            self._send(201, payload, cache=False)
        except (TrainingLockedError, ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
            tmp_path.unlink(missing_ok=True)
            self._smart_error(exc)

    def _smart_models_post(self, parts: list[str]) -> None:
        if len(parts) == 3 and parts[2] == "upload":
            self._smart_models_upload()
            return
        body = self._read_json()
        smart = self.workbench.smart_label
        try:
            if len(parts) == 3 and parts[2] == "activate":
                model_name = str(body.get("model") or DEFAULT_SMART_MODEL).strip()
                slot = str(body.get("slot") or "").strip()
                if slot and slot not in {"auto", "interactive", "semantic", "video", "yoloe_vp", "yoloe_text"}:
                    raise ValueError("不支持的模型槽位类型")
                res = smart.activate(
                    model_name,
                    slot=slot,
                    auto_default=bool(body.get("auto_default")),
                )
                self._send(200, res, cache=False)
                return
            if len(parts) == 3 and parts[2] == "download":
                model_name = str(body.get("model") or "").strip()
                custom_url = str(body.get("url") or "").strip()
                record = smart.start_download(model_name, custom_url=custom_url)
                self._send(202, record, cache=False)
                return
            if len(parts) == 3 and parts[2] == "unload":
                reason = str(body.get("reason") or "已手动停止模型")
                self._send(200, smart.unload(reason=reason), cache=False)
                return
            if len(parts) == 3 and parts[2] == "open-dir":
                folder = model_registry.ensure_sam_library()
                if not self._is_local_client():
                    self._send(403, {"error": f"仅限本机打开文件夹；模型目录为：{folder}", "dir": str(folder)}, cache=False)
                    return
                opened = False
                if os.name == "nt" and hasattr(os, "startfile"):
                    os.startfile(str(folder))
                    opened = True
                self._send(200, {"ok": True, "opened": opened, "dir": str(folder)}, cache=False)
                return
            self._send(404, {"error": "没有这个接口"})
        except (TrainingLockedError, ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
            self._smart_error(exc)

    def _parts_smart_get(self, parts: list[str]) -> None:
        """GET /api/projects/<id>/parts/smart[/status]"""
        project_id = parts[2]
        if self._open(project_id) is None:
            return
        refresh = "refresh=1" in (urlparse(self.path).query or "")
        if len(parts) == 5 or (len(parts) == 6 and parts[5] == "status"):
            self._send(200, self.workbench.smart_label.status(project_id, refresh_env=refresh), cache=False)
            return
        self._send(404, {"error": "没有这个接口"})

    def _parts_smart_post(self, parts: list[str]) -> None:
        """POST /api/projects/<id>/parts/smart/<action>"""
        project_id = parts[2]
        if self._open(project_id) is None:
            return
        body = self._read_json()
        action = parts[5] if len(parts) >= 6 else ""
        smart = self.workbench.smart_label
        try:
            if action == "activate":
                model_name = str(body.get("model") or DEFAULT_SMART_MODEL).strip()
                slot = str(body.get("slot") or "").strip()
                if slot and slot not in {"auto", "interactive", "semantic", "video", "yoloe_vp", "yoloe_text"}:
                    raise ValueError("不支持的模型槽位类型")
                res = smart.activate(
                    model_name,
                    project_id=project_id,
                    slot=slot,
                    auto_default=bool(body.get("auto_default")),
                )
                self._send(200, res, cache=False)
                return
            if action == "unload":
                reason = str(body.get("reason") or "离开标注页释放显存")
                self._send(200, smart.unload(reason=reason), cache=False)
                return
            if action == "encode":
                item_id = str(body.get("item_id") or "").strip()
                if not _valid_name(item_id):
                    raise ValueError("素材编号不合法")
                res = smart.encode_image(project_id, item_id)
                self._send(200, res, cache=False)
                return
            if action == "predict":
                item_id = str(body.get("item_id") or "").strip()
                if not _valid_name(item_id):
                    raise ValueError("素材编号不合法")
                res = smart.predict_interactive(project_id, item_id, body)
                self._send(200, res, cache=False)
                return
            if action == "batch-zeroshot":
                job = smart.start_batch_zeroshot(project_id, body)
                self._send(202, job, cache=False)
                return
            if action == "batch-track":
                job = smart.start_batch_track(project_id, body)
                self._send(202, job, cache=False)
                return
            if action == "stop":
                self._send(200, smart.stop_batch(project_id), cache=False)
                return
            if action == "download":
                model_name = str(body.get("model") or "").strip()
                custom_url = str(body.get("url") or "").strip()
                record = smart.start_download(model_name, custom_url=custom_url)
                self._send(202, record, cache=False)
                return
            self._send(404, {"error": "没有这个智能标注接口"})
        except (TrainingLockedError, ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
            self._smart_error(exc)
