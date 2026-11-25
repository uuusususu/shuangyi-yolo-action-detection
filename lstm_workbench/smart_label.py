"""智能标注管理器：Meta SAM 3 / SAM 2.1 / MobileSAM / YOLOE 常驻调度、按需下载、批量自动标注与 8GB 显存强互斥。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any
from urllib.parse import urlparse
import urllib.request
import uuid

from . import model_registry, yolo_jobs
from .parts_store import PartsStore

DEFAULT_SMART_MODEL = "sam3.pt"
IDLE_TIMEOUT_SECONDS = 600.0
VIDEO_TRACK_CHUNK_SIZE = 40
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_WORKER_SCRIPT = Path(__file__).resolve().parent / "smart_label_worker.py"


class TrainingLockedError(RuntimeError):
    """训练或重型 GPU 任务正在运行时，禁止启动或使用任何智能/自动标注（含 CPU 模式）。"""


def _kill_process_tree(pid: int) -> None:
    if pid <= 0:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                capture_output=True,
                timeout=5,
                creationflags=_NO_WINDOW,
            )
        except Exception:
            pass
    else:
        try:
            os.kill(pid, 9)
        except OSError:
            pass


class SmartLabelManager:
    """工作台全局唯一的智能标注协调器，单卡 8GB 显存下保证单槽位驻留与训练强互斥。"""

    def __init__(self, workbench: Any):
        self.workbench = workbench
        self._rpc_lock = threading.RLock()
        self._meta_lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._state: str = "idle"  # idle | loading | ready | unavailable
        self._active_model: str = ""
        self._active_family: str = ""
        self._active_slot: str = ""
        self._active_device: str = ""
        self._active_half: bool = False
        self._vram: dict = {}
        self._last_error: dict = {}
        self._last_used: float = 0.0
        self._env_cache: dict | None = None
        self._downloads: dict[str, dict] = {}
        self._download_stops: dict[str, threading.Event] = {}
        self._batch_jobs: dict[str, dict] = {}
        self._batch_stops: dict[str, threading.Event] = {}
        self._batch_threads: dict[str, threading.Thread] = {}
        self._watchdog_stop = threading.Event()
        self._watchdog: threading.Thread | None = None

    def _ensure_watchdog(self) -> None:
        with self._meta_lock:
            if self._watchdog is not None and self._watchdog.is_alive():
                return
            self._watchdog_stop.clear()
            self._watchdog = threading.Thread(target=self._idle_loop, name="smart-label-watchdog", daemon=True)
            self._watchdog.start()

    # ---------- 显存状态与训练互斥 ----------

    def is_active(self) -> bool:
        """是否有常驻推理子进程或进行中的智能批量任务。"""
        with self._meta_lock:
            proc_alive = self._proc is not None and self._proc.poll() is None
            batch_running = any(j.get("status") in {"queued", "running"} for j in self._batch_jobs.values())
            return bool(proc_alive or batch_running or self._state == "loading")

    def _check_not_training_locked(self) -> None:
        if self.workbench.is_training_locked():
            raise TrainingLockedError("训练或重型 GPU 任务正在运行，已彻底禁用智能标注（含 CPU 模式）以保障训练资源，当前请继续使用纯人工标注。")

    def stop_and_release(self, reason: str = "训练任务抢占显存", timeout: float = 5.0) -> None:
        """同步停止所有进行中的智能批量任务与常驻 Worker，并等待子进程退出彻底归还显存。"""
        # 1. 先取消全部进行中的智能批量标注线程
        with self._meta_lock:
            stops = list(self._batch_stops.values())
            threads = list(self._batch_threads.values())
            for job in self._batch_jobs.values():
                if job.get("status") in {"queued", "running"}:
                    job["status"] = "stopped"
                    job["error"] = f"已中断：{reason}"
                    job["finished_at"] = time.time()
        for ev in stops:
            ev.set()
        deadline = time.monotonic() + min(timeout, 2.5)
        for thread in threads:
            if thread.is_alive() and thread is not threading.current_thread():
                remain = max(0.05, deadline - time.monotonic())
                thread.join(timeout=remain)

        # 2. 优雅通知常驻 Worker 退出，超时强杀进程树，确保 CUDA 上下文彻底回收
        with self._rpc_lock:
            proc = self._proc
            self._proc = None
            with self._meta_lock:
                self._state = "idle"
                self._active_model = ""
                self._active_family = ""
                self._active_slot = ""
                self._active_device = ""
                self._active_half = False
                self._last_used = 0.0
                if reason:
                    self._last_error = {"code": "preempted", "message": reason}
            if proc is not None and proc.poll() is None:
                try:
                    if proc.stdin and not proc.stdin.closed:
                        proc.stdin.write(json.dumps({"id": "shutdown", "cmd": "shutdown"}) + "\n")
                        proc.stdin.flush()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=max(0.5, timeout))
                except subprocess.TimeoutExpired:
                    _kill_process_tree(proc.pid)
                    try:
                        proc.kill()
                        proc.wait(timeout=2.0)
                    except Exception:
                        pass
                finally:
                    for stream in (proc.stdin, proc.stdout, proc.stderr):
                        if stream is not None:
                            try:
                                stream.close()
                            except Exception:
                                pass

    def unload(self, reason: str = "已手动停止模型") -> dict:
        self.stop_and_release(reason=reason, timeout=4.0)
        with self._meta_lock:
            self._last_error = {"code": "stopped", "message": reason} if reason else {}
        return self.status()

    @staticmethod
    def default_slot_for_model(model_name: str) -> str:
        family = model_registry.smart_model_family(model_name)
        return "yoloe_vp" if family == "yoloe" else "interactive"

    def _idle_loop(self) -> None:
        while not self._watchdog_stop.wait(30.0):
            try:
                with self._meta_lock:
                    proc_alive = self._proc is not None and self._proc.poll() is None
                    batch_running = any(j.get("status") in {"queued", "running"} for j in self._batch_jobs.values())
                    idle_for = (time.monotonic() - self._last_used) if self._last_used > 0 else 0.0
                if not proc_alive:
                    return
                if not batch_running and idle_for >= IDLE_TIMEOUT_SECONDS:
                    self.stop_and_release(reason="智能标注空闲超时，已自动卸载释放显存", timeout=4.0)
                    return
            except Exception:
                continue

    # ---------- 环境探测与可用模型清单 ----------

    def probe_env(self, refresh: bool = False) -> dict:
        with self._meta_lock:
            if self._env_cache is not None and not refresh:
                return dict(self._env_cache)
        yolo_env = yolo_jobs.environment(refresh=refresh)
        if not yolo_env.get("ok"):
            result = {
                "ok": False,
                "python_path": "",
                "cuda": False,
                "gpu": "",
                "total_mb": 0,
                "free_mb": 0,
                "ultralytics": "",
                "clip": False,
                "mobileclip": False,
                "clip_install_cmd": "",
                "mobileclip_install_cmd": "",
                "error": yolo_env.get("error") or "未检测到 YOLO Python 环境",
            }
            with self._meta_lock:
                self._env_cache = result
            return dict(result)

        py_path = str(yolo_env["python_path"])
        clip_cmd = f'"{py_path}" -m pip install git+https://github.com/ultralytics/CLIP.git'
        mobileclip_cmd = f'"{py_path}" -m pip install git+https://github.com/ultralytics/mobileclip.git'
        probe_code = (
            "import importlib.util, json, torch, ultralytics\n"
            "cuda = torch.cuda.is_available()\n"
            "free_mb = total_mb = 0\n"
            "gpu = ''\n"
            "if cuda:\n"
            "  free_b, total_b = torch.cuda.mem_get_info(0)\n"
            "  free_mb, total_mb = int(free_b // (1024*1024)), int(total_b // (1024*1024))\n"
            "  gpu = torch.cuda.get_device_name(0)\n"
            "print(json.dumps({'cuda': cuda, 'gpu': gpu, 'total_mb': total_mb, 'free_mb': free_mb,"
            " 'ultralytics': ultralytics.__version__,"
            " 'clip': importlib.util.find_spec('clip') is not None,"
            " 'mobileclip': importlib.util.find_spec('mobileclip') is not None}))\n"
        )
        info = {
            "ok": True,
            "python_path": py_path,
            "cuda": bool(yolo_env.get("cuda")),
            "gpu": str(yolo_env.get("gpu") or ""),
            "total_mb": 0,
            "free_mb": 0,
            "ultralytics": str(yolo_env.get("ultralytics") or ""),
            "clip": False,
            "mobileclip": False,
            "clip_install_cmd": clip_cmd,
            "mobileclip_install_cmd": mobileclip_cmd,
        }
        try:
            done = subprocess.run(
                [py_path, "-c", probe_code],
                capture_output=True,
                text=True,
                timeout=30,
                encoding="utf-8",
                errors="replace",
                creationflags=_NO_WINDOW,
            )
            lines = (done.stdout or "").strip().splitlines()
            if lines:
                parsed = json.loads(lines[-1])
                info.update(parsed)
        except Exception:
            pass
        with self._meta_lock:
            self._env_cache = info
        return dict(info)

    def available_models(self, project_id: str | None = None, *, env: dict | None = None) -> list[dict]:
        env_info = env or self.probe_env()
        local_map = {item["name"].lower(): item for item in model_registry.list_sam_models()}
        with self._meta_lock:
            downloads = {k.lower(): dict(v) for k, v in self._downloads.items()}
            active = self._active_model.lower()

        models: list[dict] = []
        seen: set[str] = set()
        for entry in model_registry.SMART_MODEL_CATALOG:
            key = entry["name"].lower()
            seen.add(key)
            local = local_map.get(key)
            text_dep = entry.get("text_dep") or ""
            dep_ok = True if not text_dep else bool(env_info.get(text_dep))
            install_cmd = (
                env_info.get("clip_install_cmd", "")
                if text_dep == "clip"
                else env_info.get("mobileclip_install_cmd", "")
                if text_dep == "mobileclip"
                else ""
            )
            models.append(
                {
                    "name": entry["name"],
                    "label": entry["label"],
                    "family": entry["family"],
                    "kind": "sam_family",
                    "capabilities": list(entry["capabilities"]),
                    "vram_gb": entry["vram_gb"],
                    "gated": bool(entry.get("gated")),
                    "downloadable": bool(entry.get("downloadable")),
                    "local": local is not None,
                    "bytes": local["bytes"] if local else 0,
                    "active": active == key and self._state == "ready",
                    "text_dep": text_dep,
                    "text_dep_ok": dep_ok,
                    "install_cmd": "" if dep_ok else install_cmd,
                    "note": entry["note"],
                    "download": downloads.get(key),
                }
            )

        # 追加用户手动放到 models/sam/ 的自定义权重
        for key, local in local_map.items():
            if key in seen:
                continue
            models.append(
                {
                    "name": local["name"],
                    "label": local["label"],
                    "family": local["family"],
                    "kind": "sam_family",
                    "capabilities": local["capabilities"],
                    "vram_gb": local["vram_gb"],
                    "gated": False,
                    "downloadable": False,
                    "local": True,
                    "bytes": local["bytes"],
                    "active": active == key and self._state == "ready",
                    "text_dep": local.get("text_dep", ""),
                    "text_dep_ok": True,
                    "install_cmd": "",
                    "note": "本地自定义智能标注权重",
                    "download": None,
                }
            )

        # 补充本工程已有 YOLO-OBB 训练轮次与模型库 ONNX，让面板完整呈现可用标注模型
        if project_id:
            try:
                folder = self.workbench.store.open_dir(project_id)
                for run in yolo_jobs.list_runs(folder):
                    best = yolo_jobs.runs_dir(folder) / run["name"] / "weights" / "best.pt"
                    if best.is_file():
                        models.append(
                            {
                                "name": f"run:{run['name']}/best.pt",
                                "label": f"本工程 YOLO-OBB（{run['name']} / best.pt）",
                                "family": "yolo_obb",
                                "kind": "yolo_preannotate",
                                "run": run["name"],
                                "model": "best.pt",
                                "capabilities": ["batch_preannotate"],
                                "vram_gb": 0.6,
                                "gated": False,
                                "downloadable": False,
                                "local": True,
                                "bytes": best.stat().st_size,
                                "active": False,
                                "text_dep": "",
                                "text_dep_ok": True,
                                "install_cmd": "",
                                "note": "使用本工程已训练的 YOLO-OBB 权重批量预标注",
                                "download": None,
                            }
                        )
            except Exception:
                pass
        for item in model_registry.list_yolo_models():
            models.append(
                {
                    "name": item["name"],
                    "label": f"模型库 YOLO（{item['name']}）",
                    "family": "yolo_obb",
                    "kind": "yolo_preannotate",
                    "run": "",
                    "model": item["name"],
                    "capabilities": ["batch_preannotate"],
                    "vram_gb": 0.5,
                    "gated": False,
                    "downloadable": False,
                    "local": True,
                    "bytes": item["bytes"],
                    "active": False,
                    "text_dep": "",
                    "text_dep_ok": True,
                    "install_cmd": "",
                    "note": "使用全局模型库中的 YOLO ONNX 批量预标注",
                    "download": None,
                }
            )
        return models

    def status(self, project_id: str | None = None, *, refresh_env: bool = False) -> dict:
        locked = self.workbench.is_training_locked()
        env = self.probe_env(refresh=refresh_env)
        models = self.available_models(project_id, env=env)
        with self._meta_lock:
            proc_alive = self._proc is not None and self._proc.poll() is None
            state = "blocked_by_training" if locked else (self._state if (proc_alive or self._state != "ready") else "idle")
            if state == "idle" and self._state == "ready" and not proc_alive:
                self._state = "idle"
                self._active_model = ""
                self._active_slot = ""
            err = dict(self._last_error) if self._last_error else {}
            batch = dict(self._batch_jobs.get(project_id) or {}) if project_id else None
            downloads = {k: dict(v) for k, v in self._downloads.items()}
            active_model = "" if locked else self._active_model
            active_family = "" if locked else self._active_family
            active_slot = "" if locked else self._active_slot
            active_device = "" if locked else self._active_device
            active_half = False if locked else self._active_half
            vram = dict(self._vram) if self._vram else {
                "cuda": env.get("cuda", False),
                "gpu": env.get("gpu", ""),
                "total_mb": env.get("total_mb", 0),
                "free_mb": env.get("free_mb", 0),
            }

        reason_code = "training_locked" if locked else err.get("code", "")
        reason = (
            "训练或重型 GPU 任务进行中，已自动停止智能标注以保障 ≥8GB 显存与训练稳定；当前仅保留纯人工画框与审核。"
            if locked
            else err.get("message", "")
        )
        return {
            "state": state,
            "training_locked": locked,
            "default_model": DEFAULT_SMART_MODEL,
            "active_model": active_model,
            "active_family": active_family,
            "active_slot": active_slot,
            "device": active_device,
            "half": active_half,
            "reason_code": reason_code,
            "reason": reason,
            "show_model_panel": state == "unavailable",
            "sam_dir": str(model_registry.sam_library_dir()),
            "env": env,
            "vram": vram,
            "available_models": models,
            "downloads": downloads,
            "batch_job": batch if batch else None,
        }

    # ---------- 子进程生命周期与 RPC ----------

    def _ensure_worker_locked(self) -> subprocess.Popen:
        """需在持有 _rpc_lock 时调用。"""
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        env_info = self.probe_env()
        if not env_info.get("ok"):
            raise RuntimeError(env_info.get("error") or "未找到可用的 YOLO Python 环境")
        py_path = env_info["python_path"]
        child_env = dict(os.environ)
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["YOLO_OFFLINE"] = "1"
        child_env["YOLO_AUTOINSTALL"] = "false"
        proc = subprocess.Popen(
            [py_path, "-u", str(_WORKER_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
            creationflags=_NO_WINDOW,
        )
        # 等待 ready 握手
        assert proc.stdout is not None
        ready_line = proc.stdout.readline()
        if not ready_line:
            err_text = (proc.stderr.read() if proc.stderr else "")[:400]
            proc.kill()
            raise RuntimeError(f"智能标注子进程启动失败：{err_text or '无输出'}")
        try:
            hello = json.loads(ready_line)
            if hello.get("event") != "ready":
                raise ValueError(ready_line)
        except Exception as exc:
            proc.kill()
            raise RuntimeError(f"智能标注子进程握手失败：{exc}") from None
        self._proc = proc
        self._ensure_watchdog()
        return proc

    def _call_rpc(self, payload: dict) -> dict:
        with self._rpc_lock:
            self._check_not_training_locked()
            proc = self._ensure_worker_locked()
            req_id = uuid.uuid4().hex[:10]
            msg = dict(payload, id=req_id)
            assert proc.stdin is not None and proc.stdout is not None
            try:
                proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
            except OSError as exc:
                self._proc = None
                raise RuntimeError(f"智能标注子进程通信中断：{exc}") from None
            if not line:
                stderr_tail = ""
                if proc.stderr is not None:
                    try:
                        stderr_tail = proc.stderr.read()[:400]
                    except Exception:
                        pass
                self._proc = None
                raise RuntimeError(f"智能标注子进程意外退出：{stderr_tail or '请检查显存或模型文件'}")
            resp = json.loads(line)
            with self._meta_lock:
                self._last_used = time.monotonic()
                if isinstance(resp.get("vram"), dict):
                    self._vram = resp["vram"]
            return resp

    # ---------- 激活、预编码与交互预测 ----------

    def activate(
        self,
        model_name: str = DEFAULT_SMART_MODEL,
        *,
        project_id: str | None = None,
        slot: str = "",
        auto_default: bool = False,
    ) -> dict:
        self._check_not_training_locked()
        name = str(model_name or DEFAULT_SMART_MODEL).strip()
        resolved_slot = str(slot or "").strip()
        if not resolved_slot or resolved_slot == "auto":
            resolved_slot = self.default_slot_for_model(name)
        target = model_registry.resolve_sam_path(name)
        env = self.probe_env()
        if not env.get("ok"):
            with self._meta_lock:
                self._state = "unavailable"
                self._last_error = {"code": "missing_yolo_env", "message": env.get("error") or "未找到 YOLO Python 环境"}
            return self.status(project_id)

        if not target.is_file() or target.stat().st_size == 0:
            sam_dir = model_registry.sam_library_dir()
            if name.lower() == DEFAULT_SMART_MODEL:
                msg = (
                    f"默认模型 {name} 尚未就绪：未在 {sam_dir} 检测到权重文件。"
                    "SAM 3 受 Meta 许可门禁保护，请下载后放入该目录，或在下方直接一键下载/切换 SAM 2.1 / MobileSAM / YOLOE。"
                )
            else:
                msg = f"未在 {sam_dir} 找到模型 {name}，请先点击一键下载或手动拷入该目录。"
            with self._meta_lock:
                self._state = "unavailable"
                self._last_error = {"code": "missing_weight", "message": msg, "target_model": name}
            return self.status(project_id)

        with self._meta_lock:
            self._state = "loading"
            self._last_error = {}

        try:
            resp = self._call_rpc({"cmd": "load", "model_path": str(target), "model_name": name, "slot": resolved_slot})
        except TrainingLockedError:
            raise
        except Exception as exc:
            with self._meta_lock:
                self._state = "unavailable"
                self._last_error = {"code": "load_failed", "message": str(exc), "target_model": name}
            return self.status(project_id)

        if not resp.get("ok"):
            code = str(resp.get("code") or "load_failed")
            if code == "vram_insufficient":
                # OOM 后立即销毁子进程确保 8GB 显存彻底干净
                self.stop_and_release(reason=str(resp.get("error") or "显存不足"), timeout=3.0)
            with self._meta_lock:
                self._state = "unavailable"
                self._active_model = ""
                self._active_slot = ""
                self._last_error = {
                    "code": code,
                    "message": str(resp.get("error") or f"无法启动 {name}"),
                    "target_model": name,
                }
            return self.status(project_id)

        with self._meta_lock:
            self._state = "ready"
            self._active_model = str(resp.get("model_name") or name)
            self._active_family = str(resp.get("family") or model_registry.smart_model_family(name))
            self._active_slot = str(resp.get("slot") or resolved_slot)
            self._active_device = str(resp.get("device") or "")
            self._active_half = bool(resp.get("half"))
            self._last_error = {}
            self._last_used = time.monotonic()
        return self.status(project_id)

    def encode_image(self, project_id: str, item_id: str) -> dict:
        self._check_not_training_locked()
        with self._meta_lock:
            if self._state != "ready" or self._active_slot != "interactive":
                raise ValueError("当前没有已就绪的交互式智能标注模型，请先在可用模型面板中选择并加载")
        folder = self.workbench.store.open_dir(project_id)
        image_path = PartsStore(folder).image_path(item_id)
        resp = self._call_rpc({"cmd": "encode", "image_path": str(image_path)})
        if not resp.get("ok"):
            if resp.get("code") == "vram_insufficient":
                self.stop_and_release(reason=str(resp.get("error") or "显存不足"), timeout=3.0)
                with self._meta_lock:
                    self._state = "unavailable"
                    self._last_error = {"code": "vram_insufficient", "message": str(resp.get("error") or "显存不足")}
            raise RuntimeError(str(resp.get("error") or "图像特征编码失败"))
        return resp

    def predict_interactive(self, project_id: str, item_id: str, params: dict) -> dict:
        self._check_not_training_locked()
        with self._meta_lock:
            if self._state != "ready" or self._active_slot != "interactive":
                raise ValueError("当前没有已就绪的交互式智能标注模型，请先选择并加载 SAM 3 / SAM 2.1 / MobileSAM")
        folder = self.workbench.store.open_dir(project_id)
        image_path = PartsStore(folder).image_path(item_id)
        points = params.get("points") or []
        labels = params.get("labels") or []
        bbox = params.get("bbox")
        fit_mode = "hbb" if str(params.get("fit_mode") or "obb") == "hbb" else "obb"
        resp = self._call_rpc(
            {
                "cmd": "predict",
                "image_path": str(image_path),
                "points": points,
                "labels": labels,
                "bbox": bbox,
                "fit_mode": fit_mode,
            }
        )
        if not resp.get("ok"):
            if resp.get("code") == "vram_insufficient":
                self.stop_and_release(reason=str(resp.get("error") or "显存不足"), timeout=3.0)
                with self._meta_lock:
                    self._state = "unavailable"
                    self._last_error = {"code": "vram_insufficient", "message": str(resp.get("error") or "显存不足")}
            raise RuntimeError(str(resp.get("error") or "智能交互分割失败"))
        return resp

    # ---------- 按需模型下载 ----------

    def start_download(self, model_name: str, *, custom_url: str = "") -> dict:
        self._check_not_training_locked()
        target = model_registry.resolve_sam_path(model_name)
        catalog_entry = next((m for m in model_registry.SMART_MODEL_CATALOG if m["name"].lower() == target.name.lower()), None)
        url = str(custom_url or "").strip()
        if not url:
            if catalog_entry and catalog_entry.get("downloadable") and catalog_entry.get("download_url"):
                url = str(catalog_entry["download_url"])
            elif catalog_entry and catalog_entry.get("gated"):
                raise ValueError(
                    f"{target.name} 为 Meta 许可门禁模型，无法匿名直连下载；请手动下载后放入 {target.parent}，或在自定义下载地址中填入内网镜像直链。"
                )
            else:
                raise ValueError(f"模型 {target.name} 没有内置下载地址，请手动放入 {target.parent} 或提供 HTTP/HTTPS 直链。")

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("下载地址必须是合法的 http:// 或 https:// 链接")

        key = target.name.lower()
        with self._meta_lock:
            existing = self._downloads.get(key)
            if existing and existing.get("status") == "downloading":
                return dict(existing)
            stop_ev = threading.Event()
            record = {
                "model": target.name,
                "status": "downloading",
                "downloaded": 0,
                "total": 0,
                "percent": 0,
                "error": "",
                "started_at": time.time(),
            }
            self._downloads[key] = record
            self._download_stops[key] = stop_ev

        def run_dl() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = target.parent / f".{target.name}.{uuid.uuid4().hex[:8]}.part"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "lstm-workbench-smart-label/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    total = int(resp.headers.get("Content-Length") or 0)
                    with self._meta_lock:
                        record["total"] = total
                    written = 0
                    with tmp_path.open("wb") as out:
                        while True:
                            if stop_ev.is_set():
                                raise InterruptedError("下载已取消")
                            chunk = resp.read(256 * 1024)
                            if not chunk:
                                break
                            out.write(chunk)
                            written += len(chunk)
                            with self._meta_lock:
                                record["downloaded"] = written
                                record["percent"] = min(99, int(written * 100 / total)) if total > 0 else 0
                os.replace(tmp_path, target)
                with self._meta_lock:
                    record["status"] = "done"
                    record["percent"] = 100
                    record["finished_at"] = time.time()
            except Exception as exc:
                tmp_path.unlink(missing_ok=True)
                with self._meta_lock:
                    record["status"] = "stopped" if isinstance(exc, InterruptedError) else "failed"
                    record["error"] = str(exc)
                    record["finished_at"] = time.time()

        threading.Thread(target=run_dl, name=f"sam-download-{target.name}", daemon=True).start()
        return dict(record)

    # ---------- 三合一批量自动标注（零样本 + 分块连续帧追踪） ----------

    def _begin_batch_job(self, project_id: str, engine: str, total: int) -> tuple[dict, threading.Event]:
        self._check_not_training_locked()
        with self._meta_lock:
            cur = self._batch_jobs.get(project_id)
            if cur and cur.get("status") in {"queued", "running"}:
                raise RuntimeError("当前工程已有智能批量自动标注任务正在运行，请先等待或停止")
            stop_ev = threading.Event()
            job = {
                "id": f"smart-{uuid.uuid4().hex[:10]}",
                "engine": engine,
                "status": "running",
                "done": 0,
                "total": max(int(total), 1),
                "applied": 0,
                "skipped": 0,
                "error": "",
                "started_at": time.time(),
                "finished_at": None,
            }
            self._batch_jobs[project_id] = job
            self._batch_stops[project_id] = stop_ev
            return job, stop_ev

    def stop_batch(self, project_id: str) -> dict:
        with self._meta_lock:
            ev = self._batch_stops.get(project_id)
            if ev is not None:
                ev.set()
            job = self._batch_jobs.get(project_id)
            return dict(job) if job else {"status": "idle"}

    def start_batch_zeroshot(self, project_id: str, params: dict) -> dict:
        """零样本批量自动标注（SAM 3 语义文本/视觉示例 或 YOLOE-26-seg 视觉示例/文本）。"""
        self._check_not_training_locked()
        engine = str(params.get("engine") or "yoloe_vp").strip()
        if engine not in {"sam3_semantic", "yoloe_vp", "yoloe_text"}:
            raise ValueError("零样本引擎仅支持 sam3_semantic、yoloe_vp、yoloe_text")

        model_name = str(params.get("model") or ("sam3.pt" if engine == "sam3_semantic" else "yoloe-26s-seg.pt")).strip()
        target = model_registry.resolve_sam_path(model_name)
        if not target.is_file():
            raise FileNotFoundError(f"未找到模型 {model_name}，请先在可用模型面板中下载或放入 models/sam/")

        env = self.probe_env()
        if engine == "sam3_semantic" and not env.get("clip"):
            raise ValueError(f"SAM 3 文本/语义模式缺少 clip 依赖。请在 YOLO 环境执行：{env.get('clip_install_cmd')}")
        if engine == "yoloe_text" and not env.get("mobileclip"):
            raise ValueError(f"YOLOE 文本模式缺少 mobileclip 依赖（视觉示例框模式无需安装）。安装命令：{env.get('mobileclip_install_cmd')}")

        folder = self.workbench.store.open_dir(project_id)
        store = PartsStore(folder)
        snap = store.snapshot()
        classes = list(snap.get("classes") or [])
        if not classes:
            raise ValueError("请先设置至少一个零件类别再启动批量自动标注")

        raw_ids = params.get("items")
        all_items = snap.get("items") or []
        if isinstance(raw_ids, list) and raw_ids:
            id_set = {str(x) for x in raw_ids}
            selected = [it for it in all_items if it["id"] in id_set]
        else:
            selected = [it for it in all_items if it.get("status") not in {"reviewed", "negative"}]
        if not selected:
            raise ValueError("没有可处理的未审核素材")

        conf = float(params.get("conf", 0.25))
        if not (0.01 <= conf <= 1.0):
            raise ValueError("置信度阈值必须在 0.01 到 1.0 之间")
        fit_mode = "hbb" if str(params.get("fit_mode") or "obb") == "hbb" else "obb"
        replace = bool(params.get("replace", True))
        prompts = params.get("prompts") or []
        exemplar = params.get("visual_exemplar")
        resolved_exemplar = None
        if isinstance(exemplar, dict) and exemplar.get("item_id") and exemplar.get("bbox"):
            ref_path = store.image_path(str(exemplar["item_id"]))
            resolved_exemplar = {
                "image_path": str(ref_path),
                "bbox": [float(v) for v in exemplar["bbox"]],
                "class_id": int(exemplar.get("class_id", 0)),
            }
        if engine == "yoloe_vp" and not resolved_exemplar:
            raise ValueError("YOLOE 视觉示例模式需要在参考图上框选一个视觉示例目标")
        if engine in {"sam3_semantic", "yoloe_text"} and not resolved_exemplar and not prompts:
            raise ValueError("请至少填写一个类别的英文提示词，或提供视觉示例框")

        job, stop_ev = self._begin_batch_job(project_id, engine, len(selected))

        def run_batch() -> None:
            try:
                slot = "semantic" if engine == "sam3_semantic" else engine
                load_resp = self._call_rpc({"cmd": "load", "model_path": str(target), "model_name": model_name, "slot": slot, "conf": conf})
                if not load_resp.get("ok"):
                    raise RuntimeError(str(load_resp.get("error") or "加载零样本模型失败"))
                with self._meta_lock:
                    self._state = "ready"
                    self._active_model = model_name
                    self._active_family = str(load_resp.get("family") or model_registry.smart_model_family(model_name))
                    self._active_slot = slot

                for item in selected:
                    if stop_ev.is_set() or self.workbench.is_training_locked():
                        with self._meta_lock:
                            job["status"] = "stopped"
                            job["finished_at"] = time.time()
                        return
                    if item.get("status") in {"reviewed", "negative"}:
                        with self._meta_lock:
                            job["skipped"] += 1
                            job["done"] += 1
                        continue
                    img_path = store.image_path(item["id"])
                    resp = self._call_rpc(
                        {
                            "cmd": "predict_zeroshot_one",
                            "engine": engine,
                            "image_path": str(img_path),
                            "prompts": prompts,
                            "visual_exemplar": resolved_exemplar,
                            "conf": conf,
                            "fit_mode": fit_mode,
                        }
                    )
                    if not resp.get("ok"):
                        if resp.get("code") == "vram_insufficient":
                            raise RuntimeError(str(resp.get("error") or "显存不足"))
                        with self._meta_lock:
                            job["skipped"] += 1
                            job["done"] += 1
                        continue
                    pred_boxes = resp.get("boxes") or []
                    with self.workbench.data_lock(project_id):
                        res = store.apply_auto_boxes(
                            item["id"],
                            pred_boxes,
                            expected_classes=classes,
                            expected_revision=item["revision"],
                            replace=replace,
                        )
                    with self._meta_lock:
                        if res.get("applied"):
                            job["applied"] += 1
                        else:
                            job["skipped"] += 1
                        job["done"] += 1
                with self._meta_lock:
                    job["status"] = "done"
                    job["finished_at"] = time.time()
            except Exception as exc:
                with self._meta_lock:
                    job["status"] = "failed"
                    job["error"] = str(exc)
                    job["finished_at"] = time.time()

        t = threading.Thread(target=run_batch, name=f"smart-zeroshot-{project_id}", daemon=True)
        with self._meta_lock:
            self._batch_threads[project_id] = t
        t.start()
        return dict(job)

    def start_batch_track(self, project_id: str, params: dict) -> dict:
        """同来源组连续抽帧分块追踪传播（每块 ≤40 帧，适配 ≥8GB 显存）。"""
        self._check_not_training_locked()
        seed_id = str(params.get("seed_item_id") or "").strip()
        if not seed_id:
            raise ValueError("需要指定已标注的首帧素材编号")

        # 默认优先使用当前激活/已就绪的 SAM 3 或 SAM 2.1
        model_name = str(params.get("model") or "").strip()
        if not model_name:
            local_names = [m["name"] for m in model_registry.list_sam_models() if m["family"] in {"sam3", "sam2"}]
            for pref in ("sam3.pt", "sam2.1_b.pt", "sam2.1_s.pt", "sam2.1_t.pt"):
                if pref in local_names:
                    model_name = pref
                    break
            if not model_name and local_names:
                model_name = local_names[0]
        if not model_name:
            raise FileNotFoundError("视频连续帧追踪需要本地已就绪 SAM 3 或 SAM 2.1 权重，请先在可用模型面板中下载 sam2.1_b.pt 或 sam2.1_t.pt")
        target = model_registry.resolve_sam_path(model_name)
        if not target.is_file():
            raise FileNotFoundError(f"未找到视频追踪模型 {model_name}")
        if model_registry.smart_model_family(model_name) not in {"sam3", "sam2"}:
            raise ValueError("连续帧视频追踪仅支持 SAM 3 或 SAM 2.1 系列模型")

        folder = self.workbench.store.open_dir(project_id)
        store = PartsStore(folder)
        snap = store.snapshot()
        classes = list(snap.get("classes") or [])
        items = snap.get("items") or []
        seed_item = next((it for it in items if it["id"] == seed_id), None)
        if seed_item is None:
            raise FileNotFoundError("找不到首帧素材")
        if not seed_item.get("boxes"):
            raise ValueError("首帧还没有标注框，请先标好目标零件再向后追踪")

        group = seed_item["source_group"]
        same_group = [it for it in items if it["source_group"] == group]
        same_group.sort(key=lambda x: (x.get("frame_index") is None, x.get("frame_index") if x.get("frame_index") is not None else 0, x["name"]))
        seed_pos = next((i for i, it in enumerate(same_group) if it["id"] == seed_id), -1)
        max_follow = max(1, min(int(params.get("max_frames", 120)), 600))
        follow = same_group[seed_pos + 1 : seed_pos + 1 + max_follow]
        if not follow:
            raise ValueError("同一来源组在当前帧之后没有后续帧可追踪")

        fit_mode = "hbb" if str(params.get("fit_mode") or "obb") == "hbb" else "obb"
        replace = bool(params.get("replace", True))
        job, stop_ev = self._begin_batch_job(project_id, "video_track", len(follow))

        def run_track() -> None:
            try:
                load_resp = self._call_rpc({"cmd": "load", "model_path": str(target), "model_name": model_name, "slot": "video"})
                if not load_resp.get("ok"):
                    raise RuntimeError(str(load_resp.get("error") or "加载视频追踪模型失败"))
                with self._meta_lock:
                    self._state = "ready"
                    self._active_model = model_name
                    self._active_family = str(load_resp.get("family") or model_registry.smart_model_family(model_name))
                    self._active_slot = "video"

                anchor_item = seed_item
                anchor_boxes = list(seed_item["boxes"])
                step = VIDEO_TRACK_CHUNK_SIZE - 1  # 每块含 1 张锚帧 + 最多 39 张目标帧（总计 ≤40 帧）
                for start in range(0, len(follow), step):
                    if stop_ev.is_set() or self.workbench.is_training_locked():
                        with self._meta_lock:
                            job["status"] = "stopped"
                            job["finished_at"] = time.time()
                        return
                    chunk_targets = follow[start : start + step]
                    chunk_items = [anchor_item] + chunk_targets
                    frames_payload = [{"id": it["id"], "path": str(store.image_path(it["id"]))} for it in chunk_items]
                    resp = self._call_rpc(
                        {
                            "cmd": "track_chunk",
                            "frames": frames_payload,
                            "seed_boxes": anchor_boxes,
                            "fit_mode": fit_mode,
                        }
                    )
                    if not resp.get("ok"):
                        raise RuntimeError(str(resp.get("error") or "连续帧追踪失败"))
                    tracked_map: dict[str, list[dict]] = resp.get("tracked") or {}
                    last_valid_item = None
                    last_valid_boxes = None
                    for target_item in chunk_targets:
                        boxes_for_frame = tracked_map.get(target_item["id"]) or []
                        if boxes_for_frame:
                            last_valid_item = target_item
                            last_valid_boxes = boxes_for_frame
                        with self.workbench.data_lock(project_id):
                            res = store.apply_auto_boxes(
                                target_item["id"],
                                boxes_for_frame,
                                expected_classes=classes,
                                expected_revision=target_item["revision"],
                                replace=replace,
                            )
                        with self._meta_lock:
                            if res.get("applied"):
                                job["applied"] += 1
                            else:
                                job["skipped"] += 1
                            job["done"] += 1
                    if last_valid_item is not None and last_valid_boxes:
                        anchor_item = last_valid_item
                        anchor_boxes = last_valid_boxes
                    else:
                        break
                with self._meta_lock:
                    job["status"] = "done"
                    job["finished_at"] = time.time()
            except Exception as exc:
                with self._meta_lock:
                    job["status"] = "failed"
                    job["error"] = str(exc)
                    job["finished_at"] = time.time()

        t = threading.Thread(target=run_track, name=f"smart-track-{project_id}", daemon=True)
        with self._meta_lock:
            self._batch_threads[project_id] = t
        t.start()
        return dict(job)
