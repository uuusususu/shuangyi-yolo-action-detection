"""零件模型（YOLO）路由：训练、续训、官方图表文件、ONNX 导出/启用、独立验证/预标注与 ZIP 上传下载。"""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import threading
from urllib.parse import urlparse
import uuid

from .. import dataset_transfer, model_registry, yolo_jobs, yolo_validation
from ..http_utils import _valid_name, read_settings
from ..parts_store import PartsStore


class YoloRoutesMixin:
    def _yolo_get(self, parts: list[str]) -> None:
        folder = self._open(parts[2])
        if folder is None:
            return
        if len(parts) == 4:
            self._send(200, {
                "dataset": yolo_jobs.dataset_info(folder),
                "downloads": dataset_transfer.list_downloads(folder),
                "versions": PartsStore(folder).snapshot()["versions"],
                "runs": yolo_jobs.list_runs(folder),
                "active": (yolo_jobs.active_run(folder) or Path()).name or "",
                "defaults": {"model": yolo_jobs.DEFAULT_BASE, "bases": yolo_jobs.list_base_models()},
            })
            return
        if len(parts) == 5 and parts[4] == "downloads":
            self._send(200, {"archives": dataset_transfer.list_downloads(folder)})
            return
        if len(parts) == 6 and parts[4] == "downloads":
            try:
                archive = dataset_transfer.download_archive_path(folder, parts[5])
            except (ValueError, FileNotFoundError) as exc:
                self._parts_error(exc)
                return
            self._stream_file(archive, cache=False, attachment_name=archive.name)
            return
        if len(parts) == 5 and parts[4] == "evaluations":
            self._send(200, {"evaluations": yolo_validation.list_validations(folder)})
            return
        if len(parts) >= 6 and parts[4] == "evaluations":
            try:
                if not _valid_name(parts[5]):
                    raise ValueError("验证记录编号不合法")
                if len(parts) == 6:
                    self._send(200, yolo_validation.validation_detail(folder, parts[5]))
                    return
                if len(parts) == 8 and parts[6] == "images":
                    if not _valid_name(parts[7]):
                        raise ValueError("素材编号不合法")
                    self._stream_file(yolo_validation.image_file(folder, parts[5], parts[7]), cache=False)
                    return
            except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
                self._parts_error(exc)
                return
        if len(parts) >= 6 and parts[4] == "runs":
            if not _valid_name(parts[5]):
                self._send(400, {"error": "轮次名不合法"})
                return
            run = yolo_jobs.runs_dir(folder) / parts[5]
            if not (run / "job.json").is_file():
                self._send(404, {"error": "没有这一轮"})
                return
            if len(parts) == 6:
                self._send(200, yolo_jobs.run_detail(run), cache=False)
                return
            if len(parts) == 7 and parts[6] == "log":
                query = urlparse(self.path).query or ""
                match = re.search(r"(?:^|&)after=(\d+)", query)
                self._send(200, yolo_jobs.read_log(run, int(match.group(1)) if match else 0), cache=False)
                return
            if len(parts) >= 8 and parts[6] == "file":
                self._send_bounded_file(run, "/".join(parts[7:]))
                return
        self._send(404, {"error": "没有这个接口"})

    def _yolo_dataset_upload(self, project_id: str) -> None:
        """浏览器/内网网线直连上传 YOLO 数据集 .zip 或 .ndjson，保存 ZIP 并导入工程。"""
        workbench = self.workbench
        raw_name = Path(self._upload_name() or "dataset.zip").name
        try:
            folder = workbench.store.open_dir(project_id)
        except (FileNotFoundError, ValueError):
            self._drain_body()
            self._send(404, {"error": "工程不存在"})
            return
        if workbench.yolo_training():
            self._drain_body()
            self._send(409, {"error": "零件模型正在训练，请先等待或停止后再上传数据集"})
            return
        suffix = Path(raw_name).suffix.lower()
        if suffix not in {".zip", ".ndjson"}:
            self._drain_body()
            self._send(400, {"error": "请上传 .zip 数据集压缩包或 .ndjson 清单文件"})
            return
        sync_parts = (self.headers.get("X-Sync-Parts") or "1") != "0"
        staging_dir = folder / f".yolo-upload-{uuid.uuid4().hex[:10]}"
        staging_dir.mkdir(parents=True, exist_ok=True)
        slug = dataset_transfer._safe_slug(Path(raw_name).stem, "dataset")
        incoming = staging_dir / f"{slug}{suffix}"
        try:
            self._receive_to(incoming)
            if suffix == ".zip":
                zip_name = f"{slug}.zip"
                zip_dest = dataset_transfer.downloads_dir(folder) / zip_name
                zip_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(incoming, zip_dest)
                extracted = staging_dir / "extracted"
                dataset_transfer._safe_extract_zip(zip_dest, extracted)
                yaml_file = dataset_transfer._find_yaml_in_extracted(extracted)
                if yaml_file is None:
                    raise ValueError("ZIP 压缩包内未找到 data.yaml，若只是普通图片压缩包请在零件素材区域上传")
                names = yolo_jobs._read_yaml_names(yaml_file.read_text(encoding="utf-8", errors="replace"))
                with workbench.data_lock(project_id):
                    ds_info = yolo_jobs.import_dataset(folder, str(yaml_file))
                    if sync_parts and names:
                        dataset_transfer.sync_yolo_dir_to_parts_store(folder, yaml_file.parent, names, slug)
            else:
                dataset_out = staging_dir / slug
                converted = dataset_transfer.convert_ndjson_file_to_yolo_dir(incoming, dataset_out)
                zip_name = f"{converted['name']}.zip"
                zip_dest = dataset_transfer.downloads_dir(folder) / zip_name
                dataset_transfer._pack_directory_to_zip(dataset_out, zip_dest)
                with workbench.data_lock(project_id):
                    ds_info = yolo_jobs.import_dataset(folder, str(converted["data_yaml"]))
                    if sync_parts and converted["names"]:
                        dataset_transfer.sync_yolo_dir_to_parts_store(folder, dataset_out, converted["names"], converted["name"])
            self._send(201, {"dataset": ds_info, "zip_file": zip_name, "downloads": dataset_transfer.list_downloads(folder)})
        except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
            self._send(400, {"error": str(exc)})
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    def _yolo_post(self, parts: list[str]) -> None:
        workbench = self.workbench
        project_id = parts[2]
        if len(parts) == 6 and parts[4:] == ["dataset", "upload"]:
            self._yolo_dataset_upload(project_id)
            return
        folder = self._open(project_id)
        if folder is None:
            return
        body = self._read_json()
        if len(parts) == 5 and parts[4] in {"validate", "preannotate"}:
            stop = self._start_job(project_id, nonblocking=True)
            if stop is None:
                return
            try:
                if workbench.yolo_training():
                    self._send(409, {"error": "零件模型正在训练，请先等待或停止"})
                    return
                params = dict(body)
                params["mode"] = "preannotate" if parts[4] == "preannotate" else "validate"
                with workbench.data_lock(project_id):
                    record = yolo_validation.prepare_validation(folder, params)
                identifier = record["id"]

                def execute() -> None:
                    try:
                        yolo_validation.execute_validation(folder, identifier, stop)
                    except Exception as exc:
                        try:
                            yolo_validation.fail_validation(folder, identifier, str(exc), record)
                        except (OSError, ValueError):
                            pass
                    finally:
                        workbench.end_job(project_id)
                        with workbench._guards:
                            workbench._async_jobs.pop(identifier, None)

                thread = threading.Thread(target=execute, name=f"yolo-evaluation-{identifier}", daemon=True)
                with workbench._guards:
                    workbench._async_jobs[identifier] = thread
                try:
                    thread.start()
                except RuntimeError:
                    with workbench._guards:
                        workbench._async_jobs.pop(identifier, None)
                    raise
                self._job_project = None
                self._send(202, record)
            except (ValueError, FileNotFoundError, KeyError, OSError, RuntimeError) as exc:
                self._parts_error(exc)
            finally:
                self._end_job()
            return
        if len(parts) == 7 and parts[4] == "evaluations" and parts[6] == "feedback":
            if not _valid_name(parts[5]):
                self._send(400, {"error": "验证记录编号不合法"})
                return
            try:
                items = body.get("items")
                if not isinstance(items, list) or not items or any(not isinstance(item, str) or not _valid_name(item) for item in items):
                    raise ValueError("请选择有效的错例素材")
                with workbench.data_lock(project_id):
                    result = yolo_validation.feedback(folder, parts[5], items)
                self._send(200, result)
            except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
                self._parts_error(exc)
            return
        if len(parts) == 7 and parts[4] == "runs" and parts[6] == "resume":
            if not _valid_name(parts[5]):
                self._send(400, {"error": "轮次名不合法"})
                return
            if workbench.yolo_training() or workbench.any_job_running() or not workbench._gpu_lock.acquire(blocking=False):
                self._send(409, {"error": "已有任务正在使用显卡，请先等待或停止"})
                return
            try:
                if workbench.any_job_running():
                    self._send(409, {"error": "已有任务正在运行，请先等待或停止"})
                    return
                workbench.preempt_smart_label("零件模型续训即将启动，已停止智能标注释放显存")
                job = yolo_jobs.resume_training(folder, parts[5], body)
                self._send(201, {"run": job})
            except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
                self._parts_error(exc)
            finally:
                workbench._gpu_lock.release()
            return
        if len(parts) == 5 and parts[4] == "dataset":
            if self._start_job(project_id, nonblocking=True) is None:
                return
            try:
                if workbench.yolo_training():
                    self._send(409, {"error": "正在跑任务，先停止再导入数据"})
                    return
                with workbench.data_lock(project_id):
                    info = yolo_jobs.import_dataset(folder, str(body.get("source") or ""))
            except (ValueError, FileNotFoundError, OSError) as exc:
                self._send(400, {"error": str(exc)})
                return
            finally:
                self._end_job()
            self._send(200, {"dataset": info})
            return
        if len(parts) == 5 and parts[4] == "train":
            busy = workbench.yolo_training()
            if busy:
                self._send(409, {"error": f"零件模型已经在训练（{busy}）"})
                return
            if workbench.any_job_running():
                self._send(409, {"error": "抽特征或动作训练正在跑，显卡只有一张，先等它结束"})
                return
            try:
                if not workbench._gpu_lock.acquire(blocking=False):
                    self._send(409, {"error": "特征准备或训练正在使用显卡，请先暂停或等待"})
                    return
                try:
                    if workbench.any_job_running():
                        self._send(409, {"error": "已有任务正在运行，请先暂停或等待"})
                        return
                    workbench.preempt_smart_label("零件模型训练即将启动，已停止智能标注释放显存")
                    job = yolo_jobs.start_training(folder, body)
                finally:
                    workbench._gpu_lock.release()
            except (yolo_jobs.YoloEnvMissing, ValueError, FileNotFoundError, OSError) as exc:
                self._send(400, {"error": str(exc)})
                return
            self._send(201, {"run": job})
            return
        if len(parts) == 7 and parts[4] == "runs" and parts[6] in {"export", "use"}:
            if not _valid_name(parts[5]):
                self._send(400, {"error": "轮次名不合法"})
                return
            run = yolo_jobs.runs_dir(folder) / parts[5]
            if not (run / "job.json").is_file():
                self._send(404, {"error": "没有这一轮"})
                return
            if parts[6] == "export":
                if workbench.yolo_training():
                    self._send(409, {"error": "零件模型正在训练，导出会抢显卡和内存，等训练结束"})
                    return
                if self._start_job(project_id) is None:
                    return
                try:
                    info = yolo_jobs.export_onnx(project_id, folder, run, body.get("imgsz"))
                except (FileNotFoundError, RuntimeError, yolo_jobs.YoloEnvMissing, subprocess.TimeoutExpired) as exc:
                    self._send(400, {"error": str(exc)})
                    return
                finally:
                    self._end_job()
                self._send(200, {"export": info})
                return
            exported = yolo_jobs.run_summary(run).get("exported")
            if not exported:
                self._send(400, {"error": "这一轮还没导出 ONNX"})
                return
            if workbench.lock_for(project_id).locked():
                self._send(409, {"error": "正在跑任务，先停止再换零件模型"})
                return
            current = read_settings(workbench.store, project_id)
            probe = model_registry.probe_yolo(exported["file"], current["yolo_conf_threshold"])
            if not probe.get("ok"):
                self._send(400, {"error": probe.get("error") or "导出的模型加载失败"})
                return
            with workbench.data_lock(project_id):
                latest = read_settings(workbench.store, project_id)
                latest["yolo_path"] = exported["file"]
                latest["yolo_sha256"] = probe["sha256"]
                latest["yolo_classes"] = probe["classes"]
                workbench.store.save_settings(project_id, latest)
            workbench.preparation.request(project_id)
            self._send(200, {"settings": latest, "yolo": probe})
            return
        self._send(404, {"error": "没有这个接口"})

    def _library_with_usage(self) -> dict:
        """模型库清单，附上每个 YOLO 文件被哪些工程选用、来自哪一轮。"""
        info = model_registry.library_info()
        used: dict[str, list[str]] = {}
        for project_id in self.workbench.store.list_projects():
            try:
                chosen = str(self.workbench.store.load_settings(project_id).get("yolo_path") or "")
            except (OSError, ValueError):
                continue
            if chosen:
                used.setdefault(Path(chosen).name, []).append(project_id)
        for item in info["yolo"]["files"]:
            item["used_by"] = used.get(item["name"], [])
            meta = model_registry.YOLO_LIBRARY / Path(item["name"]).with_suffix(".json").name
            try:
                item["origin"] = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                item["origin"] = None
        return info
