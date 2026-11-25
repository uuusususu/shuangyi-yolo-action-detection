"""动作时序（LSTM）路由：工程概览、模型探测、划分确认、特征抽取、训练与视频复盘检查。"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from urllib.parse import quote

from .. import check_clip, feature_store, model_registry, service, settings as settings_module
from ..dataset import check_split
from ..http_utils import _valid_name, is_loopback_client, read_settings
from ..service import annotation_blockers, annotation_warnings, write_progress

JOB_ERRORS = (ValueError, FileNotFoundError, KeyError, RuntimeError, OSError)


class ActionRoutesMixin:
    def _run_summaries(self, project_id: str, folder: Path) -> tuple[list[str], list[dict]]:
        runs = self.workbench.store.list_dir(project_id, "runs")
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
        workbench = self.workbench
        settings = read_settings(workbench.store, project_id)
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
            "preparation": workbench.preparation.status(project_id),
        }

    def _model(self, project_id: str) -> None:
        """选定 YOLO 和手部模型，检查能不能加载，记下指纹。"""
        workbench = self.workbench
        folder = self._open(project_id)
        if folder is None:
            return
        if workbench.lock_for(project_id).locked():
            self._send(409, {"error": "正在跑任务，先停止再改模型设置"})
            return
        patch = self._read_json()
        try:
            merged = settings_module.clean(patch, workbench.store.load_settings(project_id))
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
        merged["yolo_path"] = model_registry.portable_yolo_path(
            model_registry.resolve_yolo_path(merged["yolo_path"])
        )
        with workbench.data_lock(project_id):
            latest = settings_module.clean(patch, workbench.store.load_settings(project_id))
            for key in ("yolo_path", "yolo_sha256", "yolo_classes", "hands_sha256"):
                if key in merged:
                    latest[key] = merged[key]
            workbench.store.save_settings(project_id, latest)
        workbench.preparation.request(project_id)
        self._send(200, {"settings": latest, "yolo": yolo, "hands": hands})

    def _split(self, project_id: str) -> None:
        folder = self._open(project_id)
        if folder is None:
            return
        body = self._read_json()
        with self.workbench.data_lock(project_id):
            self._split_locked(project_id, body)

    def _split_locked(self, project_id: str, body: dict) -> None:
        workbench = self.workbench
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
            warning = ""
            try:
                workbench.preparation.request(project_id)
            except (OSError, ValueError) as exc:
                warning = str(exc)
            self._send(200, {"split": service.split_overview(workbench.store, project_id, current), "preparation_error": warning})
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
            prepared = service.prepare_clips(payload, current)
            checked = check_split(prepared.clips, prepared.window, current["split_train"], current["split_val"])
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
        if not _valid_name(run_name):
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
            self._drain_body()
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
                read_settings(self.workbench.store, project_id),
                self.workbench.store.load_annotations(project_id),
                on_progress=publish,
                should_stop=stop.is_set,
            )
        except (check_clip.Stopped, feature_store.Stopped) as exc:
            write_progress(folder, {"status": "stopped", "error": str(exc), "history": []})
            self._send(200, {"stopped": True, "error": str(exc)})
            return
        except Exception as exc:
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
                read_settings(self.workbench.store, project_id),
                on_progress=publish,
                should_stop=stop.is_set,
            )
        except check_clip.Stopped as exc:
            write_progress(folder, {"status": "stopped", "error": str(exc), "history": []})
            self._send(200, {"stopped": True, "error": str(exc)})
            return
        except Exception as exc:
            self._fail_job(folder, exc)
            return
        finally:
            self._end_job()
        write_progress(folder, {"status": "check-done", "history": [], "check": {"done": 1, "total": 1}})
        self._send(200, {"overlay": True})

    def _check_media(self, run: Path) -> None:
        name = self._upload_name()
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
                read_settings(self.workbench.store, project_id),
                self.workbench.store.load_annotations(project_id),
                on_progress=publish,
                should_stop=stop.is_set,
            )
        except (check_clip.Stopped, feature_store.Stopped) as exc:
            write_progress(folder, {"status": "stopped", "error": str(exc), "history": []})
            self._send(200, {"stopped": True, "error": str(exc)})
            return
        except Exception as exc:
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
                read_settings(self.workbench.store, project_id),
                self.workbench.store.load_annotations(project_id),
                on_progress=publish,
                should_stop=stop.is_set,
            )
        except check_clip.Stopped as exc:
            write_progress(folder, {"status": "stopped", "error": str(exc), "history": []})
            self._send(200, {"stopped": True, "error": str(exc)})
            return
        except Exception as exc:
            self._fail_job(folder, exc)
            return
        finally:
            self._end_job()
        write_progress(folder, {"status": "check-done", "history": [], "check": {"done": 1, "total": 1}})
        self._send(200, {"check": {"ready": True, "segments": len(result.get("segments") or [])}})

    def _reveal_feature(self, project_id: str) -> None:
        """本机访问时在资源管理器里选中 npy；内网远端电脑访问时返回直链下载。"""
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
        client_ip = str(self.client_address[0]) if self.client_address else ""
        if not is_loopback_client(client_ip):
            download_url = f"/api/projects/{quote(project_id)}/features/{quote(npy.name)}"
            self._send(200, {"remote": True, "download": download_url})
            return
        subprocess.Popen(["explorer", "/select,", str(npy)])
        self._send(200, {"opened": str(npy)})

    def _features(self, project_id: str) -> None:
        from .. import app as app_module

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
            annotations = self.workbench.store.load_annotations(project_id)
            chosen = clip_ids if clip_ids is not None else [str(clip["id"]) for clip in annotations.get("clips") or []]
            _arrays, report = app_module._extract_tracked(
                folder, annotations, read_settings(self.workbench.store, project_id), chosen, stop.is_set
            )
        except feature_store.Stopped as exc:
            write_progress(folder, {"status": "stopped", "error": str(exc), "history": []})
            self._send(200, {"stopped": True, "error": str(exc)})
            return
        except Exception as exc:
            self._fail_job(folder, exc)
            return
        finally:
            self._end_job()
        write_progress(folder, {"status": "features-done", "history": [], "features": report})
        self._send(200, {"features": report})

    def _train(self, project_id: str) -> None:
        from .. import app as app_module

        workbench = self.workbench
        folder = self._open(project_id)
        if folder is None:
            return
        body = self._read_json()
        display_name = ""
        if isinstance(body, dict) and "display_name" in body:
            try:
                display_name = settings_module.display_name(body.pop("display_name"))
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
        stop = self._start_job(project_id)
        if stop is None:
            return
        try:
            if body:
                with workbench.data_lock(project_id):
                    merged = settings_module.clean(body, workbench.store.load_settings(project_id))
                    workbench.store.save_settings(project_id, merged)
        except ValueError as exc:
            self._end_job()
            self._send(400, {"error": str(exc)})
            return
        try:
            annotations = workbench.store.load_annotations(project_id)
            settings = read_settings(workbench.store, project_id)
            chosen = service.clips_for_training(annotations, settings)
            clip_ids = [str(clip["id"]) for clip in chosen.get("clips") or [] if clip.get("id")]
            if not clip_ids:
                raise ValueError("还没有勾选参加训练的片子")
            completion_blockers = service.annotation_completion_blockers(annotations, settings)
            if completion_blockers:
                raise ValueError(completion_blockers[0])
            features, _report = app_module._extract_tracked(folder, annotations, settings, clip_ids, stop.is_set)
            result = app_module.start_training(
                workbench.store, project_id, features, should_stop=stop.is_set, display_name=display_name,
                annotations=annotations, settings=settings,
            )
        except feature_store.Stopped as exc:
            write_progress(folder, {"status": "stopped", "error": str(exc), "history": []})
            self._send(200, {"stopped": True, "error": str(exc)})
            return
        except Exception as exc:
            self._fail_job(folder, exc)
            return
        finally:
            self._end_job()
        self._send(200, result)
