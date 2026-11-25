"""零件素材工作区路由：原生 OBB 标注、审核状态、质检、固定数据版本发布与跨机分块上传。"""
from __future__ import annotations

from pathlib import Path
import shutil
from urllib.parse import unquote
import uuid

from .. import dataset_transfer, yolo_jobs
from ..http_utils import _valid_name
from ..parts_store import PartsStore


class PartsRoutesMixin:
    def _parts_error(self, exc: Exception) -> None:
        if isinstance(exc, FileNotFoundError):
            code = 404
        elif isinstance(exc, ValueError) and "修订冲突" in str(exc):
            code = 409
        else:
            code = 400
        self._send(code, {"error": str(exc)})

    def _parts_get(self, parts: list[str]) -> None:
        project_id = parts[2]
        folder = self._open(project_id)
        if folder is None:
            return
        try:
            store = PartsStore(folder)
            with self.workbench.data_lock(project_id):
                if len(parts) == 4:
                    result = store.snapshot()
                elif len(parts) == 5 and parts[4] == "quality":
                    result = store.quality()
                elif len(parts) == 5 and parts[4] == "media-clips":
                    result = {"clips": store.list_project_media_clips()}
                elif len(parts) == 7 and parts[4] == "items" and parts[6] == "image":
                    if not _valid_name(parts[5]):
                        raise ValueError("素材编号不合法")
                    target = store.image_path(parts[5])
                    self._stream_file(target, cache=False)
                    return
                else:
                    self._send(404, {"error": "没有这个接口"})
                    return
            self._send(200, result)
        except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
            self._parts_error(exc)

    def _parts_upload(self, project_id: str) -> None:
        """浏览器/内网网线直连分块上传图片、视频或图片 ZIP 到零件工作区。"""
        workbench = self.workbench
        raw_name = Path(self._upload_name() or "upload.bin").name
        try:
            folder = workbench.store.open_dir(project_id)
        except (FileNotFoundError, ValueError):
            self._drain_body()
            self._send(404, {"error": "工程不存在"})
            return
        group_hdr = unquote(self.headers.get("X-Source-Group") or "").strip()
        try:
            interval = int(self.headers.get("X-Interval") or "30")
            max_frames = int(self.headers.get("X-Max-Frames") or "300")
        except ValueError:
            self._drain_body()
            self._send(400, {"error": "抽帧参数必须是整数"})
            return
        dedupe = (self.headers.get("X-Dedupe") or "1") != "0"

        staging_dir = folder / f".parts-upload-{uuid.uuid4().hex[:10]}"
        staging_dir.mkdir(parents=True, exist_ok=True)
        safe_leaf = dataset_transfer._safe_slug(Path(raw_name).stem, "asset") + Path(raw_name).suffix.lower()
        incoming = staging_dir / safe_leaf
        try:
            self._receive_to(incoming)
            suffix = incoming.suffix.lower()
            if suffix == ".zip":
                extracted = staging_dir / "unzipped"
                dataset_transfer._safe_extract_zip(incoming, extracted)
                yaml_in_zip = dataset_transfer._find_yaml_in_extracted(extracted)
                if yaml_in_zip is not None:
                    slug = dataset_transfer._safe_slug(Path(raw_name).stem, "dataset")
                    zip_dest = dataset_transfer.downloads_dir(folder) / f"{slug}.zip"
                    zip_dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(incoming, zip_dest)
                    names = yolo_jobs._read_yaml_names(yaml_in_zip.read_text(encoding="utf-8", errors="replace"))
                    if workbench.yolo_training():
                        raise ValueError("零件模型正在训练，请等待结束再导入数据集 ZIP")
                    with workbench.data_lock(project_id):
                        yolo_jobs.import_dataset(folder, str(yaml_in_zip))
                        if names:
                            dataset_transfer.sync_yolo_dir_to_parts_store(folder, yaml_in_zip.parent, names, slug)
                        snap = PartsStore(folder).snapshot()
                else:
                    with workbench.data_lock(project_id):
                        snap = PartsStore(folder).import_source(
                            str(extracted),
                            source_group=group_hdr or Path(raw_name).stem,
                            interval=interval,
                            max_frames=max_frames,
                            dedupe=dedupe,
                        )
            else:
                with workbench.data_lock(project_id):
                    snap = PartsStore(folder).import_source(
                        str(incoming),
                        source_group=group_hdr or (Path(raw_name).stem if suffix in dataset_transfer.VIDEO_EXTENSIONS else "浏览器上传"),
                        interval=interval,
                        max_frames=max_frames,
                        dedupe=dedupe,
                    )
            self._send(201, snap)
        except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
            self._parts_error(exc)
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    def _parts_post(self, parts: list[str]) -> None:
        project_id = parts[2]
        if len(parts) == 5 and parts[4] == "upload":
            self._parts_upload(project_id)
            return
        folder = self._open(project_id)
        if folder is None:
            return
        body = self._read_json()
        try:
            store = PartsStore(folder)
            with self.workbench.data_lock(project_id):
                if len(parts) == 5 and parts[4] == "classes":
                    revision = body.get("revision")
                    if type(revision) is not int or revision < 0:
                        raise ValueError("类别保存需要有效修订号")
                    result = store.set_classes(body.get("classes"), revision)
                elif len(parts) == 5 and parts[4] == "import":
                    interval = body.get("interval", 30)
                    max_frames = body.get("max_frames", 300)
                    if type(interval) is not int or type(max_frames) is not int:
                        raise ValueError("抽帧间隔和数量上限必须是整数")
                    dedupe = body.get("dedupe", True)
                    if type(dedupe) is not bool:
                        raise ValueError("去重开关必须是布尔值")
                    result = store.import_source(
                        body.get("source", ""), body.get("source_group", ""), interval, max_frames, dedupe
                    )
                elif len(parts) == 5 and parts[4] == "import-clip":
                    interval = body.get("interval", 30)
                    max_frames = body.get("max_frames", 300)
                    if type(interval) is not int or type(max_frames) is not int:
                        raise ValueError("抽帧间隔和数量上限必须是整数")
                    dedupe = body.get("dedupe", True)
                    if type(dedupe) is not bool:
                        raise ValueError("去重开关必须是布尔值")
                    result = store.import_project_clip(
                        body.get("file", ""),
                        source_group=body.get("source_group", ""),
                        interval=interval,
                        max_frames=max_frames,
                        dedupe=dedupe,
                    )
                elif len(parts) == 6 and parts[4] == "items":
                    if not _valid_name(parts[5]):
                        raise ValueError("素材编号不合法")
                    item = store.save_item(
                        parts[5], body.get("boxes"), body.get("status"), body.get("revision"), body.get("source_group")
                    )
                    result = {"item": item, "revision": store.snapshot()["revision"]}
                elif len(parts) == 7 and parts[4] == "items" and parts[6] == "delete":
                    if not _valid_name(parts[5]):
                        raise ValueError("素材编号不合法")
                    result = store.delete_item(parts[5], body.get("revision"))
                elif len(parts) == 5 and parts[4] == "publish":
                    result = store.publish(body.get("splits"), body.get("val_ratio", 0.2), body.get("test_ratio", 0.1))
                else:
                    self._send(404, {"error": "没有这个接口"})
                    return
            self._send(200, result)
        except (ValueError, FileNotFoundError, KeyError, TypeError, OSError) as exc:
            self._parts_error(exc)
