"""常驻智能标注推理子进程（运行在 YOLO 独立 Python 环境中）。

通信协议：
- stdin：每行一条 JSON 请求 {"id": "...", "cmd": "...", ...}
- stdout：每行一条 JSON 响应 {"id": "...", "ok": true/false, ...}
- 所有 Ultralytics / PyTorch 打印全部重定向到 stderr，避免污染 JSON-RPC 信道。

8GB 显存安全策略：
- 单槽位驻留：同一时刻只保留 1 个模型实例；切换槽位前彻底释放旧模型并 torch.cuda.empty_cache()。
- 默认 FP16（half=True）+ torch.inference_mode()；单图特征缓存仅在 GPU 保留当前图，历史 2 张卸载至 CPU。
- 捕获 CUDA OutOfMemoryError 时立即释放模型并返回 code="vram_insufficient"，由主服务引导切换轻量模型。
"""
from __future__ import annotations

from collections import OrderedDict
import contextlib
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import cv2
import numpy as np

# 确保以脚本路径直接启动时也能导入同目录的 smart_obb
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from smart_obb import bbox_xyxy_to_obb, mask_to_obb  # noqa: E402


def _read_bgr(path: str) -> np.ndarray | None:
    """支持中文路径的图片读取。"""
    p = Path(path)
    if not p.is_file():
        return None
    raw = np.fromfile(str(p), dtype=np.uint8)
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def _move_tree(obj: Any, device: Any) -> Any:
    """递归将特征树中的 Tensor 在 GPU/CPU 之间搬移，8GB 显卡上只让当前图占显存。"""
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.to(device=device, non_blocking=False)
    if isinstance(obj, dict):
        return {k: _move_tree(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_tree(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_tree(v, device) for v in obj)
    return obj


def _vram_info() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda": False, "gpu": "", "total_mb": 0, "free_mb": 0}
        free_b, total_b = torch.cuda.mem_get_info(0)
        return {
            "cuda": True,
            "gpu": torch.cuda.get_device_name(0),
            "total_mb": int(total_b // (1024 * 1024)),
            "free_mb": int(free_b // (1024 * 1024)),
        }
    except Exception:
        return {"cuda": False, "gpu": "", "total_mb": 0, "free_mb": 0}


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _min_vram_mb(model_name: str, slot: str) -> int:
    stem = Path(model_name).stem.lower()
    if "sam3" in stem:
        return 3000 if slot in {"semantic", "video"} else 2600
    if "sam2" in stem:
        if "_l" in stem:
            return 2200
        if "_b" in stem:
            return 1200
        return 800
    if "yoloe" in stem:
        return 600
    return 400


class SmartWorker:
    def __init__(self) -> None:
        self.model_name: str = ""
        self.model_path: str = ""
        self.family: str = ""
        self.slot: str = ""
        self.device: str = "cpu"
        self.half: bool = False
        self.predictor: Any = None
        self.yoloe_model: Any = None
        self._encoded_key: str = ""
        self._encoded_shape: tuple[int, int] = (0, 0)
        self._cpu_cache: OrderedDict[str, tuple[Any, tuple[int, int], Any]] = OrderedDict()
        self._max_cpu_cache = 2

    def unload(self) -> None:
        self.predictor = None
        self.yoloe_model = None
        self.model_name = ""
        self.model_path = ""
        self.family = ""
        self.slot = ""
        self._encoded_key = ""
        self._encoded_shape = (0, 0)
        self._cpu_cache.clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def probe(self) -> dict:
        import ultralytics
        import torch

        vram = _vram_info()
        return {
            "ok": True,
            "python": sys.version.split()[0],
            "python_path": sys.executable,
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "clip": _has_module("clip"),
            "mobileclip": _has_module("mobileclip"),
            **vram,
        }

    def load(self, req: dict) -> dict:
        import torch
        from ultralytics import YOLOE
        from ultralytics.models.sam import (
            Predictor,
            SAM2Predictor,
            SAM2VideoPredictor,
            SAM3Predictor,
            SAM3SemanticPredictor,
            SAM3VideoPredictor,
        )

        model_path = str(req.get("model_path") or "").strip()
        model_name = str(req.get("model_name") or Path(model_path).name).strip()
        slot = str(req.get("slot") or "interactive").strip()
        want_device = str(req.get("device") or "auto").strip()
        if not model_path or not Path(model_path).is_file():
            return {"ok": False, "code": "missing_weight", "error": f"找不到模型文件：{model_name}"}

        # 已加载同模型同槽位时直接复用
        if (
            self.model_path == model_path
            and self.slot == slot
            and (self.predictor is not None or self.yoloe_model is not None)
        ):
            return {
                "ok": True,
                "model_name": self.model_name,
                "family": self.family,
                "slot": self.slot,
                "device": self.device,
                "half": self.half,
                "vram": _vram_info(),
            }

        stem = Path(model_name).stem.lower()
        family = "sam3" if "sam3" in stem else "sam2" if "sam2" in stem else "yoloe" if "yoloe" in stem else "sam"

        if family == "sam3" and slot == "semantic" and not _has_module("clip"):
            return {
                "ok": False,
                "code": "missing_clip",
                "error": "SAM 3 文本/语义零样本模式缺少 clip 依赖（交互点选不受影响）",
            }
        if family == "yoloe" and slot == "yoloe_text" and not _has_module("mobileclip"):
            return {
                "ok": False,
                "code": "missing_mobileclip",
                "error": "YOLOE 文本提示模式缺少 mobileclip 依赖（视觉示例框模式无需安装，可直接用）",
            }

        # 先彻底卸载旧槽位再检查剩余显存
        self.unload()

        cuda_ok = torch.cuda.is_available() and want_device != "cpu"
        device = "0" if cuda_ok else "cpu"
        half = bool(cuda_ok)

        if cuda_ok:
            vram = _vram_info()
            need_mb = _min_vram_mb(model_name, slot)
            if vram["free_mb"] > 0 and vram["free_mb"] < need_mb:
                return {
                    "ok": False,
                    "code": "vram_insufficient",
                    "error": f"当前 GPU 剩余显存 {vram['free_mb']} MiB，低于 {model_name}（{slot}）所需的 {need_mb} MiB",
                    "vram": vram,
                }

        started = time.perf_counter()
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if family == "yoloe":
                    model = YOLOE(model_path)
                    if cuda_ok:
                        model.to("cuda:0")
                    self.yoloe_model = model
                    self.predictor = None
                else:
                    from ultralytics.cfg import DEFAULT_CFG_DICT

                    overrides = dict(
                        conf=float(req.get("conf", 0.25)),
                        task="segment",
                        mode="predict",
                        model=model_path,
                        device=device,
                        verbose=False,
                        save=False,
                    )
                    if "quantize" in DEFAULT_CFG_DICT:
                        overrides["quantize"] = 16 if half else None
                    else:
                        overrides["half"] = half
                    if family == "sam3":
                        overrides["imgsz"] = 1008
                        if slot == "semantic":
                            pred = SAM3SemanticPredictor(overrides=overrides)
                        elif slot == "video":
                            pred = SAM3VideoPredictor(overrides=overrides)
                        else:
                            pred = SAM3Predictor(overrides=overrides)
                    elif family == "sam2":
                        overrides["imgsz"] = 1024
                        pred = SAM2VideoPredictor(overrides=overrides) if slot == "video" else SAM2Predictor(overrides=overrides)
                    else:
                        overrides["imgsz"] = 1024
                        pred = Predictor(overrides=overrides)
                    pred.setup_model(verbose=False)
                    self.predictor = pred
                    self.yoloe_model = None
        except torch.cuda.OutOfMemoryError as exc:
            self.unload()
            return {
                "ok": False,
                "code": "vram_insufficient",
                "error": f"加载 {model_name} 触发显存不足（CUDA OOM）：{exc}",
                "vram": _vram_info(),
            }
        except Exception as exc:
            self.unload()
            return {
                "ok": False,
                "code": "load_failed",
                "error": f"加载 {model_name} 失败：{type(exc).__name__}: {exc}",
                "vram": _vram_info(),
            }

        self.model_name = model_name
        self.model_path = model_path
        self.family = family
        self.slot = slot
        self.device = f"cuda:0" if cuda_ok else "cpu"
        self.half = half
        return {
            "ok": True,
            "model_name": self.model_name,
            "family": self.family,
            "slot": self.slot,
            "device": self.device,
            "half": self.half,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "vram": _vram_info(),
        }

    def _ensure_encoded(self, image_path: str) -> tuple[int, int]:
        import torch

        if self.predictor is None:
            raise RuntimeError("尚未加载交互分割模型")
        p = Path(image_path)
        if not p.is_file():
            raise FileNotFoundError(f"找不到素材图片：{image_path}")
        stat = p.stat()
        key = f"{p.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
        if self._encoded_key == key and getattr(self.predictor, "features", None) is not None:
            return self._encoded_shape

        # 将当前 GPU 特征先卸载到 CPU 缓存，控制 8GB 卡峰值
        if self._encoded_key and getattr(self.predictor, "features", None) is not None:
            try:
                cpu_feat = _move_tree(self.predictor.features, "cpu")
                self._cpu_cache[self._encoded_key] = (cpu_feat, self._encoded_shape, getattr(self.predictor, "dataset", None))
                self._cpu_cache.move_to_end(self._encoded_key)
                while len(self._cpu_cache) > self._max_cpu_cache:
                    self._cpu_cache.popitem(last=False)
            except Exception:
                pass
            self.predictor.reset_image()

        # 命中 CPU 缓存则快速搬回 GPU
        if key in self._cpu_cache:
            cpu_feat, shape, cached_dataset = self._cpu_cache.pop(key)
            target_dev = getattr(self.predictor, "device", "cuda:0" if self.device.startswith("cuda") else "cpu")
            self.predictor.features = _move_tree(cpu_feat, target_dev)
            if cached_dataset is not None:
                self.predictor.dataset = cached_dataset
            self._encoded_key = key
            self._encoded_shape = shape
            return shape

        bgr = _read_bgr(image_path)
        if bgr is None:
            raise ValueError(f"图片无法解码：{p.name}")
        h, w = int(bgr.shape[0]), int(bgr.shape[1])
        with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
            self.predictor.set_image(bgr)
        self._encoded_key = key
        self._encoded_shape = (w, h)
        return (w, h)

    def encode(self, req: dict) -> dict:
        import torch

        started = time.perf_counter()
        try:
            w, h = self._ensure_encoded(str(req.get("image_path") or ""))
            return {
                "ok": True,
                "width": w,
                "height": h,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            }
        except torch.cuda.OutOfMemoryError as exc:
            self.unload()
            return {"ok": False, "code": "vram_insufficient", "error": f"图像编码显存不足：{exc}", "vram": _vram_info()}
        except Exception as exc:
            return {"ok": False, "code": "encode_failed", "error": f"图像编码失败：{exc}"}

    def predict(self, req: dict) -> dict:
        import torch

        started = time.perf_counter()
        image_path = str(req.get("image_path") or "")
        fit_mode = "hbb" if str(req.get("fit_mode") or "obb") == "hbb" else "obb"
        raw_points = req.get("points") or []
        raw_labels = req.get("labels") or []
        raw_bbox = req.get("bbox")
        if not raw_points and not raw_bbox:
            return {"ok": False, "code": "bad_request", "error": "至少提供一个提示点或粗略框"}

        try:
            w, h = self._ensure_encoded(image_path)
            kwargs: dict[str, Any] = {}
            if raw_points:
                pts_px = [[float(pt[0]) * w, float(pt[1]) * h] for pt in raw_points]
                lbls = [int(v) for v in raw_labels] if raw_labels else [1] * len(pts_px)
                kwargs["points"] = [pts_px]
                kwargs["labels"] = [lbls]
            if raw_bbox and len(raw_bbox) == 4:
                x1, y1, x2, y2 = [float(v) for v in raw_bbox]
                kwargs["bboxes"] = [[min(x1, x2) * w, min(y1, y2) * h, max(x1, x2) * w, max(y1, y2) * h]]

            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                results = self.predictor(**kwargs)

            if not results or results[0].masks is None or results[0].masks.data is None or len(results[0].masks.data) == 0:
                return {"ok": False, "code": "no_mask", "error": "未分割出有效前景掩码，请调整点击位置"}

            res0 = results[0]
            masks_tensor = res0.masks.data
            scores = None
            if res0.boxes is not None and getattr(res0.boxes, "conf", None) is not None and len(res0.boxes.conf) == len(masks_tensor):
                scores = res0.boxes.conf.detach().float().cpu().numpy()
            best_i = int(np.argmax(scores)) if scores is not None and len(scores) > 0 else 0
            best_score = float(scores[best_i]) if scores is not None and len(scores) > 0 else 0.9
            mask_np = masks_tensor[best_i].detach().float().cpu().numpy()

            fitted = mask_to_obb(mask_np, w, h, fit_mode=fit_mode)
            if fitted is None:
                return {"ok": False, "code": "no_mask", "error": "分割区域过小或退化，无法拟合旋转框"}

            return {
                "ok": True,
                "points": fitted["points"],
                "polygon": fitted["polygon"],
                "area_ratio": fitted["area_ratio"],
                "score": round(max(0.0, min(1.0, best_score)), 4),
                "source": self.family if self.family in {"sam3", "sam2", "sam"} else "sam",
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            }
        except torch.cuda.OutOfMemoryError as exc:
            self.unload()
            return {"ok": False, "code": "vram_insufficient", "error": f"交互推理显存不足：{exc}", "vram": _vram_info()}
        except Exception as exc:
            return {"ok": False, "code": "predict_failed", "error": f"交互推理失败：{type(exc).__name__}: {exc}"}

    def predict_zeroshot_one(self, req: dict) -> dict:
        """单图零样本分割并拟合 OBB，支持 sam3_semantic / yoloe_vp / yoloe_text。"""
        import torch

        engine = str(req.get("engine") or "").strip()
        image_path = str(req.get("image_path") or "")
        fit_mode = "hbb" if str(req.get("fit_mode") or "obb") == "hbb" else "obb"
        conf = float(req.get("conf", 0.25))
        prompts = list(req.get("prompts") or [])  # [{"text": "...", "class_id": 0}, ...]
        exemplar = req.get("visual_exemplar")  # {"image_path": ..., "bbox": [x1,y1,x2,y2], "class_id": 0}

        bgr = _read_bgr(image_path)
        if bgr is None:
            return {"ok": False, "code": "bad_image", "error": f"无法读取图片：{image_path}"}
        h, w = int(bgr.shape[0]), int(bgr.shape[1])

        boxes_out: list[dict] = []
        try:
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                if engine == "sam3_semantic":
                    if self.predictor is None:
                        return {"ok": False, "code": "not_loaded", "error": "SAM 3 语义模型未加载"}
                    self.predictor.args.conf = conf
                    if hasattr(self.predictor, "reset_prompts"):
                        self.predictor.reset_prompts()
                    self.predictor.set_image(bgr)
                    if exemplar and exemplar.get("bbox"):
                        eb = [float(v) for v in exemplar["bbox"]]
                        cls_id = int(exemplar.get("class_id", 0))
                        # 视觉示例框（仅当示例图与当前图同源或同尺寸映射）
                        res_list = self.predictor(bboxes=[[eb[0] * w, eb[1] * h, eb[2] * w, eb[3] * h]])
                        self._collect_results(res_list, w, h, fit_mode, "sam3", {0: cls_id}, conf, boxes_out)
                    elif prompts:
                        texts = [str(p["text"]).strip() for p in prompts if str(p.get("text") or "").strip()]
                        cls_map = {i: int(p.get("class_id", i)) for i, p in enumerate(prompts) if str(p.get("text") or "").strip()}
                        if texts:
                            res_list = self.predictor(text=texts)
                            self._collect_results(res_list, w, h, fit_mode, "sam3", cls_map, conf, boxes_out)
                elif engine in {"yoloe_vp", "yoloe_text"}:
                    if self.yoloe_model is None:
                        return {"ok": False, "code": "not_loaded", "error": "YOLOE 模型未加载"}
                    if engine == "yoloe_vp":
                        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

                        if not exemplar or not exemplar.get("image_path") or not exemplar.get("bbox"):
                            return {"ok": False, "code": "bad_request", "error": "YOLOE 视觉示例模式需要参考图和示例框"}
                        ref_bgr = _read_bgr(str(exemplar["image_path"]))
                        if ref_bgr is None:
                            return {"ok": False, "code": "bad_image", "error": "无法读取视觉示例参考图"}
                        rh, rw = int(ref_bgr.shape[0]), int(ref_bgr.shape[1])
                        eb = [float(v) for v in exemplar["bbox"]]
                        cls_id = int(exemplar.get("class_id", 0))
                        vp = {
                            "bboxes": np.array([[eb[0] * rw, eb[1] * rh, eb[2] * rw, eb[3] * rh]], dtype=np.float32),
                            "cls": np.array([0], dtype=np.int32),
                        }
                        res_list = self.yoloe_model.predict(
                            bgr,
                            refer_image=ref_bgr,
                            visual_prompts=vp,
                            predictor=YOLOEVPSegPredictor,
                            conf=conf,
                            verbose=False,
                            save=False,
                        )
                        self._collect_results(res_list, w, h, fit_mode, "yoloe", {0: cls_id}, conf, boxes_out)
                    else:
                        texts = [str(p["text"]).strip() for p in prompts if str(p.get("text") or "").strip()]
                        cls_map = {i: int(p.get("class_id", i)) for i, p in enumerate(prompts) if str(p.get("text") or "").strip()}
                        if texts:
                            self.yoloe_model.set_classes(texts, self.yoloe_model.get_text_pe(texts))
                            res_list = self.yoloe_model.predict(bgr, conf=conf, verbose=False, save=False)
                            self._collect_results(res_list, w, h, fit_mode, "yoloe", cls_map, conf, boxes_out)
                else:
                    return {"ok": False, "code": "bad_request", "error": f"不支持的零样本引擎：{engine}"}
        except torch.cuda.OutOfMemoryError as exc:
            self.unload()
            return {"ok": False, "code": "vram_insufficient", "error": f"零样本批量推理显存不足：{exc}", "vram": _vram_info()}
        except Exception as exc:
            return {"ok": False, "code": "zeroshot_failed", "error": f"零样本推理失败：{type(exc).__name__}: {exc}"}

        return {"ok": True, "boxes": boxes_out}

    @staticmethod
    def _collect_results(
        results: Any,
        w: int,
        h: int,
        fit_mode: str,
        source: str,
        cls_map: dict[int, int],
        conf_thresh: float,
        out: list[dict],
    ) -> None:
        if not results:
            return
        res = results[0]
        masks_data = res.masks.data if (res.masks is not None and res.masks.data is not None) else None
        boxes = res.boxes
        n = len(masks_data) if masks_data is not None else (len(boxes) if boxes is not None else 0)
        for i in range(n):
            score = 0.8
            raw_cls = 0
            if boxes is not None and i < len(boxes):
                if getattr(boxes, "conf", None) is not None:
                    score = float(boxes.conf[i].item())
                if getattr(boxes, "cls", None) is not None:
                    raw_cls = int(boxes.cls[i].item())
            if score < conf_thresh:
                continue
            class_id = cls_map.get(raw_cls, raw_cls)
            points = None
            if masks_data is not None and i < len(masks_data):
                fitted = mask_to_obb(masks_data[i].detach().float().cpu().numpy(), w, h, fit_mode=fit_mode)
                if fitted is not None:
                    points = fitted["points"]
            if points is None and boxes is not None and i < len(boxes) and getattr(boxes, "xyxy", None) is not None:
                points = bbox_xyxy_to_obb(boxes.xyxy[i].detach().float().cpu().tolist(), w, h)
            if points is not None:
                out.append({"class_id": int(class_id), "points": points, "source": source, "confidence": round(max(0.0, min(1.0, score)), 4)})

    def track_chunk(self, req: dict) -> dict:
        """对连续抽帧的一个显存安全分块（≤40帧）执行 SAM3/SAM2 视频传播，首帧种子框向后传播。"""
        import torch

        frames = list(req.get("frames") or [])  # [{"id": ..., "path": ...}, ...]
        seed_boxes = list(req.get("seed_boxes") or [])  # [{"class_id": 0, "points": [[x,y],...]}, ...]
        fit_mode = "hbb" if str(req.get("fit_mode") or "obb") == "hbb" else "obb"
        if len(frames) < 2 or not seed_boxes:
            return {"ok": False, "code": "bad_request", "error": "连续帧追踪至少需要 2 帧素材且首帧包含种子框"}
        if self.predictor is None or self.slot != "video":
            return {"ok": False, "code": "not_loaded", "error": "视频追踪模型未加载"}

        first_bgr = _read_bgr(str(frames[0]["path"]))
        if first_bgr is None:
            return {"ok": False, "code": "bad_image", "error": "无法读取首帧图片"}
        fh, fw = int(first_bgr.shape[0]), int(first_bgr.shape[1])

        # 用临时目录封装 MJPG 视频分块（满足 SAM2/SAM3VideoPredictor 对 dataset.mode == "video" 的要求，同时兼容中文路径）
        tmp_dir = Path(tempfile.mkdtemp(prefix="sam_track_"))
        try:
            tmp_video = tmp_dir / "chunk.avi"
            writer = cv2.VideoWriter(str(tmp_video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (fw, fh))
            if not writer.isOpened():
                raise RuntimeError("无法初始化临时追踪视频编码器 (MJPG)")
            try:
                for idx, entry in enumerate(frames):
                    bgr = first_bgr if idx == 0 else _read_bgr(str(entry["path"]))
                    if bgr is None:
                        raise ValueError(f"无法读取第 {idx + 1} 帧图片")
                    if bgr.shape[1] != fw or bgr.shape[0] != fh:
                        bgr = cv2.resize(bgr, (fw, fh), interpolation=cv2.INTER_LINEAR)
                    writer.write(bgr)
            finally:
                writer.release()

            bboxes_px = []
            cls_ids = []
            for box in seed_boxes:
                pts = box["points"]
                xs = [float(p[0]) * fw for p in pts]
                ys = [float(p[1]) * fh for p in pts]
                bboxes_px.append([min(xs), min(ys), max(xs), max(ys)])
                cls_ids.append(int(box["class_id"]))

            tracked: dict[str, list[dict]] = {}
            if hasattr(self.predictor, "inference_state"):
                self.predictor.inference_state = {}
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                # 逐帧流式消费，及时释放状态防止 8GB 显存积累
                stream = self.predictor(source=str(tmp_video), bboxes=bboxes_px, stream=True)
                for idx, res in enumerate(stream):
                    if idx == 0 or idx >= len(frames):
                        continue
                    frame_boxes: list[dict] = []
                    masks_data = res.masks.data if (res.masks is not None and res.masks.data is not None) else None
                    if masks_data is not None:
                        for obj_i in range(min(len(masks_data), len(cls_ids))):
                            fitted = mask_to_obb(masks_data[obj_i].detach().float().cpu().numpy(), fw, fh, fit_mode=fit_mode)
                            if fitted is not None:
                                frame_boxes.append(
                                    {
                                        "class_id": cls_ids[obj_i],
                                        "points": fitted["points"],
                                        "source": "video_track",
                                        "confidence": 0.85,
                                    }
                                )
                    tracked[str(frames[idx]["id"])] = frame_boxes

            # 清理视频记忆状态
            if hasattr(self.predictor, "inference_state"):
                self.predictor.inference_state = {}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"ok": True, "tracked": tracked}
        except torch.cuda.OutOfMemoryError as exc:
            self.unload()
            return {"ok": False, "code": "vram_insufficient", "error": f"视频帧追踪显存不足：{exc}", "vram": _vram_info()}
        except Exception as exc:
            return {"ok": False, "code": "track_failed", "error": f"视频帧追踪失败：{type(exc).__name__}: {exc}"}
        finally:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    worker = SmartWorker()
    # 就绪握手包
    sys.stdout.write(json.dumps({"event": "ready", "pid": os.getpid()}, ensure_ascii=False) + "\n")
    sys.stdout.flush()

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        req_id = ""
        try:
            req = json.loads(line)
            req_id = str(req.get("id") or "")
            cmd = str(req.get("cmd") or "")
            if cmd == "ping":
                resp = {
                    "ok": True,
                    "active_model": worker.model_name,
                    "active_slot": worker.slot,
                    "device": worker.device,
                    "vram": _vram_info(),
                }
            elif cmd == "probe":
                resp = worker.probe()
            elif cmd == "load":
                resp = worker.load(req)
            elif cmd == "encode":
                resp = worker.encode(req)
            elif cmd == "predict":
                resp = worker.predict(req)
            elif cmd == "predict_zeroshot_one":
                resp = worker.predict_zeroshot_one(req)
            elif cmd == "track_chunk":
                resp = worker.track_chunk(req)
            elif cmd == "unload":
                worker.unload()
                resp = {"ok": True, "vram": _vram_info()}
            elif cmd == "shutdown":
                worker.unload()
                sys.stdout.write(json.dumps({"id": req_id, "ok": True}, ensure_ascii=False) + "\n")
                sys.stdout.flush()
                return
            else:
                resp = {"ok": False, "code": "unknown_cmd", "error": f"未知命令：{cmd}"}
        except Exception as exc:
            resp = {"ok": False, "code": "worker_error", "error": f"{type(exc).__name__}: {exc}"}
        resp["id"] = req_id
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
