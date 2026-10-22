"""模型：确认 YOLO 和手部模型能加载、记下指纹，并给抽特征提供推理后端。"""
from __future__ import annotations

import hashlib
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
STATION_SRC = ROOT.parent / "yolo_action_detection" / "src"

# 全局模型库。所有工程共用一份，不随工程走 —— 换模型是换库里的文件，
# 不是每个工程各存一份。库放在工作台目录里，整个 lstm_workbench 文件夹搬走时模型跟着走。
MODEL_LIBRARY = ROOT / "models"
YOLO_LIBRARY = MODEL_LIBRARY / "yolo"
HANDS_LIBRARY = MODEL_LIBRARY / "hands"

# 手部模型指纹要跟着实际加载的文件走。0 是 lite，1 是 full。
HANDS_MODEL_FILES = {
    0: ("modules/palm_detection/palm_detection_lite.tflite", "modules/hand_landmark/hand_landmark_lite.tflite"),
    1: ("modules/palm_detection/palm_detection_full.tflite", "modules/hand_landmark/hand_landmark_full.tflite"),
}
# 和工位用同一个复杂度。两边的手部 21 点必须一模一样，否则特征对不上。
HANDS_MODEL_COMPLEXITY = 0
HANDS_MIN_DETECTION_CONFIDENCE = 0.5
HANDS_MIN_TRACKING_CONFIDENCE = 0.5
HANDS_PREPROCESS_VERSION = "2-isolated-video-tracking"

# 少了这个计算图 MediaPipe 就建不起图，手部直接不可用。
# 它不在 HANDS_MODEL_FILES 里，但判断库完不完整时必须一起看。
_HANDS_GRAPH = "hand_landmark/hand_landmark_tracking_cpu.binarypb"

_DEFAULT_CONF = 0.8
_TRACKER = None
_TRACKER_LOCK = threading.Lock()


def file_sha256(path: Path) -> str:
    """按块读。模型文件可能几百兆，不要一次读进内存。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _in_library(name: str) -> Path:
    """HANDS_MODEL_FILES 里的名字带 modules/ 前缀（那是 MediaPipe 包内的布局），
    库里是平铺的：models/hands/<模块名>/<文件>。"""
    parts = Path(name).parts
    if parts and parts[0] == "modules":
        parts = parts[1:]
    return HANDS_LIBRARY.joinpath(*parts)


def _describe(path: Path) -> dict:
    return {"name": path.name, "path": str(path), "bytes": path.stat().st_size}


def list_yolo_models() -> list[dict]:
    """模型库里的 ONNX 清单。目录不在或空的就是空列表，不抛。"""
    if not YOLO_LIBRARY.is_dir():
        return []
    found = sorted(
        (item for item in YOLO_LIBRARY.iterdir() if item.is_file() and item.suffix.lower() == ".onnx"),
        key=lambda item: item.name.lower(),
    )
    return [
        {
            "name": item.name,
            "path": str(item),
            "bytes": item.stat().st_size,
            "mtime": datetime.fromtimestamp(item.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        }
        for item in found
    ]


def library_info() -> dict:
    """模型库的现状。页面靠它渲染选择器，也在库为空时告诉用户该往哪放文件。

    手部文件齐了就列出实际路径。这只看磁盘，不代表 MediaPipe 已经加载成功。
    """
    source = hands_source()
    files = []
    for item in list(source["models"]) + ([source["graph"]] if source.get("graph") is not None else []):
        if item.is_file():
            files.append(_describe(item))
    return {
        "root": str(MODEL_LIBRARY),
        "yolo": {"dir": str(YOLO_LIBRARY), "exists": YOLO_LIBRARY.is_dir(), "files": list_yolo_models()},
        "hands": {
            "dir": source["dir"] or str(HANDS_LIBRARY),
            "exists": bool(files),
            "source": source["source"],
            "complexity": HANDS_MODEL_COMPLEXITY if source["source"] == "library" else None,
            "files": files,
            # 启动预热的结果：loading / ready / failed，没预热是 idle。页面据此显示「已加载」。
            "load": hands_status(),
        },
    }


def hands_source(complexity: int = HANDS_MODEL_COMPLEXITY) -> dict:
    """手部模型这次到底从哪读。库齐就用库，缺一个文件就整体退回 MediaPipe 内置 ——
    半个库比没有库更危险：那样加载的是内置模型，而页面上写的是库路径。

    models 是两个权重文件，指纹只认它们；graph 是计算图，不参与指纹但少了它跑不起来。"""
    names = HANDS_MODEL_FILES.get(complexity, ())
    from_library = [_in_library(name) for name in names]
    graph = HANDS_LIBRARY / _HANDS_GRAPH
    if from_library and all(item.is_file() and item.stat().st_size > 0 for item in from_library) and graph.is_file():
        return {"source": "library", "dir": str(HANDS_LIBRARY), "models": from_library, "graph": graph}

    try:
        import mediapipe
    except ImportError:
        return {"source": "bundled", "dir": "", "models": [], "graph": None}
    package = Path(mediapipe.__file__).resolve().parent
    return {
        "source": "bundled",
        "dir": str(package / "modules"),
        "models": [package / name for name in names],
        "graph": package / "modules" / _HANDS_GRAPH,
    }


def station_src_on_path() -> None:
    text = str(STATION_SRC)
    if text not in sys.path:
        sys.path.insert(0, text)


def hands_fingerprint(complexity: int = HANDS_MODEL_COMPLEXITY) -> dict:
    """手部模型指纹。读不到就给出原因，不抛。

    指纹只覆盖那两个权重文件，和以前逐字一致 —— 所以把模型挪进库里不会改变已有契约的
    比对口径。source / dir / files[].path 是这一轮新增的：页面要显示这次真正加载的是哪几个
    文件，而不是笼统一个哈希。计算图不参与指纹，但它在文件清单里。"""
    try:
        import mediapipe
    except ImportError as exc:
        return {"ok": False, "error": f"载入 MediaPipe 失败：{exc}"}
    version = str(getattr(mediapipe, "__version__", ""))
    source = hands_source(complexity)
    paths = source["models"]
    missing = [path.name for path in paths if not path.is_file()]
    if source.get("graph") is None or not source["graph"].is_file():
        return {"ok": False, "version": version, "error": "缺少手部追踪计算图"}
    if not paths or missing:
        return {"ok": False, "version": version, "error": "缺少手部模型文件：" + "、".join(missing or ["（未找到）"])}

    files = {}
    digest = hashlib.sha256()
    for path in paths:
        one = file_sha256(path)
        files[path.name] = one
        digest.update(f"{path.name}:{one}\n".encode("utf-8"))
    graph = source["graph"]
    return {
        "ok": True,
        "version": version,
        "model_complexity": complexity,
        "sha256": digest.hexdigest(),
        "graph_sha256": file_sha256(graph),
        "files": files,
        "source": source["source"],
        "dir": source["dir"],
        "paths": [_describe(item) for item in paths] + ([_describe(graph)] if graph is not None else []),
    }


def create_hands_tracker():
    """工作台与产线共用的创建入口；每次返回独立的连续视频会话。"""
    station_src_on_path()
    from hand_tracking.mediapipe_hands import MediaPipeHandsTracker

    source = hands_source()
    return MediaPipeHandsTracker(
        max_num_hands=2,
        model_complexity=HANDS_MODEL_COMPLEXITY,
        min_detection_confidence=HANDS_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=HANDS_MIN_TRACKING_CONFIDENCE,
        model_root=source["dir"] if source["source"] == "library" else None,
    )


def hands_cache_identity() -> dict:
    """缓存认当前磁盘模型与实际参数，不信任 settings 里可能过期的指纹。"""
    fingerprint = hands_fingerprint()
    if not fingerprint.get("ok"):
        raise RuntimeError(f"手部模型不可用：{fingerprint.get('error', '不能读取指纹')}")
    return {
        "hands_sha256": fingerprint["sha256"],
        "hands_graph_sha256": str(fingerprint.get("graph_sha256") or ""),
        "hands_model_complexity": fingerprint["model_complexity"],
        "hands_mediapipe_version": str(fingerprint.get("version") or ""),
        "hands_min_detection_confidence": HANDS_MIN_DETECTION_CONFIDENCE,
        "hands_min_tracking_confidence": HANDS_MIN_TRACKING_CONFIDENCE,
        "hands_preprocess_version": HANDS_PREPROCESS_VERSION,
    }


def hands_tracker():
    """兼容旧版单实例检查接口；视频推理必须用 create_hands_tracker 独立创建。"""
    global _TRACKER
    # 旧调用方并发检查时只创建一份；正常预热和视频抽取不再依赖它。
    with _TRACKER_LOCK:
        if _TRACKER is None or not _TRACKER.available:
            if _TRACKER is not None:
                _TRACKER.close()
            _TRACKER = create_hands_tracker()
        return _TRACKER


def probe_hands() -> dict:
    """指纹之外还要真能构造出来，才算「能加载」。"""
    fingerprint = hands_fingerprint()
    if not fingerprint.get("ok"):
        return fingerprint
    try:
        tracker = create_hands_tracker()
    except Exception as exc:
        return {**fingerprint, "ok": False, "error": f"手部模型加载失败：{exc}"}
    try:
        if not tracker.available:
            return {**fingerprint, "ok": False, "error": f"手部模型加载失败：{tracker.error}"}
        return fingerprint
    finally:
        tracker.close()


# ---------------------------------------------------------------------------
# 启动预热：训练台一起来就把手部模型真加载一遍，结果缓存住。
#
# 以前手部模型只在用户点「检查并保存」时才加载，页面上「已加载」的依据是 settings 里
# 有没有指纹 —— 新工程、或者没点过检查的工程，启动后永远显示成没就绪，哪怕模型文件
# 全在、也能加载。预热把「能不能加载」变成启动时就知道的事实，加载失败的原因也会
# 原样摆出来，而不是等到抽特征才炸。
# ---------------------------------------------------------------------------
_WARM_LOCK = threading.Lock()
_WARM_STATE: dict = {"state": "idle"}
_WARM_THREAD: threading.Thread | None = None


def _run_warmup() -> None:
    global _WARM_STATE
    started = time.monotonic()
    try:
        result = probe_hands()
    except Exception as exc:  # 预热绝不能把训练台带崩
        result = {"ok": False, "error": f"手部模型加载失败：{exc}"}
    elapsed = round(time.monotonic() - started, 2)
    with _WARM_LOCK:
        _WARM_STATE = {
            "state": "ready" if result.get("ok") else "failed",
            "seconds": elapsed,
            **result,
        }


def start_hands_warmup() -> None:
    """后台线程里加载手部模型。已经在跑或者跑完了就不重复。"""
    global _WARM_STATE, _WARM_THREAD
    with _WARM_LOCK:
        if _WARM_THREAD is not None and _WARM_THREAD.is_alive():
            return
        if _WARM_STATE.get("state") in {"ready", "loading"}:
            return
        _WARM_STATE = {"state": "loading"}
        _WARM_THREAD = threading.Thread(target=_run_warmup, name="hands-warmup", daemon=True)
        _WARM_THREAD.start()


def hands_status() -> dict:
    """页面用的手部模型状态：idle（没预热）/ loading / ready / failed。"""
    with _WARM_LOCK:
        return dict(_WARM_STATE)


def record_hands_probe(result: dict) -> None:
    """用户手动「检查并保存」也是一次真加载，结果以最新这次为准。

    不同步的话，预热失败过一次、之后手动检查通过了，页面重新拉数据仍会显示失败。
    """
    global _WARM_STATE
    with _WARM_LOCK:
        _WARM_STATE = {"state": "ready" if result.get("ok") else "failed", **result}


def reset_hands_warmup() -> None:
    """测试用：清掉缓存，让下一次预热重新来。"""
    global _WARM_STATE, _WARM_THREAD, _TRACKER
    with _WARM_LOCK:
        _WARM_STATE = {"state": "idle"}
        _WARM_THREAD = None
    with _TRACKER_LOCK:
        if _TRACKER is not None:
            _TRACKER.close()
        _TRACKER = None


def yolo_providers() -> list[str]:
    """抽特征时优先用显卡。CUDA 不可用时用 DirectML（Windows 显卡），再不行才用 CPU。"""
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "DmlExecutionProvider" in available:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def build_yolo(path: str, conf_threshold: float):
    station_src_on_path()
    from yolo_runtime.onnx_obb_processor import OnnxObbProcessor

    return OnnxObbProcessor(path, conf_threshold=conf_threshold, providers=yolo_providers())


def load_yolo(path: str, conf_threshold: float):
    """先按显卡加载。显卡会话建不起来就改回 CPU，抽特征不因此中断。"""
    processor = build_yolo(path, conf_threshold)
    try:
        processor.load()
        return processor
    except Exception:
        if yolo_providers() == ["CPUExecutionProvider"]:
            raise
    station_src_on_path()
    from yolo_runtime.onnx_obb_processor import OnnxObbProcessor

    processor = OnnxObbProcessor(path, conf_threshold=conf_threshold, providers=["CPUExecutionProvider"])
    processor.load()
    return processor


def probe_yolo(path_text: str, conf_threshold: float) -> dict:
    """真加载一次 ONNX，拿到类名和文件指纹。"""
    if not path_text:
        return {"ok": False, "error": "还没有填写 YOLO 路径"}
    path = Path(path_text)
    if not path.is_file():
        return {"ok": False, "error": f"YOLO 文件不存在：{path}"}
    try:
        processor = load_yolo(str(path), conf_threshold)
    except Exception as exc:
        return {"ok": False, "error": f"YOLO 加载失败：{exc}"}
    names = processor.get_class_names()
    return {
        "ok": True,
        "path": str(path),
        "sha256": file_sha256(path),
        "classes": [name for _index, name in sorted(names.items())],
    }


def hand_points(frame: np.ndarray) -> list[list[tuple[float, float, float]]]:
    """单帧检查用。连续视频必须使用 live_backends，不能逐帧重建追踪器。"""
    tracker = create_hands_tracker()
    try:
        if not tracker.available:
            raise RuntimeError(f"手部模型不可用：{tracker.error}")
        hands = []
        for hand in tracker.process(frame):
            depths = hand.depths or (0.0,) * len(hand.points)
            hands.append([(x, y, z) for (x, y), z in zip(hand.points, depths)])
        return hands
    finally:
        tracker.close()


def yolo_backend(settings: dict):
    """只加载 YOLO。手部开关关掉时，抽特征不创建手部追踪器。"""
    yolo_path = str(settings.get("yolo_path") or "")
    if not yolo_path:
        raise FileNotFoundError("还没有选定 YOLO，先去「模型」页检查并保存")
    conf_threshold = float(settings.get("yolo_conf_threshold") or _DEFAULT_CONF)
    processor = load_yolo(yolo_path, conf_threshold)

    def detect(frame: np.ndarray) -> list[dict]:
        state = processor.process_frame(frame)
        if state.error:
            raise RuntimeError(state.error)
        return [{
            "label": item.label,
            "conf": item.conf,
            "box": item.box,
            "polygon": [(float(x), float(y)) for x, y in item.polygon],
        } for item in state.detections]

    return detect


def live_backends(settings: dict):
    """返回 (detect, hands) 两个闭包。抽取方逐帧成对调用，保证框和手来自同一帧。"""
    detect = yolo_backend(settings)
    tracker = create_hands_tracker()
    if not tracker.available:
        tracker.close()
        raise RuntimeError(f"手部模型不可用：{tracker.error}")

    def hands(frame: np.ndarray) -> list[list[tuple[float, float, float]]]:
        result = []
        for hand in tracker.process(frame):
            depths = hand.depths or (0.0,) * len(hand.points)
            result.append([(x, y, z) for (x, y), z in zip(hand.points, depths)])
        return result

    hands.close = tracker.close
    return detect, hands
