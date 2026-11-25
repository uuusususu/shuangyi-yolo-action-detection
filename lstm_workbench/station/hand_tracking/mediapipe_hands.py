"""MediaPipe Hands 封装：识别 21 点并绘制骨架，明确区分故障和正常无手。"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import types
from typing import Iterable, Iterator

import cv2
import numpy as np

from runtime_diagnostics import get_runtime_logger

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)

_HAND_POINT_COUNT = 21
_MODULE_DIRS = ("hand_landmark", "palm_detection")
_MANIFEST_NAME = "mirror.json"
_SKELETON_COLOR_BGR = (255, 0, 0)
_POINT_RADIUS = 5
_LINE_THICKNESS = 4
_NATIVE_HANDS_LOCK = threading.RLock()


@dataclass(frozen=True)
class HandLandmarks:
    """一只手的 21 个像素坐标点（相对当前帧宽高）。depths 是 MediaPipe 的相对深度。"""

    points: tuple[tuple[float, float], ...]
    handedness: str = ""
    depths: tuple[float, ...] = ()


class MediaPipeHandsTracker:
    """单视频 Hands 会话。初始化失败由状态反馈，推理故障同时抛异常。"""

    def __init__(
        self,
        *,
        max_num_hands: int = 2,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_root: str | Path | None = None,
    ) -> None:
        self._hands = None
        self._available = False
        self._error = ""
        self._logged_process_error = False
        self._logger = get_runtime_logger()
        try:
            self._hands = _create_native_hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                model_root=model_root,
            )
            self._available = True
            self._logger.info("hand_overlay=ready max_num_hands=%s", max_num_hands)
        except Exception as exc:
            self._error = str(exc)
            self._logger.exception("hand_overlay=init_failed error=%s", exc)

    @property
    def available(self) -> bool:
        return self._available and self._hands is not None

    @property
    def error(self) -> str:
        return self._error

    def process(self, frame_bgr) -> list[HandLandmarks]:
        if not self.available:
            raise RuntimeError(f"手部模型不可用：{self._error or '追踪会话已经关闭'}")
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            return []
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._hands.process(rgb)
            return _parse_hands_results(results, frame_bgr.shape[1], frame_bgr.shape[0])
        except Exception as exc:
            self._error = f"手部模型推理失败：{exc}"
            self._available = False
            if not self._logged_process_error:
                self._logger.exception("hand_overlay=process_failed error=%s", exc)
                self._logged_process_error = True
            raise RuntimeError(self._error) from exc

    def close(self) -> None:
        hands = self._hands
        self._hands = None
        self._available = False
        if hands is None:
            return
        try:
            hands.close()
        except Exception:
            pass


def landmarks_to_payload(hands: Iterable[HandLandmarks]) -> list[dict]:
    """把内部结构转成 overlay 信号可跨线程传递的字典列表。"""
    payload = []
    for hand in hands:
        payload.append(
            {
                "handedness": str(hand.handedness or ""),
                "points": [[float(x), float(y)] for x, y in hand.points],
                "depths": [float(value) for value in hand.depths],
            }
        )
    return payload


def draw_hand_landmarks(frame: np.ndarray, hands) -> np.ndarray:
    """在 BGR 图上画 21 点和标准手指连线。hands 为空时原样返回。"""
    items = list(hands or [])
    if not items or frame is None or getattr(frame, "size", 0) == 0:
        return frame
    if not frame.flags.writeable:
        frame = frame.copy()
    for item in items:
        points = _extract_draw_points(item)
        if len(points) != _HAND_POINT_COUNT:
            continue
        for start, end in HAND_CONNECTIONS:
            p1 = points[start]
            p2 = points[end]
            cv2.line(frame, p1, p2, _SKELETON_COLOR_BGR, _LINE_THICKNESS, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, _POINT_RADIUS, _SKELETON_COLOR_BGR, -1, cv2.LINE_AA)
    return frame


def _extract_draw_points(item) -> list[tuple[int, int]]:
    raw_points = item.points if isinstance(item, HandLandmarks) else (item or {}).get("points")
    points: list[tuple[int, int]] = []
    for point in raw_points or []:
        try:
            points.append((int(round(float(point[0]))), int(round(float(point[1])))))
        except (TypeError, ValueError, IndexError):
            return []
    return points


def _parse_hands_results(results, width: int, height: int) -> list[HandLandmarks]:
    multi = getattr(results, "multi_hand_landmarks", None) or []
    handedness_list = getattr(results, "multi_handedness", None) or []
    parsed: list[HandLandmarks] = []
    for index, hand_lms in enumerate(multi):
        landmarks = getattr(hand_lms, "landmark", None) or []
        points = []
        depths = []
        for landmark in landmarks:
            points.append((float(landmark.x) * width, float(landmark.y) * height))
            depths.append(float(getattr(landmark, "z", 0.0)))
        if len(points) != _HAND_POINT_COUNT:
            continue
        parsed.append(
            HandLandmarks(
                points=tuple(points),
                handedness=_handedness_label(handedness_list, index),
                depths=tuple(depths),
            )
        )
    return parsed


def _handedness_label(handedness_list, index: int) -> str:
    if index >= len(handedness_list):
        return ""
    try:
        return str(handedness_list[index].classification[0].label or "")
    except (AttributeError, IndexError, TypeError):
        return ""


_MATPLOTLIB_STUB_MODULES = (
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.cm",
    "matplotlib.colors",
)


def _ensure_matplotlib_importable() -> None:
    """MediaPipe drawing_utils imports matplotlib; we draw with OpenCV, so stub if missing."""
    try:
        for name in _MATPLOTLIB_STUB_MODULES:
            __import__(name)
        return
    except ImportError:
        _install_matplotlib_import_stubs()


def _install_matplotlib_import_stubs() -> None:
    matplotlib = sys.modules.get("matplotlib")
    if matplotlib is None or not isinstance(matplotlib, types.ModuleType):
        matplotlib = types.ModuleType("matplotlib")
    matplotlib.__path__ = []
    sys.modules["matplotlib"] = matplotlib
    for name in _MATPLOTLIB_STUB_MODULES[1:]:
        module = sys.modules.get(name)
        if module is None or not isinstance(module, types.ModuleType):
            module = types.ModuleType(name)
            sys.modules[name] = module
        setattr(matplotlib, name.rsplit(".", 1)[-1], module)
    get_runtime_logger().info("hand_overlay=matplotlib_stubbed")


def _create_native_hands(model_root=None, **kwargs):
    # 资源根的临时替换影响整个进程，预热、工作台和相机不能同时改写它。
    with _NATIVE_HANDS_LOCK:
        return _create_native_hands_locked(model_root=model_root, **kwargs)


def _create_native_hands_locked(model_root=None, **kwargs):
    """model_root 给的是模型库里的库根（models/hands），不给就用 MediaPipe 自带的那份。

    MediaPipe 的 Hands() 没有模型路径参数，它只认包内 mediapipe/modules 下的文件。
    所以想让库生效，只能把库里的模块目录镜像到 ASCII 缓存目录，再把资源根指过去 ——
    否则库摆在那儿也不会被读，页面上写的库路径就成了假的。"""
    _ensure_matplotlib_importable()
    import mediapipe as mp
    from mediapipe.python.solutions.hands import Hands

    package = Path(mp.__file__).resolve().parent
    if model_root is None and _path_is_ascii(package.parent):
        return Hands(**kwargs)
    source_modules = Path(model_root) if model_root else package / "modules"
    cache_root = _ascii_cache_root()
    _mirror_hand_modules(source_modules, cache_root)
    with _patched_solution_base_root(cache_root):
        return Hands(**kwargs)


def _path_is_ascii(path: Path) -> bool:
    return str(path).isascii()


def _ascii_cache_root() -> Path:
    for key in ("LOCALAPPDATA", "TEMP", "TMP"):
        value = os.environ.get(key)
        if value and str(value).isascii():
            return Path(value) / "YOLOActionDetection" / "mediapipe_resources"
    return Path("C:/YOLOActionDetection/mediapipe_resources")


def _mirror_hand_modules(source_modules: Path, dest_root: Path) -> Path:
    """把 source_modules 下要用的两个模块目录镜像到 ASCII 缓存目录。

    命中与否看清单里记的源目录和各文件签名 —— 换掉库里的模型后签名就对不上，
    会重镜像。只认目标目录里有没有 marker 文件的话，换了模型也会一直沿用旧缓存。"""
    dest_modules = dest_root / "mediapipe" / "modules"
    manifest = dest_modules / _MANIFEST_NAME
    signature = {"source": str(source_modules), "modules": {}}
    for name in _MODULE_DIRS:
        src = source_modules / name
        if src.is_dir():
            signature["modules"][name] = _module_signature(src)
    if _mirror_is_current(manifest, signature):
        return dest_root

    dest_modules.mkdir(parents=True, exist_ok=True)
    for name in _MODULE_DIRS:
        src = source_modules / name
        if not src.is_dir():
            continue
        dst = dest_modules / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
    _write_manifest(manifest, signature)
    return dest_root


def _module_signature(src: Path) -> list:
    """镜像校验使用内容摘要，避免同尺寸且保留修改时间的换模仍复用旧文件。"""
    items = []
    for item in sorted(src.rglob("*")):
        if not item.is_file() or "__pycache__" in item.parts:
            continue
        stat = item.stat()
        digest = hashlib.sha256()
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        items.append([str(item.relative_to(src)), stat.st_size, digest.hexdigest()])
    return items


def _mirror_is_current(manifest: Path, signature: dict) -> bool:
    try:
        return json.loads(manifest.read_text(encoding="utf-8")) == signature
    except (OSError, ValueError):
        return False


def _write_manifest(manifest: Path, signature: dict) -> None:
    """写失败不算错：下次还会重镜像一遍，只是慢一点。"""
    try:
        manifest.write_text(json.dumps(signature, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


@contextmanager
def _patched_solution_base_root(ascii_root: Path) -> Iterator[None]:
    """MediaPipe C++ 打不开含中文的 site-packages 路径，构造期把资源根指到 ASCII 目录。"""
    import mediapipe.python.solution_base as solution_base

    original_abspath = solution_base.os.path.abspath
    real_file = Path(solution_base.__file__).resolve()
    fake_file = str(ascii_root / "mediapipe" / "python" / "solution_base.py")

    def _abspath(path) -> str:
        resolved = original_abspath(path)
        try:
            if Path(resolved).resolve() == real_file:
                return fake_file
        except OSError:
            pass
        return resolved

    solution_base.os.path.abspath = _abspath
    try:
        yield
    finally:
        solution_base.os.path.abspath = original_abspath
