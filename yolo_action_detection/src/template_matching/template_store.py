"""定位模板与形状模型的统一路径、签名和落盘。

模板由两个文件构成，同一事务原子写入：

- ``locator_template.png``  人可查看、可重建的原始框选图
- ``locator_model.npz``     权威运行工件（分层定向边缘点集 + 固化的梯度阈值）

运行期加载 ``.npz``；若只有 PNG（旧版本升级上来的现场），首次加载自动重建并写日志。
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional
from uuid import uuid4

import cv2
import numpy as np

from template_matching.shape_model import ShapeModel, ShapeModelSettings

logger = logging.getLogger("yolo_action_detection.runtime")

DEFAULT_TEMPLATE_NAME = "locator_template.png"
DEFAULT_MODEL_NAME = "locator_model.npz"


def _config_dir(config, app_base_dir: str | Path | None) -> Path:
    config_path_text = str(getattr(config, "get_config_path", lambda: "")() or "").strip()
    if config_path_text:
        return Path(config_path_text).parent
    base_dir = Path(app_base_dir) if app_base_dir is not None else Path.cwd()
    return base_dir / "config"


def resolve_template_path(config, app_base_dir: str | Path | None = None) -> Path:
    """相对模板路径始终以实际配置文件所在目录为基准。"""
    raw_path = str(getattr(config, "roi_template_path", "") or "").strip()
    if not raw_path:
        raise ValueError("定位模板文件名不能为空")
    template_path = Path(raw_path)
    if template_path.is_absolute():
        return template_path
    return _config_dir(config, app_base_dir) / template_path


def resolve_model_path(config, app_base_dir: str | Path | None = None) -> Path:
    """形状模型（.npz）路径。默认与模板同目录、同名。"""
    raw_path = str(getattr(config, "roi_template_model_path", "") or "").strip()
    if not raw_path:
        raw_path = DEFAULT_MODEL_NAME
    model_path = Path(raw_path)
    if model_path.is_absolute():
        return model_path
    return _config_dir(config, app_base_dir) / model_path


def shape_model_settings(config) -> ShapeModelSettings:
    """把配置项换算成形状匹配参数。角度步进 0 表示自动。"""
    angle_range = abs(float(getattr(config, "roi_template_angle_range", 15.0)))
    return ShapeModelSettings(
        min_score=float(getattr(config, "roi_template_match_threshold", 0.7)),
        angle_start=-angle_range,
        angle_extent=2.0 * angle_range,
        angle_step=float(getattr(config, "roi_template_angle_step", 0.0) or 0.0),
        polarity=str(getattr(config, "roi_template_polarity", "use") or "use"),
    )


def _file_signature(path: Path) -> tuple:
    if not path.is_file():
        return (str(path), "missing")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        return (str(path), "unreadable", str(exc))
    return (str(path), len(payload), hashlib.sha256(payload).hexdigest())


def template_file_signature(config, app_base_dir: str | Path | None = None) -> tuple:
    """模板与模型的内容签名，用于判断是否需要重建运行处理器。"""
    if not bool(getattr(config, "roi_template_enabled", False)):
        return (False,)
    try:
        template_path = resolve_template_path(config, app_base_dir)
        model_path = resolve_model_path(config, app_base_dir)
    except ValueError as exc:
        return (True, "invalid", str(exc))
    return (True, _file_signature(template_path), _file_signature(model_path))


def read_template_image(path: str | Path) -> np.ndarray:
    """使用字节解码，兼容 Windows 中文路径。"""
    template_path = Path(path)
    try:
        payload = template_path.read_bytes()
    except OSError as exc:
        raise OSError(f"无法读取定位模板: {template_path}: {exc}") from exc
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"定位模板不是有效图片: {template_path}")
    return image


def load_shape_model(
    config,
    app_base_dir: str | Path | None = None,
    *,
    rebuild_from_png: bool = True,
) -> ShapeModel:
    """加载运行用的形状模型。

    优先读 ``.npz``；不存在时若允许，则从 PNG 重建并落盘（旧现场平滑升级）。
    """
    settings = shape_model_settings(config)
    model_path = resolve_model_path(config, app_base_dir)
    if model_path.is_file():
        return ShapeModel.load(model_path, settings)

    template_path = resolve_template_path(config, app_base_dir)
    if not template_path.is_file():
        raise FileNotFoundError(
            f"定位模型和模板都不存在: {model_path} / {template_path}；请重新创建模板"
        )
    if not rebuild_from_png:
        raise FileNotFoundError(f"定位模型不存在: {model_path}")
    logger.warning(
        "shape_model=rebuilt_from_png template=%s model=%s "
        "（旧版本模板自动升级，梯度阈值按当前图像重新固化）",
        template_path, model_path,
    )
    model = ShapeModel(read_template_image(template_path), settings)
    try:
        model.save(model_path)
    except Exception as exc:
        logger.warning("shape_model=save_failed path=%s error=%s", model_path, exc)
    return model


def prepare_template_candidate(
    image: np.ndarray,
    template_destination: str | Path,
    model_destination: str | Path,
    settings: ShapeModelSettings,
    *,
    created_frame_size: Optional[tuple[int, int]] = None,
) -> tuple[Path, Path, ShapeModel]:
    """在正式文件旁写入并自检临时模板与模型，正式文件暂不变更。

    返回 (临时 PNG, 临时 npz, 已建好的模型)。调用方负责原子替换或清理。
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("模板框选区域无有效图像")

    template_path = Path(template_destination)
    model_path = Path(model_destination)
    template_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    temp_template = template_path.with_name(f".{template_path.stem}.{token}.tmp.png")
    temp_model = model_path.with_name(f".{model_path.stem}.{token}.tmp.npz")
    try:
        encoded_ok, encoded = cv2.imencode(".png", image)
        if not encoded_ok:
            raise OSError("OpenCV 无法编码模板 PNG")
        temp_template.write_bytes(encoded.tobytes())
        # 从落盘后的 PNG 重新解码建模，确保运行期加载到的就是这一份内容
        model = ShapeModel(
            read_template_image(temp_template),
            settings,
            created_frame_size=created_frame_size,
        )
        model.save(temp_model)
        return temp_template, temp_model, model
    except Exception:
        temp_template.unlink(missing_ok=True)
        temp_model.unlink(missing_ok=True)
        raise


def replace_file_atomically(temp_path: str | Path, destination: str | Path) -> None:
    """同目录原子替换，避免运行线程看到半写文件。"""
    os.replace(Path(temp_path), Path(destination))


def restore_file(path: str | Path, previous_payload: bytes | None) -> None:
    """回滚文件；无旧内容时删除新文件。"""
    destination = Path(path)
    if previous_payload is None:
        destination.unlink(missing_ok=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".{destination.name}.{uuid4().hex}.restore")
    temp_path.write_bytes(previous_payload)
    os.replace(temp_path, destination)


def read_optional_bytes(path: str | Path) -> Optional[bytes]:
    candidate = Path(path)
    return candidate.read_bytes() if candidate.is_file() else None
