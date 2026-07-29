"""YOLO OBB 动作检测系统入口。"""
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from config import ConfigManager
from app_state import AppState
from camera.camera_worker import CameraWorker
from detection_logging.audio_feedback import resolve_sound_file
from runtime_diagnostics import (
    close_runtime_logger,
    configure_runtime_logger,
    get_runtime_logger,
    get_runtime_log_path,
    install_unhandled_exception_hook,
    log_model_context,
    log_startup_context,
)
from yolo_runtime.onnx_obb_processor import OnnxObbProcessor
from template_matching.template_store import resolve_template_path
from step_sequence.step_sequence_engine import StepSequenceEngine
from ui.main_window import MainWindow
from ui.theme import THEME_ID, apply_application_theme, require_theme_module


@dataclass(frozen=True)
class AppPaths:
    base_dir: Path
    resource_dir: Path
    config_path: Path


def resolve_app_paths(
    *,
    frozen: bool | None = None,
    executable: str | Path | None = None,
    meipass: str | Path | None = None,
    main_file: str | Path | None = None,
) -> AppPaths:
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else bool(frozen)
    if is_frozen:
        exe = Path(executable or sys.executable).resolve()
        base_dir = exe.parent
        resource_dir = Path(meipass or getattr(sys, "_MEIPASS", base_dir)).resolve()
        return AppPaths(base_dir=base_dir, resource_dir=resource_dir, config_path=base_dir / "config.json")

    src_main = Path(main_file or __file__).resolve()
    base_dir = src_main.parents[1]
    return AppPaths(base_dir=base_dir, resource_dir=base_dir, config_path=base_dir / "config" / "config.json")


def initialize_portable_config(base_dir: Path, resource_dir: Path) -> Path:
    logger = get_runtime_logger()
    config_path = Path(base_dir) / "config.json"
    if config_path.exists():
        logger.info("portable_config=existing path=%s", config_path)
        return config_path

    for bundled_config in (
        Path(resource_dir) / "config.json",
        Path(resource_dir) / "config" / "config.json",
        Path(resource_dir) / "config" / "config.example.json",
    ):
        if bundled_config.exists():
            try:
                shutil.copyfile(bundled_config, config_path)
                logger.warning(
                    "portable_config=created source=%s destination=%s",
                    bundled_config,
                    config_path,
                )
            except OSError:
                logger.exception(
                    "portable_config=copy_failed source=%s destination=%s",
                    bundled_config,
                    config_path,
                )
            break
    if not config_path.exists():
        logger.error("portable_config=missing destination=%s", config_path)
    return config_path


def resolve_model_path(config: ConfigManager, app_base_dir: Path | str) -> Path:
    model_path = Path(str(config.yolo_model_path))
    if model_path.is_absolute():
        return model_path
    return Path(app_base_dir) / model_path


def _create_pt_processor(config: ConfigManager, model_path_text: str):
    from yolo_runtime.yolo_obb_processor import YoloObbProcessor

    return YoloObbProcessor(
        model_path=model_path_text,
        conf_threshold=config.yolo_conf_threshold,
        iou_threshold=config.yolo_iou_threshold,
        device=config.ultralytics_device,
        half=config.ultralytics_half,
        tracker=config.ultralytics_tracker,
        max_det=config.ultralytics_max_det,
        track_persist=config.ultralytics_track_persist,
    )


def create_yolo_processor(config: ConfigManager, app_base_dir: Path | str | None = None):
    """从配置创建最终运行处理器，包括可选的模板门控。"""
    base_dir = Path(app_base_dir) if app_base_dir is not None else resolve_app_paths().base_dir
    model_path = resolve_model_path(config, base_dir)
    model_path_text = str(model_path)
    if model_path.suffix.lower() == ".onnx":
        processor = OnnxObbProcessor(
            model_path=model_path_text,
            conf_threshold=config.yolo_conf_threshold,
            iou_threshold=config.yolo_iou_threshold,
            max_det=config.ultralytics_max_det,
        )
        processor.load()
    else:
        processor = _create_pt_processor(config, model_path_text)
        processor.load()
    return _wrap_with_template_gate(processor, config, base_dir)


def _wrap_with_template_gate(processor, config: ConfigManager, base_dir: Path):
    """模板启用时必须返回门控处理器，初始化失败直接阻断。"""
    if not getattr(config, "roi_inspection_enabled", False):
        return processor
    if not getattr(config, "roi_template_enabled", False):
        return processor
    template_path = resolve_template_path(config, base_dir)
    if not template_path.is_file():
        raise FileNotFoundError(f"定位模板文件不存在: {template_path}")
    search_rect = getattr(config, "roi_template_search_rect", None)
    if search_rect is None:
        raise ValueError("搜索 ROI 未配置")

    logger = get_runtime_logger()
    try:
        from template_matching.template_store import load_shape_model
        from yolo_runtime.template_gated_processor import TemplateGatedProcessor

        template_model = load_shape_model(config, base_dir)
        wrapped = TemplateGatedProcessor(processor, template_model, search_rect)
        logger.info(
            "template_gate=enabled template=%s levels=%d points=%s "
            "base_angle_step=%.3f min_search_size=%s search_roi=%s",
            template_path,
            template_model.level_count,
            template_model.point_counts,
            template_model.base_angle_step,
            template_model.min_search_size(0),
            search_rect,
        )
        return wrapped
    except Exception as exc:
        logger.exception("template_gate=failed path=%s error=%s", template_path, exc)
        raise RuntimeError(f"定位模板加载失败: {template_path}: {exc}") from exc


def load_yolo_processor_with_diagnostics(
    config: ConfigManager,
    app_base_dir: Path | str,
    *,
    include_model_hash: bool = False,
):
    """Load the configured processor and preserve the full failure traceback."""
    logger = get_runtime_logger()
    model_path = resolve_model_path(config, app_base_dir)
    log_model_context(model_path, include_hash=include_model_hash)
    try:
        processor = create_yolo_processor(config, app_base_dir=app_base_dir)
        class_names = processor.get_class_names()
        logger.info(
            "processor_load=ok type=%s classes=%s",
            type(processor).__name__,
            class_names,
        )
        return processor
    except Exception:
        logger.exception("processor_load=failed")
        return None


def create_step_engine(config: ConfigManager) -> StepSequenceEngine:
    """按当前配置创建步骤引擎。"""
    return StepSequenceEngine(
        step_class_names=config.category_names,
        step_counts=config.get_runtime_category_counts(),
        enter_stable_frames=config.action_pass_stable_frames,
        out_of_order_frames=config.action_ng_stable_frames,
        leave_stable_frames=config.action_leave_stable_frames,
        order_constraint_enabled=config.action_order_constraint_enabled,
    )


def _resolve_portable_sound(kind: str, app_base_dir: Path, resource_dir: Path) -> Path:
    resource_sound = resolve_sound_file(kind, resource_dir=resource_dir)
    if resource_sound.exists():
        return resource_sound
    base_sound = resolve_sound_file(kind, resource_dir=app_base_dir)
    if base_sound.exists():
        return base_sound
    return resource_sound


def run_portable_smoke_check(
    config_path: Path,
    app_base_dir: Path,
    resource_dir: Path | None = None,
    processor_loader=None,
) -> int:
    lines: list[str] = []
    resource_root = Path(resource_dir) if resource_dir is not None else Path(app_base_dir)

    def emit(message: str) -> None:
        print(message)
        lines.append(message)

    def finish(code: int) -> int:
        log_dir = Path(app_base_dir) / "logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "portable_smoke_test.log").write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        return code

    config = ConfigManager()
    try:
        config.load(config_path)
    except ValueError as exc:
        emit(f"[portable-smoke] config invalid: {exc}")
        return finish(1)

    structural_errors = config.inspection_validation_errors(None)
    if structural_errors:
        emit(f"[portable-smoke] inspection config invalid: {'; '.join(structural_errors)}")
        return finish(1)

    model_path = resolve_model_path(config, app_base_dir)
    emit(f"[portable-smoke] config={config_path}")
    emit(f"[portable-smoke] model={model_path}")
    if not model_path.exists():
        emit("[portable-smoke] model missing")
        return finish(2)

    if processor_loader is None:
        processor = load_yolo_processor_with_diagnostics(
            config,
            app_base_dir,
            include_model_hash=True,
        )
    else:
        processor = processor_loader(config, app_base_dir)
    if processor is None:
        emit(
            "[portable-smoke] processor load failed; "
            f"diagnostics={get_runtime_log_path() or 'unavailable'}"
        )
        return finish(6)
    if config.roi_inspection_enabled and config.roi_template_enabled:
        from template_matching.shape_model import ShapeModel
        from yolo_runtime.template_gated_processor import TemplateGatedProcessor

        if not isinstance(processor, TemplateGatedProcessor):
            emit(
                "[portable-smoke] template gate processor not loaded: "
                f"actual={type(processor).__name__}"
            )
            return finish(6)
        model = processor.template_model
        if not isinstance(model, ShapeModel):
            emit(
                "[portable-smoke] shape model not loaded: "
                f"actual={type(model).__name__}"
            )
            return finish(6)
        emit(
            "[portable-smoke] template_gate=ok "
            f"processor={type(processor).__name__} model={type(model).__name__} "
            f"template={model.template_size} levels={model.level_count} "
            f"points={model.point_counts} angle_step={model.base_angle_step:.3f} "
            f"min_search_size={model.min_search_size(0)}"
        )
    emit(
        "[portable-smoke] processor="
        f"{type(processor).__name__} classes={processor.get_class_names()}"
    )
    class_names = processor.get_class_names()
    class_values = class_names.values() if isinstance(class_names, dict) else class_names
    config_errors = config.inspection_validation_errors(class_values)
    if config_errors:
        emit(f"[portable-smoke] inspection config invalid: {'; '.join(config_errors)}")
        return finish(1)

    pass_sound = _resolve_portable_sound("pass", Path(app_base_dir), resource_root)
    fail_sound = _resolve_portable_sound("fail", Path(app_base_dir), resource_root)
    emit(f"[portable-smoke] sound pass={pass_sound}")
    emit(f"[portable-smoke] sound fail={fail_sound}")
    if not pass_sound.exists() or not fail_sound.exists():
        emit("[portable-smoke] sound missing")
        return finish(3)

    evidence_dir = Path(app_base_dir) / "outputs" / "evidence"
    try:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        probe = evidence_dir / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        emit(f"[portable-smoke] evidence_dir={evidence_dir}")
    except OSError as exc:
        emit(f"[portable-smoke] evidence dir not writable: {exc}")
        return finish(4)

    try:
        from camera.mvsdk_camera import MvSdkCamera

        if MvSdkCamera.is_available():
            emit("[portable-smoke] mvsdk=available")
        else:
            emit(f"[portable-smoke] mvsdk=unavailable: {MvSdkCamera.load_error()}")
    except Exception as exc:
        emit(f"[portable-smoke] mvsdk=error: {exc}")

    try:
        require_theme_module()
        emit(f"[portable-smoke] theme={THEME_ID}")
    except RuntimeError as exc:
        emit(f"[portable-smoke] theme unavailable: {exc}")
        return finish(5)

    emit("[portable-smoke] ok")
    return finish(0)


def main():
    """应用主函数。"""
    paths = resolve_app_paths()
    base_dir = paths.base_dir
    config_path = paths.config_path

    runtime_log_path = configure_runtime_logger(base_dir)
    install_unhandled_exception_hook()
    if getattr(sys, "frozen", False):
        config_path = initialize_portable_config(paths.base_dir, paths.resource_dir)

    log_startup_context(
        base_dir=base_dir,
        resource_dir=paths.resource_dir,
        config_path=config_path,
        frozen=bool(getattr(sys, "frozen", False)),
    )
    logger = get_runtime_logger()

    if "--portable-smoke-test" in sys.argv:
        smoke_code = run_portable_smoke_check(
            config_path=config_path,
            app_base_dir=base_dir,
            resource_dir=paths.resource_dir,
        )
        logger.info("portable_smoke_exit=%s", smoke_code)
        close_runtime_logger()
        sys.exit(smoke_code)

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("YOLO OBB 动作检测系统")
    app.setOrganizationName("WingTech")
    apply_application_theme(app)

    # 加载配置
    config = ConfigManager()
    try:
        config.load(config_path)
    except ValueError as exc:
        logger.exception("config_load=failed path=%s", config_path)
        print(f"[config] 配置加载失败: {exc}")
        print("[config] 使用默认配置")
    else:
        logger.info(
            "config_load=ok model_task=%s configured_model=%s",
            config.model_task,
            config.yolo_model_path,
        )

    print(f"[config] path={config_path}")
    print(f"[config] model_task={config.model_task}")

    state = AppState()

    # 创建 YOLO 处理器
    processor = load_yolo_processor_with_diagnostics(config, app_base_dir=base_dir)
    if processor is not None:
        class_names = processor.get_class_names()
        print(f"[yolo] 模型加载成功，类别: {class_names}")
    else:
        print("[yolo] 模型加载失败，详情见运行诊断日志")

    # 创建步骤引擎
    step_engine = create_step_engine(config)

    window = MainWindow(
        config,
        state,
        processor,
        step_engine,
        resource_dir=paths.resource_dir,
        evidence_base_dir=base_dir / "outputs" / "evidence",
        processor_factory=create_yolo_processor,
        app_base_dir=base_dir,
        runtime_log_path=runtime_log_path,
    )
    window.show()
    app.processEvents()

    if processor:
        window.worker.set_frame_processor(processor)
        logger.info("frame_processor=assigned type=%s", type(processor).__name__)
        print(f"[runtime] YOLO OBB 处理器已设置")
    else:
        logger.error("frame_processor=unavailable")
        print("[runtime] 警告：无可用处理器")

    exit_code = app.exec()
    try:
        config.save(config_path)
        logger.info("config_save=ok path=%s", config_path)
    except Exception:
        logger.exception("config_save=failed path=%s", config_path)
        raise
    finally:
        logger.info("session_end exit_code=%s", exit_code)
        close_runtime_logger()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
