"""YOLO OBB 动作检测系统入口。"""
import multiprocessing
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path

APP_WINDOW_TITLE = "双翼视觉防错系统"
ERROR_ALREADY_EXISTS = 183
_instance_mutex = None


def gui_instance_lock_required(argv: list[str] | None = None) -> bool:
    """启动自检不占界面锁，避免诊断把正在用的程序当成重复打开。"""
    args = list(sys.argv if argv is None else argv)
    return "--portable-smoke-test" not in args


def this_process_owns_instance(handle, last_error: int) -> bool:
    """互斥量创建失败时仍允许启动；只有确认已有实例才拒绝。"""
    if not handle:
        return True
    return int(last_error) != ERROR_ALREADY_EXISTS


def _create_windows_mutex(name: str):
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    handle = kernel32.CreateMutexW(None, False, name)
    return handle, ctypes.get_last_error()


def _close_windows_handle(handle) -> None:
    if not handle or sys.platform != "win32":
        return
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle(handle)


def acquire_single_instance() -> bool:
    """同一台机器只留一个检测程序。本进程已经持有锁时再次调用仍然算成功。"""
    global _instance_mutex
    if sys.platform != "win32":
        return True
    if _instance_mutex:
        return True
    handle, last_error = _create_windows_mutex("Local\\WingTech.YOLOActionDetection")
    if not this_process_owns_instance(handle, last_error):
        _close_windows_handle(handle)
        return False
    _instance_mutex = handle
    return True


def activate_titled_window(title: str) -> bool:
    if sys.platform != "win32":
        return False
    import ctypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
    user32.FindWindowW.restype = ctypes.c_void_p
    hwnd = user32.FindWindowW(None, title)
    if not hwnd:
        return False
    user32.ShowWindow(hwnd, 9)
    user32.SetForegroundWindow(hwnd)
    return True


def _tell_user_already_running() -> None:
    """重复打开时用系统对话框结束，不再加载 Qt 和模型。"""
    activate_titled_window(APP_WINDOW_TITLE)
    if sys.platform != "win32":
        print("程序正在启动或已在运行。请使用已经打开的窗口。", file=sys.stderr)
        return
    import ctypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.MessageBoxW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
    user32.MessageBoxW.restype = ctypes.c_int
    user32.MessageBoxW(
        None,
        "程序正在启动或已在运行。请使用已经打开的窗口。",
        APP_WINDOW_TITLE,
        0x00000040,
    )


def _stop_if_another_instance() -> None:
    if gui_instance_lock_required() and not acquire_single_instance():
        _tell_user_already_running()
        raise SystemExit(0)


if __name__ == "__main__":
    # 必须赶在导入 YOLO、LSTM 之前。否则连点时第二个进程会先把模型加载起来。
    multiprocessing.freeze_support()
    _stop_if_another_instance()

from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import Qt

from config import ConfigManager
from app_state import AppState
from camera.camera_worker import CameraWorker
from detection_logging.action_trace import configure_action_trace
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
from roi_inspection.geometry import usable_normalized_rect
from template_matching.template_store import resolve_template_path
from step_sequence.step_sequence_engine import StepSequenceEngine, build_step_engine
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


def show_startup_notice() -> QWidget:
    """模型加载完成前先出现一个窗口，避免用户以为没点开而再点一次。"""
    notice = QWidget()
    notice.setWindowTitle(APP_WINDOW_TITLE)
    notice.setWindowFlags(
        Qt.WindowType.Window
        | Qt.WindowType.WindowTitleHint
        | Qt.WindowType.CustomizeWindowHint
    )
    notice.setFixedSize(460, 160)
    layout = QVBoxLayout(notice)
    label = QLabel("正在启动，请稍候。")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(label)
    notice.show()
    return notice


def save_live_config(window, config_path: Path | str) -> None:
    """退出时保存窗口当前这份配置。

    配置页保存会换成新实例，并已经写入磁盘。退出时如果再写启动时的旧实例，
    刚选中的动作模型文件夹会被盖回旧路径。
    """
    window.config.save(config_path)


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
        device="",
        half=config.ultralytics_half,
        tracker=config.ultralytics_tracker,
        max_det=300,
        track_persist=config.ultralytics_track_persist,
    )


def configure_runtime_thread_limits() -> None:
    """限制 OpenCV 内部线程池，避免在低核工控机上与 ONNX Runtime / Qt 界面争抢 CPU。"""
    import os
    import cv2

    cv_threads = max(1, min(2, (os.cpu_count() or 2) // 2))
    try:
        cv2.setNumThreads(cv_threads)
    except Exception:
        pass


def create_yolo_processor(config: ConfigManager, app_base_dir: Path | str | None = None):
    """从配置创建最终运行处理器，包括可选的模板门控。"""
    configure_runtime_thread_limits()
    base_dir = Path(app_base_dir) if app_base_dir is not None else resolve_app_paths().base_dir
    model_path = resolve_model_path(config, base_dir)
    model_path_text = str(model_path)
    if model_path.suffix.lower() == ".onnx":
        processor = OnnxObbProcessor(
            model_path=model_path_text,
            conf_threshold=config.yolo_conf_threshold,
            iou_threshold=config.yolo_iou_threshold,
            max_det=300,
            intra_op_threads=0,
            graph_optimization="all",
        )
        processor.load()
    else:
        processor = _create_pt_processor(config, model_path_text)
        processor.load()
    return _wrap_with_template_gate(processor, config, base_dir)


def template_gate_can_assemble(config: ConfigManager, app_base_dir: Path | str) -> bool:
    """模板开关开、搜索 ROI 有效、模板文件存在时才具备挂门控的条件。"""
    if not getattr(config, "roi_template_enabled", False):
        return False
    if usable_normalized_rect(
        getattr(config, "roi_template_search_rect", None),
        warn_label="搜索 ROI",
    ) is None:
        return False
    try:
        template_path = resolve_template_path(config, app_base_dir)
    except ValueError:
        return False
    return template_path.is_file()


def _wrap_with_template_gate(processor, config: ConfigManager, base_dir: Path):
    """齐全时挂门控；缺一层跳过；文件损坏才阻断。"""
    if not getattr(config, "roi_template_enabled", False):
        return processor
    search_rect = getattr(config, "roi_template_search_rect", None)
    if usable_normalized_rect(search_rect, warn_label="搜索 ROI") is None:
        return processor
    try:
        template_path = resolve_template_path(config, base_dir)
    except ValueError:
        return processor
    if not template_path.is_file():
        return processor

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
    """按当前配置创建步骤引擎。

    启动、保存配置和重建引擎都走这里，避免几处构造参数各自漂移。
    """
    return build_step_engine(config)


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
    emit(f"[portable-smoke] frozen={bool(getattr(sys, 'frozen', False))} sys_path={sys.path}")
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
    if template_gate_can_assemble(config, app_base_dir):
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

    from lstm_action.bundle import load_runtime_bundle, uses_lstm_sequence

    if uses_lstm_sequence(config):
        try:
            import numpy as np
            from lstm_action.stream import TimelineSession
            from lstm_workbench.model_registry import create_hands_tracker

            bundle_path = Path(str(config.lstm_bundle_path or ""))
            if not bundle_path.is_absolute():
                bundle_path = Path(app_base_dir) / bundle_path
            values = [name for _index, name in sorted(class_names.items())] if isinstance(class_names, dict) else list(class_names)
            bundle = load_runtime_bundle(
                bundle_path,
                category_names=list(config.category_names),
                yolo_path=str(model_path),
                yolo_conf=float(config.yolo_conf_threshold),
                yolo_class_names=values,
                step_counts=list(config.category_counts),
            )
            tracker = create_hands_tracker()
            try:
                if not tracker.available:
                    raise RuntimeError(tracker.error or "手部模型不可用")
                tracker.process(np.zeros((64, 64, 3), dtype=np.uint8))
                if not tracker.available or tracker.error:
                    raise RuntimeError(tracker.error or "手部前向失败")
            finally:
                tracker.close()
            session = TimelineSession(
                bundle,
                confirm_seconds=float(getattr(config, "action_switch_confirm_seconds", 0.5)),
            )
            result = session.observe_frame(
                source_frame=0, timestamp=0.0, detections=[], hands=[], frame_w=64, frame_h=64,
            )
            if result.get("fault"):
                raise RuntimeError(result["fault"])
            emit(
                f"[portable-smoke] lstm=ok bundle={bundle_path} "
                f"weights_sha256={bundle.weights_sha256} hands_sha256={bundle.contract['hands_sha256']} "
                f"time_base_fps={bundle.time_base_fps} phase={result.get('phase')} "
                f"hands_forward=ok torch_forward=ok frozen={bool(getattr(sys, 'frozen', False))}"
            )
        except Exception as exc:
            emit(f"[portable-smoke] lstm failed: {type(exc).__name__}: {exc}")
            return finish(7)

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
    multiprocessing.freeze_support()
    _stop_if_another_instance()

    paths = resolve_app_paths()
    base_dir = paths.base_dir
    config_path = paths.config_path

    runtime_log_path = configure_runtime_logger(base_dir)
    action_trace_path = configure_action_trace(base_dir)
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
    logger.info("action_trace=%s", action_trace_path)

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
    app.setApplicationName(APP_WINDOW_TITLE)
    app.setOrganizationName("WingTech")
    apply_application_theme(app)
    app.setQuitOnLastWindowClosed(False)
    startup_notice = show_startup_notice()
    app.processEvents()

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
    startup_notice.close()
    startup_notice.deleteLater()
    app.setQuitOnLastWindowClosed(True)
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
        save_live_config(window, config_path)
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
