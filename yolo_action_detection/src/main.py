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
from yolo_runtime.onnx_obb_processor import OnnxObbProcessor
from step_sequence.step_sequence_engine import StepSequenceEngine
from ui.main_window import MainWindow


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
    config_path = Path(base_dir) / "config.json"
    if config_path.exists():
        return config_path

    for bundled_config in (
        Path(resource_dir) / "config.json",
        Path(resource_dir) / "config" / "config.json",
        Path(resource_dir) / "config" / "config.example.json",
    ):
        if bundled_config.exists():
            try:
                shutil.copyfile(bundled_config, config_path)
            except OSError:
                pass
            break
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
    """从配置创建 YOLO OBB 处理器。"""
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
        return processor

    processor = _create_pt_processor(config, model_path_text)
    processor.load()
    return processor


def create_step_engine(config: ConfigManager) -> StepSequenceEngine:
    """按当前配置创建步骤引擎。"""
    return StepSequenceEngine(
        step_class_names=config.category_names,
        step_counts=getattr(config, "category_counts", None),
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

    model_path = resolve_model_path(config, app_base_dir)
    emit(f"[portable-smoke] config={config_path}")
    emit(f"[portable-smoke] model={model_path}")
    if not model_path.exists():
        emit("[portable-smoke] model missing")
        return finish(2)

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

    emit("[portable-smoke] ok")
    return finish(0)


def main():
    """应用主函数。"""
    paths = resolve_app_paths()
    base_dir = paths.base_dir
    config_path = paths.config_path
    if getattr(sys, "frozen", False):
        config_path = initialize_portable_config(paths.base_dir, paths.resource_dir)

    if "--portable-smoke-test" in sys.argv:
        sys.exit(run_portable_smoke_check(
            config_path=config_path,
            app_base_dir=base_dir,
            resource_dir=paths.resource_dir,
        ))

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("YOLO OBB 动作检测系统")
    app.setOrganizationName("WingTech")

    # 加载配置
    config = ConfigManager()
    try:
        config.load(config_path)
    except ValueError as exc:
        print(f"[config] 配置加载失败: {exc}")
        print("[config] 使用默认配置")

    print(f"[config] path={config_path}")
    print(f"[config] model_task={config.model_task}")

    state = AppState()

    # 创建 YOLO 处理器
    try:
        processor = create_yolo_processor(config, app_base_dir=base_dir)
        class_names = processor.get_class_names()
        print(f"[yolo] 模型加载成功，类别: {class_names}")
    except Exception as exc:
        print(f"[yolo] 模型加载失败: {exc}")
        processor = None

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
    )
    window.show()
    app.processEvents()

    if processor:
        window.worker.set_frame_processor(processor)
        print(f"[runtime] YOLO OBB 处理器已设置")
    else:
        print("[runtime] 警告：无可用处理器")

    exit_code = app.exec()
    config.save(config_path)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
