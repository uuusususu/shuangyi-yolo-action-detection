"""应用入口模块"""
import sys
import shutil
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from config import ConfigManager
from state import AppState
from mediapipe_object_processor import create_frame_processor_from_config
from ui.main_window import MainWindow


def main():
    """应用主函数"""
    # 高 DPI 支持
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("手部动作识别")
    app.setOrganizationName("HandGestureApp")
    
    # 加载配置
    if getattr(sys, "frozen", False):
        base_dir = Path(sys.executable).resolve().parent
        bundle_dir = Path(getattr(sys, "_MEIPASS", base_dir))
        config_path = base_dir / "config.json"
        bundled_config = bundle_dir / "config.json"
        if (not config_path.exists()) and bundled_config.exists():
            try:
                shutil.copyfile(bundled_config, config_path)
            except OSError:
                pass
    else:
        base_dir = Path(__file__).resolve().parents[1]
        config_path = base_dir / "config.json"
    config = ConfigManager()
    config.load(config_path)

    print(f"[config] path={config_path}")
    print(f"[config] debug_gesture_overlay={bool(getattr(config, 'debug_gesture_overlay', False))}")
    
    # 创建状态管理器
    state = AppState()
    
    # 创建主窗口
    window = MainWindow(config, state)
    window.show()
    app.processEvents()
    
    # 创建帧处理器
    frame_processor = create_frame_processor_from_config(config)
    
    # 设置帧处理器
    window.worker.set_frame_processor(frame_processor)
    window._sync_runtime_model_from_processor(frame_processor, save_config=True)
    print(
        f"[runtime] startup config={config.get_config_path()} model={config.get_model_path()} "
        f"mode={config.get_detection_mode_display_name()}"
    )

    # 运行应用
    exit_code = app.exec()
    
    # 保存配置
    config.save(config_path)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
