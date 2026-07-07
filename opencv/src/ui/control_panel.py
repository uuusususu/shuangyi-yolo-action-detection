"""控制面板模块"""
from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QVBoxLayout, QGroupBox, QComboBox, QLabel
from PySide6.QtCore import Signal


class ControlPanel(QWidget):
    """控制按钮面板
    
    包含相机和检测控制按钮。
    """
    
    # 信号定义
    start_camera_clicked = Signal()
    stop_camera_clicked = Signal()
    refresh_cameras_clicked = Signal()
    camera_selected = Signal(object)
    start_inference_clicked = Signal()
    stop_inference_clicked = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._camera_on = False
        self._has_cameras = True
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self) -> None:
        """设置 UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 相机控制组
        camera_group = QGroupBox("相机控制")
        camera_layout = QVBoxLayout(camera_group)

        camera_select_layout = QHBoxLayout()
        camera_select_layout.addWidget(QLabel("相机:"))

        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(160)
        camera_select_layout.addWidget(self.camera_combo)

        self.btn_refresh_cameras = QPushButton("刷新")
        camera_select_layout.addWidget(self.btn_refresh_cameras)

        camera_layout.addLayout(camera_select_layout)

        camera_btn_layout = QHBoxLayout()
        
        self.btn_start_camera = QPushButton("开启相机")
        self.btn_stop_camera = QPushButton("停止相机")
        self.btn_stop_camera.setEnabled(False)
        
        camera_btn_layout.addWidget(self.btn_start_camera)
        camera_btn_layout.addWidget(self.btn_stop_camera)
        camera_layout.addLayout(camera_btn_layout)
        
        # 检测控制组
        detection_group = QGroupBox("检测控制")
        detection_layout = QHBoxLayout(detection_group)
        
        self.btn_start_inference = QPushButton("开始检测")
        self.btn_stop_inference = QPushButton("停止检测")
        self.btn_start_inference.setEnabled(False)
        self.btn_stop_inference.setEnabled(False)
        
        detection_layout.addWidget(self.btn_start_inference)
        detection_layout.addWidget(self.btn_stop_inference)
        
        layout.addWidget(camera_group)
        layout.addWidget(detection_group)
    
    def _connect_signals(self) -> None:
        """连接信号"""
        self.btn_start_camera.clicked.connect(self.start_camera_clicked.emit)
        self.btn_stop_camera.clicked.connect(self.stop_camera_clicked.emit)
        self.btn_refresh_cameras.clicked.connect(self.refresh_cameras_clicked.emit)
        self.btn_start_inference.clicked.connect(self.start_inference_clicked.emit)
        self.btn_stop_inference.clicked.connect(self.stop_inference_clicked.emit)

        self.camera_combo.currentIndexChanged.connect(self._on_camera_combo_changed)

    def _on_camera_combo_changed(self, _index: int) -> None:
        data = self.camera_combo.currentData()
        if data is not None and data != -1:
            self.camera_selected.emit(data)

    def set_camera_list(self, items: list[tuple[str, object]], *, selected_value: object) -> None:
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        if not items:
            self._has_cameras = False
            self.camera_combo.addItem("未检测到相机", -1)
        else:
            self._has_cameras = True
            for text, idx in items:
                self.camera_combo.addItem(text, idx)
            for i in range(self.camera_combo.count()):
                if self.camera_combo.itemData(i) == selected_value:
                    self.camera_combo.setCurrentIndex(i)
                    break
        self.camera_combo.blockSignals(False)
        self.update_camera_state(self._camera_on)
    
    def update_camera_state(self, camera_on: bool) -> None:
        """更新相机状态对应的按钮"""
        self._camera_on = camera_on
        self.btn_start_camera.setEnabled(not camera_on)
        self.btn_stop_camera.setEnabled(camera_on)

        self.camera_combo.setEnabled((not camera_on) and self._has_cameras)
        self.btn_refresh_cameras.setEnabled(not camera_on)

        if not self._has_cameras:
            self.btn_start_camera.setEnabled(False)
        
        # 相机关闭时禁用检测按钮
        if not camera_on:
            self.btn_start_inference.setEnabled(False)
            self.btn_stop_inference.setEnabled(False)
        else:
            self.btn_start_inference.setEnabled(True)
    
    def update_inference_state(self, inference_on: bool) -> None:
        """更新推理状态对应的按钮"""
        self.btn_start_inference.setEnabled(not inference_on)
        self.btn_stop_inference.setEnabled(inference_on)
