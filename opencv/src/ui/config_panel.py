"""配置面板模块。"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ConfigManager


class ConfigPanel(QWidget):
    """参数配置面板。"""

    config_changed = Signal(str, object)

    def __init__(self, config: ConfigManager, parent=None):
        super().__init__(parent)
        self.config = config
        self._setup_ui()
        self._load_values()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("参数配置")
        group_layout = QVBoxLayout(group)

        hands_layout = QHBoxLayout()
        hands_label = QLabel("最大手数量:")
        self.hands_spinbox = QSpinBox()
        self.hands_spinbox.setRange(1, 2)
        self.hands_spinbox.setToolTip("同时检测的最大手数量 (1-2)")
        hands_layout.addWidget(hands_label)
        hands_layout.addWidget(self.hands_spinbox)
        hands_layout.addStretch()

        detection_layout = QVBoxLayout()
        detection_header = QHBoxLayout()
        detection_label = QLabel("检测置信度:")
        self.detection_value_label = QLabel("50%")
        detection_header.addWidget(detection_label)
        detection_header.addStretch()
        detection_header.addWidget(self.detection_value_label)

        self.detection_slider = QSlider(Qt.Orientation.Horizontal)
        self.detection_slider.setRange(0, 100)
        self.detection_slider.setToolTip("最小检测置信度 (0-100%)")

        detection_layout.addLayout(detection_header)
        detection_layout.addWidget(self.detection_slider)

        tracking_layout = QVBoxLayout()
        tracking_header = QHBoxLayout()
        tracking_label = QLabel("跟踪置信度:")
        self.tracking_value_label = QLabel("50%")
        tracking_header.addWidget(tracking_label)
        tracking_header.addStretch()
        tracking_header.addWidget(self.tracking_value_label)

        self.tracking_slider = QSlider(Qt.Orientation.Horizontal)
        self.tracking_slider.setRange(0, 100)
        self.tracking_slider.setToolTip("最小跟踪置信度 (0-100%)")

        tracking_layout.addLayout(tracking_header)
        tracking_layout.addWidget(self.tracking_slider)

        group_layout.addLayout(hands_layout)
        group_layout.addSpacing(10)
        group_layout.addLayout(detection_layout)
        group_layout.addSpacing(10)
        group_layout.addLayout(tracking_layout)

        layout.addWidget(group)

    def _load_values(self) -> None:
        self.hands_spinbox.setValue(self.config.max_num_hands)

        detection_percent = int(self.config.min_detection_confidence * 100)
        self.detection_slider.setValue(detection_percent)
        self.detection_value_label.setText(f"{detection_percent}%")

        tracking_percent = int(self.config.min_tracking_confidence * 100)
        self.tracking_slider.setValue(tracking_percent)
        self.tracking_value_label.setText(f"{tracking_percent}%")

    def _connect_signals(self) -> None:
        self.hands_spinbox.valueChanged.connect(self._on_hands_changed)
        self.detection_slider.valueChanged.connect(self._on_detection_changed)
        self.tracking_slider.valueChanged.connect(self._on_tracking_changed)

    def _on_hands_changed(self, value: int) -> None:
        self.config.max_num_hands = value
        self.config.save()
        self.config_changed.emit("max_num_hands", value)

    def _on_detection_changed(self, value: int) -> None:
        self.detection_value_label.setText(f"{value}%")
        confidence = value / 100.0
        self.config.min_detection_confidence = confidence
        self.config.save()
        self.config_changed.emit("min_detection_confidence", confidence)

    def _on_tracking_changed(self, value: int) -> None:
        self.tracking_value_label.setText(f"{value}%")
        confidence = value / 100.0
        self.config.min_tracking_confidence = confidence
        self.config.save()
        self.config_changed.emit("min_tracking_confidence", confidence)
