"""YOLO OBB 配置页。只包含 YOLO 相关配置。"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox,
    QGroupBox, QFileDialog, QMessageBox, QScrollArea, QSpinBox,
    QDoubleSpinBox,
)

from ui.runtime_ui_tokens import (
    PAGE_BG,
    TEXT_ACCENT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    checkbox_style,
    config_button_style,
    group_box_style,
    input_style,
    scroll_area_style,
    text_style,
)


class ConfigPage(QWidget):
    """配置页：YOLO 模型、tracker、步骤类别序列、稳定帧和结果反馈。"""

    back_clicked = Signal()
    stats_changed = Signal()
    config_saved = Signal()

    def __init__(self, config, stats_manager=None) -> None:
        super().__init__()
        self.config = config
        self._stats_manager = stats_manager
        self._build_ui()
        self._load_values()
        self.refresh_stats_display()

    def _build_ui(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(scroll_area_style())
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        header = QLabel("配置中心")
        header.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        header.setStyleSheet(text_style(TEXT_PRIMARY, size=20, weight=800))
        layout.addWidget(header)

        # 生产统计区域
        if self._stats_manager is not None:
            self._stats_group = QGroupBox("生产统计")
            self._stats_group.setStyleSheet(group_box_style())
            stats_layout = QGridLayout(self._stats_group)
            stats_layout.setContentsMargins(14, 18, 14, 14)
            stats_layout.setHorizontalSpacing(12)
            stats_layout.setVerticalSpacing(10)

            self._stats_start_label = QLabel("开始时间: --")
            self._stats_start_label.setStyleSheet(text_style(TEXT_SECONDARY, size=13, weight=600))
            stats_layout.addWidget(self._stats_start_label, 0, 0, 1, 4)

            self._stats_total_label = QLabel("总数: 0")
            self._stats_total_label.setStyleSheet(text_style(TEXT_PRIMARY, size=14, weight=700))
            stats_layout.addWidget(self._stats_total_label, 1, 0)
            self._stats_ok_label = QLabel("OK: 0")
            self._stats_ok_label.setStyleSheet(text_style("#ACFFC6", size=14, weight=700))
            stats_layout.addWidget(self._stats_ok_label, 1, 1)
            self._stats_ng_label = QLabel("NG: 0")
            self._stats_ng_label.setStyleSheet(text_style("#FF6A75", size=14, weight=700))
            stats_layout.addWidget(self._stats_ng_label, 1, 2)
            self._stats_yield_label = QLabel("良率: 0.0%")
            self._stats_yield_label.setStyleSheet(text_style(TEXT_ACCENT, size=14, weight=700))
            stats_layout.addWidget(self._stats_yield_label, 1, 3)

            self._reset_btn = QPushButton("归零并保存记录")
            self._reset_btn.setStyleSheet(config_button_style("danger"))
            self._reset_btn.setFixedWidth(180)
            self._reset_btn.clicked.connect(self._on_reset_stats)
            stats_layout.addWidget(self._reset_btn, 2, 0, 1, 4)

            layout.addWidget(self._stats_group)

        # 现场常用配置
        self._operator_group = QGroupBox("现场常用配置")
        self._operator_group.setStyleSheet(group_box_style())
        operator_layout = QGridLayout(self._operator_group)
        operator_layout.setContentsMargins(14, 18, 14, 14)
        operator_layout.setHorizontalSpacing(12)
        operator_layout.setVerticalSpacing(10)

        row = 0
        operator_layout.addWidget(self._section_label("模型与阈值"), row, 0, 1, 3)
        row += 1
        operator_layout.addWidget(self._label("YOLO 模型路径:"), row, 0)
        self._model_path_input = QLineEdit()
        operator_layout.addWidget(self._model_path_input, row, 1)
        self._select_model_btn = QPushButton("选择模型")
        self._select_model_btn.clicked.connect(self._on_select_model)
        self._select_model_btn.setStyleSheet(config_button_style("secondary"))
        self._select_model_btn.setFixedWidth(100)
        operator_layout.addWidget(self._select_model_btn, row, 2)

        row += 1
        operator_layout.addWidget(self._label("置信度阈值:"), row, 0)
        self._conf_input = QDoubleSpinBox()
        self._conf_input.setRange(0.0, 1.0)
        self._conf_input.setSingleStep(0.05)
        self._conf_input.setFixedWidth(100)
        operator_layout.addWidget(self._conf_input, row, 1)

        row += 1
        operator_layout.addWidget(self._label("IoU 阈值:"), row, 0)
        self._iou_input = QDoubleSpinBox()
        self._iou_input.setRange(0.0, 1.0)
        self._iou_input.setSingleStep(0.05)
        self._iou_input.setFixedWidth(100)
        operator_layout.addWidget(self._iou_input, row, 1)

        row += 1
        operator_layout.addWidget(self._label("PASS 稳定帧:"), row, 0)
        self._action_pass_frames_input = QSpinBox()
        self._action_pass_frames_input.setRange(1, 30)
        self._action_pass_frames_input.setFixedWidth(100)
        operator_layout.addWidget(self._action_pass_frames_input, row, 1)

        row += 1
        operator_layout.addWidget(self._label("NG 稳定帧:"), row, 0)
        self._action_ng_frames_input = QSpinBox()
        self._action_ng_frames_input.setRange(1, 30)
        self._action_ng_frames_input.setFixedWidth(100)
        operator_layout.addWidget(self._action_ng_frames_input, row, 1)

        row += 1
        operator_layout.addWidget(self._section_label("结果反馈"), row, 0, 1, 3)
        row += 1
        self._sound_feedback_cb = QCheckBox("启用声音提示")
        operator_layout.addWidget(self._sound_feedback_cb, row, 0, 1, 2)
        row += 1
        self._fail_evidence_cb = QCheckBox("错误时保存证据图片")
        operator_layout.addWidget(self._fail_evidence_cb, row, 0, 1, 2)

        row += 1
        operator_layout.addWidget(self._section_label("相机参数"), row, 0, 1, 3)
        row += 1
        operator_layout.addWidget(self._label("参数模式:"), row, 0)
        self._cam_mode_input = QComboBox()
        self._cam_mode_input.addItems(["preserve", "load_group", "load_file", "manual"])
        self._cam_mode_input.setToolTip("preserve=不覆盖 load_group=加载参数组 load_file=加载文件 manual=手动设置")
        operator_layout.addWidget(self._cam_mode_input, row, 1)

        row += 1
        operator_layout.addWidget(self._label("参数组编号:"), row, 0)
        self._cam_group_input = QSpinBox()
        self._cam_group_input.setRange(0, 16)
        self._cam_group_input.setFixedWidth(100)
        operator_layout.addWidget(self._cam_group_input, row, 1)

        row += 1
        operator_layout.addWidget(self._label("参数文件路径:"), row, 0)
        self._cam_file_input = QLineEdit()
        self._cam_file_input.setPlaceholderText("load_file 模式使用")
        operator_layout.addWidget(self._cam_file_input, row, 1, 1, 2)

        row += 1
        operator_layout.addWidget(self._label("手动曝光(us):"), row, 0)
        self._cam_exp_input = QSpinBox()
        self._cam_exp_input.setRange(100, 1000000)
        self._cam_exp_input.setFixedWidth(120)
        operator_layout.addWidget(self._cam_exp_input, row, 1)

        row += 1
        operator_layout.addWidget(self._section_label("动作类别序列"), row, 0, 1, 3)

        self._step_inputs = []
        for i in range(6):
            row += 1
            operator_layout.addWidget(self._label(f"步骤{i+1} 类别:"), row, 0)
            inp = QLineEdit()
            inp.setPlaceholderText("留空表示未配置")
            operator_layout.addWidget(inp, row, 1, 1, 2)
            self._step_inputs.append(inp)

        layout.addWidget(self._operator_group)

        self._advanced_toggle = QPushButton("▶ 展开高级配置")
        self._advanced_toggle.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; color: {TEXT_ACCENT}; "
            f"font-size: 14px; font-weight: 700; text-align: left; padding: 4px 0; }}"
            f"QPushButton:hover {{ color: #A0F4FF; }}"
        )
        self._advanced_toggle.setFixedHeight(28)
        self._advanced_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._advanced_toggle.clicked.connect(self._toggle_advanced)
        layout.addWidget(self._advanced_toggle)

        self._advanced_group = QGroupBox("高级配置")
        self._advanced_group.setStyleSheet(group_box_style())
        advanced_layout = QGridLayout(self._advanced_group)
        advanced_layout.setContentsMargins(14, 18, 14, 14)
        advanced_layout.setHorizontalSpacing(12)
        advanced_layout.setVerticalSpacing(10)

        row = 0
        advanced_layout.addWidget(self._section_label("推理与追踪"), row, 0, 1, 3)
        row += 1
        advanced_layout.addWidget(self._label("推理设备:"), row, 0)
        self._device_input = QLineEdit()
        self._device_input.setPlaceholderText("留空自动选择")
        advanced_layout.addWidget(self._device_input, row, 1)

        row += 1
        advanced_layout.addWidget(self._label("追踪器:"), row, 0)
        self._tracker_input = QComboBox()
        self._tracker_input.addItems(["bytetrack.yaml", "botsort.yaml"])
        advanced_layout.addWidget(self._tracker_input, row, 1)

        row += 1
        advanced_layout.addWidget(self._label("最大目标数:"), row, 0)
        self._max_det_input = QSpinBox()
        self._max_det_input.setRange(1, 1000)
        self._max_det_input.setFixedWidth(100)
        advanced_layout.addWidget(self._max_det_input, row, 1)

        row += 1
        self._track_persist_cb = QCheckBox("追踪保持 (track persist)")
        advanced_layout.addWidget(self._track_persist_cb, row, 0, 1, 2)

        row += 1
        advanced_layout.addWidget(self._section_label("动作判定配置"), row, 0, 1, 3)

        row += 1
        self._action_order_cb = QCheckBox("启用错序 NG（只检查未来未完成步骤）")
        advanced_layout.addWidget(self._action_order_cb, row, 0, 1, 2)

        row += 1
        advanced_layout.addWidget(self._section_label("显示与产能"), row, 0, 1, 3)
        row += 1
        advanced_layout.addWidget(self._label("间隔时间(秒):"), row, 0)
        self._cooldown_input = QDoubleSpinBox()
        self._cooldown_input.setRange(0.0, 30.0)
        self._cooldown_input.setSingleStep(0.5)
        self._cooldown_input.setFixedWidth(100)
        advanced_layout.addWidget(self._cooldown_input, row, 1)

        row += 1
        advanced_layout.addWidget(self._label("今日目标产能:"), row, 0)
        self._target_input = QSpinBox()
        self._target_input.setRange(0, 99999)
        self._target_input.setFixedWidth(100)
        advanced_layout.addWidget(self._target_input, row, 1)

        row += 1
        self._show_conf_cb = QCheckBox("显示置信度")
        advanced_layout.addWidget(self._show_conf_cb, row, 0)
        self._show_tid_cb = QCheckBox("显示 Track ID")
        advanced_layout.addWidget(self._show_tid_cb, row, 1)

        self._advanced_group.setVisible(False)
        layout.addWidget(self._advanced_group)

        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        self._save_btn = QPushButton("保存配置")
        self._save_btn.clicked.connect(self._on_save)
        self._save_btn.setStyleSheet(config_button_style("primary"))
        self._back_btn = QPushButton("返回主页")
        self._back_btn.clicked.connect(self.back_clicked.emit)
        self._back_btn.setStyleSheet(config_button_style("secondary"))
        btn_layout.addStretch(1)
        btn_layout.addWidget(self._save_btn)
        btn_layout.addWidget(self._back_btn)
        layout.addLayout(btn_layout)
        layout.addStretch(1)

        scroll.setWidget(container)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
        self.setStyleSheet(
            f"QWidget {{ background-color: {PAGE_BG}; color: {TEXT_PRIMARY}; }}"
            + input_style()
            + checkbox_style()
        )

    def _label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(text_style(TEXT_SECONDARY, size=14, weight=600))
        return label

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(text_style(TEXT_ACCENT, size=15, weight=800))
        return label

    def _toggle_advanced(self) -> None:
        visible = not self._advanced_group.isVisible()
        self._advanced_group.setVisible(visible)
        self._advanced_toggle.setText("▼ 收起高级配置" if visible else "▶ 展开高级配置")

    def _load_values(self) -> None:
        c = self.config
        self._model_path_input.setText(c.yolo_model_path)
        self._conf_input.setValue(c.yolo_conf_threshold)
        self._iou_input.setValue(c.yolo_iou_threshold)
        self._device_input.setText(c.ultralytics_device)
        idx = self._tracker_input.findText(c.ultralytics_tracker)
        if idx >= 0:
            self._tracker_input.setCurrentIndex(idx)
        self._max_det_input.setValue(c.ultralytics_max_det)
        self._track_persist_cb.setChecked(c.ultralytics_track_persist)
        for i, name in enumerate(c.category_names):
            if i < 6:
                self._step_inputs[i].setText(name)
        # 相机参数
        idx = self._cam_mode_input.findText(getattr(c, "camera_parameter_mode", "preserve"))
        if idx >= 0:
            self._cam_mode_input.setCurrentIndex(idx)
        self._cam_group_input.setValue(getattr(c, "camera_parameter_group", 0))
        self._cam_file_input.setText(getattr(c, "camera_parameter_file", ""))
        self._cam_exp_input.setValue(getattr(c, "camera_manual_exposure_us", 30000))
        # 动作判定
        self._action_pass_frames_input.setValue(getattr(c, "action_pass_stable_frames", 2))
        self._action_ng_frames_input.setValue(getattr(c, "action_ng_stable_frames", 2))
        self._action_order_cb.setChecked(getattr(c, "action_order_constraint_enabled", True))
        self._cooldown_input.setValue(c.round_cooldown_seconds)
        self._target_input.setValue(c.today_target_capacity)
        self._show_conf_cb.setChecked(c.show_confidence_overlay)
        self._show_tid_cb.setChecked(c.show_track_id_overlay)
        self._sound_feedback_cb.setChecked(getattr(c, "sound_feedback_enabled", True))
        self._fail_evidence_cb.setChecked(getattr(c, "fail_evidence_enabled", True))

    def _on_select_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO OBB 模型", "", "YOLO 模型 (*.pt *.onnx)"
        )
        if path:
            self._model_path_input.setText(path)

    def refresh_stats_display(self) -> None:
        """刷新生产统计区域显示。"""
        if self._stats_manager is None or not hasattr(self, "_stats_start_label"):
            return
        from datetime import datetime
        batch = self._stats_manager.batch
        dt = datetime.fromtimestamp(batch.started_at).strftime("%Y-%m-%d %H:%M:%S")
        self._stats_start_label.setText(f"开始时间: {dt}")
        self._stats_total_label.setText(f"总数: {batch.total}")
        self._stats_ok_label.setText(f"OK: {batch.ok}")
        self._stats_ng_label.setText(f"NG: {batch.ng}")
        self._stats_yield_label.setText(f"良率: {batch.yield_rate:.1f}%")

    def _on_reset_stats(self) -> None:
        """归零并保存记录：确认后调用统计管理器。"""
        if self._stats_manager is None:
            return
        reply = QMessageBox.question(
            self,
            "归零确认",
            "系统会先保存当前统计记录，再清零并开始新批次。\n确定要归零吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        result = self._stats_manager.reset_and_archive()
        if not result.success:
            QMessageBox.warning(self, "归零失败", f"保存记录失败，统计未清零：\n{result.error}")
            return

        self.refresh_stats_display()
        self.stats_changed.emit()
        if result.archived:
            QMessageBox.information(
                self, "归零成功",
                f"已保存生产记录到：\n{result.record_path}\n\n统计已清零，新批次已开始。"
            )
        else:
            QMessageBox.information(
                self, "归零完成",
                "当前批次无计数，已刷新开始时间。"
            )

    def _on_save(self) -> None:
        c = self.config
        c.yolo_model_path = self._model_path_input.text()
        c.yolo_conf_threshold = self._conf_input.value()
        c.yolo_iou_threshold = self._iou_input.value()
        c.ultralytics_device = self._device_input.text()
        c.ultralytics_tracker = self._tracker_input.currentText()
        c.ultralytics_max_det = self._max_det_input.value()
        c.ultralytics_track_persist = self._track_persist_cb.isChecked()
        c.category_names = [inp.text() for inp in self._step_inputs]
        # 相机参数
        c.camera_parameter_mode = self._cam_mode_input.currentText()
        c.camera_parameter_group = self._cam_group_input.value()
        c.camera_parameter_file = self._cam_file_input.text()
        c.camera_manual_exposure_us = self._cam_exp_input.value()
        # 动作判定
        c.action_pass_stable_frames = self._action_pass_frames_input.value()
        c.action_ng_stable_frames = self._action_ng_frames_input.value()
        c.action_order_constraint_enabled = self._action_order_cb.isChecked()
        c.round_cooldown_seconds = self._cooldown_input.value()
        c.today_target_capacity = self._target_input.value()
        c.show_confidence_overlay = self._show_conf_cb.isChecked()
        c.show_track_id_overlay = self._show_tid_cb.isChecked()
        c.sound_feedback_enabled = self._sound_feedback_cb.isChecked()
        c.fail_evidence_enabled = self._fail_evidence_cb.isChecked()
        try:
            c.validate()
            config_path = c.get_config_path()
            if not config_path:
                raise ValueError("配置路径不存在，无法保存。")
            c.save(config_path)
            self.config_saved.emit()
            QMessageBox.information(self, "保存成功", "配置已保存。")
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "配置错误", str(exc))
