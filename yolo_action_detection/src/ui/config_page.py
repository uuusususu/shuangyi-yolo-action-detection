"""YOLO OBB 分区配置页。"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QComboBox,
    QFileDialog, QMessageBox, QScrollArea, QSpinBox, QDoubleSpinBox,
    QStackedWidget, QFrame,
)

from ui.runtime_ui_tokens import (
    CONFIG_NAV_WIDTH,
    PAGE_BG,
    PANEL_BG_DARK,
    STROKE_MAIN,
    TEXT_ACCENT,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    scroll_area_style,
    text_style,
)
from ui.widgets.native_panels import (
    ConfigNavButton,
    FooterActionBar,
    FormField,
    FormGroup,
    SwitchRow,
)


class ConfigPage(QWidget):
    """配置页：YOLO 模型、tracker、步骤类别序列、稳定帧和结果反馈。"""

    back_clicked = Signal()
    stats_changed = Signal()
    config_saved = Signal()
    camera_refresh_requested = Signal()

    def __init__(self, config, stats_manager=None) -> None:
        super().__init__()
        self.config = config
        self._stats_manager = stats_manager
        self._camera_devices = []
        self._camera_enumeration_error = ""
        self._saved_feedback_timer = QTimer(self)
        self._saved_feedback_timer.setSingleShot(True)
        self._saved_feedback_timer.timeout.connect(self._hide_saved_feedback)
        self._build_ui()
        self._load_values()
        self.refresh_stats_display()

    def _build_ui(self) -> None:
        self.setObjectName("configPage")
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar = QFrame(self)
        sidebar.setFixedWidth(CONFIG_NAV_WIDTH)
        sidebar.setStyleSheet(
            f"QFrame {{ background-color: {PANEL_BG_DARK}; border: none; border-right: 1px solid {STROKE_MAIN}; }}"
        )
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(16, 20, 16, 20)
        side_layout.setSpacing(8)
        self._sidebar_back_btn = QPushButton("返回检测")
        self._sidebar_back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._sidebar_back_btn.clicked.connect(self.back_clicked.emit)
        side_layout.addWidget(self._sidebar_back_btn)
        brand = QLabel("系统配置")
        brand.setStyleSheet(text_style(TEXT_PRIMARY, size=20, weight=800))
        side_layout.addWidget(brand)
        subtitle = QLabel("视觉行为引导系统")
        subtitle.setStyleSheet(text_style(TEXT_MUTED, size=12, weight=500))
        side_layout.addWidget(subtitle)
        side_layout.addSpacing(18)

        section_titles = ["模型与步骤", "动作判定", "工业相机", "显示与反馈", "区域检查", "生产统计"]
        self._nav_buttons = []
        for index, title in enumerate(section_titles):
            button = ConfigNavButton(title)
            button.clicked.connect(lambda _checked=False, i=index: self._show_section(i))
            side_layout.addWidget(button)
            self._nav_buttons.append(button)
        side_layout.addStretch(1)
        version = QLabel("CONFIG SCHEMA · v1")
        version.setStyleSheet(text_style(TEXT_MUTED, size=11, weight=500))
        side_layout.addWidget(version)
        root.addWidget(sidebar)

        content = QWidget(self)
        content.setObjectName("configPageContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        header = QFrame(content)
        header.setFixedHeight(92)
        header.setStyleSheet(f"QFrame {{ background: {PAGE_BG}; border: none; border-bottom: 1px solid {STROKE_MAIN}; }}")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(28, 16, 28, 14)
        title_layout = QVBoxLayout()
        title_layout.setSpacing(4)
        eyebrow = QLabel("CONFIGURATION")
        eyebrow.setStyleSheet(text_style(TEXT_ACCENT, size=11, weight=800))
        self._section_title_label = QLabel(section_titles[0])
        self._section_title_label.setStyleSheet(text_style(TEXT_PRIMARY, size=26, weight=800))
        title_layout.addWidget(eyebrow)
        title_layout.addWidget(self._section_title_label)
        header_layout.addLayout(title_layout)
        header_layout.addStretch(1)
        content_layout.addWidget(header)

        self._section_stack = QStackedWidget(content)
        self._section_stack.setStyleSheet("QStackedWidget { background: transparent; border: none; }")
        content_layout.addWidget(self._section_stack, 1)

        self._model_page, model_layout = self._create_section_page()
        self._action_page, action_layout = self._create_section_page()
        self._camera_page, camera_layout = self._create_section_page()
        self._feedback_page, feedback_layout = self._create_section_page()
        self._region_page, region_layout = self._create_section_page()
        self._stats_page, stats_layout = self._create_section_page()

        self._build_model_section(model_layout)
        self._build_action_section(action_layout)
        self._build_camera_section(camera_layout)
        self._build_feedback_section(feedback_layout)
        self._build_region_section(region_layout)
        self._build_stats_section(stats_layout)

        self._footer = FooterActionBar(content)
        self._save_btn = self._footer.save_button
        self._back_btn = self._footer.cancel_button
        self._saved_label = self._footer.saved_label
        self._save_btn.clicked.connect(self._on_save)
        self._back_btn.clicked.connect(self.back_clicked.emit)
        content_layout.addWidget(self._footer)
        root.addWidget(content, 1)

        self._apply_control_metrics()
        self.setStyleSheet(
            f"QWidget#configPage, QWidget#configPageContent {{ "
            f"background-color: {PAGE_BG}; color: {TEXT_PRIMARY}; }}"
        )
        self._show_section(0)

    def _apply_control_metrics(self) -> None:
        """Keep production-friendly hit sizes while the theme owns drawing."""
        for control_type in (QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox):
            for control in self.findChildren(control_type):
                control.setMinimumHeight(40)
        for button in self.findChildren(QPushButton):
            button.setMinimumHeight(40)

    def _create_section_page(self):
        scroll = QScrollArea(self._section_stack)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(scroll_area_style())
        page = QWidget(scroll)
        page.setObjectName(f"configSectionPage{self._section_stack.count()}")
        page.setStyleSheet(f"QWidget#{page.objectName()} {{ background: transparent; border: none; }}")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 22, 28, 28)
        layout.setSpacing(16)
        layout.addStretch(1)
        scroll.setWidget(page)
        self._section_stack.addWidget(scroll)
        return page, layout

    @staticmethod
    def _transparent_container(parent=None) -> QWidget:
        container = QWidget(parent)
        container.setObjectName("layoutContainer")
        container.setStyleSheet(
            "QWidget#layoutContainer { background: transparent; border: none; }"
        )
        return container

    def _insert_group(self, layout: QVBoxLayout, group: FormGroup) -> None:
        layout.insertWidget(layout.count() - 1, group)

    def _add_form_grid(self, group: FormGroup, fields, columns: int = 2) -> None:
        container = self._transparent_container(group)
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(14)
        row = 0
        column = 0
        for field, full_width in fields:
            if full_width:
                if column:
                    row += 1
                    column = 0
                grid.addWidget(field, row, 0, 1, columns)
                row += 1
                continue
            grid.addWidget(field, row, column)
            column += 1
            if column >= columns:
                row += 1
                column = 0
        for grid_column in range(columns):
            grid.setColumnStretch(grid_column, 1)
        group.content_layout.addWidget(container)

    def _build_model_section(self, layout: QVBoxLayout) -> None:
        model_group = FormGroup("YOLO OBB 模型")
        self._model_path_input = QLineEdit()
        self._select_model_btn = QPushButton("浏览")
        self._select_model_btn.setMinimumWidth(84)
        self._select_model_btn.clicked.connect(self._on_select_model)
        path_row = self._transparent_container()
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.setSpacing(8)
        path_layout.addWidget(self._model_path_input, 1)
        path_layout.addWidget(self._select_model_btn)
        self._conf_input = QDoubleSpinBox(); self._conf_input.setRange(0.0, 1.0); self._conf_input.setSingleStep(0.05)
        self._iou_input = QDoubleSpinBox(); self._iou_input.setRange(0.0, 1.0); self._iou_input.setSingleStep(0.05)
        self._device_input = QLineEdit(); self._device_input.setPlaceholderText("留空自动选择")
        self._tracker_input = QComboBox(); self._tracker_input.addItems(["bytetrack.yaml", "botsort.yaml"])
        self._max_det_input = QSpinBox(); self._max_det_input.setRange(1, 1000)
        self._add_form_grid(model_group, [
            (FormField("模型文件路径", path_row, "支持 .pt 与 .onnx，保存后将重新加载模型。"), True),
            (FormField("置信度阈值", self._conf_input), False),
            (FormField("IoU 阈值", self._iou_input), False),
            (FormField("推理设备", self._device_input), False),
            (FormField("跟踪器", self._tracker_input), False),
            (FormField("最大检测数量", self._max_det_input), False),
        ])
        persist_row = SwitchRow("持续跟踪", "跨帧保持目标 Track ID。")
        self._track_persist_cb = persist_row.checkbox
        model_group.content_layout.addWidget(persist_row)
        self._insert_group(layout, model_group)

        steps_group = FormGroup("类别步骤与数量")
        steps_container = self._transparent_container(steps_group)
        steps_grid = QGridLayout(steps_container)
        steps_grid.setContentsMargins(0, 0, 0, 0)
        steps_grid.setHorizontalSpacing(10)
        steps_grid.setVerticalSpacing(9)
        steps_grid.addWidget(self._label("顺序"), 0, 0)
        steps_grid.addWidget(self._label("类别名称"), 0, 1)
        steps_grid.addWidget(self._label("同帧目标数量"), 0, 2)
        self._step_inputs = []
        self._step_count_inputs = []
        for i in range(6):
            index_label = QLabel(f"STEP {i + 1:02d}")
            index_label.setStyleSheet(text_style(TEXT_ACCENT, size=12, weight=800))
            inp = QLineEdit(); inp.setPlaceholderText("留空表示未配置")
            count = QSpinBox(); count.setRange(1, 999); count.setValue(1); count.setMinimumWidth(110)
            steps_grid.addWidget(index_label, i + 1, 0)
            steps_grid.addWidget(inp, i + 1, 1)
            steps_grid.addWidget(count, i + 1, 2)
            self._step_inputs.append(inp)
            self._step_count_inputs.append(count)
        steps_grid.setColumnStretch(1, 1)
        steps_group.content_layout.addWidget(steps_container)
        self._insert_group(layout, steps_group)

    def _build_action_section(self, layout: QVBoxLayout) -> None:
        group = FormGroup("动作通过与异常判定")
        self._action_pass_frames_input = QSpinBox(); self._action_pass_frames_input.setRange(1, 30)
        self._action_ng_frames_input = QSpinBox(); self._action_ng_frames_input.setRange(1, 30)
        self._cooldown_input = QDoubleSpinBox(); self._cooldown_input.setRange(0.0, 60.0); self._cooldown_input.setSingleStep(0.5)
        self._add_form_grid(group, [
            (FormField("通过稳定帧数", self._action_pass_frames_input), False),
            (FormField("NG 稳定帧数", self._action_ng_frames_input), False),
            (FormField("完成一轮间隔（秒）", self._cooldown_input, "PASS 或 NG 后保持结果颜色，到间隔结束再开始新一轮。"), False),
        ])
        order_row = SwitchRow("步骤顺序约束", "防止跳步或越序操作。")
        self._action_order_cb = order_row.checkbox
        group.content_layout.addWidget(order_row)
        self._insert_group(layout, group)

    def _build_camera_section(self, layout: QVBoxLayout) -> None:
        group = FormGroup("工业相机参数")
        device_row = self._transparent_container(group)
        device_layout = QHBoxLayout(device_row)
        device_layout.setContentsMargins(0, 0, 0, 0)
        device_layout.setSpacing(10)
        self._camera_device_input = QComboBox(device_row)
        self._camera_device_input.setMinimumWidth(360)
        self._camera_refresh_btn = QPushButton("刷新设备", device_row)
        self._camera_refresh_btn.clicked.connect(self.camera_refresh_requested.emit)
        device_layout.addWidget(self._camera_device_input, 1)
        device_layout.addWidget(self._camera_refresh_btn)
        self._camera_device_status = QLabel("等待枚举迈德相机", group)
        self._camera_device_status.setWordWrap(True)
        self._camera_device_status.setStyleSheet(
            text_style(TEXT_MUTED, size=12, weight=600)
        )
        group.content_layout.addWidget(
            FormField(
                "迈德相机",
                device_row,
                "设备列表来自当前 SDK 枚举；保存后才应用相机切换。",
            )
        )
        group.content_layout.addWidget(self._camera_device_status)
        self._cam_mode_input = QComboBox(); self._cam_mode_input.addItems(["preserve", "load_group", "load_file", "manual"])
        self._cam_group_input = QSpinBox(); self._cam_group_input.setRange(0, 16)
        self._cam_file_input = QLineEdit(); self._cam_file_input.setPlaceholderText("load_file 模式使用")
        self._cam_exp_input = QSpinBox(); self._cam_exp_input.setRange(100, 1000000)
        self._add_form_grid(group, [
            (FormField("参数模式", self._cam_mode_input, "preserve 不覆盖；load_group / load_file 加载已有参数；manual 手动设置。"), False),
            (FormField("参数组编号", self._cam_group_input), False),
            (FormField("参数文件路径", self._cam_file_input), True),
            (FormField("手动曝光（us）", self._cam_exp_input), False),
        ])
        self._insert_group(layout, group)

    def set_camera_devices(self, devices, error: str = "") -> None:
        """Replace the in-memory MvSDK device catalog shown in the form."""
        pending_sn = str(self._camera_device_input.currentData() or "").strip()
        self._camera_devices = list(devices or [])
        self._camera_enumeration_error = str(error or "").strip()

        self._camera_device_input.clear()
        self._camera_device_input.addItem("请选择迈德相机", "")
        for device in self._camera_devices:
            self._camera_device_input.addItem(str(device), str(device.sn or "").strip())
            if not str(device.sn or "").strip():
                item = self._camera_device_input.model().item(
                    self._camera_device_input.count() - 1
                )
                if item is not None:
                    item.setEnabled(False)

        online_sns = {
            str(device.sn).strip()
            for device in self._camera_devices
            if str(device.sn or "").strip()
        }
        saved_sn = str(getattr(self.config, "mvsdk_camera_sn", "") or "").strip()
        target_sn = pending_sn if pending_sn in online_sns else ""
        if not target_sn and saved_sn in online_sns:
            target_sn = saved_sn

        legacy_name = str(
            getattr(self.config, "mvsdk_friendly_name", "") or ""
        ).strip()
        legacy_matches = []
        if not target_sn and not saved_sn and legacy_name:
            legacy_matches = [
                device
                for device in self._camera_devices
                if device.friendly_name == legacy_name and str(device.sn or "").strip()
            ]
            if len(legacy_matches) == 1:
                target_sn = legacy_matches[0].sn

        selected_index = self._camera_device_input.findData(target_sn)
        self._camera_device_input.setCurrentIndex(max(0, selected_index))

        if self._camera_enumeration_error:
            status = f"设备枚举失败：{self._camera_enumeration_error}"
            color = "#F44336"
        elif not self._camera_devices:
            status = "未发现在线迈德相机"
            color = TEXT_MUTED
        elif saved_sn and saved_sn not in online_sns:
            status = f"已保存相机不在线（SN: {saved_sn}），请重新选择"
            color = "#FF9800"
        elif not saved_sn and legacy_name and len(legacy_matches) > 1:
            status = f"发现多台名称为 {legacy_name} 的相机，请按 SN 选择"
            color = "#FF9800"
        else:
            valid_count = len(online_sns)
            status = f"发现 {valid_count} 台可选择相机"
            color = TEXT_SECONDARY
        self._camera_device_status.setText(status)
        self._camera_device_status.setStyleSheet(text_style(color, size=12, weight=600))

    def _selected_camera_device(self):
        selected_sn = str(self._camera_device_input.currentData() or "").strip()
        if not selected_sn:
            return None
        return next(
            (
                device
                for device in self._camera_devices
                if str(device.sn or "").strip() == selected_sn
            ),
            None,
        )

    def _build_feedback_section(self, layout: QVBoxLayout) -> None:
        overlay_group = FormGroup("画面叠加")
        conf_row = SwitchRow("显示置信度", "在检测框标签中显示模型分数。")
        self._show_conf_cb = conf_row.checkbox
        tid_row = SwitchRow("显示 Track ID", "显示持续跟踪目标编号。")
        self._show_tid_cb = tid_row.checkbox
        overlay_group.content_layout.addWidget(conf_row)
        overlay_group.content_layout.addWidget(tid_row)
        self._insert_group(layout, overlay_group)

        feedback_group = FormGroup("操作反馈")
        pass_sound_row = SwitchRow("PASS 提示音", "每个通过轮次最多播放一次提示音。")
        self._pass_sound_cb = pass_sound_row.checkbox
        fail_sound_row = SwitchRow("FAIL 提示音", "每个异常轮次最多播放一次提示音。")
        self._fail_sound_cb = fail_sound_row.checkbox
        evidence_row = SwitchRow("失败证据保存", "保存 NG 画面及本轮上下文。")
        self._fail_evidence_cb = evidence_row.checkbox
        feedback_group.content_layout.addWidget(pass_sound_row)
        feedback_group.content_layout.addWidget(fail_sound_row)
        feedback_group.content_layout.addWidget(evidence_row)
        self._insert_group(layout, feedback_group)

    def _build_region_section(self, layout: QVBoxLayout) -> None:
        group = FormGroup("首类别区域检查")
        self._region_title_label = group.title_label
        enabled_row = SwitchRow(
            "启用首类别区域检查",
            "类别 1 作为父类并创建追踪 ID；后续类别只在对应父类区域内按配置数量判定。FAIL 为本轮报警，补齐后可重试 PASS。",
        )
        self._first_category_region_cb = enabled_row.checkbox
        self._first_category_region_cb.stateChanged.connect(self._on_mode_changed)
        group.content_layout.addWidget(enabled_row)
        self._pcb_margin_input = QDoubleSpinBox(); self._pcb_margin_input.setRange(0.0, 1.0); self._pcb_margin_input.setSingleStep(0.05); self._pcb_margin_input.setValue(0.15)
        hint = QLabel("数量异常稳定帧数复用“动作判定 / NG 稳定帧数”；PASS 才是父类 ID 的终局完成。")
        hint.setWordWrap(True)
        hint.setStyleSheet(text_style(TEXT_MUTED, size=12, weight=600))
        group.content_layout.addWidget(hint)
        self._add_form_grid(group, [
            (FormField("归属边距比例", self._pcb_margin_input, "扩大父区域归属范围的比例。"), False),
        ])
        self._insert_group(layout, group)

    def _build_stats_section(self, layout: QVBoxLayout) -> None:
        self._stats_group = FormGroup("生产统计")
        stats_row = self._transparent_container(self._stats_group)
        stats_grid = QGridLayout(stats_row)
        stats_grid.setContentsMargins(0, 0, 0, 0)
        self._stats_start_label = QLabel("开始时间: --")
        self._stats_total_label = QLabel("总数: 0")
        self._stats_ok_label = QLabel("OK: 0")
        self._stats_ng_label = QLabel("NG: 0")
        self._stats_yield_label = QLabel("良率: 0.0%")
        for label in [self._stats_start_label, self._stats_total_label, self._stats_ok_label, self._stats_ng_label, self._stats_yield_label]:
            label.setStyleSheet(text_style(TEXT_SECONDARY, size=14, weight=700))
        stats_grid.addWidget(self._stats_start_label, 0, 0, 1, 4)
        stats_grid.addWidget(self._stats_total_label, 1, 0)
        stats_grid.addWidget(self._stats_ok_label, 1, 1)
        stats_grid.addWidget(self._stats_ng_label, 1, 2)
        stats_grid.addWidget(self._stats_yield_label, 1, 3)
        self._stats_group.content_layout.addWidget(stats_row)
        self._target_input = QSpinBox(); self._target_input.setRange(0, 99999)
        self._stats_group.content_layout.addWidget(FormField("今日目标产能", self._target_input))
        self._insert_group(layout, self._stats_group)

        actions = FormGroup("批次操作")
        hint = QLabel("归零前先保存当前生产记录，再开始新批次。无计数时只刷新批次开始时间。")
        hint.setWordWrap(True)
        hint.setStyleSheet(text_style(TEXT_MUTED, size=13, weight=500))
        actions.content_layout.addWidget(hint)
        self._reset_btn = QPushButton("归零并归档")
        self._reset_btn.setProperty("buttonRole", "danger")
        self._reset_btn.setEnabled(self._stats_manager is not None)
        self._reset_btn.clicked.connect(self._on_reset_stats)
        actions.content_layout.addWidget(self._reset_btn, 0, Qt.AlignmentFlag.AlignLeft)
        self._insert_group(layout, actions)

    def _show_section(self, index: int) -> None:
        if not 0 <= index < self._section_stack.count():
            return
        self._section_stack.setCurrentIndex(index)
        for button_index, button in enumerate(self._nav_buttons):
            button.setChecked(button_index == index)
        self._section_title_label.setText(self._nav_buttons[index].text())

    def _label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(text_style(TEXT_SECONDARY, size=14, weight=600))
        return label

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(text_style(TEXT_ACCENT, size=15, weight=800))
        return label

    def _on_mode_changed(self) -> None:
        """区域检查只复用类别顺序，开关关闭时禁用其专属判定参数。"""
        enabled = self._first_category_region_cb.isChecked()
        self._pcb_margin_input.setEnabled(enabled)

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
        # 步骤数量
        counts = getattr(c, "category_counts", [1, 1, 1, 1, 1, 1])
        for i in range(6):
            val = counts[i] if i < len(counts) else 1
            self._step_count_inputs[i].setValue(val)
        # 相机参数
        idx = self._cam_mode_input.findText(getattr(c, "camera_parameter_mode", "preserve"))
        if idx >= 0:
            self._cam_mode_input.setCurrentIndex(idx)
        self._cam_group_input.setValue(getattr(c, "camera_parameter_group", 0))
        self._cam_file_input.setText(getattr(c, "camera_parameter_file", ""))
        self._cam_exp_input.setValue(getattr(c, "camera_manual_exposure_us", 30000))
        self.set_camera_devices(self._camera_devices, self._camera_enumeration_error)
        # 动作判定
        self._action_pass_frames_input.setValue(getattr(c, "action_pass_stable_frames", 2))
        self._action_ng_frames_input.setValue(getattr(c, "action_ng_stable_frames", 10))
        self._action_order_cb.setChecked(getattr(c, "action_order_constraint_enabled", True))
        self._cooldown_input.setValue(c.round_cooldown_seconds)
        self._target_input.setValue(c.today_target_capacity)
        self._show_conf_cb.setChecked(c.show_confidence_overlay)
        self._show_tid_cb.setChecked(c.show_track_id_overlay)
        self._pass_sound_cb.setChecked(getattr(c, "pass_sound_enabled", False))
        self._fail_sound_cb.setChecked(getattr(c, "fail_sound_enabled", True))
        self._fail_evidence_cb.setChecked(getattr(c, "fail_evidence_enabled", True))
        # 首类别区域检查
        self._first_category_region_cb.setChecked(getattr(c, "first_category_region_check_enabled", False))
        self._pcb_margin_input.setValue(getattr(c, "pcb_assignment_margin_ratio", 0.15))
        self._on_mode_changed()

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
        c.category_counts = [inp.value() for inp in self._step_count_inputs]
        # 相机参数
        selected_camera = self._selected_camera_device()
        selected_index = self._camera_device_input.currentIndex()
        camera_validation_error = ""
        if selected_camera is not None:
            c.mvsdk_camera_sn = str(selected_camera.sn).strip()
            c.mvsdk_friendly_name = selected_camera.friendly_name
        elif selected_index > 0:
            camera_validation_error = "所选迈德相机缺少 SN，无法保存为稳定设备"
        elif self._camera_devices and not getattr(c, "mvsdk_camera_sn", ""):
            camera_validation_error = "请选择一台迈德相机后再保存"
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
        c.pass_sound_enabled = self._pass_sound_cb.isChecked()
        c.fail_sound_enabled = self._fail_sound_cb.isChecked()
        c.fail_evidence_enabled = self._fail_evidence_cb.isChecked()
        # 首类别区域检查；旧 pcb_* 字段仅保留加载兼容，不再作为用户入口。
        c.first_category_region_check_enabled = self._first_category_region_cb.isChecked()
        c.pcb_inspection_enabled = False
        c.pcb_class_name = ""
        c.pcb_component_class_names = []
        # 兼容字段同步到统一 NG 稳定帧数；首类别模式实际读取 action_ng_stable_frames。
        c.pcb_fail_stable_frames = c.action_ng_stable_frames
        # 兼容字段与统一轮次间隔保持一致，不再提供第二个用户配置入口。
        c.pcb_round_interval_seconds = c.round_cooldown_seconds
        c.pcb_assignment_margin_ratio = self._pcb_margin_input.value()
        try:
            if camera_validation_error:
                raise ValueError(camera_validation_error)
            c.validate()
            config_path = c.get_config_path()
            if not config_path:
                raise ValueError("配置路径不存在，无法保存。")
            c.save(config_path)
            self.config_saved.emit()
            self._saved_label.setVisible(True)
            self._saved_feedback_timer.start(2200)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "配置错误", str(exc))

    def _hide_saved_feedback(self) -> None:
        self._saved_label.setVisible(False)
