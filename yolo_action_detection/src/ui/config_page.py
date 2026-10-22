"""YOLO OBB 分区配置页。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QComboBox,
    QFileDialog, QMessageBox, QScrollArea, QSpinBox, QDoubleSpinBox,
    QStackedWidget, QFrame,
)

from ui.runtime_ui_tokens import (
    CONFIG_NAV_WIDTH,
    FONT_MICRO,
    MONO_FAMILY,
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
from config import CURRENT_CONFIG_SCHEMA_VERSION
from roi_inspection.geometry import NormalizedRect
from ui.roi_editor import RoiEditorWidget, RoiKind
from template_matching.template_store import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPLATE_NAME,
    prepare_template_candidate,
    read_optional_bytes,
    replace_file_atomically,
    restore_file,
)
from template_matching.shape_model import ShapeModel, ShapeModelSettings


class ConfigPage(QWidget):
    """配置页：YOLO 模型、tracker、步骤类别序列、稳定帧和结果反馈。"""

    back_clicked = Signal()
    stats_changed = Signal()
    config_saved = Signal()
    camera_refresh_requested = Signal()

    def __init__(
        self,
        config,
        stats_manager=None,
        roi_frame_provider=None,
        config_committer=None,
    ) -> None:
        super().__init__()
        self.config = config
        self._stats_manager = stats_manager
        self._roi_frame_provider = roi_frame_provider
        self._config_committer = config_committer
        self._camera_devices = []
        self._camera_enumeration_error = ""
        self._saved_feedback_timer = QTimer(self)
        self._saved_feedback_timer.setSingleShot(True)
        self._saved_feedback_timer.timeout.connect(self._hide_saved_feedback)
        self._build_ui()
        self._load_values()
        self._sync_model_dependent_fields()
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
        brand_row = QWidget(sidebar)
        brand_row.setStyleSheet("background: transparent;")
        brand_layout = QHBoxLayout(brand_row)
        brand_layout.setContentsMargins(4, 0, 0, 0)
        brand_layout.setSpacing(8)
        brand_dot = QLabel(brand_row)
        brand_dot.setFixedSize(8, 8)
        brand_dot.setStyleSheet(f"background-color: {TEXT_ACCENT}; border: none; border-radius: 4px;")
        brand = QLabel("系统配置", brand_row)
        brand.setStyleSheet(text_style(TEXT_PRIMARY, size=20, weight=800))
        brand_layout.addWidget(brand_dot, 0, Qt.AlignmentFlag.AlignVCenter)
        brand_layout.addWidget(brand, 0, Qt.AlignmentFlag.AlignVCenter)
        brand_layout.addStretch(1)
        side_layout.addWidget(brand_row)
        subtitle = QLabel("视觉行为引导系统")
        subtitle.setStyleSheet(text_style(TEXT_MUTED, size=12, weight=500))
        subtitle.setContentsMargins(4, 0, 0, 0)
        side_layout.addWidget(subtitle)
        side_layout.addSpacing(18)

        section_titles = ["模型与步骤", "动作判定", "工业相机", "显示与反馈", "ROI 配置", "生产统计"]
        self._nav_buttons = []
        self._nav_index_labels = []
        for index, title in enumerate(section_titles):
            row = QWidget(sidebar)
            row.setStyleSheet("background: transparent;")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(9)
            index_label = QLabel(f"{index + 1:02d}", row)
            index_label.setStyleSheet(self._nav_index_style(False))
            button = ConfigNavButton(title, row)
            button.clicked.connect(lambda _checked=False, i=index: self._show_section(i))
            row_layout.addWidget(index_label, 0, Qt.AlignmentFlag.AlignVCenter)
            row_layout.addWidget(button, 1)
            side_layout.addWidget(row)
            self._nav_buttons.append(button)
            self._nav_index_labels.append(index_label)
        side_layout.addStretch(1)
        version = QLabel(f"CONFIG SCHEMA · v{CURRENT_CONFIG_SCHEMA_VERSION}")
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
        self._model_path_input.textChanged.connect(self._sync_model_dependent_fields)

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
        model_group = FormGroup("YOLO OBB 模型", number="01 / MODEL")
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
        self._tracker_input = QComboBox(); self._tracker_input.addItems(["bytetrack.yaml", "botsort.yaml"])
        self._add_form_grid(model_group, [
            (FormField("模型文件路径", path_row, "支持 .pt 与 .onnx，保存后将重新加载模型。"), True),
            (FormField("置信度阈值", self._conf_input), False),
            (FormField("IoU 阈值", self._iou_input), False),
            (FormField("跟踪器", self._tracker_input), False),
        ])
        persist_row = SwitchRow("持续跟踪", "跨帧保持目标 Track ID。")
        self._track_persist_cb = persist_row.checkbox
        model_group.content_layout.addWidget(persist_row)
        self._insert_group(layout, model_group)

        steps_group = FormGroup("类别步骤与数量", number="02 / STEPS")
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
        group = FormGroup("动作通过与异常判定", number="03 / JUDGE")
        self._action_pass_frames_input = QSpinBox(); self._action_pass_frames_input.setRange(1, 99999)
        self._action_ng_frames_input = QSpinBox(); self._action_ng_frames_input.setRange(1, 99999)
        self._cooldown_input = QDoubleSpinBox(); self._cooldown_input.setRange(0.0, 60.0); self._cooldown_input.setSingleStep(0.5)
        self._switch_confirm_input = QDoubleSpinBox()
        self._switch_confirm_input.setRange(0.1, 5.0)
        self._switch_confirm_input.setSingleStep(0.1)
        self._switch_confirm_input.setDecimals(1)
        self._add_form_grid(group, [
            (FormField("通过稳定帧数", self._action_pass_frames_input), False),
            (FormField("NG 稳定帧数", self._action_ng_frames_input), False),
            (FormField("完成一轮间隔（秒）", self._cooldown_input, "PASS 或 NG 后保持结果颜色，到间隔结束再开始新一轮。"), False),
            (FormField("动作切换确认（秒）", self._switch_confirm_input, "切到下一个动作后，新动作连续保持这么久，上一个动作就完成。"), False),
        ])
        order_row = SwitchRow("步骤顺序约束", "防止跳步或越序操作。")
        self._action_order_cb = order_row.checkbox
        group.content_layout.addWidget(order_row)
        lstm_row = SwitchRow("时序动作", "顺序约束开启且未启用 ROI 时，由连续 LSTM 判定完整动作。这个开关不再单独放行。")
        self._lstm_action_cb = lstm_row.checkbox
        self._lstm_action_cb.setEnabled(False)
        group.content_layout.addWidget(lstm_row)
        self._lstm_contract_input = QLineEdit()
        self._lstm_contract_input.setPlaceholderText("旧契约路径，仅兼容保留")
        self._lstm_bundle_input = QLineEdit()
        self._lstm_bundle_input.setPlaceholderText("选择训练出来的模型文件夹")
        self._select_action_bundle_btn = QPushButton("浏览")
        self._select_action_bundle_btn.setMinimumWidth(84)
        self._select_action_bundle_btn.clicked.connect(self._on_select_action_bundle)
        bundle_row = self._transparent_container()
        bundle_layout = QHBoxLayout(bundle_row)
        bundle_layout.setContentsMargins(0, 0, 0, 0)
        bundle_layout.setSpacing(8)
        bundle_layout.addWidget(self._lstm_bundle_input, 1)
        bundle_layout.addWidget(self._select_action_bundle_btn)
        self._add_form_grid(group, [
            (FormField("时序契约", self._lstm_contract_input), False),
            (FormField(
                "动作模型包",
                bundle_row,
                "点浏览选择训练文件夹。有手部、没手部都可以，程序按文件夹里的契约决定。",
            ), True),
        ])
        retry_row = SwitchRow(
            "类别/数量失败重试",
            "打开后复用上面的「NG 稳定帧数」：第 1 步到位后，当前步未检出或数量不符时连续累计，满则报 NG。",
        )
        self._fail_retry_cb = retry_row.checkbox
        group.content_layout.addWidget(retry_row)
        self._insert_group(layout, group)

    def _build_camera_section(self, layout: QVBoxLayout) -> None:
        group = FormGroup("工业相机参数", number="04 / CAMERA")
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
        self._cam_exp_input = QSpinBox(); self._cam_exp_input.setRange(100, 1000000)
        self._add_form_grid(group, [
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

    def _sync_model_dependent_fields(self) -> None:
        """按模型类型置灰无效参数。

        .onnx 走的是本项目自己的 ONNX Runtime 推理与后处理，Ultralytics 的
        跟踪器、持续跟踪都不参与，也不会产出 Track ID；留着可编辑只会让现场以为调了有用。
        """
        is_onnx = str(self._model_path_input.text()).strip().lower().endswith(".onnx")
        for widget in (self._tracker_input, self._track_persist_cb,
                       self._show_tid_cb):
            widget.setEnabled(not is_onnx)

    def _build_feedback_section(self, layout: QVBoxLayout) -> None:
        overlay_group = FormGroup("画面叠加", number="05 / OVERLAY")
        conf_row = SwitchRow("显示置信度", "在检测框标签中显示模型分数。")
        self._show_conf_cb = conf_row.checkbox
        tid_row = SwitchRow("显示 Track ID", "显示持续跟踪目标编号。")
        self._show_tid_cb = tid_row.checkbox
        overlay_group.content_layout.addWidget(conf_row)
        overlay_group.content_layout.addWidget(tid_row)
        self._insert_group(layout, overlay_group)

        feedback_group = FormGroup("操作反馈", number="06 / FEEDBACK")
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
        group = FormGroup("ROI 定位检测", number="08 / ROI")
        self._region_title_label = group.title_label
        enabled_row = SwitchRow(
            "启用 ROI 定位检测",
            "只切换认法：打开后不按顺序、按配置数量同时计数。关闭则按步骤顺序计数。已保存的 ROI 两种认法都会加载。",
        )
        self._roi_enabled_cb = enabled_row.checkbox
        self._roi_enabled_cb.stateChanged.connect(self._on_mode_changed)
        group.content_layout.addWidget(enabled_row)

        # 模板匹配定位。
        template_row = SwitchRow("启用模板匹配定位", "在搜索 ROI 内用边缘模板定位触发本轮，匹配成功才跑 YOLO 识别。")
        self._roi_template_cb = template_row.checkbox
        self._roi_template_cb.stateChanged.connect(self._sync_template_inputs_enabled)
        group.content_layout.addWidget(template_row)

        self._template_threshold_input = QDoubleSpinBox()
        self._template_threshold_input.setRange(0.01, 1.0)
        self._template_threshold_input.setSingleStep(0.05)
        self._template_threshold_input.setDecimals(2)
        self._template_angle_input = QDoubleSpinBox()
        self._template_angle_input.setRange(0.0, 180.0)
        self._template_angle_input.setSingleStep(1.0)
        self._template_angle_input.setDecimals(1)
        self._template_angle_step_input = QDoubleSpinBox()
        self._template_angle_step_input.setRange(0.0, 180.0)
        self._template_angle_step_input.setSingleStep(0.5)
        self._template_angle_step_input.setDecimals(1)
        self._add_form_grid(group, [
            (FormField(
                "匹配分数阈值",
                self._template_threshold_input,
                "得分≈可见轮廓比例：0.7 表示允许 30% 轮廓缺失。默认 0.7。",
            ), False),
            (FormField("旋转角度范围", self._template_angle_input, "允许的旋转角度±度数，默认 15。"), False),
            (FormField(
                "角度步进",
                self._template_angle_step_input,
                "0 表示按模板尺寸自动推算（推荐）。金字塔搜索下放宽角度范围几乎不增加耗时。",
            ), False),
        ])

        diag_group = FormGroup("模板匹配诊断（现场测试用）", number="09 / DIAG")
        diag_row = SwitchRow(
            "记录模板匹配诊断日志",
            "每帧记录逐角度得分曲线、定位抖动、耗时和失败原因，写入 logs/template_sessions。",
        )
        self._template_diag_cb = diag_row.checkbox
        self._template_diag_cb.stateChanged.connect(self._sync_template_inputs_enabled)
        diag_group.content_layout.addWidget(diag_row)
        shadow_row = SwitchRow(
            "并行计算候选算法得分",
            "同帧再算一次梯度方向匹配得分做 A/B 对照，不影响 PASS/NG 判定，会增加约 20-40ms 耗时。",
        )
        self._template_shadow_cb = shadow_row.checkbox
        diag_group.content_layout.addWidget(shadow_row)
        self._template_sample_input = QSpinBox()
        self._template_sample_input.setRange(0, 2000)
        self._add_form_grid(diag_group, [
            (FormField(
                "抽样保存帧数",
                self._template_sample_input,
                "按得分分层保存搜索域无损 PNG，供离线重放算法。0 表示不保存。",
            ), False),
        ])
        self._diag_status_label = QLabel("诊断日志：未开始")
        self._diag_status_label.setWordWrap(True)
        self._diag_status_label.setStyleSheet(text_style(TEXT_MUTED, size=12, weight=600))
        diag_group.content_layout.addWidget(self._diag_status_label)

        editor_group = FormGroup("静态画面与 ROI", number="07 / ROI EDITOR")
        toolbar = self._transparent_container(editor_group)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(8)
        self._refresh_roi_frame_btn = QPushButton("刷新画面")
        self._roi_kind_input = QComboBox()
        self._roi_kind_input.addItem("识别 ROI", RoiKind.INSPECTION.value)
        self._roi_kind_input.addItem("模板 ROI", RoiKind.LOCATOR.value)
        self._roi_kind_input.addItem("搜索 ROI", RoiKind.SEARCH.value)
        self._create_roi_btn = QPushButton("创建 ROI")
        self._delete_roi_btn = QPushButton("删除 ROI")
        self._refresh_roi_frame_btn.clicked.connect(self._refresh_roi_frame)
        self._roi_kind_input.currentIndexChanged.connect(self._on_roi_kind_changed)
        self._create_roi_btn.clicked.connect(self._create_selected_roi)
        self._delete_roi_btn.clicked.connect(self._delete_selected_roi)
        for button in (
            self._refresh_roi_frame_btn,
            self._roi_kind_input,
            self._create_roi_btn,
            self._delete_roi_btn,
        ):
            toolbar_layout.addWidget(button)
        toolbar_layout.addStretch(1)
        editor_group.content_layout.addWidget(toolbar)
        self._roi_frame_status = QLabel("没有可用画面，请先打开相机并刷新")
        self._roi_frame_status.setStyleSheet(text_style(TEXT_MUTED, size=12, weight=600))
        editor_group.content_layout.addWidget(self._roi_frame_status)
        self._roi_editor = RoiEditorWidget(editor_group)
        self._roi_editor.rect_changed.connect(self._on_roi_rect_changed)
        editor_group.content_layout.addWidget(self._roi_editor)
        roi_save_actions = self._transparent_container(editor_group)
        roi_save_layout = QHBoxLayout(roi_save_actions)
        roi_save_layout.setContentsMargins(0, 0, 0, 0)
        self._save_roi_btn = QPushButton("保存当前 ROI")
        self._save_roi_btn.clicked.connect(self._save_selected_roi)
        roi_save_layout.addWidget(self._save_roi_btn)
        roi_save_layout.addStretch(1)
        editor_group.content_layout.addWidget(roi_save_actions)
        self._on_roi_kind_changed(self._roi_kind_input.currentIndex())
        # 面板顺序对齐设计稿：编辑器 → 定位 → 诊断
        self._insert_group(layout, editor_group)
        self._insert_group(layout, group)
        self._insert_group(layout, diag_group)
        self._update_roi_controls()
        self._sync_template_inputs_enabled()

    def _build_stats_section(self, layout: QVBoxLayout) -> None:
        self._stats_group = FormGroup("生产统计", number="10 / STATS")
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
        self._insert_group(layout, self._stats_group)

        actions = FormGroup("批次操作", number="11 / BATCH")
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

    @staticmethod
    def _nav_index_style(active: bool) -> str:
        return text_style(
            TEXT_ACCENT if active else TEXT_MUTED,
            size=FONT_MICRO,
            weight=700,
            extra=f"font-family: '{MONO_FAMILY}'; letter-spacing: 0.08em;",
        )

    def _show_section(self, index: int) -> None:
        if not 0 <= index < self._section_stack.count():
            return
        self._section_stack.setCurrentIndex(index)
        for button_index, (button, index_label) in enumerate(
            zip(self._nav_buttons, self._nav_index_labels)
        ):
            active = button_index == index
            button.setChecked(active)
            index_label.setStyleSheet(self._nav_index_style(active))
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
        """ROI 模式关闭时保留配置值，但禁用专属参数。"""
        self._sync_template_inputs_enabled()
        self._update_roi_controls()

    def _sync_template_inputs_enabled(self) -> None:
        """模板定位关闭时灰掉模板相关输入框。"""
        roi_enabled = self._roi_enabled_cb.isChecked()
        template_enabled = roi_enabled and self._roi_template_cb.isChecked()
        self._roi_template_cb.setEnabled(roi_enabled)
        self._template_threshold_input.setEnabled(template_enabled)
        self._template_angle_input.setEnabled(template_enabled)
        self._template_angle_step_input.setEnabled(template_enabled)
        if hasattr(self, "_template_diag_cb"):
            diagnostics_on = template_enabled and self._template_diag_cb.isChecked()
            self._template_diag_cb.setEnabled(template_enabled)
            self._template_shadow_cb.setEnabled(diagnostics_on)
            self._template_sample_input.setEnabled(diagnostics_on)

    def _on_roi_rect_changed(self, _kind: str, _rect) -> None:
        self._update_roi_controls()

    def _save_selected_roi(self) -> None:
        """只保存下拉框当前选中的 ROI，不提交另外两种草稿。"""
        kind = self._selected_roi_kind()
        rect = self._roi_editor.roi_rect(kind)
        if rect is None:
            self._roi_frame_status.setText("请先创建并框选当前 ROI")
            return
        if kind is RoiKind.LOCATOR:
            self._save_template_roi(rect)
            return
        self._save_plain_roi(kind, rect)

    def _template_self_check_rect(self, model: ShapeModel, template_rect: NormalizedRect) -> dict:
        """回查窗口从模板 ROI 向四周扩到算法最小窗，不够则用整图。不用搜索 ROI。"""
        frame_h, frame_w = self._roi_editor.frame_shape[:2]
        x, y, width, height = template_rect.to_pixel_rect((frame_w, frame_h))
        need_w, need_h = model.min_search_size(2)
        extra_w = max(0, int(need_w - width))
        extra_h = max(0, int(need_h - height))
        x0 = max(0, int(round(x - extra_w / 2)))
        y0 = max(0, int(round(y - extra_h / 2)))
        x1 = min(frame_w, int(round(x + width + extra_w / 2)))
        y1 = min(frame_h, int(round(y + height + extra_h / 2)))
        if (x1 - x0) < need_w or (y1 - y0) < need_h:
            return {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
        return {
            "x": x0 / frame_w,
            "y": y0 / frame_h,
            "width": (x1 - x0) / frame_w,
            "height": (y1 - y0) / frame_h,
        }

    def _roi_field(self, kind: RoiKind) -> tuple[str, str]:
        return {
            RoiKind.SEARCH: ("roi_template_search_rect", "搜索 ROI"),
            RoiKind.INSPECTION: ("roi_inspection_rect", "识别 ROI"),
            RoiKind.LOCATOR: ("roi_template_rect", "模板 ROI"),
        }[kind]

    def _persist_roi_field(self, field_name: str, label: str, value) -> bool:
        try:
            config_path = self._roi_config_path()
        except ValueError as exc:
            QMessageBox.warning(self, "ROI 保存失败", str(exc))
            return False

        previous_config = config_path.read_bytes() if config_path.is_file() else None
        previous_schema = self.config.config_schema_version
        previous_rect = deepcopy(getattr(self.config, field_name))
        temp_config = config_path.with_name(f".{config_path.name}.tmp")
        try:
            self.config.config_schema_version = CURRENT_CONFIG_SCHEMA_VERSION
            setattr(self.config, field_name, value)
            self.config.save(temp_config)
            replace_file_atomically(temp_config, config_path)
        except Exception as exc:
            temp_config.unlink(missing_ok=True)
            restore_file(config_path, previous_config)
            self.config.config_schema_version = previous_schema
            setattr(self.config, field_name, previous_rect)
            QMessageBox.warning(self, "ROI 保存失败", f"{label} 保存失败：{exc}")
            return False
        return True

    def _save_plain_roi(self, kind: RoiKind, rect: NormalizedRect) -> None:
        field_name, label = self._roi_field(kind)
        if not self._persist_roi_field(field_name, label, rect.to_mapping()):
            return
        self._roi_frame_status.setText(f"{label} 已单独保存")
        self._show_saved_feedback()

    def _save_template_roi(self, rect: NormalizedRect) -> None:
        """只用模板 ROI 创建并自检模板，不依赖搜索或识别 ROI。"""
        template_img = self._roi_editor.capture_rect_frame(RoiKind.LOCATOR)
        if template_img is None or template_img.size == 0:
            self._roi_frame_status.setText("框选区域无效，请重新框选")
            return

        try:
            config_path = self._roi_config_path()
        except ValueError as exc:
            QMessageBox.warning(self, "模板保存失败", str(exc))
            return

        template_path = config_path.parent / DEFAULT_TEMPLATE_NAME
        model_path = config_path.parent / DEFAULT_MODEL_NAME
        previous_template = read_optional_bytes(template_path)
        previous_model = read_optional_bytes(model_path)
        previous_config = read_optional_bytes(config_path)
        tracked_fields = (
            "config_schema_version",
            "roi_template_path",
            "roi_template_model_path",
            "roi_template_match_threshold",
            "roi_template_angle_range",
            "roi_template_angle_step",
            "roi_template_rect",
        )
        previous_values = {
            name: deepcopy(getattr(self.config, name))
            for name in tracked_fields
        }
        temp_template = None
        temp_model = None
        temp_config = config_path.with_name(f".{config_path.name}.tmp")
        match_result = None

        try:
            settings = self._template_settings_from_inputs()
            static_frame = self._roi_editor.frame_copy()
            frame_h, frame_w = static_frame.shape[:2]
            temp_template, temp_model, candidate_model = prepare_template_candidate(
                template_img,
                template_path,
                model_path,
                settings,
                created_frame_size=(int(frame_w), int(frame_h)),
            )

            verify_rect = self._template_self_check_rect(candidate_model, rect)
            match_result = candidate_model.find(static_frame, verify_rect)
            if not match_result.matched:
                raise ValueError(
                    f"模板在模板 ROI 内回查得分 {match_result.score:.3f}，"
                    f"未达到阈值 {settings.min_score:.3f}。\n"
                    f"失败原因：{match_result.failure or '未知'}\n"
                    f"请检查框选区域是否包含清晰轮廓，或降低匹配分数阈值。"
                )
            replace_file_atomically(temp_template, template_path)
            temp_template = None
            replace_file_atomically(temp_model, model_path)
            temp_model = None

            self.config.roi_template_model_path = model_path.name
            self.config.config_schema_version = CURRENT_CONFIG_SCHEMA_VERSION
            self.config.roi_template_path = template_path.name
            self.config.roi_template_match_threshold = self._template_threshold_input.value()
            self.config.roi_template_angle_range = self._template_angle_input.value()
            self.config.roi_template_angle_step = self._template_angle_step_input.value()
            self.config.roi_template_rect = rect.to_mapping()
            self.config.save(temp_config)
            replace_file_atomically(temp_config, config_path)

            self._roi_editor.set_template_feedback(
                success=True,
                score=match_result.score,
                box=match_result.box,
                contours=(),
            )
            self._roi_frame_status.setText(
                self._template_saved_summary(candidate_model, match_result)
            )
            self._show_saved_feedback()
            self._update_roi_controls()
        except Exception as exc:
            if temp_template is not None:
                temp_template.unlink(missing_ok=True)
            if temp_model is not None:
                temp_model.unlink(missing_ok=True)
            temp_config.unlink(missing_ok=True)
            restore_file(template_path, previous_template)
            restore_file(model_path, previous_model)
            restore_file(config_path, previous_config)
            for name, value in previous_values.items():
                setattr(self.config, name, value)
            title, message = "模板保存失败", f"定位模板保存失败：{exc}"
            status = message
            self._roi_editor.set_template_feedback(
                success=False,
                score=match_result.score if match_result is not None else None,
                box=match_result.box if match_result is not None else None,
                contours=(),
                message=status,
            )
            self._roi_frame_status.setText(status)
            QMessageBox.warning(self, title, message)
            self._update_roi_controls()

    def _template_settings_from_inputs(self):
        """用当前界面上的参数构造形状模型设置。"""
        angle_range = abs(float(self._template_angle_input.value()))
        step = float(self._template_angle_step_input.value())
        return ShapeModelSettings(
            min_score=float(self._template_threshold_input.value()),
            angle_start=-angle_range,
            angle_extent=2.0 * angle_range,
            angle_step=step,
            polarity=str(getattr(self.config, "roi_template_polarity", "use") or "use"),
        )

    def _template_saved_summary(self, model, match_result) -> str:
        """保存成功后把回查得分和建议阈值显示出来。"""
        suggested = max(0.30, round(match_result.score - 0.25, 2))
        return (
            f"模板已保存：{model.template_size[0]}×{model.template_size[1]}，"
            f"{model.level_count} 层金字塔，角度步进 {model.base_angle_step:.2f}°"
            f" ｜ 在模板 ROI 内回查得分 {match_result.score:.3f}"
            f"（当前阈值 {model.settings.min_score:.2f}，建议不高于 {suggested:.2f}）"
        )

    def _roi_config_path(self) -> Path:
        config_path_text = str(self.config.get_config_path() or "").strip()
        if not config_path_text:
            raise ValueError("配置路径不存在，无法保存 ROI")
        return Path(config_path_text)

    def _show_saved_feedback(self) -> None:
        self._saved_label.setVisible(True)
        self._saved_feedback_timer.start(2200)

    def set_template_diagnostics_status(self, text: str, *, warning: bool = False) -> None:
        """显示当前模板诊断会话的落盘位置，便于现场取回日志。"""
        if not hasattr(self, "_diag_status_label"):
            return
        self._diag_status_label.setText(text or "诊断日志：未开始")
        color = "#FF9800" if warning else TEXT_MUTED
        self._diag_status_label.setStyleSheet(text_style(color, size=12, weight=600))

    def _load_values(self) -> None:
        c = self.config
        self._model_path_input.setText(c.yolo_model_path)
        self._conf_input.setValue(c.yolo_conf_threshold)
        self._iou_input.setValue(c.yolo_iou_threshold)
        idx = self._tracker_input.findText(c.ultralytics_tracker)
        if idx >= 0:
            self._tracker_input.setCurrentIndex(idx)
        self._track_persist_cb.setChecked(c.ultralytics_track_persist)
        for i, name in enumerate(c.category_names):
            if i < 6:
                self._step_inputs[i].setText(name)
        # 步骤数量
        counts = getattr(c, "category_counts", [1, 1, 1, 1, 1, 1])
        for i in range(6):
            val = counts[i] if i < len(counts) else 1
            self._step_count_inputs[i].setValue(_display_positive_int(val))
        # 相机参数
        self._cam_exp_input.setValue(getattr(c, "camera_manual_exposure_us", 30000))
        self.set_camera_devices(self._camera_devices, self._camera_enumeration_error)
        # 动作判定
        self._action_pass_frames_input.setValue(getattr(c, "action_pass_stable_frames", 2))
        self._action_ng_frames_input.setValue(getattr(c, "action_ng_stable_frames", 10))
        self._action_order_cb.setChecked(getattr(c, "action_order_constraint_enabled", True))
        self._lstm_action_cb.setChecked(bool(getattr(c, "lstm_action_enabled", False)))
        self._lstm_contract_input.setText(str(getattr(c, "lstm_contract_path", "") or ""))
        self._lstm_bundle_input.setText(str(getattr(c, "lstm_bundle_path", "") or ""))
        self._fail_retry_cb.setChecked(getattr(c, "action_fail_retry_enabled", False))
        self._cooldown_input.setValue(c.round_cooldown_seconds)
        self._switch_confirm_input.setValue(float(getattr(c, "action_switch_confirm_seconds", 0.5)))
        self._show_conf_cb.setChecked(c.show_confidence_overlay)
        self._show_tid_cb.setChecked(c.show_track_id_overlay)
        self._pass_sound_cb.setChecked(getattr(c, "pass_sound_enabled", False))
        self._fail_sound_cb.setChecked(getattr(c, "fail_sound_enabled", True))
        self._fail_evidence_cb.setChecked(getattr(c, "fail_evidence_enabled", True))
        # ROI 定位检测
        self._roi_enabled_cb.setChecked(getattr(c, "roi_inspection_enabled", False))
        self._roi_template_cb.setChecked(getattr(c, "roi_template_enabled", False))
        self._template_threshold_input.setValue(float(getattr(c, "roi_template_match_threshold", 0.6)))
        self._template_angle_input.setValue(float(getattr(c, "roi_template_angle_range", 15.0)))
        self._template_angle_step_input.setValue(float(getattr(c, "roi_template_angle_step", 1.0)))
        self._template_diag_cb.setChecked(bool(getattr(c, "template_diagnostics_enabled", False)))
        self._template_shadow_cb.setChecked(
            bool(getattr(c, "template_diagnostics_shadow_enabled", True))
        )
        self._template_sample_input.setValue(
            int(getattr(c, "template_diagnostics_sample_frames", 120))
        )
        for kind, value in (
            (RoiKind.LOCATOR, getattr(c, "roi_template_rect", None)),
            (RoiKind.SEARCH, getattr(c, "roi_template_search_rect", None)),
            (RoiKind.INSPECTION, getattr(c, "roi_inspection_rect", None)),
        ):
            if value:
                try:
                    self._roi_editor.set_rect(kind, NormalizedRect.from_mapping(value))
                except ValueError:
                    pass
        self._sync_action_quantity_inputs()
        self._action_order_cb.toggled.connect(self._sync_action_quantity_inputs)
        self._roi_enabled_cb.toggled.connect(self._sync_action_quantity_inputs)
        self._on_mode_changed()

    def _sync_action_quantity_inputs(self) -> None:
        action_mode = self._action_order_cb.isChecked() and not self._roi_enabled_cb.isChecked()
        for box in self._step_count_inputs:
            box.setEnabled(not action_mode)

    def _refresh_roi_frame(self) -> None:
        frame = self._roi_frame_provider() if callable(self._roi_frame_provider) else None
        self._roi_editor.set_frame(frame)
        if self._roi_editor.has_frame:
            height, width = self._roi_editor.frame_shape[:2]
            self._roi_frame_status.setText(f"静态画面：{width} × {height}")
        else:
            self._roi_frame_status.setText("没有可用画面，请先打开相机并刷新")
        self._update_roi_controls()

    def _create_default_roi(self, kind: RoiKind) -> None:
        if not self._roi_editor.has_frame:
            return
        self._roi_editor.set_active_kind(kind)
        if kind == RoiKind.LOCATOR:
            rect = NormalizedRect(0.10, 0.20, 0.30, 0.40)
        elif kind == RoiKind.SEARCH:
            rect = NormalizedRect(0.05, 0.10, 0.45, 0.80)
        else:
            rect = NormalizedRect(0.55, 0.10, 0.40, 0.80)
        self._roi_editor.set_rect(kind, rect)

    def _selected_roi_kind(self) -> RoiKind:
        return RoiKind(self._roi_kind_input.currentData())

    def _on_roi_kind_changed(self, _index: int) -> None:
        self._roi_editor.set_active_kind(self._selected_roi_kind())
        self._update_roi_controls()

    def _create_selected_roi(self) -> None:
        self._create_default_roi(self._selected_roi_kind())

    def _delete_selected_roi(self) -> None:
        kind = self._selected_roi_kind()
        field_name, label = self._roi_field(kind)
        self._roi_editor.delete_rect(kind)
        if getattr(self.config, field_name, None) is None:
            self._roi_frame_status.setText(f"{label} 已删除")
            self._update_roi_controls()
            return
        if self._persist_roi_field(field_name, label, None):
            self._roi_frame_status.setText(f"{label} 已删除并保存")
            self._show_saved_feedback()
        self._update_roi_controls()

    def _update_roi_controls(self) -> None:
        has_frame = hasattr(self, "_roi_editor") and self._roi_editor.has_frame
        if hasattr(self, "_create_roi_btn"):
            self._create_roi_btn.setEnabled(has_frame)
        if hasattr(self, "_delete_roi_btn"):
            selected_rect = self._roi_editor.roi_rect(self._selected_roi_kind())
            self._delete_roi_btn.setEnabled(selected_rect is not None)
        if hasattr(self, "_save_roi_btn"):
            selected_rect = self._roi_editor.roi_rect(self._selected_roi_kind())
            self._save_roi_btn.setEnabled(has_frame and selected_rect is not None)

    def _on_select_model(self) -> None:
        current = str(self._model_path_input.text()).strip()
        start = str(Path(current).parent) if current else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO OBB 模型", start, "YOLO 模型 (*.pt *.onnx)"
        )
        if path:
            self._model_path_input.setText(path)

    def _on_select_action_bundle(self) -> None:
        from lstm_action.bundle import bundle_directory_for_model_file

        current = str(self._lstm_bundle_input.text()).strip()
        start = current if current else ""
        path = QFileDialog.getExistingDirectory(self, "选择动作模型文件夹", start)
        if not path:
            return
        try:
            directory = bundle_directory_for_model_file(path)
        except ValueError as exc:
            QMessageBox.warning(self, "动作模型", str(exc))
            return
        self._lstm_bundle_input.setText(str(directory))

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
        c = deepcopy(self.config)
        try:
            self._write_form_into_config(c)
            camera_validation_error = self._camera_validation_error()
            if camera_validation_error:
                raise ValueError(camera_validation_error)
            if self._config_committer is None and (
                not c.roi_inspection_enabled
                and c.action_order_constraint_enabled
            ):
                from lstm_action.bundle import load_runtime_bundle

                if not str(c.lstm_bundle_path or "").strip():
                    raise ValueError("顺序模式需要动作模型包，不能按 YOLO 放行")
                load_runtime_bundle(
                    c.lstm_bundle_path,
                    category_names=list(c.category_names or []),
                    yolo_path=str(c.yolo_model_path or ""),
                    yolo_conf=float(c.yolo_conf_threshold),
                    step_counts=list(c.category_counts or []),
                )
            c.validate()
            c.require_valid_inspection_config(
                None,
                require_roi_rects=False,
            )
            config_path = c.get_config_path()
            if not config_path:
                raise ValueError("配置路径不存在，无法保存。")
            if self._config_committer is not None:
                self._config_committer(c)
                self.config = c
            else:
                c.save(config_path)
                self.config.__dict__.update(c.__dict__)
            self.config_saved.emit()
            self._saved_label.setVisible(True)
            self._saved_feedback_timer.start(2200)
        except Exception as exc:
            QMessageBox.warning(self, "配置错误", str(exc))

    def _write_form_into_config(self, c) -> None:
        c.yolo_model_path = self._model_path_input.text()
        c.yolo_conf_threshold = self._conf_input.value()
        c.yolo_iou_threshold = self._iou_input.value()
        c.ultralytics_tracker = self._tracker_input.currentText()
        c.ultralytics_track_persist = self._track_persist_cb.isChecked()
        c.category_names = [inp.text() for inp in self._step_inputs]
        c.category_counts = [inp.value() for inp in self._step_count_inputs]
        selected_camera = self._selected_camera_device()
        if selected_camera is not None:
            c.mvsdk_camera_sn = str(selected_camera.sn).strip()
            c.mvsdk_friendly_name = selected_camera.friendly_name
        c.camera_manual_exposure_us = self._cam_exp_input.value()
        c.action_pass_stable_frames = self._action_pass_frames_input.value()
        c.action_ng_stable_frames = self._action_ng_frames_input.value()
        c.action_order_constraint_enabled = self._action_order_cb.isChecked()
        c.lstm_action_enabled = self._lstm_action_cb.isChecked()
        c.lstm_contract_path = self._lstm_contract_input.text().strip()
        c.lstm_bundle_path = self._lstm_bundle_input.text().strip()
        c.action_fail_retry_enabled = self._fail_retry_cb.isChecked()
        c.round_cooldown_seconds = self._cooldown_input.value()
        c.action_switch_confirm_seconds = self._switch_confirm_input.value()
        c.show_confidence_overlay = self._show_conf_cb.isChecked()
        c.show_track_id_overlay = self._show_tid_cb.isChecked()
        c.pass_sound_enabled = self._pass_sound_cb.isChecked()
        c.fail_sound_enabled = self._fail_sound_cb.isChecked()
        c.fail_evidence_enabled = self._fail_evidence_cb.isChecked()
        c.config_schema_version = CURRENT_CONFIG_SCHEMA_VERSION
        c.roi_inspection_enabled = self._roi_enabled_cb.isChecked()
        c.roi_template_enabled = self._roi_template_cb.isChecked()
        c.roi_template_match_threshold = self._template_threshold_input.value()
        c.roi_template_angle_range = self._template_angle_input.value()
        c.roi_template_angle_step = self._template_angle_step_input.value()
        c.template_diagnostics_enabled = self._template_diag_cb.isChecked()
        c.template_diagnostics_shadow_enabled = self._template_shadow_cb.isChecked()
        c.template_diagnostics_sample_frames = self._template_sample_input.value()

    def _camera_validation_error(self) -> str:
        selected_camera = self._selected_camera_device()
        selected_index = self._camera_device_input.currentIndex()
        if selected_camera is None and selected_index > 0:
            return "所选迈德相机缺少 SN，无法保存为稳定设备"
        if selected_camera is None and self._camera_devices and not getattr(self.config, "mvsdk_camera_sn", ""):
            return "请选择一台迈德相机后再保存"
        return ""

    def _hide_saved_feedback(self) -> None:
        self._saved_label.setVisible(False)


def _display_positive_int(value) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 1
    return parsed if parsed > 0 else 1
