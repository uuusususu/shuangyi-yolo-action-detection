"""双翼视觉防错系统 — 五分区系统设置页。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from config import CURRENT_CONFIG_SCHEMA_VERSION
from roi_inspection.geometry import NormalizedRect
from template_matching.shape_model import ShapeModel, ShapeModelSettings
from template_matching.template_store import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPLATE_NAME,
    prepare_template_candidate,
    read_optional_bytes,
    replace_file_atomically,
    restore_file,
)
from ui.roi_editor import RoiEditorWidget, RoiKind
from ui.runtime_ui_tokens import (
    CONFIG_FORM_MAX_WIDTH,
    CONFIG_NAV_WIDTH,
    FONT_MICRO,
    FONT_SMALL,
    MONO_FAMILY,
    PAGE_BG,
    PANEL_BG_DARK,
    STROKE_SUBTLE,
    TEXT_ACCENT,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    scroll_area_style,
    text_style,
)
from ui.widgets.native_panels import (
    CollapsibleSection,
    ConfigNavButton,
    FooterActionBar,
    FormField,
    FormGroup,
    KpiRow,
    SwitchRow,
)


_SECTION_META = (
    ("检测方案", "选择检测模式、配置视觉与动作模型及作业目标清单"),
    ("判定规则", "调整防抖确认帧数、结果保持时长与错序报警策略"),
    ("区域与定位", "在静态画面上框选识别区域、定位模板与搜索区域"),
    ("相机与采集", "扫描选择工业相机设备并调整曝光等采集参数"),
    ("反馈与统计", "配置画面叠加、声光追溯反馈并管理当前批次统计"),
)


class ConfigPage(QWidget):
    """系统设置页：按产线换线动线划分为五个配置分区。"""

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
        self._loading_values = False
        self._dirty = False
        self._saved_feedback_timer = QTimer(self)
        self._saved_feedback_timer.setSingleShot(True)
        self._saved_feedback_timer.timeout.connect(self._hide_saved_feedback)
        self._build_ui()
        self._load_values()
        self._bind_dirty_tracking()
        self._sync_model_dependent_fields()
        self.refresh_stats_display()

    def _build_ui(self) -> None:
        self.setObjectName("configPage")
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 左侧五分区导航栏
        sidebar = QFrame(self)
        sidebar.setObjectName("configSidebar")
        sidebar.setFixedWidth(CONFIG_NAV_WIDTH)
        sidebar.setStyleSheet(
            f"QFrame#configSidebar {{ background-color: {PANEL_BG_DARK}; border: none; "
            f"border-right: 1px solid {STROKE_SUBTLE}; }}"
        )
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(16, 22, 16, 18)
        side_layout.setSpacing(8)

        brand_row = QWidget(sidebar)
        brand_row.setStyleSheet("background: transparent;")
        brand_layout = QHBoxLayout(brand_row)
        brand_layout.setContentsMargins(4, 0, 0, 0)
        brand_layout.setSpacing(8)
        brand_dot = QLabel(brand_row)
        brand_dot.setFixedSize(9, 9)
        brand_dot.setStyleSheet(f"background-color: {TEXT_ACCENT}; border: none; border-radius: 4px;")
        brand = QLabel("系统设置", brand_row)
        brand.setStyleSheet(text_style(TEXT_PRIMARY, size=20, weight=800))
        brand_layout.addWidget(brand_dot, 0, Qt.AlignmentFlag.AlignVCenter)
        brand_layout.addWidget(brand, 0, Qt.AlignmentFlag.AlignVCenter)
        brand_layout.addStretch(1)
        side_layout.addWidget(brand_row)

        subtitle = QLabel("双翼视觉防错系统", sidebar)
        subtitle.setStyleSheet(text_style(TEXT_MUTED, size=12, weight=500))
        subtitle.setContentsMargins(4, 0, 0, 0)
        side_layout.addWidget(subtitle)
        side_layout.addSpacing(18)

        self._nav_buttons: list[ConfigNavButton] = []
        self._nav_index_labels: list[QLabel] = []
        for index, (title, _desc) in enumerate(_SECTION_META):
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
        foot_note = QLabel("本地工位配置", sidebar)
        foot_note.setStyleSheet(text_style(TEXT_MUTED, size=11, weight=500))
        foot_note.setContentsMargins(4, 0, 0, 0)
        side_layout.addWidget(foot_note)
        root.addWidget(sidebar)

        # 右侧主内容列
        content = QWidget(self)
        content.setObjectName("configPageContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        header = QFrame(content)
        header.setObjectName("configSectionHeader")
        header.setFixedHeight(78)
        header.setStyleSheet(
            f"QFrame#configSectionHeader {{ background: {PAGE_BG}; border: none; "
            f"border-bottom: 1px solid {STROKE_SUBTLE}; }}"
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(28, 14, 28, 12)
        title_layout = QVBoxLayout()
        title_layout.setSpacing(3)
        self._section_title_label = QLabel(_SECTION_META[0][0], header)
        self._section_title_label.setStyleSheet(text_style(TEXT_PRIMARY, size=22, weight=800))
        self._section_subtitle_label = QLabel(_SECTION_META[0][1], header)
        self._section_subtitle_label.setStyleSheet(text_style(TEXT_MUTED, size=FONT_SMALL, weight=500))
        title_layout.addWidget(self._section_title_label)
        title_layout.addWidget(self._section_subtitle_label)
        header_layout.addLayout(title_layout)
        header_layout.addStretch(1)
        content_layout.addWidget(header)

        self._section_stack = QStackedWidget(content)
        self._section_stack.setStyleSheet("QStackedWidget { background: transparent; border: none; }")
        content_layout.addWidget(self._section_stack, 1)

        # 5 个配置分区（前 4 个表单页定宽 900px，区域与定位采用全宽左右双栏）
        self._model_page, model_layout = self._create_section_page(max_width=CONFIG_FORM_MAX_WIDTH)
        self._action_page, action_layout = self._create_section_page(max_width=CONFIG_FORM_MAX_WIDTH)
        self._region_page, region_layout = self._create_full_width_page()
        self._camera_page, camera_layout = self._create_section_page(max_width=CONFIG_FORM_MAX_WIDTH)
        self._feedback_page, feedback_layout = self._create_section_page(max_width=CONFIG_FORM_MAX_WIDTH)
        self._stats_page = self._feedback_page

        self._build_model_section(model_layout)
        self._build_action_section(action_layout)
        self._build_region_section(region_layout)
        self._build_camera_section(camera_layout)
        self._build_feedback_and_stats_section(feedback_layout)
        self._model_path_input.textChanged.connect(self._sync_model_dependent_fields)

        self._footer = FooterActionBar(content)
        self._save_btn = self._footer.save_button
        self._back_btn = self._footer.cancel_button
        self._saved_label = self._footer.saved_label
        self._dirty_label = self._footer.dirty_label
        self._save_btn.clicked.connect(self._on_save)
        self._back_btn.clicked.connect(self._on_back_clicked)
        content_layout.addWidget(self._footer)
        root.addWidget(content, 1)

        self._apply_control_metrics()
        self.setStyleSheet(
            f"QWidget#configPage, QWidget#configPageContent {{ "
            f"background-color: {PAGE_BG}; color: {TEXT_PRIMARY}; }}"
        )
        self._show_section(0)

    def _apply_control_metrics(self) -> None:
        """保持产线友好点击高度，控件绘制完全交由开源主题接管。"""
        for control_type in (QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox):
            for control in self.findChildren(control_type):
                control.setMinimumHeight(40)
        for button in self.findChildren(QPushButton):
            if button.property("segmented") is True:
                button.setMinimumHeight(38)
            else:
                button.setMinimumHeight(40)

    def _create_section_page(self, *, max_width: int = CONFIG_FORM_MAX_WIDTH):
        scroll = QScrollArea(self._section_stack)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(scroll_area_style())
        wrapper = QWidget(scroll)
        wrapper.setObjectName(f"configSectionWrapper{self._section_stack.count()}")
        wrapper.setStyleSheet(f"QWidget#{wrapper.objectName()} {{ background: transparent; border: none; }}")
        outer = QHBoxLayout(wrapper)
        outer.setContentsMargins(28, 20, 28, 28)
        outer.setSpacing(0)

        page = QWidget(wrapper)
        page.setObjectName(f"configSectionPage{self._section_stack.count()}")
        page.setStyleSheet(f"QWidget#{page.objectName()} {{ background: transparent; border: none; }}")
        page.setMaximumWidth(max_width)
        page.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        layout.addStretch(1)

        outer.addWidget(page, 1)
        outer.addStretch(0)
        scroll.setWidget(wrapper)
        self._section_stack.addWidget(scroll)
        return page, layout

    def _create_full_width_page(self):
        page = QWidget(self._section_stack)
        page.setObjectName(f"configFullPage{self._section_stack.count()}")
        page.setStyleSheet(f"QWidget#{page.objectName()} {{ background: transparent; border: none; }}")
        layout = QHBoxLayout(page)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(18)
        self._section_stack.addWidget(page)
        return page, layout

    @staticmethod
    def _transparent_container(parent=None) -> QWidget:
        container = QWidget(parent)
        container.setObjectName("layoutContainer")
        container.setStyleSheet("QWidget#layoutContainer { background: transparent; border: none; }")
        return container

    def _insert_group(self, layout: QVBoxLayout, widget: QWidget) -> None:
        layout.insertWidget(layout.count() - 1, widget)

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

    # ------------------------------------------------------------------
    # 01 检测方案
    # ------------------------------------------------------------------
    def _build_model_section(self, layout: QVBoxLayout) -> None:
        mode_group = FormGroup(
            "检测模式",
            subtitle="先确定工位运行模式，后续参数将自动按当前模式联动显示。",
        )
        seg_row = self._transparent_container(mode_group)
        seg_layout = QHBoxLayout(seg_row)
        seg_layout.setContentsMargins(0, 0, 0, 0)
        seg_layout.setSpacing(10)
        self._mode_sequence_btn = QPushButton("顺序防错模式", seg_row)
        self._mode_sequence_btn.setCheckable(True)
        self._mode_sequence_btn.setProperty("segmented", True)
        self._mode_roi_btn = QPushButton("固定工位计数模式", seg_row)
        self._mode_roi_btn.setCheckable(True)
        self._mode_roi_btn.setProperty("segmented", True)
        self._mode_sequence_btn.clicked.connect(lambda: self._roi_enabled_cb.setChecked(False))
        self._mode_roi_btn.clicked.connect(lambda: self._roi_enabled_cb.setChecked(True))
        seg_layout.addWidget(self._mode_sequence_btn, 1)
        seg_layout.addWidget(self._mode_roi_btn, 1)
        mode_group.content_layout.addWidget(seg_row)

        roi_mode_row = SwitchRow(
            "启用固定工位计数模式（ROI 区域判定）",
            "开启后在识别区域内并行检查各类目标数量；关闭则按配置步骤顺序逐项防错。",
        )
        self._roi_enabled_cb = roi_mode_row.checkbox
        self._roi_enabled_cb.stateChanged.connect(self._on_mode_changed)
        mode_group.content_layout.addWidget(roi_mode_row)
        self._insert_group(layout, mode_group)

        model_group = FormGroup("视觉检测模型")
        self._model_path_input = QLineEdit()
        self._select_model_btn = QPushButton("浏览")
        self._select_model_btn.setMinimumWidth(88)
        self._select_model_btn.clicked.connect(self._on_select_model)
        path_row = self._transparent_container()
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.setSpacing(8)
        path_layout.addWidget(self._model_path_input, 1)
        path_layout.addWidget(self._select_model_btn)

        self._conf_input = QDoubleSpinBox()
        self._conf_input.setRange(0.0, 1.0)
        self._conf_input.setSingleStep(0.05)
        self._iou_input = QDoubleSpinBox()
        self._iou_input.setRange(0.0, 1.0)
        self._iou_input.setSingleStep(0.05)

        self._lstm_bundle_input = QLineEdit()
        self._lstm_bundle_input.setPlaceholderText("选择训练导出的动作识别模型文件夹")
        self._select_action_bundle_btn = QPushButton("浏览")
        self._select_action_bundle_btn.setMinimumWidth(88)
        self._select_action_bundle_btn.clicked.connect(self._on_select_action_bundle)
        bundle_row = self._transparent_container()
        bundle_layout = QHBoxLayout(bundle_row)
        bundle_layout.setContentsMargins(0, 0, 0, 0)
        bundle_layout.setSpacing(8)
        bundle_layout.addWidget(self._lstm_bundle_input, 1)
        bundle_layout.addWidget(self._select_action_bundle_btn)

        self._action_bundle_field = FormField(
            "动作识别模型目录",
            bundle_row,
            "顺序防错模式下选择训练好的动作模型目录，系统将自动读取模型配置。",
        )

        self._add_form_grid(
            model_group,
            [
                (FormField("模型文件路径", path_row, "支持 .onnx 与 .pt 模型文件，保存后重新加载。"), True),
                (FormField("检测置信度", self._conf_input), False),
                (FormField("重叠过滤阈值 (IoU)", self._iou_input), False),
                (self._action_bundle_field, True),
            ],
        )

        self._tracker_input = QComboBox()
        self._tracker_input.addItems(["bytetrack.yaml", "botsort.yaml"])
        persist_row = SwitchRow("跨帧目标持续跟踪", "跨帧保持目标编号，仅在 .pt 模型推理时生效。")
        self._track_persist_cb = persist_row.checkbox

        self._advanced_model_section = CollapsibleSection("高级跟踪参数（仅 .pt 模型生效）", model_group, expanded=False)
        self._advanced_model_section.content_layout.addWidget(FormField("跟踪算法配置", self._tracker_input))
        self._advanced_model_section.content_layout.addWidget(persist_row)
        model_group.content_layout.addWidget(self._advanced_model_section)
        self._insert_group(layout, model_group)

        steps_group = FormGroup(
            "作业步骤与目标清单",
            subtitle="按工艺顺序填写步骤或目标名称；留空行将自动忽略。",
        )
        steps_container = self._transparent_container(steps_group)
        steps_grid = QGridLayout(steps_container)
        steps_grid.setContentsMargins(0, 0, 0, 0)
        steps_grid.setHorizontalSpacing(12)
        steps_grid.setVerticalSpacing(9)
        steps_grid.addWidget(self._label("序号"), 0, 0)
        steps_grid.addWidget(self._label("步骤或目标名称"), 0, 1)
        steps_grid.addWidget(self._label("目标数量"), 0, 2)
        self._step_inputs: list[QLineEdit] = []
        self._step_count_inputs: list[QSpinBox] = []
        for i in range(6):
            index_label = QLabel(f"{i + 1:02d}")
            index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            index_label.setMinimumWidth(36)
            index_label.setStyleSheet(
                text_style(TEXT_ACCENT, size=12, weight=800, extra=f"font-family: '{MONO_FAMILY}';")
            )
            inp = QLineEdit()
            inp.setPlaceholderText("留空表示未配置")
            count = QSpinBox()
            count.setRange(1, 999)
            count.setValue(1)
            count.setMinimumWidth(116)
            steps_grid.addWidget(index_label, i + 1, 0)
            steps_grid.addWidget(inp, i + 1, 1)
            steps_grid.addWidget(count, i + 1, 2)
            self._step_inputs.append(inp)
            self._step_count_inputs.append(count)
        steps_grid.setColumnStretch(1, 1)
        steps_group.content_layout.addWidget(steps_container)
        self._insert_group(layout, steps_group)

    # ------------------------------------------------------------------
    # 02 判定规则
    # ------------------------------------------------------------------
    def _build_action_section(self, layout: QVBoxLayout) -> None:
        timing_group = FormGroup("防抖与节拍参数")
        self._action_pass_frames_input = QSpinBox()
        self._action_pass_frames_input.setRange(1, 99999)
        self._action_ng_frames_input = QSpinBox()
        self._action_ng_frames_input.setRange(1, 99999)
        self._cooldown_input = QDoubleSpinBox()
        self._cooldown_input.setRange(0.0, 60.0)
        self._cooldown_input.setSingleStep(0.5)
        self._switch_confirm_input = QDoubleSpinBox()
        self._switch_confirm_input.setRange(0.1, 5.0)
        self._switch_confirm_input.setSingleStep(0.1)
        self._switch_confirm_input.setDecimals(1)
        self._add_form_grid(
            timing_group,
            [
                (FormField("合格防抖帧数", self._action_pass_frames_input, "连续命中达到该帧数后确认当前步骤合格。"), False),
                (FormField("异常确认帧数", self._action_ng_frames_input, "连续检测到异常达到该帧数后触发报警。"), False),
                (
                    FormField(
                        "单件结果保持时间 (s)",
                        self._cooldown_input,
                        "单件判定 PASS 或异常后保持当前高亮提示，计时结束后自动进入下一轮。",
                    ),
                    False,
                ),
                (
                    FormField(
                        "步骤切换确认时长 (s)",
                        self._switch_confirm_input,
                        "动作切换到下一步并连续保持达到该时长后，确认上一步完成。",
                    ),
                    False,
                ),
            ],
        )
        self._insert_group(layout, timing_group)

        policy_group = FormGroup("顺序与容错策略")
        order_row = SwitchRow("步骤顺序约束", "严格校验作业顺序，防止跳步、漏步或越序操作。")
        self._action_order_cb = order_row.checkbox
        policy_group.content_layout.addWidget(order_row)

        retry_row = SwitchRow(
            "漏检 / 数量异常报警",
            "首步完成后，当前步骤目标未检出或数量不符并达到异常确认帧数时触发报警。",
        )
        self._fail_retry_cb = retry_row.checkbox
        policy_group.content_layout.addWidget(retry_row)
        self._insert_group(layout, policy_group)

    # ------------------------------------------------------------------
    # 03 区域与定位（左右双栏标定布局）
    # ------------------------------------------------------------------
    def _build_region_section(self, layout: QHBoxLayout) -> None:
        # 左侧 65%：静态画面与区域设置画布
        canvas_card = FormGroup("静态画面与区域设置")
        canvas_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._region_title_label = canvas_card.title_label

        top_bar = self._transparent_container(canvas_card)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(12)
        self._refresh_roi_frame_btn = QPushButton("刷新画面", top_bar)
        self._refresh_roi_frame_btn.setMinimumWidth(96)
        self._refresh_roi_frame_btn.clicked.connect(self._refresh_roi_frame)
        self._roi_frame_status = QLabel("没有可用画面，请先打开相机并刷新", top_bar)
        self._roi_frame_status.setWordWrap(True)
        self._roi_frame_status.setStyleSheet(text_style(TEXT_MUTED, size=12, weight=600))
        top_layout.addWidget(self._refresh_roi_frame_btn)
        top_layout.addWidget(self._roi_frame_status, 1)
        canvas_card.content_layout.addWidget(top_bar)

        self._roi_editor = RoiEditorWidget(canvas_card)
        self._roi_editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._roi_editor.rect_changed.connect(self._on_roi_rect_changed)
        canvas_card.content_layout.addWidget(self._roi_editor, 1)
        layout.addWidget(canvas_card, 13)

        # 右侧 35%：独立滚动参数控制面板
        right_scroll = QScrollArea(self._region_page)
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(350)
        right_scroll.setMaximumWidth(440)
        right_scroll.setStyleSheet(scroll_area_style())

        right_panel = QWidget(right_scroll)
        right_panel.setObjectName("roiRightPanel")
        right_panel.setStyleSheet("QWidget#roiRightPanel { background: transparent; border: none; }")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 4, 4)
        right_layout.setSpacing(14)

        # 区域操作卡
        op_group = FormGroup("区域操作", subtitle="切换区域类型后在左侧画布框选并保存。")
        self._roi_kind_input = QComboBox(op_group)
        self._roi_kind_input.addItem("识别区域", RoiKind.INSPECTION.value)
        self._roi_kind_input.addItem("定位模板", RoiKind.LOCATOR.value)
        self._roi_kind_input.addItem("搜索区域", RoiKind.SEARCH.value)
        self._roi_kind_input.currentIndexChanged.connect(self._on_roi_kind_changed)

        seg_wrap = self._transparent_container(op_group)
        seg_layout = QHBoxLayout(seg_wrap)
        seg_layout.setContentsMargins(0, 0, 0, 0)
        seg_layout.setSpacing(6)
        self._roi_kind_buttons: list[QPushButton] = []
        for idx in range(self._roi_kind_input.count()):
            btn = QPushButton(self._roi_kind_input.itemText(idx), seg_wrap)
            btn.setCheckable(True)
            btn.setProperty("segmented", True)
            btn.clicked.connect(lambda _checked=False, i=idx: self._roi_kind_input.setCurrentIndex(i))
            seg_layout.addWidget(btn, 1)
            self._roi_kind_buttons.append(btn)
        op_group.content_layout.addWidget(seg_wrap)

        # 保留原生下拉框以同步键盘/自动化选择，默认紧凑隐藏或作为辅控
        self._roi_kind_input.setVisible(False)
        op_group.content_layout.addWidget(self._roi_kind_input)

        btn_row = self._transparent_container(op_group)
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(8)
        self._create_roi_btn = QPushButton("新建区域", btn_row)
        self._delete_roi_btn = QPushButton("删除区域", btn_row)
        self._create_roi_btn.clicked.connect(self._create_selected_roi)
        self._delete_roi_btn.clicked.connect(self._delete_selected_roi)
        btn_layout.addWidget(self._create_roi_btn, 1)
        btn_layout.addWidget(self._delete_roi_btn, 1)
        op_group.content_layout.addWidget(btn_row)

        self._save_roi_btn = QPushButton("保存当前区域", op_group)
        self._save_roi_btn.setProperty("buttonRole", "primary")
        self._save_roi_btn.clicked.connect(self._save_selected_roi)
        op_group.content_layout.addWidget(self._save_roi_btn)

        save_tip = QLabel("注：「保存当前区域」立即写入当前选中区域或生成定位模板；其余参数通过底部「保存并生效」统一提交。", op_group)
        save_tip.setWordWrap(True)
        save_tip.setStyleSheet(text_style(TEXT_MUTED, size=11, weight=500))
        op_group.content_layout.addWidget(save_tip)
        right_layout.addWidget(op_group)

        # 模板定位参数卡
        template_group = FormGroup("边缘模板定位")
        template_row = SwitchRow("启用边缘模板定位", "在搜索区域内定位工件轮廓，匹配成功后再执行识别。")
        self._roi_template_cb = template_row.checkbox
        self._roi_template_cb.stateChanged.connect(self._sync_template_inputs_enabled)
        template_group.content_layout.addWidget(template_row)

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

        template_group.content_layout.addWidget(
            FormField("匹配分数阈值", self._template_threshold_input, "得分约等于可见轮廓比例，默认 0.70。")
        )
        self._add_form_grid(
            template_group,
            [
                (FormField("允许旋转角度 (±°)", self._template_angle_input, "默认 ±15°。"), False),
                (FormField("角度搜索步进 (°)", self._template_angle_step_input, "0 为自动推算。"), False),
            ],
        )
        right_layout.addWidget(template_group)

        # 研发诊断项默认折叠
        self._diag_section = CollapsibleSection("高级诊断（现场排查用）", right_panel, expanded=False)
        diag_row = SwitchRow(
            "记录模板匹配诊断日志",
            "记录每帧匹配得分、耗时与失败原因至 logs/template_sessions。",
        )
        self._template_diag_cb = diag_row.checkbox
        self._template_diag_cb.stateChanged.connect(self._sync_template_inputs_enabled)
        self._diag_section.content_layout.addWidget(diag_row)

        shadow_row = SwitchRow(
            "并行计算对照算法得分",
            "同帧额外计算候选算法得分做评估对照，会增加约 20–40ms 耗时。",
        )
        self._template_shadow_cb = shadow_row.checkbox
        self._diag_section.content_layout.addWidget(shadow_row)

        self._template_sample_input = QSpinBox()
        self._template_sample_input.setRange(0, 2000)
        self._diag_section.content_layout.addWidget(
            FormField("抽样保存帧数", self._template_sample_input, "按得分分层保存搜索区域图像，0 表示不保存。")
        )
        self._diag_status_label = QLabel("诊断日志：未开始")
        self._diag_status_label.setWordWrap(True)
        self._diag_status_label.setStyleSheet(text_style(TEXT_MUTED, size=12, weight=600))
        self._diag_section.content_layout.addWidget(self._diag_status_label)
        right_layout.addWidget(self._diag_section)
        right_layout.addStretch(1)

        right_scroll.setWidget(right_panel)
        layout.addWidget(right_scroll, 7)

        self._on_roi_kind_changed(self._roi_kind_input.currentIndex())
        self._update_roi_controls()
        self._sync_template_inputs_enabled()

    # ------------------------------------------------------------------
    # 04 相机与采集
    # ------------------------------------------------------------------
    def _build_camera_section(self, layout: QVBoxLayout) -> None:
        device_group = FormGroup("工业相机设备")
        device_row = self._transparent_container(device_group)
        device_layout = QHBoxLayout(device_row)
        device_layout.setContentsMargins(0, 0, 0, 0)
        device_layout.setSpacing(10)
        self._camera_device_input = QComboBox(device_row)
        self._camera_device_input.setMinimumWidth(340)
        self._camera_refresh_btn = QPushButton("刷新设备", device_row)
        self._camera_refresh_btn.setMinimumWidth(96)
        self._camera_refresh_btn.clicked.connect(self.camera_refresh_requested.emit)
        device_layout.addWidget(self._camera_device_input, 1)
        device_layout.addWidget(self._camera_refresh_btn)

        self._camera_device_status = QLabel("正在等待扫描工业相机设备", device_group)
        self._camera_device_status.setWordWrap(True)
        self._camera_device_status.setStyleSheet(text_style(TEXT_MUTED, size=12, weight=600))
        device_group.content_layout.addWidget(
            FormField("已连接设备", device_row, "按相机序列号（SN）绑定设备；点击「保存并生效」后应用切换。")
        )
        device_group.content_layout.addWidget(self._camera_device_status)
        self._insert_group(layout, device_group)

        capture_group = FormGroup("采集参数")
        self._cam_exp_input = QSpinBox()
        self._cam_exp_input.setRange(100, 1000000)
        self._add_form_grid(
            capture_group,
            [(FormField("手动曝光时间 (μs)", self._cam_exp_input, "控制画面亮度与运动拖影，建议在现场照明下微调。"), False)],
        )
        self._insert_group(layout, capture_group)

    def set_camera_devices(self, devices, error: str = "") -> None:
        """刷新下拉框中的在线工业相机列表。"""
        previous_loading = self._loading_values
        self._loading_values = True
        try:
            pending_sn = str(self._camera_device_input.currentData() or "").strip()
            self._camera_devices = list(devices or [])
            self._camera_enumeration_error = str(error or "").strip()

            self._camera_device_input.clear()
            self._camera_device_input.addItem("请选择工业相机", "")
            for device in self._camera_devices:
                self._camera_device_input.addItem(str(device), str(device.sn or "").strip())
                if not str(device.sn or "").strip():
                    item = self._camera_device_input.model().item(self._camera_device_input.count() - 1)
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

            legacy_name = str(getattr(self.config, "mvsdk_friendly_name", "") or "").strip()
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
                status = f"设备扫描失败：{self._camera_enumeration_error}"
                color = "#F44336"
            elif not self._camera_devices:
                status = "未检测到在线工业相机"
                color = TEXT_MUTED
            elif saved_sn and saved_sn not in online_sns:
                status = f"已保存相机当前不在线（SN: {saved_sn}），请重新选择"
                color = "#FF9800"
            elif not saved_sn and legacy_name and len(legacy_matches) > 1:
                status = f"发现多台同名相机（{legacy_name}），请按 SN 选择"
                color = "#FF9800"
            else:
                status = f"已发现 {len(online_sns)} 台可用相机"
                color = TEXT_SECONDARY
            self._camera_device_status.setText(status)
            self._camera_device_status.setStyleSheet(text_style(color, size=12, weight=600))
        finally:
            self._loading_values = previous_loading

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
        """ONNX 模式下置灰不生效的跟踪参数并收起高级跟踪区。"""
        is_onnx = str(self._model_path_input.text()).strip().lower().endswith(".onnx")
        for widget in (self._tracker_input, self._track_persist_cb, self._show_tid_cb):
            widget.setEnabled(not is_onnx)
        if hasattr(self, "_advanced_model_section") and is_onnx:
            self._advanced_model_section.set_expanded(False)

    # ------------------------------------------------------------------
    # 05 反馈与统计
    # ------------------------------------------------------------------
    def _build_feedback_and_stats_section(self, layout: QVBoxLayout) -> None:
        # 批次统计置顶，便于换班/清零高频直达
        self._stats_group = FormGroup("当前批次生产统计")
        self._stats_start_label = QLabel("批次开始时间：--")
        self._stats_start_label.setStyleSheet(text_style(TEXT_MUTED, size=12, weight=600))
        self._stats_group.content_layout.addWidget(self._stats_start_label)

        self._stats_kpi_row = KpiRow(self._stats_group)
        self._stats_group.content_layout.addWidget(self._stats_kpi_row)

        # 兼容标签同步属性（不额外占视觉版面）
        self._stats_total_label = QLabel("总数: 0", self._stats_group)
        self._stats_ok_label = QLabel("OK: 0", self._stats_group)
        self._stats_ng_label = QLabel("NG: 0", self._stats_group)
        self._stats_yield_label = QLabel("良率: 0.0%", self._stats_group)
        for compat in (
            self._stats_total_label,
            self._stats_ok_label,
            self._stats_ng_label,
            self._stats_yield_label,
        ):
            compat.setVisible(False)

        batch_bar = self._transparent_container(self._stats_group)
        batch_layout = QHBoxLayout(batch_bar)
        batch_layout.setContentsMargins(0, 4, 0, 0)
        batch_layout.setSpacing(12)
        hint = QLabel("归零前自动归档当前批次生产记录，再开启新批次计数。", batch_bar)
        hint.setWordWrap(True)
        hint.setStyleSheet(text_style(TEXT_MUTED, size=12, weight=500))
        self._reset_btn = QPushButton("批次归零并归档", batch_bar)
        self._reset_btn.setProperty("buttonRole", "danger")
        self._reset_btn.setEnabled(self._stats_manager is not None)
        self._reset_btn.clicked.connect(self._on_reset_stats)
        batch_layout.addWidget(hint, 1)
        batch_layout.addWidget(self._reset_btn)
        self._stats_group.content_layout.addWidget(batch_bar)
        self._insert_group(layout, self._stats_group)

        overlay_group = FormGroup("画面叠加与告警反馈")
        conf_row = SwitchRow("显示置信度", "在实时画面检测框标签中显示识别分数。")
        self._show_conf_cb = conf_row.checkbox
        tid_row = SwitchRow("显示目标跟踪编号", "仅在 .pt 模型开启持续跟踪时显示。")
        self._show_tid_cb = tid_row.checkbox
        pass_sound_row = SwitchRow("PASS 提示音", "单件检测通过时播放一次提示音。")
        self._pass_sound_cb = pass_sound_row.checkbox
        fail_sound_row = SwitchRow("FAIL 提示音", "单件检测异常时播放报警提示音。")
        self._fail_sound_cb = fail_sound_row.checkbox
        evidence_row = SwitchRow("保存异常证据图像", "判定异常时自动保存现场画面与追溯记录。")
        self._fail_evidence_cb = evidence_row.checkbox

        for row in (conf_row, tid_row, pass_sound_row, fail_sound_row, evidence_row):
            overlay_group.content_layout.addWidget(row)
        self._insert_group(layout, overlay_group)

    # ------------------------------------------------------------------
    # 导航与脏状态管理
    # ------------------------------------------------------------------
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
        title, desc = _SECTION_META[index]
        self._section_title_label.setText(title)
        self._section_subtitle_label.setText(desc)

    def _label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(text_style(TEXT_SECONDARY, size=13, weight=600))
        return label

    def _bind_dirty_tracking(self) -> None:
        for line_edit in (self._model_path_input, self._lstm_bundle_input, *self._step_inputs):
            line_edit.textChanged.connect(self._mark_dirty)
        for spin in (
            self._conf_input,
            self._iou_input,
            self._action_pass_frames_input,
            self._action_ng_frames_input,
            self._cooldown_input,
            self._switch_confirm_input,
            self._cam_exp_input,
            self._template_threshold_input,
            self._template_angle_input,
            self._template_angle_step_input,
            self._template_sample_input,
            *self._step_count_inputs,
        ):
            spin.valueChanged.connect(self._mark_dirty)
        for combo in (self._tracker_input, self._camera_device_input):
            combo.currentIndexChanged.connect(self._mark_dirty)
        for cb in (
            self._roi_enabled_cb,
            self._track_persist_cb,
            self._action_order_cb,
            self._fail_retry_cb,
            self._roi_template_cb,
            self._template_diag_cb,
            self._template_shadow_cb,
            self._show_conf_cb,
            self._show_tid_cb,
            self._pass_sound_cb,
            self._fail_sound_cb,
            self._fail_evidence_cb,
        ):
            cb.toggled.connect(self._mark_dirty)

    def _mark_dirty(self, *_args) -> None:
        if self._loading_values:
            return
        self._dirty = True
        self._footer.set_dirty(True)

    def has_unsaved_changes(self) -> bool:
        return bool(self._dirty)

    def _on_back_clicked(self) -> None:
        if self._dirty:
            reply = QMessageBox.question(
                self,
                "未保存的修改",
                "当前设置存在尚未保存的修改，是否放弃修改并返回主屏？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            self._dirty = False
            self._footer.set_dirty(False)
            self._load_values()
        self.back_clicked.emit()

    # ------------------------------------------------------------------
    # 模式联动与 ROI 标定逻辑
    # ------------------------------------------------------------------
    def _on_mode_changed(self) -> None:
        """同步模式分段按钮、动作模型显隐与模板参数可用性。"""
        roi_mode = self._roi_enabled_cb.isChecked()
        if hasattr(self, "_mode_sequence_btn") and hasattr(self, "_mode_roi_btn"):
            self._mode_sequence_btn.setChecked(not roi_mode)
            self._mode_roi_btn.setChecked(roi_mode)
        if hasattr(self, "_action_bundle_field"):
            self._action_bundle_field.setVisible(not roi_mode)
        self._sync_template_inputs_enabled()
        self._update_roi_controls()

    def _sync_template_inputs_enabled(self) -> None:
        """模板定位关闭时置灰模板相关参数。"""
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
        """只保存当前选中的区域，不提交另外两种草稿。"""
        kind = self._selected_roi_kind()
        rect = self._roi_editor.roi_rect(kind)
        if rect is None:
            self._roi_frame_status.setText("请先新建并框选当前区域")
            return
        if kind is RoiKind.LOCATOR:
            self._save_template_roi(rect)
            return
        self._save_plain_roi(kind, rect)

    def _template_self_check_rect(self, model: ShapeModel, template_rect: NormalizedRect) -> dict:
        """回查窗口从定位模板向四周扩到算法最小窗，不够则用整图。"""
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
            RoiKind.SEARCH: ("roi_template_search_rect", "搜索区域"),
            RoiKind.INSPECTION: ("roi_inspection_rect", "识别区域"),
            RoiKind.LOCATOR: ("roi_template_rect", "定位模板"),
        }[kind]

    def _persist_roi_field(self, field_name: str, label: str, value) -> bool:
        try:
            config_path = self._roi_config_path()
        except ValueError as exc:
            QMessageBox.warning(self, "区域保存失败", str(exc))
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
            QMessageBox.warning(self, "区域保存失败", f"{label} 保存失败：{exc}")
            return False
        return True

    def _save_plain_roi(self, kind: RoiKind, rect: NormalizedRect) -> None:
        field_name, label = self._roi_field(kind)
        if not self._persist_roi_field(field_name, label, rect.to_mapping()):
            return
        self._roi_frame_status.setText(f"{label} 已单独保存")
        self._show_saved_feedback()

    def _save_template_roi(self, rect: NormalizedRect) -> None:
        """只用定位模板区域创建并自检模板，不依赖搜索或识别区域。"""
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
        previous_values = {name: deepcopy(getattr(self.config, name)) for name in tracked_fields}
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
                    f"模板自检得分 {match_result.score:.3f}，"
                    f"未达到阈值 {settings.min_score:.3f}。\n"
                    f"失败原因：{match_result.failure or '未知'}\n"
                    f"请检查框选区域是否包含清晰轮廓，或适当降低匹配分数阈值。"
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
            self._roi_frame_status.setText(self._template_saved_summary(candidate_model, match_result))
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
            self._roi_editor.set_template_feedback(
                success=False,
                score=match_result.score if match_result is not None else None,
                box=match_result.box if match_result is not None else None,
                contours=(),
                message=message,
            )
            self._roi_frame_status.setText(message)
            QMessageBox.warning(self, title, message)
            self._update_roi_controls()

    def _template_settings_from_inputs(self):
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
        suggested = max(0.30, round(match_result.score - 0.25, 2))
        return (
            f"定位模板已保存：{model.template_size[0]}×{model.template_size[1]}，"
            f"{model.level_count} 层金字塔，角度步进 {model.base_angle_step:.2f}°"
            f" ｜ 回查得分 {match_result.score:.3f}"
            f"（当前阈值 {model.settings.min_score:.2f}，建议不高于 {suggested:.2f}）"
        )

    def _roi_config_path(self) -> Path:
        config_path_text = str(self.config.get_config_path() or "").strip()
        if not config_path_text:
            raise ValueError("配置路径不存在，无法保存区域")
        return Path(config_path_text)

    def _show_saved_feedback(self) -> None:
        self._saved_label.setVisible(True)
        self._saved_feedback_timer.start(2200)

    def set_template_diagnostics_status(self, text: str, *, warning: bool = False) -> None:
        if not hasattr(self, "_diag_status_label"):
            return
        self._diag_status_label.setText(text or "诊断日志：未开始")
        color = "#FF9800" if warning else TEXT_MUTED
        self._diag_status_label.setStyleSheet(text_style(color, size=12, weight=600))

    # ------------------------------------------------------------------
    # 值加载与保存
    # ------------------------------------------------------------------
    def _load_values(self) -> None:
        self._loading_values = True
        try:
            c = self.config
            self._model_path_input.setText(c.yolo_model_path)
            self._conf_input.setValue(c.yolo_conf_threshold)
            self._iou_input.setValue(c.yolo_iou_threshold)
            idx = self._tracker_input.findText(c.ultralytics_tracker)
            if idx >= 0:
                self._tracker_input.setCurrentIndex(idx)
            self._track_persist_cb.setChecked(c.ultralytics_track_persist)
            for i in range(6):
                name = c.category_names[i] if i < len(c.category_names) else ""
                self._step_inputs[i].setText(name)
            counts = getattr(c, "category_counts", [1, 1, 1, 1, 1, 1])
            for i in range(6):
                val = counts[i] if i < len(counts) else 1
                self._step_count_inputs[i].setValue(_display_positive_int(val))

            self._cam_exp_input.setValue(getattr(c, "camera_manual_exposure_us", 30000))
            self.set_camera_devices(self._camera_devices, self._camera_enumeration_error)

            self._action_pass_frames_input.setValue(getattr(c, "action_pass_stable_frames", 2))
            self._action_ng_frames_input.setValue(getattr(c, "action_ng_stable_frames", 10))
            self._action_order_cb.setChecked(getattr(c, "action_order_constraint_enabled", True))
            self._lstm_bundle_input.setText(str(getattr(c, "lstm_bundle_path", "") or ""))
            self._fail_retry_cb.setChecked(getattr(c, "action_fail_retry_enabled", False))
            self._cooldown_input.setValue(c.round_cooldown_seconds)
            self._switch_confirm_input.setValue(float(getattr(c, "action_switch_confirm_seconds", 0.5)))

            self._show_conf_cb.setChecked(c.show_confidence_overlay)
            self._show_tid_cb.setChecked(c.show_track_id_overlay)
            self._pass_sound_cb.setChecked(getattr(c, "pass_sound_enabled", False))
            self._fail_sound_cb.setChecked(getattr(c, "fail_sound_enabled", True))
            self._fail_evidence_cb.setChecked(getattr(c, "fail_evidence_enabled", True))

            self._roi_enabled_cb.setChecked(getattr(c, "roi_inspection_enabled", False))
            self._roi_template_cb.setChecked(getattr(c, "roi_template_enabled", False))
            self._template_threshold_input.setValue(float(getattr(c, "roi_template_match_threshold", 0.6)))
            self._template_angle_input.setValue(float(getattr(c, "roi_template_angle_range", 15.0)))
            self._template_angle_step_input.setValue(float(getattr(c, "roi_template_angle_step", 1.0)))
            self._template_diag_cb.setChecked(bool(getattr(c, "template_diagnostics_enabled", False)))
            self._template_shadow_cb.setChecked(bool(getattr(c, "template_diagnostics_shadow_enabled", True)))
            self._template_sample_input.setValue(int(getattr(c, "template_diagnostics_sample_frames", 120)))

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
            self._on_mode_changed()
            self._dirty = False
            self._footer.set_dirty(False)
        finally:
            self._loading_values = False

        # 确保联动信号只绑定一次
        if not getattr(self, "_quantity_signals_bound", False):
            self._action_order_cb.toggled.connect(self._sync_action_quantity_inputs)
            self._roi_enabled_cb.toggled.connect(self._sync_action_quantity_inputs)
            self._quantity_signals_bound = True

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

    def _on_roi_kind_changed(self, index: int) -> None:
        self._roi_editor.set_active_kind(self._selected_roi_kind())
        for btn_idx, btn in enumerate(getattr(self, "_roi_kind_buttons", ())):
            btn.setChecked(btn_idx == index)
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
            self, "选择视觉检测模型", start, "模型文件 (*.onnx *.pt)"
        )
        if path:
            self._model_path_input.setText(path)

    def _on_select_action_bundle(self) -> None:
        from lstm_action.bundle import bundle_directory_for_model_file

        current = str(self._lstm_bundle_input.text()).strip()
        start = current if current else ""
        path = QFileDialog.getExistingDirectory(self, "选择动作识别模型目录", start)
        if not path:
            return
        try:
            directory = bundle_directory_for_model_file(path)
        except ValueError as exc:
            QMessageBox.warning(self, "动作识别模型", str(exc))
            return
        self._lstm_bundle_input.setText(str(directory))

    def refresh_stats_display(self) -> None:
        """刷新批次生产统计卡片显示。"""
        if self._stats_manager is None or not hasattr(self, "_stats_start_label"):
            return
        from datetime import datetime

        batch = self._stats_manager.batch
        dt = datetime.fromtimestamp(batch.started_at).strftime("%Y-%m-%d %H:%M:%S")
        self._stats_start_label.setText(f"批次开始时间：{dt}")
        self._stats_total_label.setText(f"总数: {batch.total}")
        self._stats_ok_label.setText(f"OK: {batch.ok}")
        self._stats_ng_label.setText(f"NG: {batch.ng}")
        self._stats_yield_label.setText(f"良率: {batch.yield_rate:.1f}%")
        if hasattr(self, "_stats_kpi_row"):
            self._stats_kpi_row.set_value("output", str(batch.total))
            self._stats_kpi_row.set_value("yield", f"{batch.yield_rate:.1f}%")
            self._stats_kpi_row.set_value("ok", str(batch.ok))
            self._stats_kpi_row.set_value("ng", str(batch.ng))

    def _on_reset_stats(self) -> None:
        """批次归零并归档：二次确认后执行。"""
        if self._stats_manager is None:
            return
        reply = QMessageBox.question(
            self,
            "批次归零确认",
            "系统将先归档保存当前批次统计记录，再清零并开启新批次。\n确定执行归零操作吗？",
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
                self,
                "归零成功",
                f"已归档生产记录至：\n{result.record_path}\n\n统计已清零，新批次已开始。",
            )
        else:
            QMessageBox.information(self, "归零完成", "当前批次暂无计数，已刷新批次开始时间。")

    def _on_save(self) -> None:
        c = deepcopy(self.config)
        try:
            self._write_form_into_config(c)
            camera_validation_error = self._camera_validation_error()
            if camera_validation_error:
                raise ValueError(camera_validation_error)
            if self._config_committer is None and (
                not c.roi_inspection_enabled and c.action_order_constraint_enabled
            ):
                from lstm_action.bundle import load_runtime_bundle

                if not str(c.lstm_bundle_path or "").strip():
                    raise ValueError("顺序防错模式需要配置动作识别模型目录")
                load_runtime_bundle(
                    c.lstm_bundle_path,
                    category_names=list(c.category_names or []),
                    yolo_path=str(c.yolo_model_path or ""),
                    yolo_conf=float(c.yolo_conf_threshold),
                    step_counts=list(c.category_counts or []),
                )
            c.validate()
            c.require_valid_inspection_config(None, require_roi_rects=False)
            config_path = c.get_config_path()
            if not config_path:
                raise ValueError("配置路径不存在，无法保存。")
            if self._config_committer is not None:
                self._config_committer(c)
                self.config = c
            else:
                c.save(config_path)
                self.config.__dict__.update(c.__dict__)
            self._dirty = False
            self._footer.set_dirty(False)
            self.config_saved.emit()
            self._show_saved_feedback()
        except Exception as exc:
            QMessageBox.warning(self, "配置校验失败", str(exc))

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
        # 保留既有兼容字段，不通过常规表单覆盖
        c.lstm_action_enabled = bool(getattr(self.config, "lstm_action_enabled", False))
        c.lstm_contract_path = str(getattr(self.config, "lstm_contract_path", "") or "").strip()
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
            return "所选工业相机缺少 SN，无法保存为稳定设备"
        if selected_camera is None and self._camera_devices and not getattr(self.config, "mvsdk_camera_sn", ""):
            return "请选择一台工业相机后再保存"
        return ""

    def _hide_saved_feedback(self) -> None:
        self._saved_label.setVisible(False)


def _display_positive_int(value) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 1
    return parsed if parsed > 0 else 1
