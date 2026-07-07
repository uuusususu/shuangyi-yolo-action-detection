"""配置页面模块。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from config import ConfigManager
from ui.runtime_ui_tokens import (
    CARD_PADDING,
    CARD_SPACING,
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_WINDOW_WIDTH,
    FONT_BODY,
    FONT_SECTION,
    FONT_SMALL,
    PAGE_BG,
    PAGE_MARGIN,
    PAGE_SPACING,
    PANEL_BG,
    PANEL_BG_ALT,
    STROKE_MAIN,
    STROKE_SOFT,
    TEXT_ACCENT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    chip_button_style,
    frame_style,
    text_style,
)


class ConfigPage(QWidget):
    """基于 pencil-new.pen 的配置页面。"""

    config_changed = Signal(str, object)
    config_saved = Signal()
    back_clicked = Signal()

    SLOT_COUNT = 6

    def __init__(self, config: ConfigManager, parent=None):
        super().__init__(parent)
        self.config = config

        self.header_card: QFrame | None = None
        self.info_card: QFrame | None = None
        self.base_config_card: QFrame | None = None
        self.category_config_card: QFrame | None = None
        self.tracking_config_card: QGroupBox | None = None
        self.actions_card: QFrame | None = None

        self.category_inputs: list[QLineEdit] = []
        self._model_switch_callback: Optional[Callable[[str, str], bool]] = None

        self._setup_ui()
        self._load_values()
        self._connect_signals()

    def _create_card(self, *, title: str | None = None, fixed_height: int | None = None) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setFrameShape(QFrame.Shape.NoFrame)
        card.setStyleSheet(frame_style(PANEL_BG))
        if fixed_height is not None:
            card.setFixedHeight(fixed_height)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(CARD_PADDING, CARD_PADDING, CARD_PADDING, CARD_PADDING)
        layout.setSpacing(CARD_SPACING)

        if title:
            layout.addWidget(self._create_card_title(title))

        return card, layout

    def _create_group_box(self, title: str) -> tuple[QGroupBox, QVBoxLayout]:
        group = QGroupBox(title, self)
        group.setStyleSheet(
            f"""
            QGroupBox {{
                color: {TEXT_ACCENT};
                font-size: {FONT_SECTION}px;
                font-weight: 700;
                border: 1px solid {STROKE_MAIN};
                border-radius: 10px;
                margin-top: 14px;
                padding-top: 10px;
                background-color: {PANEL_BG};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px 0 6px;
            }}
            """
        )
        layout = QVBoxLayout(group)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setSpacing(12)
        return group, layout

    def _create_label(self, text: str, *, accent: bool = True) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(text_style(TEXT_SECONDARY if accent else TEXT_PRIMARY, size=FONT_BODY))
        return label

    def _create_card_title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(text_style(TEXT_ACCENT, size=FONT_SECTION, weight=700))
        return label

    def _set_input_height(self, widget: QWidget, height: int = 34) -> None:
        widget.setMinimumHeight(height)
        widget.setMaximumHeight(height)

    def _setup_ui(self) -> None:
        self.setWindowTitle("系统配置")
        self.setMinimumSize(1120, 760)
        self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        self.setStyleSheet(
            f"""
            QWidget {{
                background-color: {PAGE_BG};
                color: {TEXT_PRIMARY};
                font-family: 'Microsoft YaHei UI';
            }}
            QLabel {{
                background: transparent;
                border: none;
            }}
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                background-color: #19345F;
                border: 1px solid #3E78C5;
                border-radius: 8px;
                color: #D4F8FF;
                font-size: {FONT_SMALL}px;
                padding: 4px 8px;
                min-height: 32px;
            }}
            QCheckBox {{
                color: #D4F8FF;
                spacing: 8px;
                background-color: #19345F;
                border: 1px solid #3E78C5;
                border-radius: 8px;
                padding: 4px 8px;
                font-size: {FONT_SMALL}px;
                min-height: 32px;
            }}
            QPushButton {{
                border-radius: 8px;
                font-weight: 700;
            }}
            """
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN)
        root.setSpacing(PAGE_SPACING)

        self.header_card, header_layout = self._create_card(fixed_height=56)
        header_layout.setContentsMargins(18, 10, 18, 10)
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(12)

        left_logo = QLabel("AI视觉")
        left_logo.setStyleSheet(text_style("#35D7FF", size=FONT_SECTION, weight=700))
        center_title = QLabel("系统配置")
        center_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_title.setStyleSheet(
            "background-color: #163A69; "
            f"border: 1px solid {STROKE_MAIN}; "
            "border-radius: 6px; "
            f"color: {TEXT_ACCENT}; font-size: {FONT_SECTION}px; font-weight: 700; padding: 6px 22px;"
        )
        right_tag = QLabel("配置页")
        right_tag.setStyleSheet(text_style(TEXT_SECONDARY, size=FONT_BODY, weight=600))

        header_row.addWidget(left_logo)
        header_row.addStretch()
        header_row.addWidget(center_title)
        header_row.addStretch()
        header_row.addWidget(right_tag)
        header_layout.addLayout(header_row)
        root.addWidget(self.header_card)

        self.info_card, info_layout = self._create_card(fixed_height=50)
        info_layout.setContentsMargins(18, 10, 18, 10)
        info_label = QLabel("说明: 修改参数后点击保存配置。选择模型会立即尝试切换。")
        info_label.setStyleSheet(text_style(TEXT_SECONDARY, size=FONT_BODY, weight=600))
        info_layout.addWidget(info_label)
        root.addWidget(self.info_card)

        content = QHBoxLayout()
        content.setSpacing(18)

        self.base_config_card, base_layout = self._create_group_box("基础配置")
        base_layout.setContentsMargins(12, 12, 12, 12)
        base_layout.setSpacing(8)
        self.camera_name_input = QLineEdit()
        self.camera_name_input.setPlaceholderText("输入相机名称")
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("输入 MediaPipe .task 模型文件路径")
        self.model_task_input = QComboBox()
        self.model_task_input.addItems(["mediapipe_gesture"])
        self.enable_gesture_detection_checkbox = QCheckBox("启用手势检测")
        self.enable_object_detection_checkbox = QCheckBox("启用目标检测")
        self.btn_select_model = QPushButton("选择模型")
        self.btn_select_model.setMinimumWidth(92)
        self.btn_select_model.setMaximumHeight(34)
        self.btn_select_model.setStyleSheet(
            chip_button_style(
                background="#163A69",
                border="#55A6FF",
                color="#EAF4FF",
                font_size=FONT_SMALL,
                min_height=34,
            )
        )
        self.conf_threshold_input = QDoubleSpinBox()
        self.conf_threshold_input.setDecimals(2)
        self.conf_threshold_input.setRange(0.0, 1.0)
        self.conf_threshold_input.setSingleStep(0.01)
        self.object_model_path_input = QLineEdit()
        self.object_model_path_input.setPlaceholderText("输入目标检测 .tflite 模型文件路径")
        self.object_score_threshold_input = QDoubleSpinBox()
        self.object_score_threshold_input.setDecimals(2)
        self.object_score_threshold_input.setRange(0.0, 1.0)
        self.object_score_threshold_input.setSingleStep(0.01)
        self.object_max_results_input = QSpinBox()
        self.object_max_results_input.setRange(1, 100)
        self.object_result_hold_ms_input = QSpinBox()
        self.object_result_hold_ms_input.setRange(0, 10000)
        self.yolo_model_path_input = QLineEdit()
        self.yolo_model_path_input.setPlaceholderText("输入 YOLO OBB .pt 模型文件路径")
        self.yolo_conf_threshold_input = QDoubleSpinBox()
        self.yolo_conf_threshold_input.setDecimals(2)
        self.yolo_conf_threshold_input.setRange(0.0, 1.0)
        self.yolo_conf_threshold_input.setSingleStep(0.01)
        self.yolo_iou_threshold_input = QDoubleSpinBox()
        self.yolo_iou_threshold_input.setDecimals(2)
        self.yolo_iou_threshold_input.setRange(0.0, 1.0)
        self.yolo_iou_threshold_input.setSingleStep(0.01)
        self.ultralytics_device_input = QLineEdit()
        self.ultralytics_device_input.setPlaceholderText("空=自动，0=GPU0，cpu=CPU")
        self.ultralytics_tracker_input = QLineEdit()
        self.ultralytics_tracker_input.setPlaceholderText("botsort.yaml")
        self.ultralytics_max_det_input = QSpinBox()
        self.ultralytics_max_det_input.setRange(1, 9999)
        self.ultralytics_track_persist_checkbox = QCheckBox("保持追踪 ID")
        self.mediapipe_detection_conf_input = QDoubleSpinBox()
        self.mediapipe_detection_conf_input.setDecimals(2)
        self.mediapipe_detection_conf_input.setRange(0.0, 1.0)
        self.mediapipe_detection_conf_input.setSingleStep(0.01)
        self.mediapipe_presence_conf_input = QDoubleSpinBox()
        self.mediapipe_presence_conf_input.setDecimals(2)
        self.mediapipe_presence_conf_input.setRange(0.0, 1.0)
        self.mediapipe_presence_conf_input.setSingleStep(0.01)
        self.mediapipe_tracking_conf_input = QDoubleSpinBox()
        self.mediapipe_tracking_conf_input.setDecimals(2)
        self.mediapipe_tracking_conf_input.setRange(0.0, 1.0)
        self.mediapipe_tracking_conf_input.setSingleStep(0.01)
        self.gesture_window_size_input = QSpinBox()
        self.gesture_window_size_input.setRange(1, 30)
        self.gesture_enter_frames_input = QSpinBox()
        self.gesture_enter_frames_input.setRange(1, 30)
        self.gesture_exit_frames_input = QSpinBox()
        self.gesture_exit_frames_input.setRange(1, 60)
        self.show_confidence_checkbox = QCheckBox("勾选后显示检测置信度")
        self.round_cooldown_input = QDoubleSpinBox()
        self.round_cooldown_input.setDecimals(1)
        self.round_cooldown_input.setRange(0.1, 60.0)
        self.round_cooldown_input.setSingleStep(0.1)
        self.today_target_capacity_input = QSpinBox()
        self.today_target_capacity_input.setRange(1, 999999)

        for widget in (
            self.camera_name_input,
            self.model_path_input,
            self.model_task_input,
            self.enable_gesture_detection_checkbox,
            self.enable_object_detection_checkbox,
            self.conf_threshold_input,
            self.object_model_path_input,
            self.object_score_threshold_input,
            self.object_max_results_input,
            self.object_result_hold_ms_input,
            self.yolo_model_path_input,
            self.yolo_conf_threshold_input,
            self.yolo_iou_threshold_input,
            self.ultralytics_device_input,
            self.ultralytics_tracker_input,
            self.ultralytics_max_det_input,
            self.ultralytics_track_persist_checkbox,
            self.mediapipe_detection_conf_input,
            self.mediapipe_presence_conf_input,
            self.mediapipe_tracking_conf_input,
            self.gesture_window_size_input,
            self.gesture_enter_frames_input,
            self.gesture_exit_frames_input,
            self.show_confidence_checkbox,
            self.round_cooldown_input,
            self.today_target_capacity_input,
        ):
            self._set_input_height(widget)

        self.base_config_grid = QGridLayout()
        self.base_config_grid.setHorizontalSpacing(12)
        self.base_config_grid.setVerticalSpacing(6)
        self.base_config_grid.setColumnStretch(0, 0)
        self.base_config_grid.setColumnStretch(1, 1)
        self.base_config_grid.setColumnStretch(2, 0)
        self.base_config_grid.setColumnStretch(3, 1)

        def add_base_item(row: int, column_pair: int, label_text: str, widget: QWidget) -> None:
            label = self._create_label(label_text)
            label.setMinimumWidth(82)
            label.setMaximumWidth(118)
            label.setStyleSheet(text_style(TEXT_SECONDARY, size=FONT_SMALL, weight=600))
            label_col = 0 if column_pair == 0 else 2
            widget_col = 1 if column_pair == 0 else 3
            self.base_config_grid.addWidget(label, row, label_col)
            self.base_config_grid.addWidget(widget, row, widget_col)

        model_path_row = QHBoxLayout()
        model_path_row.setContentsMargins(0, 0, 0, 0)
        model_path_row.setSpacing(6)
        model_path_row.addWidget(self.model_path_input, 1)
        model_path_row.addWidget(self.btn_select_model)
        model_path_widget = QWidget()
        model_path_widget.setLayout(model_path_row)
        model_path_widget.setMaximumHeight(36)

        interval_row = QHBoxLayout()
        interval_row.setContentsMargins(0, 0, 0, 0)
        interval_row.setSpacing(6)
        interval_row.addWidget(self.round_cooldown_input)
        interval_row.addWidget(self._create_label("秒", accent=False))
        interval_widget = QWidget()
        interval_widget.setLayout(interval_row)
        interval_widget.setMaximumHeight(36)

        capacity_row = QHBoxLayout()
        capacity_row.setContentsMargins(0, 0, 0, 0)
        capacity_row.setSpacing(6)
        capacity_row.addWidget(self.today_target_capacity_input)
        capacity_row.addWidget(self._create_label("PCS", accent=False))
        capacity_widget = QWidget()
        capacity_widget.setLayout(capacity_row)
        capacity_widget.setMaximumHeight(36)

        add_base_item(0, 0, "相机名称", self.camera_name_input)
        add_base_item(0, 1, "任务类型", self.model_task_input)
        add_base_item(1, 0, "手势模型", model_path_widget)
        add_base_item(1, 1, "目标模型", self.object_model_path_input)
        add_base_item(2, 0, "手势检测", self.enable_gesture_detection_checkbox)
        add_base_item(2, 1, "目标检测", self.enable_object_detection_checkbox)
        add_base_item(3, 0, "手势阈值", self.conf_threshold_input)
        add_base_item(3, 1, "目标阈值", self.object_score_threshold_input)
        add_base_item(4, 0, "手部检测", self.mediapipe_detection_conf_input)
        add_base_item(4, 1, "目标数量", self.object_max_results_input)
        add_base_item(5, 0, "手部存在", self.mediapipe_presence_conf_input)
        add_base_item(5, 1, "目标保持", self.object_result_hold_ms_input)
        add_base_item(6, 0, "跟踪阈值", self.mediapipe_tracking_conf_input)
        add_base_item(6, 1, "稳定窗口", self.gesture_window_size_input)
        add_base_item(7, 0, "进入帧数", self.gesture_enter_frames_input)
        add_base_item(7, 1, "退出帧数", self.gesture_exit_frames_input)
        add_base_item(8, 0, "显示置信度", self.show_confidence_checkbox)
        add_base_item(8, 1, "间隔时间", interval_widget)
        add_base_item(9, 0, "今日产能", capacity_widget)
        add_base_item(9, 1, "YOLO模型", self.yolo_model_path_input)
        add_base_item(10, 0, "YOLO置信度", self.yolo_conf_threshold_input)
        add_base_item(10, 1, "YOLO IoU", self.yolo_iou_threshold_input)
        add_base_item(11, 0, "推理设备", self.ultralytics_device_input)
        add_base_item(11, 1, "追踪器", self.ultralytics_tracker_input)
        add_base_item(12, 0, "最大目标数", self.ultralytics_max_det_input)
        add_base_item(12, 1, "追踪保持", self.ultralytics_track_persist_checkbox)
        base_layout.addLayout(self.base_config_grid)

        tip_card = QFrame()
        tip_card.setStyleSheet(frame_style(PANEL_BG_ALT, border=STROKE_SOFT, radius=8))
        tip_layout = QVBoxLayout(tip_card)
        tip_layout.setContentsMargins(14, 14, 14, 14)
        tip_layout.setSpacing(8)
        tip_layout.addWidget(self._create_card_title("配置说明"))
        tip_text = QLabel("模型路径支持手动输入和文件选择。通过“选择模型”切换失败时会保留旧模型。")
        tip_text.setWordWrap(True)
        tip_text.setStyleSheet(text_style(TEXT_PRIMARY, size=FONT_SMALL))
        tip_layout.addWidget(tip_text)
        base_layout.addWidget(tip_card)
        base_layout.addStretch()

        self.category_config_card, category_layout = self._create_group_box("应用类别配置")
        category_grid = QGridLayout()
        category_grid.setHorizontalSpacing(18)
        category_grid.setVerticalSpacing(18)
        self.category_inputs = []
        for idx in range(self.SLOT_COUNT):
            item_card = QFrame()
            item_card.setStyleSheet(frame_style(PANEL_BG_ALT, border=STROKE_SOFT, radius=8))
            item_layout = QVBoxLayout(item_card)
            item_layout.setContentsMargins(14, 14, 14, 14)
            item_layout.setSpacing(10)
            item_layout.addWidget(self._create_label(f"应用类别 {idx + 1}"))
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(f"输入应用类别 {idx + 1}")
            self._set_input_height(line_edit)
            item_layout.addWidget(line_edit)
            item_layout.addStretch()
            self.category_inputs.append(line_edit)
            category_grid.addWidget(item_card, idx // 2, idx % 2)
        category_layout.addLayout(category_grid)
        category_layout.addStretch()

        self.tracking_config_card, tracking_layout = self._create_group_box("扭力枪-洞位联合追踪")
        tracking_layout.setContentsMargins(12, 12, 12, 12)
        tracking_layout.setSpacing(8)
        self.joint_tracking_enabled_checkbox = QCheckBox("启用联合追踪")
        self.tool_class_name_input = QLineEdit()
        self.tool_class_name_input.setPlaceholderText("扭力枪")
        self.track_seed_stable_frames_input = QSpinBox()
        self.track_seed_stable_frames_input.setRange(1, 120)
        self.track_enter_stable_frames_input = QSpinBox()
        self.track_enter_stable_frames_input.setRange(1, 120)
        self.track_leave_stable_frames_input = QSpinBox()
        self.track_leave_stable_frames_input.setRange(1, 120)
        self.track_gun_lost_frames_input = QSpinBox()
        self.track_gun_lost_frames_input.setRange(1, 120)
        self.track_hole_occlusion_grace_frames_input = QSpinBox()
        self.track_hole_occlusion_grace_frames_input.setRange(1, 300)
        self.track_enter_threshold_input = QDoubleSpinBox()
        self.track_enter_threshold_input.setDecimals(2)
        self.track_enter_threshold_input.setRange(0.0, 1.0)
        self.track_enter_threshold_input.setSingleStep(0.01)
        self.track_leave_threshold_input = QDoubleSpinBox()
        self.track_leave_threshold_input.setDecimals(2)
        self.track_leave_threshold_input.setRange(0.0, 1.0)
        self.track_leave_threshold_input.setSingleStep(0.01)
        self.track_min_score_gap_input = QDoubleSpinBox()
        self.track_min_score_gap_input.setDecimals(2)
        self.track_min_score_gap_input.setRange(0.0, 1.0)
        self.track_min_score_gap_input.setSingleStep(0.01)
        self.track_order_constraint_checkbox = QCheckBox("限制顺序")
        self.track_order_rule_input = QLineEdit()
        self.track_order_rule_input.setPlaceholderText("configured_order")
        self.track_out_of_order_frames_input = QSpinBox()
        self.track_out_of_order_frames_input.setRange(1, 120)
        self.track_step_leave_frames_input = QSpinBox()
        self.track_step_leave_frames_input.setRange(1, 120)
        self.track_debug_csv_checkbox = QCheckBox("输出调试CSV")

        for widget in (
            self.joint_tracking_enabled_checkbox,
            self.tool_class_name_input,
            self.track_seed_stable_frames_input,
            self.track_enter_stable_frames_input,
            self.track_leave_stable_frames_input,
            self.track_gun_lost_frames_input,
            self.track_hole_occlusion_grace_frames_input,
            self.track_enter_threshold_input,
            self.track_leave_threshold_input,
            self.track_min_score_gap_input,
            self.track_order_constraint_checkbox,
            self.track_order_rule_input,
            self.track_out_of_order_frames_input,
            self.track_step_leave_frames_input,
            self.track_debug_csv_checkbox,
        ):
            self._set_input_height(widget)

        tracking_grid = QGridLayout()
        tracking_grid.setHorizontalSpacing(10)
        tracking_grid.setVerticalSpacing(6)
        tracking_grid.setColumnStretch(0, 0)
        tracking_grid.setColumnStretch(1, 1)

        def add_tracking_item(row: int, label_text: str, widget: QWidget) -> None:
            label = self._create_label(label_text)
            label.setMinimumWidth(94)
            label.setStyleSheet(text_style(TEXT_SECONDARY, size=FONT_SMALL, weight=600))
            tracking_grid.addWidget(label, row, 0)
            tracking_grid.addWidget(widget, row, 1)

        add_tracking_item(0, "联合追踪", self.joint_tracking_enabled_checkbox)
        add_tracking_item(1, "扭力枪类别", self.tool_class_name_input)
        add_tracking_item(2, "种子稳定帧", self.track_seed_stable_frames_input)
        add_tracking_item(3, "进入稳定帧", self.track_enter_stable_frames_input)
        add_tracking_item(4, "离开稳定帧", self.track_leave_stable_frames_input)
        add_tracking_item(5, "枪丢失帧", self.track_gun_lost_frames_input)
        add_tracking_item(6, "遮挡宽限帧", self.track_hole_occlusion_grace_frames_input)
        add_tracking_item(7, "进入阈值", self.track_enter_threshold_input)
        add_tracking_item(8, "离开阈值", self.track_leave_threshold_input)
        add_tracking_item(9, "最小分差", self.track_min_score_gap_input)
        add_tracking_item(10, "顺序约束", self.track_order_constraint_checkbox)
        add_tracking_item(11, "顺序规则", self.track_order_rule_input)
        add_tracking_item(12, "错序稳定帧", self.track_out_of_order_frames_input)
        add_tracking_item(13, "离开保护帧", self.track_step_leave_frames_input)
        add_tracking_item(14, "调试CSV", self.track_debug_csv_checkbox)
        tracking_layout.addLayout(tracking_grid)
        tracking_layout.addStretch()

        content.addWidget(self.base_config_card, 5)
        content.addWidget(self.category_config_card, 5)
        content.addWidget(self.tracking_config_card, 4)
        root.addLayout(content, 1)

        self.actions_card, actions_layout = self._create_card(fixed_height=108)
        actions_layout.setContentsMargins(18, 18, 18, 18)
        actions_row = QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(14)
        actions_row.addStretch()

        self.btn_image_save = QPushButton("图片保存")
        self.btn_image_save.setMinimumWidth(132)
        self.btn_image_save.setStyleSheet(
            chip_button_style(
                background="#136B7E",
                border="#54D5FF",
                color="#E8FBFF",
                font_size=FONT_BODY,
                min_height=44,
            )
        )
        self.btn_save = QPushButton("保存配置")
        self.btn_save.setMinimumWidth(132)
        self.btn_save.setStyleSheet(
            chip_button_style(
                background="#1C8E4F",
                border="#52E896",
                color="#E8FFF0",
                font_size=FONT_BODY,
                min_height=44,
            )
        )
        self.btn_back = QPushButton("返回主页")
        self.btn_back.setMinimumWidth(132)
        self.btn_back.setStyleSheet(
            chip_button_style(
                background="#214CCB",
                border="#55A6FF",
                color="#EAF4FF",
                font_size=FONT_BODY,
                min_height=44,
            )
        )

        actions_row.addWidget(self.btn_image_save)
        actions_row.addWidget(self.btn_save)
        actions_row.addWidget(self.btn_back)
        actions_layout.addLayout(actions_row)
        root.addWidget(self.actions_card)

    def _load_values(self) -> None:
        self.camera_name_input.setText(str(getattr(self.config, "mvsdk_friendly_name", "") or ""))
        self.sync_model_config_from_runtime(self.config.get_model_path(), self.config.get_model_task())
        self.enable_gesture_detection_checkbox.setChecked(bool(getattr(self.config, "enable_gesture_detection", True)))
        self.enable_object_detection_checkbox.setChecked(bool(getattr(self.config, "enable_object_detection", False)))
        self.conf_threshold_input.setValue(float(getattr(self.config, "mediapipe_score_threshold", 0.65)))
        self.object_model_path_input.setText(str(getattr(self.config, "object_model_path", "") or ""))
        self.object_score_threshold_input.setValue(float(getattr(self.config, "object_score_threshold", 0.3)))
        self.object_max_results_input.setValue(int(getattr(self.config, "object_max_results", 5)))
        self.object_result_hold_ms_input.setValue(int(getattr(self.config, "object_result_hold_ms", 250)))
        self.yolo_model_path_input.setText(str(getattr(self.config, "yolo_model_path", "../models/best.pt") or ""))
        self.yolo_conf_threshold_input.setValue(float(getattr(self.config, "yolo_conf_threshold", 0.5)))
        self.yolo_iou_threshold_input.setValue(float(getattr(self.config, "yolo_iou_threshold", 0.45)))
        self.ultralytics_device_input.setText(str(getattr(self.config, "ultralytics_device", "") or ""))
        self.ultralytics_tracker_input.setText(str(getattr(self.config, "ultralytics_tracker", "botsort.yaml") or "botsort.yaml"))
        self.ultralytics_max_det_input.setValue(int(getattr(self.config, "ultralytics_max_det", 300)))
        self.ultralytics_track_persist_checkbox.setChecked(
            bool(getattr(self.config, "ultralytics_track_persist", True))
        )
        self.mediapipe_detection_conf_input.setValue(
            float(getattr(self.config, "mediapipe_min_hand_detection_confidence", 0.5))
        )
        self.mediapipe_presence_conf_input.setValue(
            float(getattr(self.config, "mediapipe_min_hand_presence_confidence", 0.5))
        )
        self.mediapipe_tracking_conf_input.setValue(
            float(getattr(self.config, "mediapipe_min_tracking_confidence", 0.5))
        )
        self.gesture_window_size_input.setValue(int(getattr(self.config, "gesture_window_size", 5)))
        self.gesture_enter_frames_input.setValue(int(getattr(self.config, "gesture_enter_frames", 3)))
        self.gesture_exit_frames_input.setValue(int(getattr(self.config, "gesture_exit_frames", 5)))
        self.show_confidence_checkbox.setChecked(bool(getattr(self.config, "show_confidence_overlay", True)))
        self.round_cooldown_input.setValue(float(getattr(self.config, "round_cooldown_seconds", 2.0)))
        self.today_target_capacity_input.setValue(int(getattr(self.config, "today_target_capacity", 300)))

        current_names = list(getattr(self.config, "category_names", []))
        for idx, line_edit in enumerate(self.category_inputs):
            line_edit.setText(current_names[idx] if idx < len(current_names) else "")

        self.joint_tracking_enabled_checkbox.setChecked(bool(getattr(self.config, "joint_tracking_enabled", True)))
        self.tool_class_name_input.setText(str(getattr(self.config, "tool_class_name", "扭力枪") or "扭力枪"))
        self.track_seed_stable_frames_input.setValue(int(getattr(self.config, "track_seed_stable_frames", 3)))
        self.track_enter_stable_frames_input.setValue(int(getattr(self.config, "track_enter_stable_frames", 2)))
        self.track_leave_stable_frames_input.setValue(int(getattr(self.config, "track_leave_stable_frames", 4)))
        self.track_gun_lost_frames_input.setValue(int(getattr(self.config, "track_gun_lost_frames", 6)))
        self.track_hole_occlusion_grace_frames_input.setValue(
            int(getattr(self.config, "track_hole_occlusion_grace_frames", 12))
        )
        self.track_enter_threshold_input.setValue(float(getattr(self.config, "track_enter_threshold", 0.18)))
        self.track_leave_threshold_input.setValue(float(getattr(self.config, "track_leave_threshold", 0.08)))
        self.track_min_score_gap_input.setValue(float(getattr(self.config, "track_min_score_gap", 0.05)))
        self.track_order_constraint_checkbox.setChecked(
            bool(getattr(self.config, "track_order_constraint_enabled", True))
        )
        self.track_order_rule_input.setText(
            str(getattr(self.config, "track_order_rule", "configured_order") or "configured_order")
        )
        self.track_out_of_order_frames_input.setValue(int(getattr(self.config, "track_out_of_order_frames", 2)))
        self.track_step_leave_frames_input.setValue(int(getattr(self.config, "track_step_leave_frames", 4)))
        self.track_debug_csv_checkbox.setChecked(bool(getattr(self.config, "track_debug_csv_enabled", False)))

    def _connect_signals(self) -> None:
        self.btn_save.clicked.connect(self._on_save_clicked)
        self.btn_back.clicked.connect(self.back_clicked.emit)
        self.btn_image_save.clicked.connect(self._on_image_save_clicked)
        self.btn_select_model.clicked.connect(self._on_select_model_clicked)

    def set_model_switch_callback(self, callback: Callable[[str, str], bool]) -> None:
        self._model_switch_callback = callback

    def sync_model_config_from_runtime(self, model_path: str, model_task: str) -> None:
        self.model_path_input.setText(str(model_path or ""))
        task_index = self.model_task_input.findText(str(model_task or "mediapipe_gesture").strip().lower())
        if task_index < 0:
            task_index = self.model_task_input.findText("mediapipe_gesture")
        self.model_task_input.setCurrentIndex(max(0, task_index))

    def _normalize_model_path_for_display(self, file_path: str) -> str:
        selected = Path(file_path).resolve()
        config_path = getattr(self.config, "_config_path", None)
        if config_path:
            base_dir = Path(config_path).resolve().parent
            try:
                import os

                return Path(os.path.relpath(selected, base_dir)).as_posix()
            except ValueError:
                pass
        return str(selected)

    def _resolve_model_dialog_start_dir(self) -> str:
        current_text = self.model_path_input.text().strip() or self.config.get_model_path()
        config_path = getattr(self.config, "_config_path", None)
        base_dir = Path(config_path).resolve().parent if config_path else Path.cwd()
        candidate = Path(current_text).expanduser()

        try:
            if candidate.is_absolute():
                resolved = candidate.resolve()
            else:
                resolved = (base_dir / candidate).resolve()
        except Exception:
            resolved = base_dir

        if resolved.exists():
            return str(resolved if resolved.is_dir() else resolved.parent)

        parent = resolved.parent
        if parent.exists():
            return str(parent)
        return str(base_dir)

    def _attempt_model_switch(self, model_path: str, model_task: str) -> bool:
        if self._model_switch_callback is None:
            return True
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            return bool(self._model_switch_callback(model_path, model_task))
        finally:
            QApplication.restoreOverrideCursor()

    def _on_select_model_clicked(self) -> None:
        start_dir = self._resolve_model_dialog_start_dir()

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 MediaPipe task 模型",
            start_dir,
            "MediaPipe task 模型 (*.task)",
        )
        if not file_path:
            return

        normalized_path = self._normalize_model_path_for_display(file_path)
        previous_path = self.config.get_model_path()
        previous_task = self.config.get_model_task()
        requested_task = "mediapipe_gesture"
        self.model_path_input.setText(normalized_path)

        if not self._attempt_model_switch(normalized_path, requested_task):
            self.sync_model_config_from_runtime(previous_path, previous_task)
            return

        self.sync_model_config_from_runtime(self.config.get_model_path(), self.config.get_model_task())

    def _on_detection_changed(self, value: int) -> None:
        """兼容旧测试与旧交互。"""
        confidence = value / 100.0
        if hasattr(self, "conf_threshold_input") and self.conf_threshold_input is not None:
            self.conf_threshold_input.setValue(confidence)
        if hasattr(self, "detection_value_label") and self.detection_value_label is not None:
            self.detection_value_label.setText(f"{value}%")
        self.config.conf_threshold = confidence
        self.config.mediapipe_score_threshold = confidence
        self.config.min_detection_confidence = confidence
        self.config.save()
        self.config_changed.emit("mediapipe_score_threshold", confidence)

    def _on_image_save_clicked(self) -> None:
        output_dir = Path(__file__).resolve().parents[2] / "offline_validation"
        output_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_dir)))

    def _on_save_clicked(self) -> None:
        previous = self.config.to_dict()
        requested_model_path = self.model_path_input.text().strip() or self.config.get_model_path()
        requested_model_task = "mediapipe_gesture"

        self.config.mvsdk_friendly_name = self.camera_name_input.text().strip()
        self.config.mediapipe_task_path = requested_model_path
        self.config.model_path = requested_model_path
        self.config.model_task = requested_model_task
        self.config.enable_gesture_detection = bool(self.enable_gesture_detection_checkbox.isChecked())
        self.config.enable_object_detection = bool(self.enable_object_detection_checkbox.isChecked())
        self.config.object_model_path = self.object_model_path_input.text().strip()
        self.config.object_score_threshold = float(self.object_score_threshold_input.value())
        self.config.object_max_results = int(self.object_max_results_input.value())
        self.config.object_result_hold_ms = int(self.object_result_hold_ms_input.value())
        self.config.yolo_model_path = self.yolo_model_path_input.text().strip()
        self.config.yolo_conf_threshold = float(self.yolo_conf_threshold_input.value())
        self.config.yolo_iou_threshold = float(self.yolo_iou_threshold_input.value())
        self.config.ultralytics_device = self.ultralytics_device_input.text().strip()
        self.config.ultralytics_tracker = self.ultralytics_tracker_input.text().strip()
        self.config.ultralytics_max_det = int(self.ultralytics_max_det_input.value())
        self.config.ultralytics_track_persist = bool(self.ultralytics_track_persist_checkbox.isChecked())
        self.config.mediapipe_score_threshold = float(self.conf_threshold_input.value())
        self.config.mediapipe_min_hand_detection_confidence = float(self.mediapipe_detection_conf_input.value())
        self.config.mediapipe_min_hand_presence_confidence = float(self.mediapipe_presence_conf_input.value())
        self.config.mediapipe_min_tracking_confidence = float(self.mediapipe_tracking_conf_input.value())
        self.config.gesture_window_size = int(self.gesture_window_size_input.value())
        self.config.gesture_enter_frames = int(self.gesture_enter_frames_input.value())
        self.config.gesture_exit_frames = int(self.gesture_exit_frames_input.value())
        self.config.conf_threshold = float(self.conf_threshold_input.value())
        self.config.min_detection_confidence = float(self.mediapipe_detection_conf_input.value())
        self.config.show_confidence_overlay = bool(self.show_confidence_checkbox.isChecked())
        self.config.round_cooldown_seconds = float(self.round_cooldown_input.value())
        self.config.today_target_capacity = int(self.today_target_capacity_input.value())
        self.config.category_names = [line_edit.text().strip() for line_edit in self.category_inputs]
        self.config.joint_tracking_enabled = bool(self.joint_tracking_enabled_checkbox.isChecked())
        self.config.tool_class_name = self.tool_class_name_input.text().strip()
        self.config.track_seed_stable_frames = int(self.track_seed_stable_frames_input.value())
        self.config.track_enter_stable_frames = int(self.track_enter_stable_frames_input.value())
        self.config.track_leave_stable_frames = int(self.track_leave_stable_frames_input.value())
        self.config.track_gun_lost_frames = int(self.track_gun_lost_frames_input.value())
        self.config.track_hole_occlusion_grace_frames = int(self.track_hole_occlusion_grace_frames_input.value())
        self.config.track_enter_threshold = float(self.track_enter_threshold_input.value())
        self.config.track_leave_threshold = float(self.track_leave_threshold_input.value())
        self.config.track_min_score_gap = float(self.track_min_score_gap_input.value())
        self.config.track_order_constraint_enabled = bool(self.track_order_constraint_checkbox.isChecked())
        self.config.track_order_rule = self.track_order_rule_input.text().strip()
        self.config.track_out_of_order_frames = int(self.track_out_of_order_frames_input.value())
        self.config.track_step_leave_frames = int(self.track_step_leave_frames_input.value())
        self.config.track_debug_csv_enabled = bool(self.track_debug_csv_checkbox.isChecked())
        self.config.validate()

        runtime_keys = (
            "mediapipe_task_path",
            "enable_gesture_detection",
            "enable_object_detection",
            "object_model_path",
            "object_score_threshold",
            "object_max_results",
            "object_result_hold_ms",
            "mediapipe_score_threshold",
            "mediapipe_min_hand_detection_confidence",
            "mediapipe_min_hand_presence_confidence",
            "mediapipe_min_tracking_confidence",
            "gesture_window_size",
            "gesture_enter_frames",
            "gesture_exit_frames",
            "category_names",
        )
        runtime_changed = any(previous.get(key) != self.config.to_dict().get(key) for key in runtime_keys)
        if runtime_changed and not self._attempt_model_switch(self.config.get_model_path(), requested_model_task):
            self.config.update(**previous)
            self.sync_model_config_from_runtime(self.config.get_model_path(), self.config.get_model_task())
            return

        self.config.save()

        self.config_changed.emit("mvsdk_friendly_name", self.config.mvsdk_friendly_name)
        self.config_changed.emit("model_path", self.config.get_model_path())
        self.config_changed.emit("model_task", self.config.get_model_task())
        self.config_changed.emit("enable_gesture_detection", self.config.enable_gesture_detection)
        self.config_changed.emit("enable_object_detection", self.config.enable_object_detection)
        self.config_changed.emit("object_model_path", self.config.object_model_path)
        self.config_changed.emit("object_score_threshold", self.config.object_score_threshold)
        self.config_changed.emit("object_max_results", self.config.object_max_results)
        self.config_changed.emit("object_result_hold_ms", self.config.object_result_hold_ms)
        self.config_changed.emit("yolo_model_path", self.config.yolo_model_path)
        self.config_changed.emit("yolo_conf_threshold", self.config.yolo_conf_threshold)
        self.config_changed.emit("yolo_iou_threshold", self.config.yolo_iou_threshold)
        self.config_changed.emit("ultralytics_device", self.config.ultralytics_device)
        self.config_changed.emit("ultralytics_tracker", self.config.ultralytics_tracker)
        self.config_changed.emit("ultralytics_max_det", self.config.ultralytics_max_det)
        self.config_changed.emit("ultralytics_track_persist", self.config.ultralytics_track_persist)
        self.config_changed.emit("mediapipe_score_threshold", self.config.mediapipe_score_threshold)
        self.config_changed.emit(
            "mediapipe_min_hand_detection_confidence",
            self.config.mediapipe_min_hand_detection_confidence,
        )
        self.config_changed.emit(
            "mediapipe_min_hand_presence_confidence",
            self.config.mediapipe_min_hand_presence_confidence,
        )
        self.config_changed.emit("mediapipe_min_tracking_confidence", self.config.mediapipe_min_tracking_confidence)
        self.config_changed.emit("gesture_window_size", self.config.gesture_window_size)
        self.config_changed.emit("gesture_enter_frames", self.config.gesture_enter_frames)
        self.config_changed.emit("gesture_exit_frames", self.config.gesture_exit_frames)
        self.config_changed.emit("show_confidence_overlay", self.config.show_confidence_overlay)
        self.config_changed.emit("round_cooldown_seconds", self.config.round_cooldown_seconds)
        self.config_changed.emit("today_target_capacity", self.config.today_target_capacity)
        self.config_changed.emit("category_names", list(self.config.category_names))
        self.config_changed.emit("joint_tracking_enabled", self.config.joint_tracking_enabled)
        self.config_changed.emit("tool_class_name", self.config.tool_class_name)
        self.config_changed.emit("track_seed_stable_frames", self.config.track_seed_stable_frames)
        self.config_changed.emit("track_enter_stable_frames", self.config.track_enter_stable_frames)
        self.config_changed.emit("track_leave_stable_frames", self.config.track_leave_stable_frames)
        self.config_changed.emit("track_gun_lost_frames", self.config.track_gun_lost_frames)
        self.config_changed.emit("track_hole_occlusion_grace_frames", self.config.track_hole_occlusion_grace_frames)
        self.config_changed.emit("track_enter_threshold", self.config.track_enter_threshold)
        self.config_changed.emit("track_leave_threshold", self.config.track_leave_threshold)
        self.config_changed.emit("track_min_score_gap", self.config.track_min_score_gap)
        self.config_changed.emit("track_order_constraint_enabled", self.config.track_order_constraint_enabled)
        self.config_changed.emit("track_order_rule", self.config.track_order_rule)
        self.config_changed.emit("track_out_of_order_frames", self.config.track_out_of_order_frames)
        self.config_changed.emit("track_step_leave_frames", self.config.track_step_leave_frames)
        self.config_changed.emit("track_debug_csv_enabled", self.config.track_debug_csv_enabled)
        self.config_saved.emit()
        QMessageBox.information(self, "保存成功", "配置已保存，立即生效。")
