"""主窗口模块。"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QPoint, QTimer, Qt, QUrl, Slot
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QPolygon
from PySide6.QtMultimedia import QSoundEffect
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from camera_worker import CameraWorker
from config import ConfigManager
from mediapipe_frame_processor import MediaPipeGestureProcessor
from mediapipe_object_processor import create_frame_processor_from_config
from models import DetectionOverlayState
from mvsdk_camera import MvSdkCamera
from state import AppState
from ui.config_page import ConfigPage
from ui.runtime_ui_tokens import (
    CARD_PADDING,
    CARD_SPACING,
    CHIP_BG,
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_WINDOW_WIDTH,
    FONT_BODY,
    FONT_METRIC,
    FONT_METRIC_LARGE,
    FONT_SECTION,
    FONT_SMALL,
    PAGE_BG,
    PAGE_MARGIN,
    PAGE_SPACING,
    PANEL_BG,
    PANEL_BG_ALT,
    PANEL_BG_DARK,
    PANEL_BG_MID,
    PANEL_BG_SOFT,
    STROKE_MAIN,
    STROKE_MUTED,
    STROKE_SOFT,
    TABLE_ROW_HEIGHT,
    TEXT_ACCENT,
    TEXT_DANGER,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_SUCCESS,
    TEXT_WARNING,
    VIEWPORT_BG,
    chip_button_style,
    frame_style,
    table_style,
    text_style,
)
from ui.widgets.native_panels import DefectChip, MetricCard, ResultRowWidget, StatusBadge, ThumbnailTile


class MainWindow(QMainWindow):
    """基于 pencil-new.pen 的运行时主窗口。"""

    PAGE_MAIN = 0
    PAGE_CONFIG = 1
    VISIBLE_SLOT_COUNT = 6

    def __init__(self, config: ConfigManager, state: AppState):
        super().__init__()
        self.config = config
        self.state = state

        self.worker: CameraWorker | None = None
        self.config_page: ConfigPage | None = None

        self._active_gesture_names: set[str] = set()
        self._round_completed = False
        self._total_rounds = 0
        self._good_rounds = 0
        self._action_ng_count = 0
        self._product_ng_count = 0
        self._last_cycle_seconds = 0.0

        self.dashboard_root: QFrame | None = None
        self.dashboard_main_viewport: QFrame | None = None
        self.dashboard_result_tag_label: QPushButton | None = None
        self.dashboard_open_camera_button: QPushButton | None = None
        self.dashboard_start_button: QPushButton | None = None
        self.dashboard_stop_button: QPushButton | None = None
        self.dashboard_test_button: QPushButton | None = None
        self.dashboard_reset_button: QPushButton | None = None
        self.dashboard_config_button: QPushButton | None = None
        self.dashboard_time_label: QLabel | None = None
        self.dashboard_project_label: QLabel | None = None
        self.dashboard_inspector_label: QLabel | None = None
        self.dashboard_mode_value_label: QLabel | None = None
        self.dashboard_state_value_label: QLabel | None = None
        self.dashboard_status_fps_label: QLabel | None = None
        self.dashboard_capacity_value_label: QLabel | None = None
        self.dashboard_capacity_target_label: QLabel | None = None
        self.dashboard_yield_value_label: QLabel | None = None
        self.dashboard_total_rounds_label: QLabel | None = None
        self.dashboard_good_rounds_label: QLabel | None = None
        self.dashboard_action_ng_label: QLabel | None = None
        self.dashboard_product_ng_label: QLabel | None = None
        self.dashboard_cycle_time_label: QLabel | None = None
        self.dashboard_fastline_label: QLabel | None = None
        self.dashboard_choose_button: QPushButton | None = None
        self.dashboard_close_button: QPushButton | None = None
        self.video_label: QLabel | None = None
        self._gesture_labels: list[QLabel] = []
        self._gesture_name_to_index: dict[str, int] = {}
        self._thumbnail_status_labels: list[QLabel] = []
        self._thumbnail_name_labels: list[QLabel] = []
        self._summary_step_name_labels: list[QLabel] = []
        self._summary_step_result_labels: list[QLabel] = []
        self._stats_ok_labels: list[QTableWidgetItem] = []
        self._stats_ng_labels: list[QTableWidgetItem] = []
        self._stats_rate_labels: list[QTableWidgetItem] = []
        self._stats_pt_labels: list[QTableWidgetItem] = []
        self.dashboard_defect_name_labels: list[QLabel] = []
        self.dashboard_defect_value_labels: list[QLabel] = []
        self.dashboard_stats_table: QTableWidget | None = None
        self.dashboard_process_rows: list[QWidget] = []

        self._setup_ui()
        self._setup_worker()
        self._setup_audio()
        self._connect_signals()
        self._setup_shortcuts()

        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(33)
        self._preview_timer.timeout.connect(self._poll_latest_frame)

        self._clock_timer = QTimer(self)
        self._clock_timer.setInterval(1000)
        self._clock_timer.timeout.connect(self._update_clock_text)
        self._clock_timer.start()

        self._update_clock_text()
        self._refresh_cameras()
        self._apply_runtime_status("READY", state_text="待机")
        self._refresh_model_runtime_labels()
        self._update_dashboard_category_texts()
        self._refresh_dashboard_metrics()
        self._refresh_dashboard_action_states()

    def _setup_ui(self) -> None:
        self.setWindowTitle("双翼科技视觉行为引导系统")
        self.setMinimumSize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        self.setStyleSheet(
            f"""
            QWidget {{
                background-color: {PAGE_BG};
                color: {TEXT_PRIMARY};
                font-family: 'Microsoft YaHei UI';
            }}
            QFrame {{
                border: none;
            }}
            QLabel {{
                background: transparent;
            }}
            QPushButton {{
                border-radius: 8px;
                font-weight: 700;
                font-size: {FONT_BODY}px;
            }}
            """
        )

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.main_page = QWidget()
        self._setup_main_page()
        self.stacked_widget.addWidget(self.main_page)

        self.config_page = ConfigPage(self.config)
        self.config_page.set_model_switch_callback(self._try_switch_model)
        self.config_page.back_clicked.connect(self._show_main_page)
        self.config_page.config_changed.connect(self._on_config_changed)
        self.config_page.config_saved.connect(self._on_config_saved)
        self.stacked_widget.addWidget(self.config_page)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.fps_label = QLabel("FPS: --")
        self.camera_status_label = QLabel("相机: 关闭")
        self.inference_status_label = QLabel("检测: 关闭")
        self.status_bar.addPermanentWidget(self.fps_label)
        self.status_bar.addPermanentWidget(self.camera_status_label)
        self.status_bar.addPermanentWidget(self.inference_status_label)

    def _create_card(self, *, background: str = PANEL_BG, border: str = STROKE_MAIN) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setStyleSheet(frame_style(background, border=border))
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(CARD_PADDING, CARD_PADDING, CARD_PADDING, CARD_PADDING)
        layout.setSpacing(CARD_SPACING)
        return frame, layout

    def _create_section_title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(text_style(TEXT_ACCENT, size=FONT_SECTION, weight=700))
        return label

    def _create_outline_button(
        self,
        text: str,
        *,
        background: str = "#163A69",
        border: str = "#2876D9",
        color: str = "#D7F8FF",
        min_width: int = 88,
        min_height: int = 36,
        font_size: int = FONT_BODY,
    ) -> QPushButton:
        button = QPushButton(text)
        button.setMinimumWidth(min_width)
        button.setStyleSheet(
            chip_button_style(
                background=background,
                border=border,
                color=color,
                font_size=font_size,
                min_height=min_height,
                disabled_background="#0F2148",
            )
        )
        return button

    def _create_metric_value(self, text: str, *, color: str = TEXT_PRIMARY, size: int = FONT_METRIC) -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(text_style(color, size=size, weight=700))
        return label

    def _setup_main_page(self) -> None:
        pen_scale = 1.24
        s = lambda value: int(round(value * pen_scale))

        root = QVBoxLayout(self.main_page)
        root.setContentsMargins(PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN)
        root.setSpacing(PAGE_SPACING)

        self.dashboard_root, dashboard_layout = self._create_card(background=PANEL_BG_DARK, border=STROKE_MAIN)
        self.dashboard_root.setMaximumWidth(s(1038))
        dashboard_layout.setContentsMargins(18, 18, 18, 18)
        dashboard_layout.setSpacing(14)
        root.addWidget(self.dashboard_root)
        root.setAlignment(self.dashboard_root, Qt.AlignmentFlag.AlignHCenter)

        header_card, header_layout = self._create_card(background=PANEL_BG_SOFT)
        header_card.setFixedHeight(72)
        header_layout.setContentsMargins(18, 8, 18, 8)
        header_row = QHBoxLayout()
        header_row.setSpacing(14)
        left_logo = QLabel("AI视觉")
        left_logo.setStyleSheet(text_style("#35D7FF", size=FONT_SECTION, weight=700))
        center_title = QLabel("双翼科技视觉行为引导系统")
        center_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_title.setStyleSheet(
            f"background-color: #12386F; border: 1px solid {STROKE_MAIN}; border-radius: 6px; "
            f"color: {TEXT_PRIMARY}; font-size: 18px; font-weight: 800; padding: 6px 28px;"
        )
        self.dashboard_time_label = QLabel("--")
        self.dashboard_time_label.setStyleSheet(text_style(TEXT_SECONDARY, size=15, weight=600))
        user_icon = QLabel("👤")
        user_icon.setStyleSheet(text_style("#35D7FF", size=15, weight=600))
        user_text = QLabel("张三")
        user_text.setStyleSheet(text_style(TEXT_SECONDARY, size=15, weight=600))
        self.dashboard_config_button = self._create_outline_button("配置", background="#163A69", border=STROKE_MAIN, min_width=84)
        self.dashboard_close_button = self._create_outline_button("关闭", background="#163A69", border=STROKE_MAIN, min_width=84)

        header_row.addWidget(left_logo)
        header_row.addStretch()
        header_row.addWidget(center_title)
        header_row.addStretch()
        header_row.addWidget(self.dashboard_time_label)
        header_row.addWidget(user_icon)
        header_row.addWidget(user_text)
        header_row.addWidget(self.dashboard_config_button)
        header_row.addWidget(self.dashboard_close_button)
        header_layout.addLayout(header_row)
        dashboard_layout.addWidget(header_card)

        info_card, info_layout = self._create_card(background="#0F2148", border=STROKE_SOFT)
        info_card.setFixedHeight(74)
        info_layout.setContentsMargins(16, 8, 16, 8)
        info_row = QHBoxLayout()
        info_row.setSpacing(14)
        self.dashboard_project_label = QLabel("项目: 动作检测")
        self.dashboard_project_label.setStyleSheet(text_style(TEXT_SECONDARY, size=15, weight=600))
        self.dashboard_choose_button = self._create_outline_button("选择", background="#163A69", border=STROKE_MAIN, min_width=72)
        self.dashboard_inspector_label = QLabel("检测员: 系统")
        self.dashboard_inspector_label.setStyleSheet(text_style(TEXT_SECONDARY, size=15, weight=600))
        self.dashboard_mode_value_label = QLabel("模式: 自动模式")
        self.dashboard_mode_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dashboard_mode_value_label.setMinimumWidth(168)
        self.dashboard_mode_value_label.setStyleSheet(text_style(TEXT_SECONDARY, size=18, weight=700))
        state_prefix = QLabel("状态:")
        state_prefix.setStyleSheet(text_style(TEXT_SECONDARY, size=18, weight=700))
        self.dashboard_state_value_label = QLabel("待机")
        self.dashboard_state_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dashboard_state_value_label.setMinimumWidth(72)
        self.dashboard_state_value_label.setStyleSheet(text_style("#28E26D", size=20, weight=800))
        result_prefix = QLabel("结果:")
        result_prefix.setStyleSheet(text_style(TEXT_SECONDARY, size=18, weight=700))

        self.dashboard_result_tag_label = QPushButton("READY")
        self.dashboard_result_tag_label.setEnabled(False)
        self.dashboard_result_tag_label.setMinimumWidth(140)
        self.dashboard_result_tag_label.setStyleSheet(
            chip_button_style(
                background="#155084",
                border="#35D7FF",
                color=TEXT_PRIMARY,
                font_size=18,
                min_height=42,
                padding="6px 18px",
            )
        )

        info_row.addWidget(self.dashboard_project_label)
        info_row.addWidget(self.dashboard_choose_button)
        info_row.addWidget(self.dashboard_inspector_label)
        info_row.addStretch()
        info_row.addWidget(self.dashboard_mode_value_label)
        info_row.addWidget(state_prefix)
        info_row.addWidget(self.dashboard_state_value_label)
        info_row.addWidget(result_prefix)
        info_row.addWidget(self.dashboard_result_tag_label)
        info_layout.addLayout(info_row)
        dashboard_layout.addWidget(info_card)

        content_grid_container = QWidget(self.dashboard_root)
        content_grid = QGridLayout(content_grid_container)
        content_grid.setContentsMargins(0, 0, 0, 0)
        content_grid.setHorizontalSpacing(16)
        content_grid.setVerticalSpacing(16)
        content_grid.setColumnStretch(0, 0)
        content_grid.setColumnStretch(1, 0)
        content_grid.setColumnStretch(2, 0)
        content_grid.setRowStretch(0, 0)
        content_grid.setRowStretch(1, 0)

        main_preview_card, main_preview_layout = self._create_card(background="#101D41")
        main_preview_card.setFixedWidth(s(411))
        main_preview_card.setFixedHeight(410)
        main_preview_layout.addWidget(self._create_section_title("视频主区域"))
        self.dashboard_main_viewport = QFrame()
        self.dashboard_main_viewport.setStyleSheet(
            frame_style(VIEWPORT_BG, border=STROKE_MUTED, radius=8)
        )
        self.dashboard_main_viewport.setMinimumHeight(320)
        preview_layout = QVBoxLayout(self.dashboard_main_viewport)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        self.video_label = QLabel("点击“打开相机”开始")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumWidth(480)
        self.video_label.setMinimumHeight(286)
        self.video_label.setMouseTracking(True)
        self.video_label.setStyleSheet(
            f"QLabel {{ background-color: {VIEWPORT_BG}; color: {TEXT_PRIMARY}; border: none; font-size: {FONT_BODY}px; }}"
        )
        preview_layout.addWidget(self.video_label, 1)
        main_preview_layout.addWidget(self.dashboard_main_viewport, 1)

        preview_footer = QHBoxLayout()
        preview_footer.setSpacing(12)
        self.dashboard_status_fps_label = QLabel("预览 0.0 / 推理 0.0 / 丢帧 0")
        self.dashboard_status_fps_label.setStyleSheet(text_style(TEXT_SECONDARY, size=FONT_SMALL, weight=600))
        preview_footer.addStretch()
        preview_footer.addWidget(self.dashboard_status_fps_label)
        main_preview_layout.addLayout(preview_footer)
        content_grid.addWidget(main_preview_card, 0, 0, alignment=Qt.AlignmentFlag.AlignTop)

        middle_column_container = QWidget(self.dashboard_root)
        middle_column = QVBoxLayout(middle_column_container)
        middle_column.setSpacing(16)
        middle_column.setContentsMargins(0, 0, 0, 0)

        defect_card, defect_layout = self._create_card(background=PANEL_BG_MID)
        defect_card.setFixedWidth(s(235))
        defect_card.setFixedHeight(s(180))
        defect_layout.addWidget(self._create_section_title("不良率"))
        defect_grid = QGridLayout()
        defect_grid.setHorizontalSpacing(10)
        defect_grid.setVerticalSpacing(10)
        self.dashboard_defect_name_labels = []
        self.dashboard_defect_value_labels = []
        for idx in range(self.VISIBLE_SLOT_COUNT):
            chip = DefectChip(f"类别{idx + 1}", "0.00%", self)
            chip.setMinimumSize(s(100), s(28) + 6)
            self.dashboard_defect_name_labels.append(chip.name_label)
            self.dashboard_defect_value_labels.append(chip.value_label)
            defect_grid.addWidget(chip, idx // 2, idx % 2)
        defect_layout.addLayout(defect_grid)
        middle_column.addWidget(defect_card)

        capacity_card, capacity_layout = self._create_card(background=PANEL_BG_MID)
        capacity_card.setFixedWidth(s(235))
        capacity_card.setFixedHeight(s(118) + 10)
        capacity_layout.addWidget(self._create_section_title("产能"))
        capacity_row = QHBoxLayout()
        capacity_row.setSpacing(12)
        yield_card = QFrame()
        yield_card.setStyleSheet(frame_style(CHIP_BG, border=STROKE_SOFT, radius=8))
        yield_layout = QVBoxLayout(yield_card)
        yield_layout.setContentsMargins(10, 10, 10, 10)
        yield_layout.setSpacing(4)
        yield_title = QLabel("良率")
        yield_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        yield_title.setStyleSheet(text_style(TEXT_SECONDARY, size=FONT_SMALL, weight=600))
        self.dashboard_yield_value_label = self._create_metric_value("0.00%", color=TEXT_SUCCESS, size=22)
        yield_layout.addWidget(yield_title)
        yield_layout.addWidget(self.dashboard_yield_value_label)
        capacity_row.addWidget(yield_card, 2)

        capacity_metric = QVBoxLayout()
        capacity_metric.setSpacing(6)
        self.dashboard_capacity_value_label = self._create_metric_value("0/300", color=TEXT_PRIMARY, size=FONT_METRIC_LARGE)
        self.dashboard_capacity_target_label = QLabel("PCS / 今日")
        self.dashboard_capacity_target_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dashboard_capacity_target_label.setStyleSheet(text_style(TEXT_SECONDARY, size=FONT_BODY, weight=600))
        capacity_metric.addWidget(self.dashboard_capacity_value_label)
        capacity_metric.addWidget(self.dashboard_capacity_target_label)
        capacity_metric.addStretch()
        capacity_row.addLayout(capacity_metric, 5)
        capacity_layout.addLayout(capacity_row)
        capacity_layout.addStretch()
        middle_column.addWidget(capacity_card)
        content_grid.addWidget(middle_column_container, 0, 1, alignment=Qt.AlignmentFlag.AlignTop)

        right_column_container = QWidget(self.dashboard_root)
        right_column_container.setFixedWidth(s(345))
        right_column_layout = QVBoxLayout(right_column_container)
        right_column_layout.setContentsMargins(0, 0, 0, 0)
        right_column_layout.setSpacing(16)

        summary_card, summary_layout = self._create_card(background=PANEL_BG_MID)
        summary_card.setFixedWidth(s(345))
        summary_card.setFixedHeight(310)
        summary_layout.setSpacing(6)
        summary_layout.addWidget(self._create_section_title("结果总览"))

        metrics_grid = QGridLayout()
        metrics_grid.setHorizontalSpacing(6)
        metrics_grid.setVerticalSpacing(4)
        self.dashboard_total_rounds_label = self._create_metric_value("0", size=14)
        self.dashboard_good_rounds_label = self._create_metric_value("0", size=14)
        self.dashboard_action_ng_label = self._create_metric_value("0", color=TEXT_WARNING, size=14)
        self.dashboard_product_ng_label = self._create_metric_value("0", color=TEXT_DANGER, size=14)
        metric_pairs = [
            ("检测数量", self.dashboard_total_rounds_label),
            ("良品数", self.dashboard_good_rounds_label),
            ("动作NG", self.dashboard_action_ng_label),
            ("产品NG", self.dashboard_product_ng_label),
        ]
        for idx, (title, value_label) in enumerate(metric_pairs):
            metric_card = MetricCard(title, "0", self)
            metric_card.replace_value_widget(value_label)
            metrics_grid.addWidget(metric_card, 0, idx)
        summary_layout.addLayout(metrics_grid)

        process_header = QHBoxLayout()
        process_header.setSpacing(8)
        for text, stretch in [("序号", 1), ("步骤", 5), ("结果", 2)]:
            label = QLabel(text)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumHeight(34)
            label.setStyleSheet(
                f"background-color: #143562; border: 1px solid {STROKE_MAIN}; border-radius: 4px; "
                f"color: {TEXT_ACCENT}; font-size: 16px; font-weight: 800; padding: 5px 8px;"
            )
            process_header.addWidget(label, stretch)
        summary_layout.addLayout(process_header)

        process_rows_container = QWidget()
        process_rows_layout = QVBoxLayout(process_rows_container)
        process_rows_layout.setContentsMargins(0, 0, 0, 0)
        process_rows_layout.setSpacing(5)
        self._summary_step_name_labels = []
        self._summary_step_result_labels = []
        self.dashboard_process_rows = []
        for idx in range(self.VISIBLE_SLOT_COUNT):
            row_widget = ResultRowWidget(idx + 1, f"步骤 {idx + 1}", self)
            self._summary_step_name_labels.append(row_widget.step_label)
            self._summary_step_result_labels.append(row_widget.result_badge)
            self.dashboard_process_rows.append(row_widget)
            process_rows_layout.addWidget(row_widget)
        summary_layout.addWidget(process_rows_container)
        right_column_layout.addWidget(summary_card)

        thumbnails_card, thumbnails_layout = self._create_card(background="#101F43")
        thumbnails_card.setFixedWidth(s(650))
        thumbnails_card.setFixedHeight(240)
        thumbnails_layout.addWidget(self._create_section_title("步骤缩略图"))
        thumbnails_grid = QGridLayout()
        thumbnails_grid.setHorizontalSpacing(10)
        thumbnails_grid.setVerticalSpacing(0)
        self._thumbnail_name_labels = []
        self._thumbnail_status_labels = []
        for idx in range(self.VISIBLE_SLOT_COUNT):
            tile = ThumbnailTile(f"步骤 {idx + 1}", self)
            self._thumbnail_name_labels.append(tile.title_label)
            self._thumbnail_status_labels.append(tile.status_badge)
            thumbnails_grid.addWidget(tile, 0, idx)
        thumbnails_layout.addLayout(thumbnails_grid)
        content_grid.addWidget(thumbnails_card, 1, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignTop)

        stats_card, stats_layout = self._create_card(background=PANEL_BG_MID)
        stats_card.setFixedWidth(s(345))
        stats_card.setFixedHeight(360)
        stats_layout.addWidget(self._create_section_title("步骤统计"))
        summary_title_row = QHBoxLayout()
        summary_title = QLabel("步骤统计: CT(s)")
        summary_title.setStyleSheet(text_style(TEXT_ACCENT, size=15, weight=700))
        self.dashboard_cycle_time_label = QLabel("0.00")
        self.dashboard_cycle_time_label.setStyleSheet(text_style(TEXT_PRIMARY, size=15, weight=700))
        summary_title_row.addWidget(summary_title)
        summary_title_row.addStretch()
        summary_title_row.addWidget(self.dashboard_cycle_time_label)
        stats_layout.addLayout(summary_title_row)

        self.dashboard_stats_table = QTableWidget(self.VISIBLE_SLOT_COUNT, 6)
        self.dashboard_stats_table.setHorizontalHeaderLabels(["序号", "步骤", "OK数", "NG数", "不良率", "PT"])
        self.dashboard_stats_table.verticalHeader().setVisible(False)
        self.dashboard_stats_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.dashboard_stats_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.dashboard_stats_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.dashboard_stats_table.setShowGrid(False)
        self.dashboard_stats_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.dashboard_stats_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.dashboard_stats_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.dashboard_stats_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.dashboard_stats_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.dashboard_stats_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.dashboard_stats_table.setStyleSheet(table_style())

        self._stats_ok_labels = []
        self._stats_ng_labels = []
        self._stats_rate_labels = []
        self._stats_pt_labels = []
        for idx in range(self.VISIBLE_SLOT_COUNT):
            seq = QTableWidgetItem(str(idx + 1))
            seq.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            name = QTableWidgetItem(f"步骤 {idx + 1}")
            ok = QTableWidgetItem("0")
            ng = QTableWidgetItem("0")
            rate = QTableWidgetItem("0.00%")
            pt = QTableWidgetItem("0")
            for item in (ok, ng, rate, pt):
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.dashboard_stats_table.setItem(idx, 0, seq)
            self.dashboard_stats_table.setItem(idx, 1, name)
            self.dashboard_stats_table.setItem(idx, 2, ok)
            self.dashboard_stats_table.setItem(idx, 3, ng)
            self.dashboard_stats_table.setItem(idx, 4, rate)
            self.dashboard_stats_table.setItem(idx, 5, pt)
            self._stats_ok_labels.append(ok)
            self._stats_ng_labels.append(ng)
            self._stats_rate_labels.append(rate)
            self._stats_pt_labels.append(pt)
            self.dashboard_stats_table.setRowHeight(idx, TABLE_ROW_HEIGHT)
        stats_layout.addWidget(self.dashboard_stats_table)

        controls_block = QVBoxLayout()
        controls_block.setSpacing(8)

        controls_row = QHBoxLayout()
        controls_row.setContentsMargins(0, 0, 0, 0)
        controls_row.setSpacing(10)
        self.dashboard_start_button = self._create_outline_button("开始", background="#1C8E4F", border="#52E896", color="#E8FFF0", min_width=76)
        self.dashboard_stop_button = self._create_outline_button("停止", background="#7E1F31", border="#FF6A75", color="#FFF0F2", min_width=76)
        self.dashboard_open_camera_button = self._create_outline_button("打开相机", background="#214CCB", border="#55A6FF", color="#EAF4FF", min_width=104)
        self.dashboard_test_button = self._create_outline_button("测试", background="#136B7E", border="#54D5FF", color="#E8FBFF", min_width=76)
        self.dashboard_reset_button = self._create_outline_button("清零", background="#163A69", border="#35D7FF", color="#D7F8FF", min_width=76)
        self.dashboard_start_button.setEnabled(False)
        self.dashboard_stop_button.setEnabled(False)
        self.dashboard_test_button.setEnabled(False)
        self.dashboard_fastline_label = QLabel("快线运行: 00:00:00")
        self.dashboard_fastline_label.setStyleSheet(text_style(TEXT_SECONDARY, size=FONT_SMALL, weight=600))
        controls_row.addWidget(self.dashboard_start_button)
        controls_row.addWidget(self.dashboard_stop_button)
        controls_row.addWidget(self.dashboard_open_camera_button)
        controls_row.addWidget(self.dashboard_test_button)
        controls_row.addWidget(self.dashboard_reset_button)
        controls_row.addStretch()
        controls_block.addLayout(controls_row)

        fastline_row = QHBoxLayout()
        fastline_row.setContentsMargins(0, 0, 0, 0)
        fastline_row.addStretch()
        fastline_row.addWidget(self.dashboard_fastline_label)
        controls_block.addLayout(fastline_row)

        stats_layout.addLayout(controls_block)
        right_column_layout.addWidget(stats_card)

        content_grid.addWidget(right_column_container, 0, 2, 2, 1, alignment=Qt.AlignmentFlag.AlignTop)

        dashboard_layout.addWidget(content_grid_container)

    def _setup_worker(self) -> None:
        self.worker = CameraWorker(self.config, self.state)

    def _sync_runtime_model_from_processor(self, processor, *, save_config: bool = False) -> None:
        if processor is None:
            return

        runtime_task_getter = getattr(processor, "get_runtime_model_task", None)
        runtime_path_getter = getattr(processor, "get_runtime_model_path", None)
        runtime_task = (
            str(runtime_task_getter()).strip().lower()
            if callable(runtime_task_getter)
            else self.config.get_model_task()
        )
        runtime_model_path = (
            str(runtime_path_getter()).strip()
            if callable(runtime_path_getter)
            else self.config.get_model_path()
        )

        config_changed = False
        if runtime_task and runtime_task != self.config.get_model_task():
            self.config.model_task = runtime_task
            config_changed = True

        if save_config and config_changed:
            self.config.save()

        if self.config_page is not None:
            self.config_page.sync_model_config_from_runtime(
                self.config.get_model_path(),
                self.config.get_model_task(),
            )

        self._refresh_model_runtime_labels(
            {
                "task_type": runtime_task,
                "model_path": runtime_model_path,
                "config_path": str(self.config.get_config_path() or ""),
            }
        )

    def _setup_audio(self) -> None:
        self._sound_effect = QSoundEffect(self)
        audio_path = self._get_audio_path()
        if audio_path and audio_path.exists():
            self._sound_effect.setSource(QUrl.fromLocalFile(str(audio_path)))
            self._sound_effect.setVolume(1.0)
        else:
            print(f"[WARNING] 音频文件未找到: {audio_path}")

    def _get_audio_path(self) -> Path:
        if hasattr(sys, "_MEIPASS"):
            return Path(sys._MEIPASS) / "Pass.wav"
        return Path(__file__).parent.parent.parent / "Pass.wav"

    def _connect_signals(self) -> None:
        assert self.worker is not None
        self.dashboard_open_camera_button.clicked.connect(self._toggle_camera)
        self.dashboard_start_button.clicked.connect(self._start_detection_clicked)
        self.dashboard_stop_button.clicked.connect(self._stop_detection_clicked)
        self.dashboard_test_button.clicked.connect(self._run_single_test_clicked)
        self.dashboard_reset_button.clicked.connect(self._reset_dashboard_clicked)
        self.dashboard_config_button.clicked.connect(self._show_config_page)
        self.dashboard_choose_button.clicked.connect(self._refresh_cameras)
        self.dashboard_close_button.clicked.connect(self.close)

        self.worker.error_occurred.connect(self._show_error)
        self.worker.state_changed.connect(self._on_state_changed)
        self.worker.fps_updated.connect(self._update_fps)
        self.worker.gesture_state_changed.connect(self._on_gesture_state_changed)
        self.worker.round_completed.connect(self._on_round_completed)
        self.worker.round_progress_changed.connect(self._on_round_progress_changed)
        self.worker.pipeline_stats_updated.connect(self._on_pipeline_stats_updated)

    def _setup_shortcuts(self) -> None:
        action_start_camera = QAction(self)
        action_start_camera.setShortcut(QKeySequence("Ctrl+O"))
        action_start_camera.triggered.connect(self._toggle_camera)
        self.addAction(action_start_camera)

        action_stop_camera = QAction(self)
        action_stop_camera.setShortcut(QKeySequence("Ctrl+Shift+S"))
        action_stop_camera.triggered.connect(self._stop_all_runtime)
        self.addAction(action_stop_camera)

        action_toggle_inference = QAction(self)
        action_toggle_inference.setShortcut(QKeySequence("Ctrl+D"))
        action_toggle_inference.triggered.connect(self._toggle_inference)
        self.addAction(action_toggle_inference)

        action_quit = QAction(self)
        action_quit.setShortcut(QKeySequence("Ctrl+Q"))
        action_quit.triggered.connect(self.close)
        self.addAction(action_quit)

    def _update_clock_text(self) -> None:
        if self.dashboard_time_label is not None:
            self.dashboard_time_label.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def _apply_runtime_status(self, result_text: str, *, state_text: str) -> None:
        result_text = str(result_text).upper().strip()
        state_text = str(state_text).strip()
        if getattr(self, "dashboard_state_value_label", None) is not None:
            self.dashboard_state_value_label.setText(state_text)

        if getattr(self, "dashboard_result_tag_label", None) is None:
            return

        styles = {
            "READY": ("#155084", "#35D7FF", "#D7F8FF"),
            "TESTING": ("#9B6A00", "#FFD34D", "#FFF5C7"),
            "PASS": ("#1C8E4F", "#52E896", "#E8FFF0"),
            "STOPPED": ("#7E1F31", "#FF6A75", "#FFF0F2"),
        }
        background, border, color = styles.get(result_text, styles["READY"])
        self.dashboard_result_tag_label.setStyleSheet(
            chip_button_style(
                background=background,
                border=border,
                color=color,
                font_size=FONT_BODY,
                min_height=34,
                padding="5px 14px",
                radius=6,
            )
        )
        self.dashboard_result_tag_label.setText(result_text)

    def _refresh_dashboard_metrics(self) -> None:
        total_target = int(getattr(getattr(self, "config", None), "today_target_capacity", 300))
        total_rounds = int(getattr(self, "_total_rounds", 0))
        good_rounds = int(getattr(self, "_good_rounds", 0))
        action_ng_count = int(getattr(self, "_action_ng_count", 0))
        product_ng_count = int(getattr(self, "_product_ng_count", 0))
        last_cycle_seconds = float(getattr(self, "_last_cycle_seconds", 0.0))
        yield_rate = (good_rounds / total_rounds * 100.0) if total_rounds > 0 else 0.0

        if getattr(self, "dashboard_total_rounds_label", None) is not None:
            self.dashboard_total_rounds_label.setText(str(total_rounds))
        if getattr(self, "dashboard_good_rounds_label", None) is not None:
            self.dashboard_good_rounds_label.setText(str(good_rounds))
        if getattr(self, "dashboard_action_ng_label", None) is not None:
            self.dashboard_action_ng_label.setText(str(action_ng_count))
        if getattr(self, "dashboard_product_ng_label", None) is not None:
            self.dashboard_product_ng_label.setText(str(product_ng_count))
        if getattr(self, "dashboard_capacity_value_label", None) is not None:
            self.dashboard_capacity_value_label.setText(f"{good_rounds}/{total_target}")
        if getattr(self, "dashboard_capacity_target_label", None) is not None:
            self.dashboard_capacity_target_label.setText(f"PCS / 今日 {total_target}")
        if getattr(self, "dashboard_yield_value_label", None) is not None:
            self.dashboard_yield_value_label.setText(f"{yield_rate:.2f}%")
        if getattr(self, "dashboard_cycle_time_label", None) is not None:
            self.dashboard_cycle_time_label.setText(f"{last_cycle_seconds:.2f}")

    def _refresh_model_runtime_labels(self, payload: Optional[dict] = None) -> None:
        task_display = self.config.get_model_task_display_name()
        model_path = self.config.get_model_path()
        if isinstance(payload, dict):
            task_display = str(payload.get("task_type", "") or task_display)
            model_path = str(payload.get("model_path", "") or model_path)

        model_name = Path(model_path).name if str(model_path).strip() else "未配置"
        if getattr(self, "dashboard_mode_value_label", None) is not None:
            self.dashboard_mode_value_label.setText(f"模式: 自动模式 / 任务: {task_display}")
        if getattr(self, "dashboard_project_label", None) is not None:
            self.dashboard_project_label.setText(f"模型: {model_name}")

    def _build_candidate_config(self, model_path: str, model_task: str) -> ConfigManager:
        candidate = ConfigManager()
        candidate.update(**self.config.to_dict())
        candidate._config_path = getattr(self.config, "_config_path", None)
        requested_path = str(model_path).strip() or self.config.get_model_path()
        candidate.mediapipe_task_path = requested_path
        candidate.model_path = requested_path
        candidate.model_task = "mediapipe_gesture"
        candidate.validate()
        return candidate

    def _try_switch_model(self, model_path: str, model_task: str) -> bool:
        requested_path = str(model_path).strip()
        requested_task = "mediapipe_gesture"
        if not requested_path:
            QMessageBox.warning(self, "模型切换失败", "模型路径不能为空。")
            return False
        if not requested_path.lower().endswith(".task"):
            QMessageBox.warning(self, "模型切换失败", "当前仅支持选择 MediaPipe .task 模型文件。")
            return False

        candidate_config = self._build_candidate_config(requested_path, requested_task)
        if not candidate_config.is_supported_model_task():
            QMessageBox.warning(self, "模型切换失败", f"不支持的任务类型: {requested_task}")
            return False

        new_processor = MediaPipeGestureProcessor(candidate_config)
        if new_processor.model is None:
            error_text = str(getattr(new_processor, "_last_init_error", "") or "模型加载失败")
            QMessageBox.warning(self, "模型切换失败", error_text)
            return False

        old_processor = self.worker.frame_processor if self.worker is not None else None
        was_inference_on = bool(self.state.inference_on)
        if was_inference_on and self.worker is not None:
            self.worker.stop_inference()

        try:
            self.config.mediapipe_task_path = candidate_config.mediapipe_task_path
            self.config.model_path = candidate_config.model_path
            self.config.model_task = "mediapipe_gesture"
            self.config.save()
            if self.worker is not None:
                self.worker.set_frame_processor(new_processor)
            if old_processor is not None and old_processor is not new_processor:
                old_processor.release()
            if self.config_page is not None:
                self.config_page.sync_model_config_from_runtime(
                    self.config.get_model_path(),
                    self.config.get_model_task(),
                )
            self._refresh_model_runtime_labels()
            self.status_bar.showMessage(
                f"模型切换成功: {Path(self.config.get_model_path()).name} ({self.config.get_model_task_display_name()})",
                3000,
            )
            if was_inference_on and self.worker is not None:
                self.worker.start_inference()
            return True
        except Exception as exc:
            if self.worker is not None and old_processor is not None:
                self.worker.set_frame_processor(old_processor)
            if was_inference_on and self.worker is not None:
                self.worker.start_inference()
            QMessageBox.warning(self, "模型切换失败", str(exc))
            return False

    def _update_dashboard_category_texts(self) -> None:
        names = list(getattr(self.config, "category_names", []))
        padded_names = names + [""] * max(0, self.VISIBLE_SLOT_COUNT - len(names))

        self._gesture_name_to_index.clear()
        self._gesture_labels = list(self._summary_step_result_labels)

        for idx in range(self.VISIBLE_SLOT_COUNT):
            name = padded_names[idx].strip() if idx < len(padded_names) else ""
            display_name = name or f"应用类别 {idx + 1}"
            if idx < len(self.dashboard_defect_name_labels):
                self.dashboard_defect_name_labels[idx].setText(display_name)
            if idx < len(self.dashboard_defect_value_labels):
                self.dashboard_defect_value_labels[idx].setText("0.00%")
            if idx < len(self._thumbnail_name_labels):
                self._thumbnail_name_labels[idx].setText(display_name)
            if idx < len(self._summary_step_name_labels):
                if hasattr(self._summary_step_name_labels[idx], "setText"):
                    self._summary_step_name_labels[idx].setText(display_name)
            if getattr(self, "dashboard_stats_table", None) is not None:
                item = self.dashboard_stats_table.item(idx, 1)
                if item is not None:
                    item.setText(display_name)
            if name:
                self._gesture_name_to_index[name] = idx

        self._reset_all_gesture_labels()

    def _toggle_camera(self) -> None:
        if self.state.camera_on:
            self._stop_all_runtime()
        else:
            assert self.worker is not None
            self.worker.start_camera()

    def _start_detection_clicked(self) -> None:
        if self.worker is None:
            return
        self.worker.start_inference()

    def _stop_detection_clicked(self) -> None:
        if self.worker is None:
            return
        self.worker.stop_inference()

    def _run_single_test_clicked(self) -> None:
        if self.worker is None:
            return
        if not self.state.camera_on:
            self._show_error("请先开启相机。")
            return
        if self.state.inference_on:
            self._show_error("当前正在连续检测，请先停止后再执行测试。")
            return

        run_single_test = getattr(self.worker, "run_single_test", None)
        if callable(run_single_test):
            run_single_test()
        if getattr(self, "status_bar", None) is not None:
            self.status_bar.showMessage("已触发单次测试。", 3000)
        self._apply_runtime_status("TESTING", state_text="单次测试")

    def _refresh_dashboard_action_states(self) -> None:
        camera_open = bool(getattr(self.state, "camera_on", False))
        detecting = bool(getattr(self.state, "inference_on", False))
        if self.dashboard_open_camera_button is not None:
            self.dashboard_open_camera_button.setEnabled(True)
            self.dashboard_open_camera_button.setText("关闭相机" if camera_open else "打开相机")
        if self.dashboard_start_button is not None:
            self.dashboard_start_button.setEnabled(camera_open and not detecting)
        if self.dashboard_stop_button is not None:
            self.dashboard_stop_button.setEnabled(detecting)
        if self.dashboard_test_button is not None:
            self.dashboard_test_button.setEnabled(camera_open and not detecting)
        if self.dashboard_reset_button is not None:
            self.dashboard_reset_button.setEnabled(True)

    def _stop_all_runtime(self) -> None:
        if self.worker is None:
            return
        self.worker.stop_inference()
        self.worker.stop_camera()

    def _toggle_inference(self) -> None:
        if self.worker is None:
            return
        if self.state.inference_on:
            self.worker.stop_inference()
        else:
            self.worker.start_inference()

    def _reset_dashboard_clicked(self) -> None:
        self._total_rounds = 0
        self._good_rounds = 0
        self._action_ng_count = 0
        self._product_ng_count = 0
        self._last_cycle_seconds = 0.0
        self._active_gesture_names.clear()
        self._round_completed = False
        self._refresh_dashboard_metrics()
        self._reset_all_gesture_labels()
        self._apply_runtime_status("READY", state_text="待机")
        self.status_bar.showMessage("统计已清零", 2000)

    def _show_main_page(self) -> None:
        self.stacked_widget.setCurrentIndex(self.PAGE_MAIN)

    def _show_config_page(self) -> None:
        self.stacked_widget.setCurrentIndex(self.PAGE_CONFIG)

    def _on_config_changed(self, key: str, value) -> None:
        if key == "category_names":
            self._update_gesture_labels(list(value))
        elif key in {"today_target_capacity"}:
            self._refresh_dashboard_metrics()
        elif key in {
            "model_path",
            "model_task",
            "enable_gesture_detection",
            "enable_object_detection",
            "object_model_path",
            "object_score_threshold",
            "object_max_results",
            "object_result_hold_ms",
        }:
            self._refresh_model_runtime_labels()

    def _on_config_saved(self) -> None:
        self._update_gesture_labels(list(self.config.category_names))
        self._refresh_dashboard_metrics()
        self._refresh_model_runtime_labels()

    def _refresh_cameras(self) -> None:
        devices = MvSdkCamera.enumerate_devices()
        if devices:
            selected = str(getattr(self.config, "mvsdk_friendly_name", "") or devices[0].friendly_name)
            self.config.mvsdk_friendly_name = selected
            self.status_bar.showMessage(f"已检测到 {len(devices)} 台相机，当前: {selected}", 3000)
        else:
            self.status_bar.showMessage("未检测到相机", 3000)

    def _draw_detection_overlay(self, qimage: QImage, overlay_state: DetectionOverlayState) -> QImage:
        result = qimage.copy()
        painter = QPainter(result)

        status = str(getattr(overlay_state, "status", "idle") or "idle")
        if status in {"session_unavailable", "inference_error"}:
            painter.setPen(QColor(255, 80, 80))
            painter.drawText(20, 40, str(getattr(overlay_state, "error", status)))
            painter.end()
            return result

        painter.setPen(QPen(QColor(57, 255, 106), 2))
        for det in list(getattr(overlay_state, "detections", []) or []):
            polygon = det.get("polygon")
            if polygon:
                points = [QPoint(int(pt[0]), int(pt[1])) for pt in polygon]
                painter.drawPolygon(QPolygon(points))
                continue

            box = det.get("box", [])
            if len(box) >= 4:
                x1, y1, x2, y2 = (int(v) for v in box[:4])
                painter.drawRect(x1, y1, max(0, x2 - x1), max(0, y2 - y1))

        painter.end()
        return result

    @Slot(QImage)
    def _update_frame(self, qimage: QImage, overlay_state: DetectionOverlayState | None = None) -> None:
        if overlay_state is not None:
            qimage = self._draw_detection_overlay(qimage, overlay_state)
        pixmap = QPixmap.fromImage(qimage)
        target_size = self.video_label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        scaled_pixmap = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled_pixmap)

    @Slot(str)
    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self, "错误", message)

    @Slot(str, bool)
    def _on_state_changed(self, state_name: str, value: bool) -> None:
        if state_name == "camera_on":
            self.camera_status_label.setText(f"相机: {'开启' if value else '关闭'}")
            self._refresh_dashboard_action_states()
            if value:
                if not self._preview_timer.isActive():
                    self._preview_timer.start()
                self._apply_runtime_status(
                    "READY" if not self.state.inference_on else "TESTING",
                    state_text="运行" if self.state.inference_on else "待机",
                )
            else:
                if self._preview_timer.isActive():
                    self._preview_timer.stop()
                empty_pixmap = QPixmap(self.video_label.size())
                empty_pixmap.fill(QColor("#0A1228"))
                self.video_label.setPixmap(empty_pixmap)
                self.video_label.setText("点击“打开相机”开始")
                self._apply_runtime_status("READY", state_text="待机")
                self.fps_label.setText("FPS: --")
        elif state_name == "inference_on":
            self.inference_status_label.setText(f"检测: {'开启' if value else '关闭'}")
            self._refresh_dashboard_action_states()
            if value:
                self._apply_runtime_status("TESTING", state_text="运行")
            else:
                self._active_gesture_names.clear()
                self._round_completed = False
                self._reset_all_gesture_labels()
                self._apply_runtime_status("READY", state_text="待机" if self.state.camera_on else "停止")

    def _poll_latest_frame(self) -> None:
        if self.worker is None:
            return
        qimage, overlay_state = self.worker.get_latest_display_payload()
        if qimage is None:
            return
        self._update_frame(qimage, overlay_state)

    @Slot(float)
    def _update_fps(self, fps: float) -> None:
        self.fps_label.setText(f"FPS: {fps:.1f}")

    @Slot(dict)
    def _on_pipeline_stats_updated(self, payload: dict) -> None:
        preview_fps = float(payload.get("preview_fps", 0.0))
        infer_fps = float(payload.get("infer_fps", 0.0))
        dropped = int(payload.get("dropped_for_infer", 0))
        result_age_ms = float(payload.get("display_result_age_ms", 0.0))
        display_strategy = str(payload.get("display_strategy", "preview_overlay"))
        text = (
            f"预览 {preview_fps:.1f} / 推理 {infer_fps:.1f} / 丢帧 {dropped} "
            f"/ 结果年龄 {result_age_ms:.1f}ms / {display_strategy}"
        )
        self.fps_label.setText(f"FPS: {text}")
        if self.dashboard_status_fps_label is not None:
            self.dashboard_status_fps_label.setText(text)
        if self.dashboard_fastline_label is not None:
            self.dashboard_fastline_label.setText(f"快线运行: {result_age_ms:.1f}ms")
        self._refresh_model_runtime_labels(payload)

    def _target_gesture_names(self) -> list[str]:
        return [name for name in self._gesture_name_to_index.keys() if str(name).strip()]

    def _set_gesture_visual_state(self, index: int, *, active: bool) -> None:
        gesture_labels = getattr(self, "_gesture_labels", [])
        thumbnail_labels = getattr(self, "_thumbnail_status_labels", [])
        if index < len(gesture_labels):
            label = gesture_labels[index]
            if hasattr(label, "set_status"):
                label.set_status("PASS" if active else "READY", active=active)
            else:
                if active:
                    label.setStyleSheet(
                        chip_button_style(
                            background="#2EA846",
                            border="#52E896",
                            color=TEXT_SUCCESS,
                            font_size=FONT_BODY,
                            min_height=28,
                            padding="4px 8px",
                            radius=6,
                        )
                    )
                    if hasattr(label, "setText"):
                        label.setText("PASS")
                else:
                    label.setStyleSheet(
                        chip_button_style(
                            background="#155084",
                            border="#35D7FF",
                            color=TEXT_PRIMARY,
                            font_size=FONT_BODY,
                            min_height=28,
                            padding="4px 8px",
                            radius=6,
                        )
                    )
                    if hasattr(label, "setText"):
                        label.setText("READY")

        if index < len(thumbnail_labels):
            label = thumbnail_labels[index]
            if hasattr(label, "set_status"):
                label.set_status("PASS" if active else "READY", active=active)
            else:
                if active:
                    label.setStyleSheet(
                        f"color: #E8FFF0; font-size: {FONT_SMALL}px; font-weight: 700; background-color: #1C8E4F; border-radius: 6px; padding: 5px 10px;"
                    )
                    if hasattr(label, "setText"):
                        label.setText("PASS")
                else:
                    label.setStyleSheet(
                        f"color: {TEXT_PRIMARY}; font-size: {FONT_SMALL}px; font-weight: 700; background-color: #155084; border-radius: 6px; padding: 5px 10px;"
                    )
                    if hasattr(label, "setText"):
                        label.setText("READY")

    @Slot(str, bool)
    def _on_gesture_state_changed(self, name: str, active: bool) -> None:
        idx = self._gesture_name_to_index.get(name)
        if idx is None:
            return
        if active:
            self._active_gesture_names.add(name)
        else:
            self._active_gesture_names.discard(name)
            if not self._active_gesture_names:
                self._round_completed = False
        self._set_gesture_visual_state(idx, active=active)

    @Slot(dict)
    def _on_round_progress_changed(self, payload: dict) -> None:
        completed_count = int(payload.get("completed_count", 0))
        target_count = max(1, int(payload.get("target_count", 1)))
        if self.dashboard_state_value_label is not None and not payload.get("holding", False):
            self.dashboard_state_value_label.setText("运行" if self.state.inference_on else "待机")
        if self.dashboard_cycle_time_label is not None and not self._round_completed:
            self.dashboard_cycle_time_label.setText(f"{completed_count}/{target_count}")

    @Slot(dict)
    def _on_round_completed(self, payload: dict) -> None:
        self._round_completed = True
        self._total_rounds = int(getattr(self, "_total_rounds", 0)) + 1
        self._good_rounds = int(getattr(self, "_good_rounds", 0)) + 1
        self._last_cycle_seconds = float(payload.get("duration_ms", 0)) / 1000.0
        for name in self._target_gesture_names():
            self._active_gesture_names.add(name)
            idx = self._gesture_name_to_index.get(name)
            if idx is not None:
                self._set_gesture_visual_state(idx, active=True)
                stats_ok_labels = getattr(self, "_stats_ok_labels", [])
                stats_rate_labels = getattr(self, "_stats_rate_labels", [])
                stats_pt_labels = getattr(self, "_stats_pt_labels", [])
                if idx < len(stats_ok_labels):
                    current_ok = int(stats_ok_labels[idx].text())
                    stats_ok_labels[idx].setText(str(current_ok + 1))
                    if idx < len(stats_rate_labels):
                        stats_rate_labels[idx].setText("0.00%")
                    if idx < len(stats_pt_labels):
                        stats_pt_labels[idx].setText(str(current_ok + 1))
        self._refresh_dashboard_metrics()
        self._apply_runtime_status("PASS", state_text="完成")
        self._on_sequence_complete()

    def _on_sequence_complete(self) -> None:
        if getattr(self, "_sound_effect", None) is not None and self._sound_effect.isLoaded():
            self._sound_effect.play()
        if getattr(self, "status_bar", None) is not None:
            config = getattr(self, "config", None)
            timeout = int(float(getattr(config, "round_cooldown_seconds", 2.0)) * 1000)
            self.status_bar.showMessage("✓ 类别匹配完成一轮！", timeout)

    def _reset_all_gesture_labels(self) -> None:
        for idx in range(len(self._gesture_labels)):
            self._set_gesture_visual_state(idx, active=False)

    def _update_gesture_labels(self, names: list) -> None:
        _ = names
        self._active_gesture_names.clear()
        self._round_completed = False
        self._update_dashboard_category_texts()

    def _refresh_model_runtime_labels(self, payload: Optional[dict] = None) -> None:
        task_value = self.config.get_detection_mode_task_type()
        model_path = self.config.get_model_path()
        config_path = str(self.config.get_config_path() or "")
        if isinstance(payload, dict):
            task_value = str(payload.get("task_type", "") or task_value).strip().lower()
            model_path = str(payload.get("model_path", "") or model_path)
            config_path = str(payload.get("config_path", "") or config_path)

        task_display = self.config.get_model_task_display_name(task_value)
        model_name = Path(model_path).name if str(model_path).strip() else "未配置"
        config_name = Path(config_path).name if str(config_path).strip() else "未配置"
        if getattr(self, "dashboard_mode_value_label", None) is not None:
            self.dashboard_mode_value_label.setText(f"模式: 自动模式 / 任务: {task_display}")
        if getattr(self, "dashboard_project_label", None) is not None:
            self.dashboard_project_label.setText(f"模型: {model_name} / 配置: {config_name}")

    def _try_switch_model(self, model_path: str, model_task: str) -> bool:
        requested_path = str(model_path).strip()
        requested_task = "mediapipe_gesture"
        if not bool(self.config.has_enabled_detection_mode()):
            QMessageBox.warning(self, "检测配置失败", "至少启用一种检测模式。")
            return False
        if bool(getattr(self.config, "enable_gesture_detection", True)) and not requested_path:
            QMessageBox.warning(self, "模型切换失败", "模型路径不能为空。")
            return False
        if bool(getattr(self.config, "enable_gesture_detection", True)) and not requested_path.lower().endswith(".task"):
            QMessageBox.warning(self, "模型切换失败", "当前仅支持选择 MediaPipe .task 模型文件。")
            return False

        candidate_config = self._build_candidate_config(requested_path, requested_task)
        if not candidate_config.is_supported_model_task(requested_task):
            QMessageBox.warning(self, "模型切换失败", f"不支持的任务类型: {requested_task}")
            return False

        try:
            new_processor = create_frame_processor_from_config(
                candidate_config,
                gesture_processor_cls=MediaPipeGestureProcessor,
            )
        except Exception as exc:
            QMessageBox.warning(self, "检测配置失败", str(exc))
            return False

        if not self._processor_runtime_ready(new_processor):
            error_text = str(getattr(new_processor, "_last_init_error", "") or "模型加载失败")
            QMessageBox.warning(self, "模型切换失败", error_text)
            return False

        actual_task = str(getattr(new_processor, "get_runtime_model_task", lambda: requested_task)()).strip().lower()
        actual_model_path = str(getattr(new_processor, "get_runtime_model_path", lambda: requested_path)()).strip()
        old_processor = self.worker.frame_processor if self.worker is not None else None
        was_inference_on = bool(self.state.inference_on)
        if was_inference_on and self.worker is not None:
            self.worker.stop_inference()

        try:
            self.config.mediapipe_task_path = candidate_config.mediapipe_task_path
            self.config.model_path = candidate_config.model_path
            self.config.model_task = "mediapipe_gesture"
            self.config.save()
            if self.worker is not None:
                self.worker.set_frame_processor(new_processor)
            if old_processor is not None and old_processor is not new_processor:
                old_processor.release()
            self._sync_runtime_model_from_processor(new_processor, save_config=False)
            print(
                f"[runtime] model_switch config={self.config.get_config_path()} requested_task={requested_task} "
                f"actual_task={self.config.get_model_task()} model_path={self.config.get_model_path()} "
                f"resolved_model_path={actual_model_path}"
            )
            self.status_bar.showMessage(
                f"模型切换成功: {Path(actual_model_path or self.config.get_model_path()).name} "
                f"({self.config.get_model_task_display_name(self.config.get_model_task())})",
                3000,
            )
            if was_inference_on and self.worker is not None:
                self.worker.start_inference()
            return True
        except Exception as exc:
            if self.worker is not None and old_processor is not None:
                self.worker.set_frame_processor(old_processor)
            if was_inference_on and self.worker is not None:
                self.worker.start_inference()
            QMessageBox.warning(self, "模型切换失败", str(exc))
            return False

    def _processor_runtime_ready(self, processor) -> bool:
        children = getattr(processor, "processors", None)
        if children is not None:
            return all(self._processor_runtime_ready(child) for child in children)
        return getattr(processor, "model", object()) is not None

    def set_config_panel(self, panel) -> None:
        _ = panel

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            self.worker.stop_camera()
        self.config.save()
        event.accept()
