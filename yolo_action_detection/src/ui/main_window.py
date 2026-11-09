"""YOLO OBB 动作检测主界面。"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import logging
import threading
import time
import hashlib
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QEvent, QEventLoop, QTimer, Slot
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QMessageBox, QPushButton, QStackedWidget, QFrame, QScrollArea, QSizePolicy,
)

from calibration.calibrator import extract_tip_point, distance_mm
from calibration.models import HoleZone
from camera.camera_worker import CameraWorker
from camera.mvsdk_camera import MvSdkCamera
from detection_logging.coordinate_logger import (
    CoordinateSessionLogger,
    build_coordinate_frame_record,
)
from detection_logging.audio_feedback import SoundFeedback
from detection_logging.action_trace import trace
from detection_logging.fail_evidence import FailEvidenceContext, FailEvidenceSaver
from detection_logging.production_stats import ProductionStatsManager
from runtime_diagnostics import get_runtime_logger
from roi_inspection.engine import RoiInspectionEngine
from roi_inspection.geometry import (
    NormalizedRect,
    detection_center_in_rect,
    usable_normalized_rect,
)
from roi_inspection.models import InspectionOutcome, RoiInspectionConfig, RoundPhase
from template_matching.match_diagnostics import get_recorder as get_template_diagnostics
from ui.production_stats_store import ProductionStatsStore, batch_from_config
from ui.roi_fail_alert_dialog import RoiFailAlertDialog
from lstm_action.engine import LstmActionEngine
from step_sequence.step_sequence_engine import (
    RoundResult,
    StepStatus,
    build_step_engine,
)
from ui.runtime_ui_tokens import (
    FONT_MICRO,
    FONT_TITLE,
    FONT_SMALL,
    MONO_FAMILY,
    PAGE_BG,
    PAGE_MARGIN,
    PANEL_BG,
    PANEL_BG_DARK,
    PRIMARY_CONTROL_HEIGHT,
    STEP_NG_BORDER,
    STEP_PASS_BORDER,
    STROKE_MAIN,
    TEXT_ACCENT,
    TEXT_DANGER,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TOP_BAR_HEIGHT,
    VIEWPORT_BG,
    frame_style,
    scroll_area_style,
    text_style,
)
from hand_tracking.mediapipe_hands import draw_hand_landmarks
from ui.frame_text import draw_bgr_texts
from ui.widgets.native_panels import (
    KpiRow,
    RecognitionListItem,
)
from yolo_runtime.yolo_result_models import DetectionOverlayState, ObbDetection
from template_matching.template_store import resolve_template_path, template_file_signature, restore_file


def _status_badge_style(color: str) -> str:
    """顶栏主态徽章：描边 + 同色文字，取代裸文字状态。"""
    return (
        f"color: {color}; border: 1px solid {color}; border-radius: 6px; "
        "padding: 3px 12px; font-size: 13px; font-weight: 700; background: transparent;"
    )


class MainWindow(QMainWindow):
    """YOLO OBB 动作检测主窗口（科技风双栏）。"""

    def __init__(
        self,
        config,
        state,
        processor=None,
        step_engine=None,
        *,
        sound_feedback=None,
        fail_evidence_saver=None,
        resource_dir=None,
        evidence_base_dir=None,
        processor_factory: Optional[Callable] = None,
        app_base_dir=None,
        runtime_log_path=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.state = state
        self.processor = processor
        self.step_engine = step_engine
        self.lstm_action = LstmActionEngine()
        self._processor_factory = processor_factory
        self._model_loading = False
        self._close_after_model_load = False
        self._app_base_dir = (
            Path(app_base_dir)
            if app_base_dir is not None
            else Path(__file__).resolve().parents[2]
        )
        self._runtime_model_signature = self._model_runtime_signature(config)
        self._step_engine_signature = self._step_engine_config_signature(config)
        self._roi_engine_signature: Optional[tuple] = None
        self._last_model_reload_error = ""
        self._runtime_logger = get_runtime_logger()
        self._runtime_log_path = Path(runtime_log_path) if runtime_log_path else None

        self.setWindowTitle("双翼科技视觉行为引导系统")
        self.setMinimumSize(1200, 760)
        self.setStyleSheet(f"QMainWindow {{ background-color: {PAGE_BG}; }}")

        self.worker = CameraWorker(config, state)
        self._configure_lstm_action()
        self._camera_devices = []
        self._camera_enumeration_error = ""
        self._refresh_camera_devices()
        self._display_timer = QTimer(self)
        self._display_timer.timeout.connect(self._update_display)

        # 生产统计（延后到 evidence_dir 之后初始化）
        self._stats_manager: Optional[ProductionStatsManager] = None
        # 计数去重键：同一轮同一结果只计一次
        self._last_counted_pass_round = -1
        self._last_counted_ng_key: Optional[tuple] = None
        self._played_result_sound_keys: dict[tuple, None] = {}
        self._handled_roi_result_keys: dict[tuple, None] = {}
        self._round_start_time = 0.0
        # 固定工位 ROI 定位检测引擎（按需创建）。
        self._roi_engine: Optional[RoiInspectionEngine] = None
        self._roi_fail_alert: Optional[RoiFailAlertDialog] = None
        self._latest_runtime_overlay = DetectionOverlayState()
        self._last_render_key: Optional[tuple] = None
        self._init_roi_engine()
        # 下一轮启动定时器：完成一轮后保持当前步骤框颜色，到间隔结束再开始新一轮。
        self._round_pass_timer = QTimer(self)
        self._round_pass_timer.setSingleShot(True)
        self._round_pass_timer.timeout.connect(self._on_round_pass_settled)
        self._lstm_progress_step = -1
        self._lstm_progress_round = None
        self._pending_step_engine_rebuild = False

        # 保留 _step_cards 名称以兼容现有状态映射和测试，实际控件为紧凑识别项。
        self._step_cards: List[RecognitionListItem] = []
        self._steps_hint: Optional[QLabel] = None
        self._sticky_ng_step_idx = -1
        self._last_step_focus_key: Optional[tuple] = None

        # 配置页
        self._config_page = None
        self._last_camera_params = {}
        self._last_pipeline_stats = {}
        self._last_overlay_data = {}
        evidence_dir = (
            Path(evidence_base_dir)
            if evidence_base_dir is not None
            else Path("outputs") / "evidence"
        )
        self._sound_feedback = sound_feedback or SoundFeedback(
            enabled=(
                bool(getattr(config, "pass_sound_enabled", False))
                or bool(getattr(config, "fail_sound_enabled", True))
            ),
            resource_dir=resource_dir,
        )
        self._fail_evidence_saver = fail_evidence_saver or FailEvidenceSaver(
            enabled=getattr(config, "fail_evidence_enabled", True),
            base_dir=evidence_dir,
        )
        self._stats_manager = ProductionStatsManager(
            records_dir=evidence_dir.parent / "production_records",
        )
        self._stats_manager.restore(batch_from_config(self.config))
        self._stats_store = ProductionStatsStore(
            self._stats_manager,
            config_provider=lambda: self.config,
            on_error=self._set_notice,
            parent=self,
        )
        self._coordinate_logger = CoordinateSessionLogger(
            log_dir=getattr(config, "coordinate_logging_dir", "logs/coordinate_sessions"),
            max_queue=getattr(config, "coordinate_logging_max_queue", 1000),
            summary_enabled=getattr(config, "coordinate_logging_summary_enabled", True),
            max_file_mb=getattr(config, "coordinate_logging_max_file_mb", 0),
        )

        self._build_ui()
        self._connect_signals()
        self._refresh_steps()
        self._update_kpi()
        self._sync_button_states()
        self._runtime_logger.info(
            "main_window=ready processor_available=%s log_path=%s",
            self.processor is not None,
            self._runtime_log_path or "unavailable",
        )
        if self.processor is None:
            self._show_processor_log_notice()

    def _show_processor_log_notice(self) -> None:
        if self._runtime_log_path is not None:
            self._set_notice(f"处理器未加载，请复制日志文件：{self._runtime_log_path}")
        else:
            self._set_notice("处理器未加载，且诊断日志目录不可写。")

    def _init_roi_engine(self) -> None:
        """根据当前有效配置创建或清空 ROI 轮次引擎。"""
        self._played_result_sound_keys.clear()
        self._handled_roi_result_keys.clear()
        self._roi_engine_signature = self._roi_engine_config_signature(self.config)
        self._roi_engine = None
        if not getattr(self.config, "roi_inspection_enabled", False):
            return
        if self.config.inspection_validation_errors(self._current_model_class_names()):
            return
        from yolo_runtime.template_gated_processor import TemplateGatedProcessor

        self._roi_engine = RoiInspectionEngine(
            RoiInspectionConfig.from_config(
                self.config,
                template_gate_active=isinstance(self.processor, TemplateGatedProcessor),
            )
        )

    def _current_model_class_names(self) -> List[str]:
        if self.processor is None or not hasattr(self.processor, "get_class_names"):
            return []
        try:
            names = self.processor.get_class_names()
        except Exception:
            return []
        values = names.values() if isinstance(names, dict) else names
        return [str(name).strip() for name in values if str(name).strip()]

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("mainCentral")
        central.setStyleSheet(f"QWidget#mainCentral {{ background-color: {PAGE_BG}; }}")
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN)
        outer.setSpacing(12)

        self._stacked = QStackedWidget()
        outer.addWidget(self._stacked)

        # 主页面
        self._main_page = QWidget()
        self._main_page.setObjectName("mainPage")
        self._main_page.setStyleSheet("QWidget#mainPage { background: transparent; }")
        main_layout = QVBoxLayout(self._main_page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(12)

        # 顶部标题栏
        self._header = QFrame()
        self._header.setStyleSheet(frame_style(PANEL_BG_DARK, border=STROKE_MAIN, radius=10))
        self._header.setFixedHeight(TOP_BAR_HEIGHT)
        self._header_layout = QGridLayout(self._header)
        self._header_layout.setContentsMargins(16, 6, 16, 6)
        self._header_layout.setHorizontalSpacing(0)
        self._header_layout.setColumnStretch(0, 1)
        self._header_layout.setColumnStretch(2, 1)

        self._title_label = QLabel("双翼科技AI系统")
        self._title_label.setFixedSize(280, 42)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setStyleSheet(
            f"background-color: #0B2343; border: 1px solid {TEXT_ACCENT}; border-radius: 7px; "
            f"color: {TEXT_PRIMARY}; font-size: 20px; font-weight: 700;"
        )
        self._module_tag = QLabel("VISION MODULE · 01", self._header)
        self._module_tag.setStyleSheet(
            text_style(
                TEXT_MUTED,
                size=FONT_MICRO,
                weight=700,
                extra=f"font-family: '{MONO_FAMILY}'; letter-spacing: 0.12em;",
            )
        )
        self._status_label = QLabel("READY")
        self._status_label.setStyleSheet(_status_badge_style(TEXT_ACCENT))
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._result_label = QLabel("")
        self._result_label.setFont(QFont("Microsoft YaHei UI", 14, QFont.Bold))
        self._result_label.setMinimumWidth(64)
        self._result_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._result_label.setVisible(False)
        self._btn_config = QPushButton("配置")
        self._btn_config.setFixedSize(72, 40)
        self._btn_close = QPushButton("关闭")
        self._btn_close.setFixedSize(72, 40)
        self._btn_close.setProperty("buttonRole", "danger")

        actions = QWidget(self._header)
        actions.setObjectName("topBarActions")
        actions.setStyleSheet("QWidget#topBarActions { background: transparent; border: none; }")
        actions_layout = QHBoxLayout(actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(10)
        actions_layout.addWidget(self._status_label, 0, Qt.AlignmentFlag.AlignVCenter)
        actions_layout.addWidget(self._result_label, 0, Qt.AlignmentFlag.AlignVCenter)
        actions_layout.addWidget(self._btn_config, 0, Qt.AlignmentFlag.AlignVCenter)
        actions_layout.addWidget(self._btn_close, 0, Qt.AlignmentFlag.AlignVCenter)

        self._header_layout.addWidget(
            self._module_tag, 0, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._header_layout.addWidget(self._title_label, 0, 1, Qt.AlignmentFlag.AlignCenter)
        self._header_layout.addWidget(actions, 0, 2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        main_layout.addWidget(self._header)

        # 中部双栏：左实时画面 + 右业务面板
        mid_layout = QHBoxLayout()
        mid_layout.setSpacing(12)

        # 左侧实时检测主窗口
        video_frame = QFrame()
        video_frame.setStyleSheet(frame_style(VIEWPORT_BG, border=STROKE_MAIN, radius=12))
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(8, 8, 8, 8)
        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setStyleSheet(
            f"background-color: {VIEWPORT_BG}; border: none; border-radius: 8px;"
        )
        self._video_label.setMinimumSize(640, 480)
        self._video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        video_layout.addWidget(self._video_label)
        self._timing_label = QLabel("")
        self._timing_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._timing_label.setStyleSheet(text_style(TEXT_SECONDARY, size=12, weight=600))
        self._timing_label.setVisible(False)
        video_layout.addWidget(self._timing_label)
        self._video_frame = video_frame
        self._build_viewfinder_overlays(video_frame)
        mid_layout.addWidget(video_frame, 7)

        # 右侧业务面板
        right_panel = QFrame()
        right_panel.setStyleSheet(frame_style(PANEL_BG, border=STROKE_MAIN, radius=12))
        right_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(12)

        # 右上：KPI 单行，清零先归档再把配置里的计数写成 0
        self._kpi_row = KpiRow()
        self._btn_clear_stats = QPushButton("清零")
        self._btn_clear_stats.setFixedSize(72, 72)
        self._btn_clear_stats.setProperty("buttonRole", "danger")
        self._btn_clear_stats.clicked.connect(self._on_clear_production_stats)
        kpi_line = QHBoxLayout()
        kpi_line.setContentsMargins(0, 0, 0, 0)
        kpi_line.setSpacing(8)
        kpi_line.addWidget(self._kpi_row, 1)
        kpi_line.addWidget(self._btn_clear_stats)
        right_layout.addLayout(kpi_line)

        # 右中：紧凑识别进度列表
        steps_head = QHBoxLayout()
        steps_title = QLabel("步骤检测")
        steps_title.setStyleSheet(text_style(TEXT_ACCENT, size=15, weight=700))
        self._recognized_count_label = QLabel("0 / 0")
        self._recognized_count_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._recognized_count_label.setStyleSheet(text_style(TEXT_SECONDARY, size=FONT_SMALL, weight=700))
        steps_head.addWidget(steps_title)
        steps_head.addStretch(1)
        steps_head.addWidget(self._recognized_count_label)
        right_layout.addLayout(steps_head)

        self._steps_scroll = QScrollArea()
        self._steps_scroll.setWidgetResizable(True)
        self._steps_scroll.setMinimumHeight(280)
        self._steps_scroll.setStyleSheet(
            scroll_area_style()
            + f"QScrollArea {{ background-color: #081A34; border: 1px solid {TEXT_ACCENT}; border-radius: 8px; }}"
        )
        self._steps_container = QWidget()
        self._steps_container.setObjectName("recognitionListContainer")
        self._steps_container.setStyleSheet(
            "QWidget#recognitionListContainer { background: transparent; border: none; }"
        )
        self._steps_layout = QVBoxLayout(self._steps_container)
        self._steps_layout.setContentsMargins(10, 10, 10, 10)
        self._steps_layout.setSpacing(8)
        self._steps_layout.addStretch(1)
        self._steps_scroll.setWidget(self._steps_container)
        right_layout.addWidget(self._steps_scroll, 1)

        self._lstm_status_label = QLabel("时序动作：未启用")
        self._lstm_status_label.setWordWrap(True)
        self._lstm_status_label.setStyleSheet(text_style(TEXT_SECONDARY, size=FONT_SMALL, weight=600))
        right_layout.addWidget(self._lstm_status_label)
        self._notice_label = QLabel("")
        self._notice_label.setWordWrap(True)
        self._notice_label.setVisible(False)
        self._notice_label.setStyleSheet(
            f"background-color: #321521; border: 1px solid {TEXT_DANGER}; border-radius: 8px; "
            "color: #FFD8DC; font-size: 13px; font-weight: 600; padding: 10px 12px;"
        )
        right_layout.addWidget(self._notice_label)

        # 右下：核心操作区
        ctrl_layout = QGridLayout()
        ctrl_layout.setContentsMargins(0, 4, 0, 0)
        ctrl_layout.setHorizontalSpacing(10)
        ctrl_layout.setVerticalSpacing(8)

        self._btn_camera = QPushButton("打开相机")
        self._btn_camera.setMinimumHeight(PRIMARY_CONTROL_HEIGHT)
        self._btn_camera.setProperty("buttonRole", "primary")
        self._btn_detect = QPushButton("开始检测")
        self._btn_detect.setMinimumHeight(PRIMARY_CONTROL_HEIGHT)
        self._btn_detect.setProperty("buttonRole", "primary")

        ctrl_layout.addWidget(self._btn_camera, 0, 0)
        ctrl_layout.addWidget(self._btn_detect, 0, 1)
        right_layout.addLayout(ctrl_layout)

        mid_layout.addWidget(right_panel, 4)
        main_layout.addLayout(mid_layout, 1)

        self._stacked.addWidget(self._main_page)

    def _connect_signals(self) -> None:
        self.worker.error_occurred.connect(self._on_error)
        self.worker.overlay_state_updated.connect(self._on_overlay_updated)
        self.worker.pipeline_stats_updated.connect(self._on_stats_updated)
        self.worker.camera_params_updated.connect(self._on_camera_params)

        self._btn_camera.clicked.connect(self._on_toggle_camera)
        self._btn_detect.clicked.connect(self._on_toggle_detect)
        self._btn_config.clicked.connect(self._on_config_clicked)
        self._btn_close.clicked.connect(self._on_close_clicked)

    # ------------------------------------------------------------------
    # 动态识别进度项
    # ------------------------------------------------------------------
    def _effective_step_names(self) -> List[str]:
        """配置中的有效（非空）步骤名称，保留顺序。"""
        names = getattr(self.config, "category_names", []) or []
        return [n.strip() for n in names if n and n.strip()]

    def _refresh_steps(self) -> None:
        """根据配置有效步骤重建识别进度项。"""
        # 清空旧卡片
        for card in self._step_cards:
            card.setParent(None)
            card.deleteLater()
        self._step_cards = []
        self._last_step_focus_key = None
        if self._steps_hint is not None:
            self._steps_hint.setParent(None)
            self._steps_hint.deleteLater()
            self._steps_hint = None

        names = self._effective_step_names()
        counts = self.config.get_runtime_category_counts()
        for i, name in enumerate(names):
            engine_idx = self._card_to_engine_index(i)
            required = counts[engine_idx] if 0 <= engine_idx < len(counts) else 1
            card = RecognitionListItem(i + 1, name, required_count=required)
            card.setVisible(False)
            self._step_cards.append(card)
            # 插在 stretch 之前
            self._steps_layout.insertWidget(self._steps_layout.count() - 1, card)

        # 无有效步骤时给出提示
        if not self._step_cards:
            hint = QLabel("未配置有效步骤，请进入配置页设置步骤类别")
            hint.setStyleSheet(text_style(TEXT_SECONDARY, size=13, weight=500))
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hint.setWordWrap(True)
            hint.setMinimumHeight(180)
            self._steps_layout.insertWidget(self._steps_layout.count() - 1, hint)
            self._steps_hint = hint

        self._recognized_count_label.setText(f"0 / {len(self._step_cards)}")

        if self._roi_engine is not None:
            self._sync_roi_cards()
        else:
            self._update_step_display(self.step_engine.get_state() if self.step_engine else None)
        self._last_step_focus_key = None

    # ------------------------------------------------------------------
    # 显示与状态同步
    # ------------------------------------------------------------------
    def _update_display(self) -> None:
        overlay = self._current_display_overlay()
        render_key = self._display_render_key(overlay)
        if render_key == self._last_render_key:
            return
        render_started = time.perf_counter()
        frame, frame_id = self._resolve_display_source()
        if frame is None:
            return
        if self._should_draw_runtime_overlay(overlay, frame_id):
            frame = self._draw_overlay(frame, overlay)
        else:
            frame = self._draw_roi_overlay(frame)
        self._show_frame(frame)
        self._last_render_key = render_key
        self.worker.record_render_latency((time.perf_counter() - render_started) * 1000.0)

    def _display_render_key(self, overlay) -> tuple:
        """渲染内容的标识。内容未变时跳过渲染，避免定时器空转重复绘制。"""
        label_size = self._video_label.size()
        return (
            bool(self.state.inference_on),
            self._display_source_id(),
            int(getattr(overlay, "source_frame_id", 0) or 0),
            label_size.width(),
            label_size.height(),
            tuple(sorted((getattr(self.config, "roi_template_search_rect", None) or {}).items())),
            tuple(sorted((getattr(self.config, "roi_inspection_rect", None) or {}).items())),
        )

    def _display_source_id(self) -> int:
        """待显示帧的编号，取值口径与 `_resolve_display_source` 一致。"""
        if self.state.inference_on:
            display_id = int(self.worker.get_display_frame_id() or 0)
            if display_id > 0:
                return display_id
        return int(self.worker.get_latest_preview_frame_id() or 0)

    def _invalidate_render_cache(self) -> None:
        """绘制内容的外部依赖变化后强制下一次重绘。"""
        self._last_render_key = None

    def _resolve_display_source(self) -> tuple:
        """检测中显示被推理的同帧画面，预览时才追最新相机帧。"""
        if self.state.inference_on:
            frame, frame_id = self.worker.get_display_frame()
            if frame is not None:
                return frame, int(frame_id or 0)
        return self.worker.get_latest_preview_frame(), 0

    def _current_display_overlay(self):
        overlay = self._latest_runtime_overlay
        if getattr(overlay, "detections", None) or getattr(overlay, "hand_landmarks", None):
            return overlay
        return self.worker.get_latest_overlay()

    def _should_draw_runtime_overlay(self, overlay, frame_id: int) -> bool:
        """检测停止后立即停止画框：预览帧持续推进，旧框贴上去必然错位。"""
        if not self.state.inference_on:
            return False
        has_detections = bool(overlay and getattr(overlay, "detections", None))
        has_hands = bool(overlay and getattr(overlay, "hand_landmarks", None))
        if not has_detections and not has_hands:
            return False
        overlay_id = int(getattr(overlay, "source_frame_id", 0) or 0)
        return overlay_id > 0 and overlay_id == int(frame_id)

    def _sequential_ng_labels(self) -> set[str]:
        """顺序模式本轮 ACTION_NG 对应的类别名；冷却未开新轮前保持，供画红框。"""
        if self._roi_engine is not None or self.step_engine is None:
            return set()
        state = self.step_engine.get_state()
        if getattr(state, "round_result", None) != RoundResult.ACTION_NG:
            return set()
        idx = int(getattr(state, "action_ng_step", -1))
        steps = getattr(state, "steps", None) or []
        if 0 <= idx < len(steps):
            name = str(getattr(steps[idx], "class_name", "") or "").strip()
            if name:
                return {name}
        return set()

    def _overlay_box_color(self, label: str) -> tuple[int, int, int]:
        if str(label or "").strip() in self._sequential_ng_labels():
            return (0, 0, 255)
        return (0, 255, 128)

    def _draw_overlay(self, frame: np.ndarray, overlay) -> np.ndarray:
        if overlay:
            show_conf = self.config.show_confidence_overlay
            show_tid = self.config.show_track_id_overlay
            ng_labels = self._sequential_ng_labels()
            labels: list[tuple[str, tuple[int, int], tuple[int, int, int]]] = []
            for det in overlay.detections:
                if len(det.polygon) < 4:
                    continue
                pts = np.array(det.polygon, dtype=np.int32)
                color = self._overlay_box_color(det.label)
                thickness = 3 if str(det.label or "").strip() in ng_labels else 2
                cv2.polylines(frame, [pts], True, color, thickness)
                label_parts = [det.label]
                if show_conf:
                    label_parts.append(f"{det.conf:.2f}")
                if show_tid and det.track_id is not None:
                    label_parts.append(f"#{det.track_id}")
                label = " ".join(part for part in label_parts if part)
                tx, ty = int(det.polygon[0][0]), int(det.polygon[0][1]) - 5
                labels.append((label, (tx, max(15, ty)), color))
            frame = draw_bgr_texts(frame, labels, pixel_size=16)
            frame = self._draw_tip_and_holes(frame, overlay)
            frame = draw_hand_landmarks(frame, getattr(overlay, "hand_landmarks", None))

        return self._draw_roi_overlay(frame)

    def _draw_roi_overlay(self, frame: np.ndarray) -> np.ndarray:
        """画出当前有效的识别 ROI 与搜索 ROI，不画模板 ROI。"""
        height, width = frame.shape[:2]
        frame_size = (width, height)
        for mapping, color in (
            (getattr(self.config, "roi_template_search_rect", None), (71, 181, 255)),
            (getattr(self.config, "roi_inspection_rect", None), (87, 217, 126)),
        ):
            rect = usable_normalized_rect(mapping)
            if rect is None:
                continue
            x, y, rect_width, rect_height = rect.to_pixel_rect(frame_size)
            p1 = (int(round(x)), int(round(y)))
            p2 = (int(round(x + rect_width)), int(round(y + rect_height)))
            cv2.rectangle(frame, p1, p2, color, 2)
        return frame

    @staticmethod
    def _draw_roi_ng_detection_outlines(
        frame: np.ndarray,
        detections: list,
    ) -> np.ndarray:
        """在 NG 同帧副本上只绘制全部有效 YOLO OBB 轮廓。"""
        rendered = frame.copy()
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            try:
                points = np.asarray(detection.get("polygon", []), dtype=np.int32).reshape(-1, 2)
            except (TypeError, ValueError):
                continue
            if len(points) < 4:
                continue
            cv2.polylines(rendered, [points], True, (0, 255, 128), 2)
        return rendered

    def _draw_tip_and_holes(self, frame: np.ndarray, overlay) -> np.ndarray:
        """绘制枪头点、洞口中心、内外圈和距离。"""
        from calibration.models import CalibrationTransform, HoleDefinition
        cal_data = getattr(self.config, "calibration_points", [])
        holes_data = getattr(self.config, "holes", [])
        tool_class = self.config.tool_class_name

        if not cal_data or not holes_data:
            return frame

        # 标定和孔位只随配置变化，按内容缓存，避免每次渲染都重建。
        cache_key = (repr(cal_data), repr(holes_data))
        cached = getattr(self, "_tip_hole_cache", None)
        if cached is None or cached[0] != cache_key:
            calibrator = CalibrationTransform.from_dict({"points": cal_data})
            holes = [HoleDefinition.from_dict(h) for h in holes_data]
            self._tip_hole_cache = (cache_key, calibrator, holes)
        _, calibrator, holes = self._tip_hole_cache
        if not calibrator.is_valid:
            return frame

        obb_dets = []
        for d in overlay.detections:
            obb_dets.append(ObbDetection(
                class_id=d.class_id, label=d.label, conf=d.conf,
                track_id=d.track_id, polygon=d.polygon, box=d.box, center=d.center,
            ))
        tip, _ = extract_tip_point(obb_dets, tool_class, calibrator, overlay.source_frame_id)

        labels: list[tuple[str, tuple[int, int], tuple[int, int, int]]] = []
        for hole in holes:
            if not hole.enabled:
                continue
            cx, cy = int(hole.center_px[0]), int(hole.center_px[1])
            inner_r = self._mm_to_pixel_radius(hole.inner_radius_mm, calibrator)
            cv2.circle(frame, (cx, cy), max(1, inner_r), (0, 255, 0), 2)
            outer_r = self._mm_to_pixel_radius(hole.outer_radius_mm, calibrator)
            cv2.circle(frame, (cx, cy), max(1, outer_r), (0, 255, 255), 1)
            labels.append((hole.name, (cx - 10, cy - max(1, outer_r) - 5), (255, 255, 255)))

        if tip:
            tx, ty = int(tip.px), int(tip.py)
            cv2.circle(frame, (tx, ty), 6, (0, 0, 255), -1)
            cv2.circle(frame, (tx, ty), 8, (255, 255, 255), 1)

            if self.step_engine:
                st = self.step_engine.get_state()
                if st.current_step_index >= 0:
                    for hole in holes:
                        if hole.step_index == st.current_step_index and hole.enabled:
                            dist = distance_mm((tip.mm_x, tip.mm_y), hole.center_mm)
                            labels.append((f"{dist:.1f}mm", (tx + 10, ty - 10), (0, 255, 255)))
                            cv2.line(frame, (tx, ty),
                                     (int(hole.center_px[0]), int(hole.center_px[1])),
                                     (0, 255, 255), 1)
                            break

        return draw_bgr_texts(frame, labels, pixel_size=14)

    def _mm_to_pixel_radius(self, radius_mm: float, calibrator) -> int:
        if not calibrator.is_valid:
            return int(radius_mm)
        origin_px = calibrator.mm_to_pixel(0.0, 0.0)
        edge_px = calibrator.mm_to_pixel(radius_mm, 0.0)
        return int(((edge_px[0] - origin_px[0]) ** 2 + (edge_px[1] - origin_px[1]) ** 2) ** 0.5)

    def _show_frame(self, frame: np.ndarray) -> None:
        # 先缩到控件尺寸再做色彩转换与 Qt 构图：后面每一步的数据量都按显示分辨率计算，
        # 而不是按相机分辨率，这在高像素相机上是渲染耗时的主要来源。
        scaled = self._scale_frame_to_label(frame)
        rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        # QImage 直接引用 numpy 缓冲区，QPixmap.fromImage 在光栅平台上可能共享这块内存；
        # rgb 回收后 pixmap 就指向已释放内存。先 copy 让 Qt 自己持有像素（显示尺寸，代价很小）。
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self._video_label.setPixmap(QPixmap.fromImage(qimg))

    def _scale_frame_to_label(self, frame: np.ndarray) -> np.ndarray:
        """按 KeepAspectRatio 把整帧缩放到显示控件尺寸。"""
        height, width = frame.shape[:2]
        label_size = self._video_label.size()
        target_width = label_size.width()
        target_height = label_size.height()
        if width <= 0 or height <= 0 or target_width <= 0 or target_height <= 0:
            return frame
        ratio = min(target_width / width, target_height / height)
        new_width = max(1, int(round(width * ratio)))
        new_height = max(1, int(round(height * ratio)))
        if new_width == width and new_height == height:
            return frame
        interpolation = cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(frame, (new_width, new_height), interpolation=interpolation)

    # ------------------------------------------------------------------
    # 步骤状态 → 卡片状态映射
    # ------------------------------------------------------------------
    def _lstm_progress_index(self, state, live_action: str) -> int:
        """当前轮里正在识别的那一步。上一步还没结算完时，不把黄标签切到下一步。"""
        if (
            not self._uses_lstm_sequence()
            or state is None
            or state.round_result != RoundResult.IN_PROGRESS
        ):
            self._lstm_progress_step = -1
            self._lstm_progress_round = None
            return -1
        if state.round_id != self._lstm_progress_round:
            self._lstm_progress_round = state.round_id
            self._lstm_progress_step = -1
        live_name = str(live_action or "").strip()
        current = state.current_step_index
        if live_name and 0 <= current < len(state.steps):
            step = state.steps[current]
            if (
                step.configured
                and step.class_name == live_name
                and step.status not in (StepStatus.PASS, StepStatus.NG)
            ):
                self._lstm_progress_step = current
                return current
        held = self._lstm_progress_step
        if 0 <= held < len(state.steps):
            held_step = state.steps[held]
            if held_step.status not in (StepStatus.PASS, StepStatus.NG):
                return held
        self._lstm_progress_step = -1
        return -1

    def _update_step_display(self, state, live_action: str = "") -> None:
        """根据步骤引擎状态更新识别进度项与 KPI。"""
        if not self._step_cards:
            return

        progress_idx = self._lstm_progress_index(state, live_action)
        terminal_ng_step_idx = -1
        if state is not None and state.round_result == RoundResult.ACTION_NG:
            terminal_ng_step_idx = state.action_ng_step
        has_unresolved_ng = terminal_ng_step_idx >= 0
        ng_step_idx = terminal_ng_step_idx

        current_idx = state.current_step_index if state else -1

        recognized_count = 0
        for card_i, card in enumerate(self._step_cards):
            # card_i 对应有效步骤序号；引擎 steps 可能比卡片多（含空步骤），
            # 需要把卡片序号映射到引擎 steps 中的对应索引。
            engine_idx = self._card_to_engine_index(card_i)
            step_state = StepStatus.WAITING
            if state is not None and 0 <= engine_idx < len(state.steps):
                step_state = state.steps[engine_idx].status

            in_progress = engine_idx == progress_idx and step_state not in (
                StepStatus.PASS,
                StepStatus.NG,
            )
            # 状态映射
            if step_state == StepStatus.PASS:
                card.set_step_state("pass")
            elif step_state == StepStatus.NG:
                card.set_step_state("ng")
            elif in_progress:
                card.set_step_state("progress")
            elif has_unresolved_ng and engine_idx > ng_step_idx:
                # 整轮已经结束后，未判定的后续步骤锁定。进行中的红步不锁后面的类别。
                card.set_step_state("locked")
            elif engine_idx == current_idx:
                card.set_step_state("active")
            elif engine_idx < current_idx:
                # 已经过的步骤应已 PASS（防御性）
                card.set_step_state("pass")
            else:
                card.set_step_state("waiting")

            # 只按步骤顺序出现：未轮到的类别即使被认出也不显示。
            if state is not None and 0 <= engine_idx < len(state.steps):
                ss = state.steps[engine_idx]
                card.set_quantity_progress(ss.current_count, ss.required_count)
                should_show = (
                    step_state == StepStatus.PASS
                    or step_state == StepStatus.NG
                    or in_progress
                    or engine_idx == ng_step_idx
                    or (engine_idx == current_idx and ss.current_count > 0)
                )
                card.setVisible(should_show)
                recognized_count += int(should_show)
            else:
                card.setVisible(False)

        self._sync_recognition_empty_state(recognized_count)

        # KPI 与结果标签
        if state is not None:
            if state.round_result == RoundResult.PASS:
                self._set_notice("")
                if state.round_id != self._last_counted_pass_round:
                    self._last_counted_pass_round = state.round_id
                    self._stats_manager.record_pass()
                    self._play_pass_feedback(("action", state.round_id, "pass"))
                    ct = ""
                    if self._round_start_time > 0:
                        ct = f"{(time.time() - self._round_start_time):.1f}s"
                    if ct:
                        self._status_label.setText(f"PASS  CT {ct}")
                    self._set_result_label("PASS", "#4CAF50")
                    self._update_kpi()
                    self._start_next_round_after_pass()
                    return
                self._set_result_label("PASS", "#4CAF50")
                self._update_kpi()
            elif state.round_result == RoundResult.ACTION_NG:
                notice = state.action_ng_reason or "检测到顺序异常，请按配置顺序重新执行当前步骤。"
                if self._uses_lstm_sequence() or "数量不符" in notice or "未检出" in notice:
                    self._set_notice(notice)
                else:
                    self._set_notice("检测到顺序异常，请按配置顺序重新执行当前步骤。")
                ng_key = (state.round_id, state.action_ng_step)
                if ng_key != self._last_counted_ng_key:
                    self._last_counted_ng_key = ng_key
                    self._stats_manager.record_ng()
                    self._handle_action_ng_feedback(state)
                    self._set_result_label("NG", "#F44336")
                    self._status_label.setText("NG 已记录，下一轮检测中")
                    self._update_kpi()
                    self._start_next_round_after_ng()
                    return
                self._set_result_label("NG", "#F44336")
                self._status_label.setText("NG 已记录")
                if getattr(self, "_action_evidence_warning", ""):
                    self._set_notice(f"{notice}；{self._action_evidence_warning}")
                self._update_kpi()

        self._sync_step_focus_scroll(state)

    def _sync_recognition_empty_state(self, recognized_count: int) -> None:
        if self._steps_hint is not None:
            self._steps_hint.setVisible(not self._step_cards)
        self._recognized_count_label.setText(f"{recognized_count} / {len(self._step_cards)}")

    def _set_notice(self, text: str) -> None:
        self._notice_label.setText(text)
        self._notice_label.setVisible(bool(text))

    def _clear_stale_roi_notice(self) -> None:
        text = self._notice_label.text()
        roi_error_markers = (
            "识别 ROI 未配置",
            "模板 ROI 未配置",
            "搜索 ROI 未配置",
            "ROI 检测配置无法初始化",
            "请检查定位模板、识别 ROI",
        )
        if any(marker in text for marker in roi_error_markers):
            self._set_notice("")

    def _cooldown_interval_ms(self) -> int:
        return int(max(0.0, float(getattr(self.config, "round_cooldown_seconds", 0.0))) * 1000)

    def _resync_round_cooldown_timer(self) -> None:
        if not self._round_pass_timer.isActive():
            return
        self._round_pass_timer.start(self._cooldown_interval_ms())

    def _start_next_round_after_pass(self) -> None:
        if not self.step_engine:
            return
        self._round_pass_timer.start(self._cooldown_interval_ms())

    def _start_next_round_after_ng(self) -> None:
        if not self.step_engine:
            return
        self._round_pass_timer.start(self._cooldown_interval_ms())
        return

    def _refresh_ng_alarm_for_new_round(self) -> None:
        """新一轮允许再响一次 NG；同一轮仍靠 result_key 只报一次。"""
        self._last_counted_ng_key = None
        self._action_evidence_warning = ""
        for key in [
            item
            for item in self._played_result_sound_keys
            if item[:1] == ("action",) and item[-1:] == ("fail",)
        ]:
            self._played_result_sound_keys.pop(key, None)

    def _start_new_round_now(self) -> None:
        if not self.step_engine:
            return
        require_rearm = bool(self._last_counted_ng_key is not None)
        self._begin_step_round(require_first_step_rearm=require_rearm, continue_from_previous=True)
        self._round_start_time = time.time()
        self._last_counted_pass_round = -1
        self._refresh_ng_alarm_for_new_round()
        self._sticky_ng_step_idx = -1
        self._last_step_focus_key = None
        self._set_notice("")
        state = self.step_engine.get_state()
        self._update_step_display(state)
        self._sync_step_focus_scroll(state, reason="ng-next-round", force=True)

    def _begin_step_round(self, *, continue_from_previous: bool = False, **kwargs) -> None:
        """每轮记录真实源帧门槛，保留连续时序隐藏状态。

        延续上一轮时引擎优先用上一轮结束边界，预览帧只作没有边界时的回退。
        """
        kwargs["action_min_source_frame"] = self.worker.get_latest_preview_frame_id() + 1
        if continue_from_previous:
            kwargs["continue_from_previous"] = True
        self.step_engine.start_round(**kwargs)

    @staticmethod
    def _remember_result_key(cache: dict[tuple, None], key: tuple) -> bool:
        """登记有界结果键；返回该键是否首次出现。"""
        if key in cache:
            return False
        cache[key] = None
        if len(cache) > 2048:
            cache.pop(next(iter(cache)))
        return True

    def _play_pass_feedback(self, result_key: Optional[tuple] = None) -> None:
        if result_key is not None and not self._remember_result_key(
            self._played_result_sound_keys,
            result_key,
        ):
            return
        if not getattr(self.config, "pass_sound_enabled", False):
            return
        try:
            self._sound_feedback.play_pass()
        except Exception:
            self._runtime_logger.warning("sound_feedback=pass_failed", exc_info=True)

    def _play_fail_feedback(self, result_key: Optional[tuple] = None) -> None:
        if result_key is not None and not self._remember_result_key(
            self._played_result_sound_keys,
            result_key,
        ):
            return
        if not getattr(self.config, "fail_sound_enabled", True):
            return
        try:
            self._sound_feedback.play_fail()
        except Exception:
            self._runtime_logger.warning("sound_feedback=fail_failed", exc_info=True)

    def _handle_action_ng_feedback(self, state) -> None:
        self._play_fail_feedback(
            ("action", int(getattr(state, "round_id", 0)), "fail")
        )
        if getattr(self.config, "fail_evidence_enabled", True):
            self._save_fail_evidence(state)

    def _save_fail_evidence(self, state) -> None:
        try:
            frame, frame_id = self._get_feedback_frame()
            data = self._last_overlay_data or {}
            source_frame_id = int(data.get("source_frame_id") or frame_id or 0)
            boundary = int(getattr(state, "action_boundary_frame", 0) or 0)
            decision = int(getattr(state, "action_decision_frame", 0) or 0)
            if boundary:
                evidence = self.worker.get_action_boundary_evidence(boundary)
                if evidence is None:
                    candidate = getattr(self, "_action_feedback_evidence", {}).get(boundary)
                    if candidate is not None and candidate[2].get("runtime_generation") == self.worker.runtime_generation:
                        evidence = candidate
                if evidence is None:
                    self._action_evidence_warning = f"动作边界第 {boundary} 帧证据不可用"
                    self._set_notice(f"NG 已记录，{self._action_evidence_warning}")
                    self._runtime_logger.error("action_evidence=unavailable boundary=%s decision=%s", boundary, decision)
                    return
                frame, boundary_detections, metadata = evidence
                source_frame_id = int(metadata.get("source_frame_id", boundary))
                data = dict(metadata, detections=boundary_detections)
            step_idx = int(getattr(state, "action_ng_step", -1))
            names = getattr(self.config, "category_names", []) or []
            step_name = names[step_idx] if 0 <= step_idx < len(names) else ""
            ng_step = None
            if state is not None and 0 <= step_idx < len(getattr(state, "steps", []) or []):
                ng_step = state.steps[step_idx]
            context = FailEvidenceContext(
                round_id=int(getattr(state, "round_id", 0)),
                action_ng_step=step_idx,
                step_name=step_name,
                source_frame_id=source_frame_id,
                detections=list(data.get("detections", []) or []),
                model_path=str(data.get("model_path", "")),
                timestamp=float(data.get("timestamp") or time.time()),
                actual_count=None if ng_step is None else int(ng_step.current_count),
                required_count=None if ng_step is None else int(ng_step.required_count),
                fail_retry_count=None if ng_step is None else int(ng_step.fail_retry_count),
                ng_reason=str(getattr(state, "action_ng_reason", "") or ""),
                boundary_frame=boundary,
                decision_frame=decision or int(data.get("source_frame_id") or 0),
                seen_parts=list(getattr(state, "action_seen_parts", []) or []),
                missing_parts=list(getattr(state, "action_missing_parts", []) or []),
                model_sha256=str(getattr(state, "action_weights_sha256", "") or ""),
                action_start_time=float(getattr(state, "action_start_time", 0.0)) if boundary else None,
                action_end_time=float(getattr(state, "action_end_time", 0.0)) if boundary else None,
                action_decision_time=float(getattr(state, "action_decision_time", 0.0)) if boundary else None,
            )
            saved_path = self._fail_evidence_saver.save_fail_event(frame, context)
            if (
                saved_path is None
                and frame is not None
                and getattr(self._fail_evidence_saver, "enabled", True)
            ):
                self._runtime_logger.error(
                    "action_ng_evidence=save_failed round_id=%s frame_id=%s",
                    int(getattr(state, "round_id", 0) or 0),
                    source_frame_id,
                )
                self._set_notice("NG 证据保存失败，请检查磁盘空间和输出目录权限")
        except Exception:
            # NG 追溯记录丢失必须留痕，不能静默吞掉。
            self._runtime_logger.exception(
                "action_ng_evidence=save_failed reason=exception round_id=%s",
                int(getattr(state, "round_id", 0) or 0),
            )
            self._set_notice("NG 证据保存失败，请检查磁盘空间和输出目录权限")

    def _sync_roi_cards(self) -> None:
        """把 ROI 平级类别的实时数量和锁存状态同步到识别列表。"""
        if self._roi_engine is None:
            return
        snapshot = self._roi_engine.snapshot
        recognized_count = 0
        for card in self._step_cards:
            name = card._name_label.text() if hasattr(card, "_name_label") else ""
            category = snapshot.category_states.get(name)
            if category is None:
                card.setVisible(False)
                continue
            should_show = (
                category.observed_count > 0
                or category.passed
                or bool(snapshot.result and name in snapshot.result.failed_categories)
            )
            card.setVisible(should_show)
            if not should_show:
                continue
            card.set_quantity_progress(category.observed_count, category.required_count)
            recognized_count += int(category.observed_count > 0 or category.passed)
            if category.passed:
                card.set_step_state("pass", hint="数量匹配并已锁存")
            elif snapshot.result and name in snapshot.result.failed_categories:
                card.set_step_state("ng", hint="数量不符")
            elif snapshot.phase == RoundPhase.WAITING_FOR_LOCATOR:
                card.set_step_state("waiting", hint="等待定位")
            elif snapshot.phase == RoundPhase.RESULT_HOLD:
                card.set_step_state("waiting", hint="等待下一轮")
            else:
                card.set_step_state("active", hint="数量判定中")
        self._sync_recognition_empty_state(recognized_count)

    def _get_feedback_frame(self):
        try:
            frame, frame_id = self.worker.get_display_frame()
            if frame is not None:
                return frame, frame_id
        except Exception:
            pass
        try:
            frame = self.worker.get_latest_preview_frame()
            frame_id = self.worker.get_latest_preview_frame_id()
            return frame, frame_id
        except Exception:
            return None, 0

    def _card_to_engine_index(self, card_i: int) -> int:
        """把第 card_i 个有效步骤卡片映射到引擎 steps 索引。

        引擎 steps 按完整 category_names（含空步骤）构造，卡片只对应非空步骤，
        因此需要跳过空步骤找到对应的引擎索引。
        """
        names = getattr(self.config, "category_names", []) or []
        count = 0
        for i, name in enumerate(names):
            if name and name.strip():
                if count == card_i:
                    return i
                count += 1
        return -1

    def _engine_to_card_index(self, engine_idx: int) -> int:
        """把引擎 steps 索引映射到可见步骤卡片索引。"""
        names = getattr(self.config, "category_names", []) or []
        if engine_idx < 0 or engine_idx >= len(names):
            return -1
        count = 0
        for i, name in enumerate(names):
            if name and name.strip():
                if i == engine_idx:
                    return count
                count += 1
        return -1

    def _focused_step_card_index(self, state) -> int:
        """计算当前应该保持可见的步骤卡片。"""
        if state is None:
            return 0 if self._step_cards else -1
        action_ng_step = getattr(state, "action_ng_step", -1)
        if state.round_result == RoundResult.ACTION_NG and action_ng_step >= 0:
            card_i = self._engine_to_card_index(action_ng_step)
            if card_i >= 0:
                return card_i
        card_i = self._engine_to_card_index(getattr(state, "current_step_index", -1))
        if card_i >= 0:
            return card_i
        return 0 if self._step_cards else -1

    def _sync_step_focus_scroll(self, state, *, reason: str = "focus", force: bool = False) -> None:
        """状态焦点变化时滚动到对应步骤，避免普通帧刷新重复抢滚动。"""
        card_i = self._focused_step_card_index(state)
        if card_i < 0:
            return
        round_id = getattr(state, "round_id", -1) if state is not None else -1
        round_result = getattr(state, "round_result", None) if state is not None else None
        ng_step = getattr(state, "action_ng_step", -1) if state is not None else -1
        focus_key = (round_id, card_i, round_result, ng_step, reason)
        if not force and focus_key == self._last_step_focus_key:
            return
        self._last_step_focus_key = focus_key
        self._scroll_to_step_card(card_i, reason=reason)

    def _scroll_to_step_card(self, card_i: int, *, reason: str = "focus") -> None:
        """滚动步骤列表，让目标卡片进入可视区域。"""
        if 0 <= card_i < len(self._step_cards):
            self._steps_scroll.ensureWidgetVisible(self._step_cards[card_i], 0, 32)

    def _update_kpi(self) -> None:
        batch = self._stats_manager.batch
        self._kpi_row.set_value("output", f"{batch.total}")
        self._kpi_row.set_value("yield", f"{batch.yield_rate:.1f}%")
        self._kpi_row.set_value("ok", str(batch.ok))
        self._kpi_row.set_value("ng", str(batch.ng))

    def _on_round_pass_settled(self) -> None:
        """LSTM 只结束结果展示；原 YOLO 模式仍按间隔开始下一轮。"""
        if self._pending_step_engine_rebuild:
            self._pending_step_engine_rebuild = False
            self._rebuild_step_engine()
            self._refresh_steps()
        self._clear_result_label()
        self._status_label.setText("检测中")
        self._status_label.setStyleSheet(_status_badge_style(TEXT_ACCENT))
        if self._uses_lstm_sequence():
            # 连续作业时，下一件的首步可能在展示期间就已经开轮（引擎回到进行中），
            # 这时不能重置，否则会清掉下一件已经完成的步骤。
            # 但如果展示结束时引擎仍停在上一件的 PASS/NG，就必须开新轮：
            # 以前这里直接 return，下一件没动作之前界面一直显示上一件的标签，不会消失。
            if self.step_engine and self.step_engine.get_state().round_result != RoundResult.IN_PROGRESS:
                self._start_new_round_now()
            return
        self._start_new_round_now()

    def _set_result_label(self, text: str, color: str) -> None:
        """显示顶部 PASS/NG 结果，避免无结果时保留空白占位。"""
        # 实心徽章：PASS 绿底黑字 / NG 红底白字，远距离可读；签名保持不变。
        key = text.strip().upper()
        if key == "PASS":
            background, foreground = STEP_PASS_BORDER, "#04160E"
        elif key == "NG":
            background, foreground = STEP_NG_BORDER, "#FFFFFF"
        else:
            background, foreground = "", color
        self._result_label.setText(text)
        if background:
            self._result_label.setStyleSheet(
                f"background-color: {background}; color: {foreground}; border: none; "
                "border-radius: 6px; padding: 4px 14px; font-size: 14px; font-weight: 800;"
            )
        else:
            self._result_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self._result_label.setVisible(True)

    def _clear_result_label(self) -> None:
        self._result_label.setText("")
        self._result_label.setStyleSheet("")
        self._result_label.setVisible(False)

    # ------------------------------------------------------------------
    # 按钮状态同步
    # ------------------------------------------------------------------
    def _sync_button_states(self) -> None:
        """按钮文本与可用状态跟真实运行态同步。"""
        cam_on = self.state.camera_on
        infer_on = self.state.inference_on
        self._btn_camera.setText("关闭相机" if cam_on else "打开相机")
        self._btn_detect.setText("停止检测" if infer_on else "开始检测")
        # 检测按钮依赖相机和处理器
        can_detect = cam_on and self.processor is not None
        self._btn_detect.setVisible(True)
        self._btn_detect.setEnabled(infer_on or can_detect)
        self._update_timing_display()
        if not can_detect and not infer_on:
            if cam_on and self.processor is None:
                self._status_label.setText("处理器未加载")
                self._status_label.setStyleSheet(_status_badge_style(TEXT_SECONDARY))

    # ------------------------------------------------------------------
    # 信号槽
    # ------------------------------------------------------------------
    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._runtime_logger.error("runtime_error=%s", msg)
        self._status_label.setText(f"错误: {msg}")
        self._status_label.setStyleSheet("color: #F44336; font-weight: bold;")
        self._set_notice(msg)

    def _overlay_image_size(self, data: dict) -> tuple:
        raw_image_size = tuple((data or {}).get("image_size") or ())
        if len(raw_image_size) == 2 and all(float(value) > 0 for value in raw_image_size):
            return (int(raw_image_size[0]), int(raw_image_size[1]))
        return self._display_frame_size()

    def _display_frame_size(self) -> tuple:
        """显示帧尺寸。测试替身可能只实现 get_display_frame，此时退回读整帧。"""
        size_of = getattr(self.worker, "get_display_frame_size", None)
        if callable(size_of):
            return tuple(size_of())
        display_frame, _ = self.worker.get_display_frame()
        if display_frame is None:
            return ()
        frame_height, frame_width = display_frame.shape[:2]
        return (int(frame_width), int(frame_height))

    def _judgement_detections(self, detections: list, image_size: tuple) -> list:
        """有效识别 ROI 只把中心在框内的目标交给判定；空框整图。"""
        rect = usable_normalized_rect(getattr(self.config, "roi_inspection_rect", None))
        if rect is None or not image_size:
            return list(detections)
        return [
            detection
            for detection in detections
            if detection_center_in_rect(detection, rect, image_size)
        ]

    def _log_step_sequence_frame(
        self,
        data: dict,
        raw_detections: list,
        judged: list,
        state,
        frozen: bool,
    ) -> None:
        """顺序模式逐帧记录判定输入与结果，用于区分「没检到」「被 ROI 滤掉」「数量不符」。"""
        raw_counts = Counter(str(det.label) for det in raw_detections if getattr(det, "label", ""))
        judged_counts = Counter(str(det.label) for det in judged if getattr(det, "label", ""))
        if state is not None and 0 <= state.current_step_index < len(state.steps):
            step = state.steps[state.current_step_index]
            expected = (
                f"idx={step.index} class={step.class_name} required={step.required_count} "
                f"current={step.current_count} status={step.status.value}"
            )
        else:
            expected = "none"
        roi = getattr(self.config, "roi_inspection_rect", None)
        # 每帧一条，只在 DEBUG 下输出，避免运行日志被逐帧刷屏。
        self._runtime_logger.debug(
            "step_seq_update frame_id=%s frozen=%s roi=%s raw=%s judged=%s "
            "expected=[%s] round=%s round_id=%s reason=%s",
            data.get("source_frame_id", 0),
            frozen,
            roi,
            dict(raw_counts),
            dict(judged_counts),
            expected,
            getattr(state, "round_result", None).value if state is not None else None,
            getattr(state, "round_id", 0),
            getattr(state, "action_ng_reason", "") or "",
        )

    @Slot(dict)
    def _on_overlay_updated(self, data: dict) -> None:
        try:
            self._handle_overlay_update(data)
        finally:
            # 无论是否丢弃，都要释放推理线程的积压名额。
            ack = getattr(self.worker, "ack_overlay_event", None)
            if callable(ack):
                ack()

    def _handle_overlay_update(self, data: dict) -> None:
        if data.get("runtime_generation", self.worker.runtime_generation) != self.worker.runtime_generation:
            return
        self._last_overlay_data = dict(data or {})
        detections = data.get("detections", [])
        obb_dets = []
        for d in detections:
            obb_dets.append(ObbDetection(
                class_id=d.get("class_id", 0),
                label=d.get("label", ""),
                conf=d.get("conf", 0.0),
                track_id=d.get("track_id"),
                polygon=[tuple(p) for p in d.get("polygon", [])],
                box=tuple(d.get("box", (0, 0, 0, 0))),
                center=tuple(d.get("center", d.get("center_px", (0, 0)))),
            ))

        state = None
        image_size = self._overlay_image_size(data)
        judged = self._judgement_detections(obb_dets, image_size)
        if self._roi_engine is not None:
            # 每帧一条，只在 DEBUG 下组装和输出。
            if self._runtime_logger.isEnabledFor(logging.DEBUG):
                image_size_source = "overlay" if data.get("image_size") else "derived"
                sample_centers = []
                for d in detections[:3]:
                    c = d.get("center", d.get("center_px", (0, 0)))
                    if c:
                        sample_centers.append((float(c[0]), float(c[1])))
                self._runtime_logger.debug(
                    "roi_overlay_received frame_id=%s det_count=%d image_size=%s image_size_source=%s "
                    "sample_centers=%s labels=%s",
                    data.get("source_frame_id", 0),
                    len(detections),
                    tuple(image_size) if image_size else None,
                    image_size_source,
                    sample_centers,
                    {d.get("label", ""): 1 for d in detections},
                )

            roi_results = []
            if image_size:
                template_match = {
                    "matched": data.get("template_matched", False),
                    "score": data.get("template_score", 0.0),
                    "center": data.get("template_center"),
                    "angle": data.get("template_angle"),
                }
                previous_round_id = self._roi_engine.snapshot.round_id
                roi_results = self._roi_engine.update(
                    judged,
                    image_size=image_size,
                    source_frame_id=data.get("source_frame_id", 0),
                    timestamp=data.get("timestamp") or None,
                    template_match=template_match,
                )
                if self._roi_engine.snapshot.round_id != previous_round_id and not roi_results:
                    self._clear_result_label()
                    self._set_notice("")
            else:
                self._runtime_logger.error(
                    "roi_update=skipped reason=image_size_unavailable frame_id=%s",
                    data.get("source_frame_id", 0),
                )
                self._set_notice("ROI 定位失败：无法获取原始帧尺寸")
            self._handle_roi_results(
                roi_results,
                data,
            )
            self._sync_roi_cards()
        elif self.step_engine and self._uses_lstm_sequence():
            report = data.get("lstm_action") or {}
            frozen = self._round_pass_timer.isActive()
            if report.get("fault") or getattr(self, "_action_runtime_fault", ""):
                fault = str(report.get("fault") or self._action_runtime_fault)
                self._action_runtime_fault = fault
                self.worker.stop_inference()
                self._round_pass_timer.stop()
                self._set_notice(f"模型故障，已停止检测，禁止放行：{fault}")
                self._status_label.setText("模型故障，检测已停止")
                self._status_label.setStyleSheet(_status_badge_style(TEXT_DANGER))
                self._lstm_status_label.setText(f"时序动作：故障，{fault}")
                self._clear_result_label()
                self._sync_button_states()
                state = self.step_engine.get_state()
            else:
                state = self.step_engine.get_state()
                events = list(report.get("events") or [])
                if self.state.inference_on:
                    try:
                        for event in events:
                            state = self._consume_lstm_action_event(event, str(report.get("action") or ""))
                    except RuntimeError as exc:
                        self._action_runtime_fault = str(exc)
                        self.worker.stop_inference()
                        self._round_pass_timer.stop()
                        self._set_notice(f"动作处理故障，已停止检测，禁止放行：{exc}")
                        self._runtime_logger.exception("action_processing=failed")
                else:
                    for event in events:
                        trace(
                            "event_decision",
                            result="reject",
                            reason="inference_off",
                            note="检测未开启",
                            name=event.get("name") or "-",
                            start_frame=event.get("start_frame"),
                            end_frame=event.get("end_frame"),
                            session=event.get("session_id"),
                            event=event.get("event_id"),
                        )
                self._show_action_progress(report)
            self._log_step_sequence_frame(data, obb_dets, judged, state, frozen)
            if (
                not getattr(self, "_action_runtime_fault", "")
                and state is not None
                and state.round_result == RoundResult.IN_PROGRESS
            ):
                self._update_step_display(state, live_action=str(report.get("action") or ""))
        elif self.step_engine:
            frozen = self._round_pass_timer.isActive()
            if frozen:
                state = self.step_engine.get_state()
            else:
                try:
                    state = self.step_engine.update(judged, frame_id=data.get("source_frame_id", 0))
                except TypeError:
                    state = self.step_engine.update(judged)
            self._log_step_sequence_frame(data, obb_dets, judged, state, frozen)
            self._update_step_display(state)
        if not self._uses_lstm_sequence():
            self._update_lstm_action(data, obb_dets, image_size)

        overlay = DetectionOverlayState(
            source_frame_id=data.get("source_frame_id", 0),
            timestamp=data.get("timestamp", 0.0),
            model_path=data.get("model_path", ""),
            task_type=data.get("task_type", ""),
            detections=obb_dets,
            status=data.get("status", ""),
            error=data.get("error", ""),
            round_id=data.get("round_id", 0),
            latency_ms=data.get("latency_ms", 0.0),
            hand_landmarks=self._normalize_hand_landmarks(data.get("hand_landmarks")),
        )
        # 渲染不在事件路径上进行：积压时判定仍逐帧推进，而画面由定时器统一取最新状态渲染。
        self._latest_runtime_overlay = overlay
        self._record_coordinate_log(overlay, state)

    def _consume_lstm_action_event(self, event: dict, live_action: str = ""):
        """每个事件及时记账；结果显示停留不参与业务换轮。"""
        if self.step_engine.get_state().round_result != RoundResult.IN_PROGRESS:
            self._start_new_round_now()
        generation = self.worker.runtime_generation
        if getattr(self, "_action_feedback_evidence_generation", None) != generation:
            self._action_feedback_evidence = {}
            self._action_feedback_evidence_generation = generation
        evidence = event.get("_boundary_evidence")
        boundary = int(event.get("end_frame") or 0)
        if (
            evidence is not None
            and evidence[2].get("source_frame_id") == boundary
            and evidence[2].get("runtime_generation") == generation
        ):
            self._action_feedback_evidence[boundary] = evidence
        state = self.step_engine.apply_action_event(event)
        self._update_step_display(state, live_action=live_action)
        # 配置的类别全部完成（PASS 或 NG），这一件就结束：计数、声音、证据在上面
        # _update_step_display 里已经做完，这里立刻开新一轮，识别列表马上清空，
        # 等下一件检测到动作再显示。顶部 PASS/NG 徽章按"间隔时间"保留，到点由
        # _on_round_pass_settled 清掉。新轮可能重放暂存动作又结束一件，所以循环。
        finished = None
        notice = self._notice_label.text()
        while state.round_result != RoundResult.IN_PROGRESS:
            finished = state
            self._start_new_round_now()
            state = self.step_engine.get_state()
        if finished is not None and finished.round_result == RoundResult.ACTION_NG and notice:
            # 开新轮会清提示，NG 原因要留给操作员看。
            self._set_notice(notice)
        keep = set(getattr(state, "action_failed_boundary_frames", []))
        keep.update(list(self._action_feedback_evidence)[-2:])
        self._action_feedback_evidence = {
            frame: owned for frame, owned in self._action_feedback_evidence.items() if frame in keep
        }
        return state

    def _handle_roi_results(
        self,
        results: list,
        overlay_data: dict,
    ) -> None:
        """整轮结果只接入一次统计、声音和证据。"""
        for result in results:
            result_key = ("roi-result", result.feedback_key)
            is_new_result = self._remember_result_key(
                self._handled_roi_result_keys,
                result_key,
            )
            if not is_new_result:
                continue
            if result.outcome == InspectionOutcome.PASS:
                self._stats_manager.record_pass()
                self._play_pass_feedback(("roi", result.feedback_key))
                self._set_result_label("PASS", "#4CAF50")
                self._status_label.setText("ROI 轮次 PASS")
                self._set_notice("")
            elif result.outcome == InspectionOutcome.FAIL:
                self._stats_manager.record_ng()
                self._save_roi_fail_evidence(result, overlay_data)
                failed = " / ".join(result.failed_categories)
                self._set_result_label("NG", "#F44336")
                self._status_label.setText(f"ROI 轮次 NG：{failed}")
                self._set_notice(f"识别 ROI 数量异常：{failed}")
                self._show_roi_fail_alert(result, overlay_data)
            self._update_kpi()

    def _prepare_roi_ng_annotated_frame(
        self,
        result,
        overlay_data: dict,
        *,
        log_key: str,
    ) -> Optional[tuple[np.ndarray, list[dict], dict[str, int]]]:
        """生成同帧识别 ROI 裁剪图、局部 detections 和原图 ROI。"""
        result_frame_id = int(getattr(result, "source_frame_id", 0) or 0)
        event_frame_id = int(overlay_data.get("source_frame_id", 0) or 0)
        source_frame = overlay_data.get("source_frame")
        if result_frame_id != event_frame_id:
            self._runtime_logger.error(
                f"{log_key}=skipped reason=frame_id_mismatch "
                "result_frame_id=%s event_frame_id=%s",
                result_frame_id,
                event_frame_id,
            )
            return None
        if not isinstance(source_frame, np.ndarray) or source_frame.size == 0:
            self._runtime_logger.error(
                f"{log_key}=skipped reason=source_frame_unavailable frame_id=%s",
                event_frame_id,
            )
            return None

        try:
            inspection_roi = NormalizedRect.from_mapping(
                getattr(self.config, "roi_inspection_rect", None)
            )
        except ValueError as exc:
            self._runtime_logger.error(
                f"{log_key}=skipped reason=invalid_inspection_roi "
                "frame_id=%s error=%s",
                event_frame_id,
                exc,
            )
            return None

        source_height, source_width = source_frame.shape[:2]
        roi_x, roi_y, roi_width, roi_height = inspection_roi.to_pixel_rect(
            (source_width, source_height)
        )
        left = max(0, min(source_width, int(round(roi_x))))
        top = max(0, min(source_height, int(round(roi_y))))
        right = max(0, min(source_width, int(round(roi_x + roi_width))))
        bottom = max(0, min(source_height, int(round(roi_y + roi_height))))
        if right <= left or bottom <= top:
            self._runtime_logger.error(
                f"{log_key}=skipped reason=empty_inspection_roi frame_id=%s "
                "pixel_rect=(%s,%s,%s,%s) source_size=(%s,%s)",
                event_frame_id,
                left,
                top,
                right,
                bottom,
                source_width,
                source_height,
            )
            return None

        local_detections = []
        for detection in list(overlay_data.get("detections", []) or []):
            if not isinstance(detection, dict):
                continue
            center = detection.get("center", detection.get("center_px"))
            try:
                if center is None or len(center) < 2:
                    continue
                center_point = (float(center[0]), float(center[1]))
            except (TypeError, ValueError):
                continue
            if not inspection_roi.contains_pixel_point(
                center_point,
                (source_width, source_height),
            ):
                continue
            local_detections.append(
                self._translate_detection_to_roi(detection, left, top)
            )

        cropped_frame = source_frame[top:bottom, left:right].copy()
        rendered = self._draw_roi_ng_detection_outlines(
            cropped_frame,
            local_detections,
        )
        source_roi = {
            "x": left,
            "y": top,
            "width": right - left,
            "height": bottom - top,
            "source_width": source_width,
            "source_height": source_height,
        }
        return rendered, local_detections, source_roi

    @staticmethod
    def _translate_detection_to_roi(
        detection: dict,
        offset_x: int,
        offset_y: int,
    ) -> dict:
        """深复制 detection 并把像素坐标平移到 ROI 局部坐标。"""
        translated = deepcopy(detection)

        polygon = translated.get("polygon")
        if isinstance(polygon, (list, tuple)):
            translated_polygon = []
            for point in polygon:
                try:
                    if len(point) < 2:
                        translated_polygon.append(point)
                        continue
                    translated_polygon.append(
                        [point[0] - offset_x, point[1] - offset_y]
                    )
                except (TypeError, ValueError):
                    translated_polygon.append(point)
            translated["polygon"] = translated_polygon

        box = translated.get("box")
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            try:
                translated["box"] = [
                    box[0] - offset_x,
                    box[1] - offset_y,
                    box[2] - offset_x,
                    box[3] - offset_y,
                ]
            except (TypeError, ValueError):
                pass

        for center_key in ("center", "center_px"):
            center = translated.get(center_key)
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                try:
                    translated[center_key] = [
                        center[0] - offset_x,
                        center[1] - offset_y,
                    ]
                except (TypeError, ValueError):
                    pass
        return translated

    def _save_roi_fail_evidence(self, result, overlay_data: dict) -> None:
        """保存产生本次 ROI FAIL 的同帧绿色 OBB 图和元数据。"""
        if not getattr(self.config, "fail_evidence_enabled", True):
            return
        event_frame_id = int(overlay_data.get("source_frame_id", 0) or 0)
        try:
            prepared = self._prepare_roi_ng_annotated_frame(
                result,
                overlay_data,
                log_key="roi_ng_evidence",
            )
            if prepared is None:
                return
            frame, detections, source_roi = prepared
            context = FailEvidenceContext(
                round_id=int(result.round_id),
                action_ng_step=-1,
                step_name=f"ROI 数量异常: {'/'.join(result.failed_categories)}",
                source_frame_id=event_frame_id,
                detections=detections,
                model_path=str(overlay_data.get("model_path", "")),
                timestamp=float(
                    overlay_data.get("timestamp")
                    or getattr(result, "timestamp", 0.0)
                    or time.time()
                ),
                source_roi=source_roi,
            )
            saved_path = self._fail_evidence_saver.save_fail_event(frame, context)
            if saved_path is None:
                self._runtime_logger.error(
                    "roi_ng_evidence=save_failed round_id=%s frame_id=%s",
                    int(result.round_id),
                    event_frame_id,
                )
                self._set_notice("NG 证据保存失败，请检查磁盘空间和输出目录权限")
        except Exception as exc:
            self._runtime_logger.error(
                "roi_ng_evidence=save_failed reason=exception round_id=%s "
                "frame_id=%s error=%s",
                int(getattr(result, "round_id", 0) or 0),
                event_frame_id,
                exc,
            )

    def _show_roi_fail_alert(self, result, overlay_data: dict) -> None:
        """使用产生本次 FAIL 的同帧图像和 detections 显示纯框告警。"""
        prepared = self._prepare_roi_ng_annotated_frame(
            result,
            overlay_data,
            log_key="roi_ng_alert",
        )
        if prepared is None:
            self._play_fail_feedback(
                (
                    "roi-alert-unavailable",
                    int(getattr(result, "round_id", 0) or 0),
                    int(getattr(result, "source_frame_id", 0) or 0),
                )
            )
            return
        frame, _detections, _source_roi = prepared

        if self._roi_fail_alert is not None and self._roi_fail_alert.isVisible():
            self._roi_fail_alert.refresh(frame)
            return

        self._sound_feedback.play_fail_loop()
        self._roi_fail_alert = RoiFailAlertDialog(self)
        self._roi_fail_alert.alert_closed.connect(self._close_roi_fail_alert)
        self._roi_fail_alert.refresh(frame)
        self._roi_fail_alert.show()

    def _close_roi_fail_alert(self) -> None:
        """关闭 FAIL 告警弹窗并停止循环声音。"""
        dialog = self._roi_fail_alert
        self._roi_fail_alert = None
        if dialog is not None:
            dialog.alert_closed.disconnect(self._close_roi_fail_alert)
        self._sound_feedback.stop_loop()

    @Slot(dict)
    def _on_stats_updated(self, stats: dict) -> None:
        self._last_pipeline_stats = dict(stats)
        self._update_timing_display()

    def _update_timing_display(self) -> None:
        """把四段耗时和瓶颈段直接摆在画面下方，现场不用翻日志就能定位卡在哪。"""
        stats = self._last_pipeline_stats
        if not stats or not self.state.inference_on:
            self._timing_label.setVisible(False)
            return
        segments = [
            ("采集", float(stats.get("capture_ms", 0.0) or 0.0)),
            ("定位", float(stats.get("template_ms", 0.0) or 0.0)),
            ("推理", float(stats.get("yolo_ms", 0.0) or 0.0)),
            ("渲染", float(stats.get("render_ms", 0.0) or 0.0)),
        ]
        measured = [(name, value) for name, value in segments if value > 0.0]
        if not measured:
            self._timing_label.setVisible(False)
            return
        slowest = max(measured, key=lambda item: item[1])[0]
        parts = "   ".join(f"{name} {value:.1f}ms" for name, value in measured)
        fps = float(stats.get("infer_fps", 0.0) or 0.0)
        self._timing_label.setText(f"{parts}   瓶颈 {slowest}   推理 {fps:.1f} fps")
        self._timing_label.setVisible(True)

    @Slot(dict)
    def _on_camera_params(self, params: dict) -> None:
        self._last_camera_params = dict(params)

    def _record_coordinate_log(self, overlay, step_state) -> None:
        """把当前推理帧写入坐标诊断日志（底层能力，不在主界面显示状态）。"""
        if not self._coordinate_logger.active:
            return
        size = self._display_frame_size()
        image_size = list(size) if size else None
        record = build_coordinate_frame_record(
            overlay,
            self.config,
            camera_params=self._last_camera_params,
            pipeline_stats=self._last_pipeline_stats,
            image_size=image_size,
            step_state=step_state,
        )
        self._coordinate_logger.log_frame(record)

    # ------------------------------------------------------------------
    # 按钮处理
    # ------------------------------------------------------------------
    @Slot()
    def _on_toggle_camera(self) -> None:
        if self.state.camera_on:
            self._runtime_logger.info("camera_action=close")
            self._close_camera_session()
            self._status_label.setText("READY")
            self._status_label.setStyleSheet(_status_badge_style(TEXT_ACCENT))
        else:
            self._runtime_logger.info("camera_action=open")
            if self.worker.open_camera():
                self._runtime_logger.info("camera_open=ok")
                self._display_timer.start(33)
                self._status_label.setText("预览中")
                self._status_label.setStyleSheet(_status_badge_style(TEXT_ACCENT))
            else:
                self._runtime_logger.error("camera_open=failed")
                self._status_label.setText("相机打开失败")
                self._status_label.setStyleSheet("color: #F44336; font-weight: bold;")
        self._sync_button_states()

    @Slot()
    def _on_toggle_detect(self) -> None:
        if self.state.inference_on:
            self._runtime_logger.info("detection_action=stop")
            self.worker.stop_inference()
            self.worker.reset_action_session()
            self._round_pass_timer.stop()
            self._pending_step_engine_rebuild = False
            self._stop_template_diagnostics()
            self._status_label.setText("已停止")
            self._status_label.setStyleSheet(text_style(TEXT_SECONDARY, size=14, weight=500))
        else:
            if not self.state.camera_on:
                self._runtime_logger.warning("detection_start=rejected reason=camera_not_open")
                self._status_label.setText("请先打开相机")
                self._status_label.setStyleSheet("color: #FF9800; font-weight: bold;")
                self._sync_button_states()
                return
            if self.processor is None:
                self._runtime_logger.error(
                    "detection_start=rejected reason=processor_unavailable log_path=%s",
                    self._runtime_log_path or "unavailable",
                )
                self._status_label.setText("处理器未加载，无法检测")
                self._status_label.setStyleSheet("color: #F44336; font-weight: bold;")
                self._show_processor_log_notice()
                self._sync_button_states()
                return
            template_runtime_error = self._template_runtime_validation_error()
            if template_runtime_error:
                self._runtime_logger.error(
                    "detection_start=rejected reason=template_runtime_invalid error=%s",
                    template_runtime_error,
                )
                self._status_label.setText(template_runtime_error)
                self._status_label.setStyleSheet("color: #F44336; font-weight: bold;")
                self._set_notice(template_runtime_error)
                self._sync_button_states()
                return
            config_errors = self.config.inspection_validation_errors(
                self._current_model_class_names()
            )
            if config_errors:
                message = "；".join(config_errors)
                self._runtime_logger.error(
                    "detection_start=rejected reason=invalid_config errors=%s",
                    message,
                )
                self._status_label.setText(config_errors[0])
                self._status_label.setStyleSheet("color: #F44336; font-weight: bold;")
                self._set_notice(message)
                self._sync_button_states()
                return
            if self._uses_lstm_sequence():
                self._reload_configured_action_model()
            if self._uses_lstm_sequence() and (getattr(self, "_action_block_reason", "") or getattr(self, "_action_runtime_fault", "")):
                message = self._action_block_reason or f"模型故障，请重新保存配置以加载模型：{self._action_runtime_fault}"
                self._status_label.setText(message)
                self._status_label.setStyleSheet("color: #F44336; font-weight: bold;")
                self._set_notice(message)
                self._sync_button_states()
                return
            if getattr(self.config, "roi_inspection_enabled", False):
                self._init_roi_engine()
                if self._roi_engine is None:
                    self._status_label.setText("ROI 检测配置无法初始化")
                    self._set_notice("请检查待检类别和数量配置")
                    self._sync_button_states()
                    return
                self._sync_roi_cards()
            elif self.step_engine:
                self._begin_step_round()
                self._round_start_time = time.time()
                self._round_id_reset()
                self._last_counted_pass_round = -1
                self._refresh_ng_alarm_for_new_round()
                self._update_step_display(self.step_engine.get_state())
            self._start_template_diagnostics()
            self.worker.start_inference()
            self._clear_stale_roi_notice()
            self._runtime_logger.info(
                "detection_start=ok processor=%s",
                type(self.processor).__name__,
            )
            self._status_label.setText("检测中")
            self._status_label.setStyleSheet(_status_badge_style(TEXT_ACCENT))
            self._clear_result_label()
        self._sync_hand_overlay_enabled()
        self._sync_button_states()

    # ------------------------------------------------------------------
    # 模板匹配诊断会话
    # ------------------------------------------------------------------
    def _start_template_diagnostics(self) -> None:
        """开始检测时按配置启动模板诊断会话。失败只提示，不阻断检测。"""
        recorder = get_template_diagnostics()
        if recorder.active:
            return
        from yolo_runtime.template_gated_processor import TemplateGatedProcessor

        enabled = bool(
            getattr(self.config, "template_diagnostics_enabled", False)
            and isinstance(self.processor, TemplateGatedProcessor)
        )
        recorder.configure(
            enabled=enabled,
            log_dir=getattr(self.config, "template_diagnostics_dir", "logs/template_sessions"),
            base_dir=self._app_base_dir,
            shadow_enabled=bool(
                getattr(self.config, "template_diagnostics_shadow_enabled", True)
            ),
            sample_frames=int(getattr(self.config, "template_diagnostics_sample_frames", 120)),
            max_queue=int(getattr(self.config, "template_diagnostics_max_queue", 600)),
        )
        if not enabled:
            self._set_template_diagnostics_status("诊断日志：未启用")
            return
        session_id = recorder.start_session(context=self._template_diagnostics_context())
        if session_id:
            self._set_template_diagnostics_status(f"诊断日志记录中：{recorder.log_path}")
        else:
            message = recorder.last_error or "模板诊断日志无法创建"
            self._set_template_diagnostics_status(f"诊断日志启动失败：{message}", warning=True)
            self._set_notice(message)

    def _stop_template_diagnostics(self) -> None:
        recorder = get_template_diagnostics()
        if not recorder.active:
            return
        summary = recorder.stop_session()
        frames = summary.get("frames", 0)
        matched_rate = float(summary.get("matched_rate") or 0.0)
        sampled = summary.get("sampled_frames", 0)
        self._set_template_diagnostics_status(
            f"诊断日志已保存：{recorder.summary_path}\n"
            f"共 {frames} 帧，命中率 {matched_rate * 100:.1f}%，抽样 {sampled} 帧"
        )
        self._runtime_logger.info(
            "template_diagnostics_session=closed frames=%s matched_rate=%.3f summary=%s",
            frames,
            matched_rate,
            recorder.summary_path,
        )

    def _template_diagnostics_context(self) -> dict:
        """把定位相关配置一并写进日志，避免事后对不上参数。"""
        config = self.config
        return {
            "model_path": str(getattr(config, "yolo_model_path", "")),
            "yolo_conf_threshold": float(getattr(config, "yolo_conf_threshold", 0.0)),
            "template_path": str(getattr(config, "roi_template_path", "")),
            "match_threshold": float(getattr(config, "roi_template_match_threshold", 0.6)),
            "angle_range": float(getattr(config, "roi_template_angle_range", 15.0)),
            "angle_step": float(getattr(config, "roi_template_angle_step", 1.0)),
            "template_rect": getattr(config, "roi_template_rect", None),
            "search_rect": getattr(config, "roi_template_search_rect", None),
            "inspection_rect": getattr(config, "roi_inspection_rect", None),
            "category_names": list(getattr(config, "category_names", []) or []),
            "category_counts": list(config.get_runtime_category_counts()),
            "pass_stable_frames": int(getattr(config, "action_pass_stable_frames", 1)),
            "ng_stable_frames": int(getattr(config, "action_ng_stable_frames", 10)),
            "round_cooldown_seconds": float(getattr(config, "round_cooldown_seconds", 0.0)),
            "camera_sn": str(getattr(config, "mvsdk_camera_sn", "")),
            "camera_exposure_us": int(getattr(config, "camera_manual_exposure_us", 0)),
            "processor": type(self.processor).__name__ if self.processor else "None",
        }

    def _set_template_diagnostics_status(self, text: str, *, warning: bool = False) -> None:
        if self._config_page is not None:
            self._config_page.set_template_diagnostics_status(text, warning=warning)

    def _sync_template_diagnostics_status(self) -> None:
        """打开配置页时回显当前诊断状态和上一次会话路径。"""
        recorder = get_template_diagnostics()
        status = recorder.get_status()
        if status.get("active"):
            self._set_template_diagnostics_status(
                f"诊断日志记录中：{status.get('log_path')}\n"
                f"已写入 {status.get('written')} 帧，抽样 {status.get('sampled_frames')} 帧"
            )
        elif status.get("summary_path"):
            self._set_template_diagnostics_status(
                f"上次诊断日志：{status.get('summary_path')}"
            )
        elif not getattr(self.config, "template_diagnostics_enabled", False):
            self._set_template_diagnostics_status("诊断日志：未启用")
        else:
            self._set_template_diagnostics_status("诊断日志：将在开始检测时创建")

    def _restart_template_diagnostics_if_running(self) -> None:
        """保存配置后让诊断开关立即生效，无需重启检测。"""
        if not self.state.inference_on:
            return
        self._stop_template_diagnostics()
        self._start_template_diagnostics()

    def _template_runtime_validation_error(self) -> str:
        """仅在模板层会实际启用时检查门控处理器、模型和搜索余量。"""
        from template_matching.template_store import load_shape_model, resolve_template_path
        from yolo_runtime.template_gated_processor import TemplateGatedProcessor

        if not bool(getattr(self.config, "roi_template_enabled", False)):
            return ""
        if usable_normalized_rect(getattr(self.config, "roi_template_search_rect", None)) is None:
            return ""
        try:
            template_path = resolve_template_path(self.config, self._app_base_dir)
        except ValueError:
            return ""
        if not template_path.is_file():
            return ""

        if not isinstance(self.processor, TemplateGatedProcessor):
            return "模板定位处理器未加载，请先验证并保存"
        if self.worker.frame_processor is not self.processor:
            return "相机线程未使用模板定位处理器，请完成三种 ROI 后点击验证并保存"
        try:
            model = load_shape_model(self.config, self._app_base_dir)
        except Exception as exc:
            return f"定位模型不可用：{exc}"

        # 搜索框余量必须够，否则运行期会因旋转候选被丢弃而随机不触发。
        search = getattr(self.config, "roi_template_search_rect", None)
        frame = self.worker.get_latest_preview_frame()
        if frame is not None:
            height, width = frame.shape[:2]
            roi_w = float(search.get("width", 0.0)) * width
            roi_h = float(search.get("height", 0.0)) * height
            travel = model.center_travel((int(roi_w), int(roi_h)))
            need = max(
                int(getattr(self.config, "roi_template_min_travel_px", 40)),
                int(round(min(model.template_size)
                          * float(getattr(self.config, "roi_template_min_travel_ratio", 0.30)))),
            )
            if travel[0] < need or travel[1] < need:
                required = model.min_search_size(need)
                return (
                    f"搜索框太小：当前 {int(roi_w)}×{int(roi_h)}，"
                    f"只剩 {travel[0]}×{travel[1]}px 平移余量，"
                    f"至少需要 {required[0]}×{required[1]}"
                )
        return ""

    def _round_id_reset(self) -> None:
        """开始检测时重置本轮去重键（round_id 由 start_round 自增）。"""
        self._last_counted_pass_round = -1

    @Slot()
    def _on_config_clicked(self) -> None:
        if self._config_page is None:
            from ui.config_page import ConfigPage
            self._config_page = ConfigPage(
                self.config,
                stats_manager=self._stats_manager,
                roi_frame_provider=self.worker.get_latest_preview_frame,
                config_committer=self._commit_candidate_config,
            )
            self._config_page.back_clicked.connect(self._on_config_back)
            self._config_page.stats_changed.connect(self._on_stats_changed)
            self._config_page.camera_refresh_requested.connect(
                self._on_camera_refresh_requested
            )
            self._stacked.addWidget(self._config_page)
        self._config_page.set_camera_devices(
            self._camera_devices,
            self._camera_enumeration_error,
        )
        self._config_page.refresh_stats_display()
        self._sync_template_diagnostics_status()
        self._stacked.setCurrentWidget(self._config_page)
        self._status_label.setText("配置页")
        self._status_label.setStyleSheet("color: #9C27B0; font-weight: bold;")

    def _refresh_camera_devices(self) -> None:
        """Refresh the in-memory MvSDK device catalog without changing cameras."""
        try:
            self._camera_devices = list(self.worker.enumerate_devices())
            self._camera_enumeration_error = ""
        except Exception as exc:
            self._camera_devices = []
            self._camera_enumeration_error = str(exc)

    @Slot()
    def _on_camera_refresh_requested(self) -> None:
        self._refresh_camera_devices()
        if self._config_page is not None:
            self._config_page.set_camera_devices(
                self._camera_devices,
                self._camera_enumeration_error,
            )

    def _on_config_back(self) -> None:
        """从配置页返回主界面并刷新对应模式引擎。"""
        self._stacked.setCurrentWidget(self._main_page)
        self._apply_feedback_config()
        self._apply_step_engine_config()
        self._apply_roi_engine_config()
        self._invalidate_render_cache()
        self._sync_button_states()
        self._update_kpi()
        self._status_label.setText("READY" if not self.state.camera_on else "预览中")
        self._status_label.setStyleSheet(_status_badge_style(TEXT_ACCENT))

    def flush_production_stats(self) -> None:
        """立即写入还在等待合并的计数（关窗时调用）。"""
        self._stats_store.flush()

    def _persist_production_stats(self) -> None:
        self._stats_store.persist()

    def _on_clear_production_stats(self) -> None:
        reply = QMessageBox.question(
            self,
            "清零确认",
            "系统会先保存当前统计记录，再清零并开始新批次。\n确定要清零吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        result = self._stats_manager.reset_and_archive()
        if not result.success:
            QMessageBox.warning(self, "清零失败", f"保存记录失败，统计未清零：\n{result.error}")
            return
        self._on_stats_changed()
        if result.archived:
            QMessageBox.information(
                self, "清零成功",
                f"已保存生产记录到：\n{result.record_path}\n\n统计已清零，新批次已开始。",
            )
        else:
            QMessageBox.information(self, "清零完成", "当前批次无计数，已刷新开始时间。")

    def _on_stats_changed(self) -> None:
        """归零后把 0 写入配置，并同步主界面和配置页。"""
        self._persist_production_stats()
        self._update_kpi()
        if self._config_page is not None:
            self._config_page.refresh_stats_display()

    def _on_config_saved(self) -> bool:
        """配置页保存后立即应用运行时配置。"""
        camera_switch_result = self._apply_camera_selection()
        self._apply_feedback_config()
        model_ok = self._apply_runtime_model_config()
        self._apply_step_engine_config()
        self._reload_configured_action_model()
        if (
            model_ok
            and self._uses_lstm_sequence()
            and self._resolved_bundle_dir(self.config) is not None
            and not self._action_session_matches_config()
        ):
            self._status_label.setText(self._action_block_reason or "动作模型未加载")
            self._status_label.setStyleSheet("color: #F44336; font-weight: bold;")
            return False
        self._resync_round_cooldown_timer()
        self._init_roi_engine()
        self._invalidate_render_cache()
        self._sync_hand_overlay_enabled()
        self._sync_button_states()
        self._update_kpi()
        if not model_ok:
            self._status_label.setText(self._last_model_reload_error or "模型加载失败")
            self._status_label.setStyleSheet("color: #F44336; font-weight: bold;")
            return False
        elif camera_switch_result is None and not self.state.inference_on:
            self._status_label.setText(self._config_applied_text())
            self._status_label.setStyleSheet(_status_badge_style(TEXT_ACCENT))
        self._restart_template_diagnostics_if_running()
        self._sync_template_diagnostics_status()
        self._clear_stale_roi_notice()
        return True

    def _run_model_load(self, load: Callable):
        """在后台线程加载模型，期间界面照常重绘；返回结果或原样抛出异常。

        调用方（配置页保存）依赖同步的成功/失败语义，所以这里等加载完再返回，
        只是等待时跑本地事件循环，不再让 ONNX/torch 加载冻住整个窗口。
        加载函数不能碰 Qt 控件，只做纯模型构造。
        """
        if self._model_loading:
            raise RuntimeError("模型正在加载，请稍候再保存")
        outcome: dict = {}

        def worker() -> None:
            try:
                outcome["value"] = load()
            except BaseException as exc:  # 原样交回界面线程处理
                outcome["error"] = exc

        self._model_loading = True
        previous_text = self._status_label.text()
        self._status_label.setText("模型加载中...")
        # 加载期间界面只重绘不响应操作：开始检测、清零、返回、再次保存都要等加载结束，
        # 否则嵌套事件循环里的操作会和正在提交的配置交错。
        was_enabled = self.centralWidget().isEnabled() if self.centralWidget() else True
        if self.centralWidget() is not None:
            self.centralWidget().setEnabled(False)
        loop = None
        try:
            thread = threading.Thread(target=worker, name="model-load", daemon=True)
            thread.start()
            # 界面线程自己的定时器轮询线程是否结束，不跨线程调 Qt 对象。
            loop = QEventLoop(self)
            poll = QTimer(loop)
            poll.setInterval(20)
            poll.timeout.connect(lambda: None if thread.is_alive() else loop.quit())
            poll.start()
            if thread.is_alive():
                loop.exec()
            poll.stop()
            thread.join()
        finally:
            self._model_loading = False
            if loop is not None:
                loop.deleteLater()
            if self.centralWidget() is not None:
                self.centralWidget().setEnabled(was_enabled)
            if self._status_label.text() == "模型加载中...":
                self._status_label.setText(previous_text)
            if self._close_after_model_load:
                self._close_after_model_load = False
                QTimer.singleShot(0, self.close)
        if "error" in outcome:
            raise outcome["error"]
        return outcome["value"]

    def _close_unused_processor(self, processor) -> None:
        if processor is None or processor is self.processor:
            return
        close = getattr(processor, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                self._runtime_logger.warning("frame_processor=close_failed", exc_info=True)

    def _commit_candidate_config(self, candidate) -> None:
        """候选实例全部就绪后提交；失败时旧配置、模型和会话均保留。"""
        from lstm_action.bundle import uses_lstm_sequence

        candidate.validate()

        def load_runtime():
            processor = self._create_processor_from_config(candidate)
            action_session, action_reason = None, ""
            if uses_lstm_sequence(candidate):
                action_session, action_reason = self._load_action_session(candidate, processor)
            return processor, action_session, action_reason

        new_processor, session, reason = self._run_model_load(load_runtime)
        try:
            raw_names = new_processor.get_class_names() if hasattr(new_processor, "get_class_names") else []
            class_names = list(raw_names.values()) if isinstance(raw_names, dict) else list(raw_names)
            candidate.require_valid_inspection_config(class_names, require_roi_rects=False)
            if uses_lstm_sequence(candidate) and session is None:
                raise ValueError(reason or "动作模型未能加载，保留原运行配置")
        except Exception:
            # 校验没过，新加载的模型不会被用上，立即释放会话。
            self._close_unused_processor(new_processor)
            raise
        new_step_engine = build_step_engine(candidate)
        path = Path(candidate.get_config_path())
        old_bytes = path.read_bytes() if path.is_file() else None
        was_running = bool(self.state.inference_on)
        self.worker.stop_inference()
        saved = False
        # 候选配置是点保存时拷贝的，加载模型期间可能又出了件，用最新计数覆盖。
        self._stats_store.apply_to(candidate)
        try:
            candidate.save(path)
            saved = True
            self.worker.replace_runtime(candidate, new_processor, session)
        except Exception:
            if saved:
                restore_file(path, old_bytes)
            if self.worker.frame_processor is not new_processor:
                self._close_unused_processor(new_processor)
            if was_running and self.state.camera_on:
                self.worker.start_inference()
            raise

        self.config = candidate
        self.processor = new_processor
        self.step_engine = new_step_engine
        self._round_pass_timer.stop()
        self._pending_step_engine_rebuild = False
        self._last_overlay_data = {}
        self._last_model_reload_error = ""
        self._action_block_reason = ""
        self._action_runtime_fault = ""
        self._runtime_model_signature = self._model_runtime_signature(candidate)
        self._step_engine_signature = self._step_engine_config_signature(candidate)
        self.lstm_action.state.loaded = session is not None
        self.lstm_action.state.reason = "已加载" if session is not None else "未启用"
        self._show_lstm_state(self.lstm_action.state)
        if self._config_page is not None:
            self._config_page.config = candidate
        self._apply_feedback_config()
        self._init_roi_engine()
        self._refresh_steps()
        self._invalidate_render_cache()
        self._clear_result_label()
        self._set_notice("")
        self._refresh_ng_alarm_for_new_round()
        self._sticky_ng_step_idx = -1
        if was_running and self.state.camera_on:
            self._begin_step_round()
            self._round_start_time = time.time()
            self.worker.start_inference()
        camera_switch_result = self._apply_camera_selection()
        self._sync_hand_overlay_enabled()
        self._sync_button_states()
        self._update_kpi()
        if camera_switch_result is not False:
            self._status_label.setText("检测中" if self.state.inference_on else self._config_applied_text())
        else:
            self._set_notice("配置和模型已保存，相机切换失败，请检查所选相机后重新打开")
        self._restart_template_diagnostics_if_running()
        self._sync_template_diagnostics_status()

    def _apply_camera_selection(self) -> Optional[bool]:
        """Apply a saved SN change only when a camera session is currently open."""
        active_device = self.worker.active_device
        if not self.state.camera_on or active_device is None:
            return None

        active_sn = str(active_device.sn or "").strip()
        target_sn = str(getattr(self.config, "mvsdk_camera_sn", "") or "").strip()
        if target_sn == active_sn:
            return None

        if self.state.inference_on:
            self.worker.stop_inference()
        self._close_camera_session()

        if self.worker.open_camera():
            self._display_timer.start(33)
            self._status_label.setText("预览中")
            self._status_label.setStyleSheet(_status_badge_style(TEXT_ACCENT))
            return True

        self._status_label.setText(f"相机切换失败（SN: {target_sn}）")
        self._status_label.setStyleSheet("color: #F44336; font-weight: bold;")
        return False

    def _close_camera_session(self, *, clear_visuals: bool = True) -> None:
        """Stop detection and camera threads before releasing the SDK session."""
        if self.state.inference_on:
            self.worker.stop_inference()
        self.worker.reset_action_session()
        self._display_timer.stop()
        self._round_pass_timer.stop()
        self._stop_template_diagnostics()
        if self._coordinate_logger.active:
            self._coordinate_logger.stop_session()
        self.worker.close_camera()
        self._sync_hand_overlay_enabled()
        if clear_visuals:
            self._clear_camera_session_visuals()

    def _clear_camera_session_visuals(self) -> None:
        """Remove all camera-specific frames, overlays, results, and trackers."""
        self._video_label.clear()
        self._latest_runtime_overlay = DetectionOverlayState()
        self._invalidate_render_cache()
        self._last_overlay_data = {}
        self._last_pipeline_stats = {}
        self._last_camera_params = {}
        self._last_counted_pass_round = -1
        self._refresh_ng_alarm_for_new_round()
        self._round_start_time = 0.0
        self._sticky_ng_step_idx = -1
        self._last_step_focus_key = None
        self._clear_result_label()
        self._set_notice("")
        self._rebuild_step_engine()
        self._init_roi_engine()
        self._refresh_steps()

    def _build_viewfinder_overlays(self, frame: QFrame) -> None:
        """取景器 HUD：四角 L 标（无辉光/模糊，只用描边）。"""
        self._viewfinder_corners = []
        for v_edge, h_edge in (("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")):
            mark = QFrame(frame)
            mark.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            v_side = "border-top" if v_edge == "top" else "border-bottom"
            h_side = "border-left" if h_edge == "left" else "border-right"
            mark.setStyleSheet(
                f"QFrame {{ background: transparent; border: none; "
                f"{v_side}: 2px solid {TEXT_ACCENT}; {h_side}: 2px solid {TEXT_ACCENT}; }}"
            )
            mark.setFixedSize(22, 22)
            mark.raise_()
            self._viewfinder_corners.append((mark, v_edge, h_edge))

        frame.installEventFilter(self)
        self._position_hud_overlays()

    def _position_hud_overlays(self) -> None:
        frame = getattr(self, "_video_frame", None)
        corners = getattr(self, "_viewfinder_corners", None)
        if frame is None or not corners:
            return
        inset = 12
        width = frame.width()
        height = frame.height()
        for mark, v_edge, h_edge in corners:
            x = inset if h_edge == "left" else max(0, width - inset - mark.width())
            y = inset if v_edge == "top" else max(0, height - inset - mark.height())
            mark.move(x, y)
            mark.raise_()

    def eventFilter(self, obj, event):
        if obj is getattr(self, "_video_frame", None) and event.type() == QEvent.Type.Resize:
            self._position_hud_overlays()
        return super().eventFilter(obj, event)

    def _sync_hand_overlay_enabled(self) -> None:
        """顺序模式且正在检测时才开 Hands；ROI / 停止检测立即关掉。"""
        enabled = bool(self.state.inference_on) and not bool(
            getattr(self.config, "roi_inspection_enabled", False)
        )
        self.worker.set_hand_overlay_enabled(enabled)

    @staticmethod
    def _normalize_hand_landmarks(raw) -> list:
        if not raw:
            return []
        hands = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            points = item.get("points") or []
            if len(points) != 21:
                continue
            cleaned = []
            try:
                for point in points:
                    cleaned.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError, IndexError):
                continue
            if len(cleaned) != 21:
                continue
            hands.append(
                {
                    "handedness": str(item.get("handedness") or ""),
                    "points": cleaned,
                }
            )
        return hands

    def _apply_feedback_config(self) -> None:
        """同步保存后可立即生效的反馈配置。"""
        self._sound_feedback.enabled = (
            bool(getattr(self.config, "pass_sound_enabled", False))
            or bool(getattr(self.config, "fail_sound_enabled", True))
        )
        self._fail_evidence_saver.enabled = bool(getattr(self.config, "fail_evidence_enabled", True))

    def _step_engine_config_signature(self, config) -> tuple:
        """Configuration fields that require resetting the normal step engine."""
        names = tuple(str(name).strip() for name in (getattr(config, "category_names", []) or []))
        counts = tuple(config.get_runtime_category_counts())
        return (
            names,
            counts,
            int(getattr(config, "action_pass_stable_frames", 1)),
            int(getattr(config, "action_ng_stable_frames", 1)),
            int(getattr(config, "action_leave_stable_frames", 4)),
            bool(getattr(config, "action_order_constraint_enabled", True)),
            bool(getattr(config, "action_fail_retry_enabled", False)),
            self._action_bundle_signature(config),
        )

    @staticmethod
    def _roi_engine_config_signature(config) -> tuple:
        """影响 ROI 轮次运行态的配置字段。"""
        enabled = bool(getattr(config, "roi_inspection_enabled", False))
        if not enabled:
            return (False,)
        names = list(getattr(config, "category_names", []) or [])
        counts = config.get_runtime_category_counts()
        required_counts = []
        for index, name in enumerate(names):
            normalized = str(name).strip()
            if normalized:
                count = counts[index] if index < len(counts) else 1
                required_counts.append((normalized, max(1, int(count))))
        inspection_rect = getattr(config, "roi_inspection_rect", None) or {}
        template_rect = getattr(config, "roi_template_rect", None) or {}
        search_rect = getattr(config, "roi_template_search_rect", None) or {}
        return (
            True,
            bool(getattr(config, "roi_template_enabled", False)),
            str(getattr(config, "roi_template_path", "")).strip(),
            float(getattr(config, "roi_template_match_threshold", 0.6)),
            float(getattr(config, "roi_template_angle_range", 15.0)),
            float(getattr(config, "roi_template_angle_step", 1.0)),
            tuple(sorted(template_rect.items())),
            tuple(sorted(search_rect.items())),
            tuple(sorted(inspection_rect.items())),
            tuple(required_counts),
            int(getattr(config, "action_pass_stable_frames", 1)),
            int(getattr(config, "action_ng_stable_frames", 1)),
            float(getattr(config, "round_cooldown_seconds", 0.0)),
        )

    def _apply_step_engine_config(self) -> bool:
        new_signature = self._step_engine_config_signature(self.config)
        if new_signature == self._step_engine_signature:
            return False
        if self._round_pass_timer.isActive() and not self._uses_lstm_sequence():
            self._pending_step_engine_rebuild = True
            return False
        self._rebuild_step_engine()
        self._refresh_steps()
        if self.state.inference_on and self.step_engine:
            self._begin_step_round()
        return True

    def _apply_roi_engine_config(self) -> bool:
        new_signature = self._roi_engine_config_signature(self.config)
        if new_signature == self._roi_engine_signature:
            return False
        self._init_roi_engine()
        self._refresh_steps()
        return True

    def _model_runtime_signature(self, config) -> tuple:
        """影响最终运行处理器实例的模型和模板配置签名。"""
        return (
            str(getattr(config, "yolo_model_path", "")),
            float(getattr(config, "yolo_conf_threshold", 0.0)),
            float(getattr(config, "yolo_iou_threshold", 0.0)),
            bool(getattr(config, "ultralytics_half", False)),
            str(getattr(config, "ultralytics_tracker", "")),
            bool(getattr(config, "ultralytics_track_persist", True)),
            bool(getattr(config, "roi_template_enabled", False)),
            str(getattr(config, "roi_template_path", "") or "").strip(),
            float(getattr(config, "roi_template_match_threshold", 0.6)),
            float(getattr(config, "roi_template_angle_range", 15.0)),
            float(getattr(config, "roi_template_angle_step", 1.0)),
            tuple(sorted((getattr(config, "roi_template_search_rect", None) or {}).items())),
            template_file_signature(config, self._app_base_dir),
            self._action_bundle_signature(config),
        )

    def _action_bundle_signature(self, config) -> tuple:
        path = Path(str(getattr(config, "lstm_bundle_path", "") or "").strip())
        if not path.is_absolute():
            path = self._app_base_dir / path
        files = []
        for name in ("best_lstm.pt", "contract.json", "runtime_manifest.json"):
            target = path / name
            try:
                digest = hashlib.sha256()
                with target.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                files.append((name, digest.hexdigest()))
            except OSError:
                files.append((name, None))
        return (
            bool(getattr(config, "action_order_constraint_enabled", False)),
            bool(getattr(config, "roi_inspection_enabled", False)),
            str(path.resolve()), tuple(files),
        )

    def _create_processor_from_config(self, config=None):
        """创建新处理器。测试可注入 factory，生产路径复用 main.create_yolo_processor。"""
        if self._processor_factory is not None:
            return self._processor_factory(config or self.config, app_base_dir=self._app_base_dir)
        from main import create_yolo_processor

        return create_yolo_processor(config or self.config, app_base_dir=self._app_base_dir)

    def _apply_runtime_model_config(self) -> bool:
        """模型相关配置变化时重载处理器，失败时保留旧处理器。"""
        new_signature = self._model_runtime_signature(self.config)
        needs_reload = new_signature != self._runtime_model_signature or self.processor is None
        if not needs_reload:
            return True

        was_inference_on = bool(self.state.inference_on)
        if was_inference_on:
            self.worker.stop_inference()

        self._status_label.setText("模型加载中...")
        self._status_label.setStyleSheet(_status_badge_style(TEXT_SECONDARY))

        try:
            new_processor = (
                self._run_model_load(self._create_processor_from_config)
                if self.processor is None or new_signature[:-1] != self._runtime_model_signature[:-1]
                else self.processor
            )
        except Exception as exc:
            self._runtime_logger.exception(
                "processor_reload=failed configured_model=%s",
                getattr(self.config, "yolo_model_path", ""),
            )
            self._last_model_reload_error = f"模型加载失败: {exc}"
            if was_inference_on and self.processor is not None and self.state.camera_on:
                self.worker.start_inference()
            self._status_label.setText(self._last_model_reload_error)
            self._status_label.setStyleSheet("color: #F44336; font-weight: bold;")
            self._sync_button_states()
            return False

        self.processor = new_processor
        self.worker.set_frame_processor(new_processor)
        self._runtime_model_signature = new_signature
        self._last_model_reload_error = ""
        self._last_overlay_data = {}
        self._runtime_logger.info(
            "processor_reload=ok type=%s configured_model=%s",
            type(new_processor).__name__,
            getattr(self.config, "yolo_model_path", ""),
        )
        if self._notice_label.text().startswith("处理器未加载"):
            self._set_notice("")

        if was_inference_on and self.state.camera_on:
            self.worker.start_inference()
        # 按真实状态显示，不按加载前记下的状态猜。
        self._status_label.setText("检测中" if self.state.inference_on else "模型已加载")
        self._status_label.setStyleSheet(_status_badge_style(TEXT_ACCENT))
        return True

    def _rebuild_step_engine(self) -> None:
        """根据最新配置重建步骤引擎，保证 UI 卡片和判定顺序一致。"""
        self._step_engine_signature = self._step_engine_config_signature(self.config)
        self._configure_lstm_action()
        if self.step_engine is None:
            return
        self.step_engine = build_step_engine(self.config)
        self._sticky_ng_step_idx = -1
        self._last_step_focus_key = None

    def _uses_lstm_sequence(self) -> bool:
        from lstm_action.bundle import uses_lstm_sequence

        return uses_lstm_sequence(self.config)

    def _resolved_bundle_dir(self, config) -> Path | None:
        raw = str(getattr(config, "lstm_bundle_path", "") or "").strip()
        if not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = self._app_base_dir / path
        try:
            return path.resolve()
        except OSError:
            return path

    def _loaded_action_folder_name(self) -> str:
        session = self.worker.get_action_session() if hasattr(self, "worker") else None
        directory = getattr(getattr(session, "bundle", None), "directory", None)
        if not directory:
            return ""
        return Path(directory).name

    def _config_applied_text(self) -> str:
        folder = self._loaded_action_folder_name()
        if self._uses_lstm_sequence() and folder:
            return f"配置已应用，动作模型已重新加载：{folder}"
        return "配置已应用"

    def _action_session_matches_config(self) -> bool:
        session = self.worker.get_action_session() if hasattr(self, "worker") else None
        expected = self._resolved_bundle_dir(self.config)
        if expected is None:
            return session is None
        directory = getattr(getattr(session, "bundle", None), "directory", None)
        if directory is None:
            return session is not None
        try:
            return Path(directory).resolve() == expected
        except OSError:
            return Path(directory) == expected

    def _reload_configured_action_model(self) -> None:
        """保存后的动作模型如果还不是当前会话，立刻换成这份。"""
        if not self._uses_lstm_sequence():
            return
        if not self._action_session_matches_config():
            self._configure_lstm_action()
        if hasattr(self, "lstm_action"):
            self._show_lstm_state(self.lstm_action.state)

    def _configure_lstm_action(self) -> None:
        if not hasattr(self, "lstm_action"):
            return
        self._action_block_reason = ""
        self._action_runtime_fault = ""
        session = None
        if self._uses_lstm_sequence():
            session, reason = self._load_action_session()
            self._action_block_reason = reason
            state = self.lstm_action.state
            state.loaded = session is not None
            state.reason = "已加载" if session is not None else (reason or "未加载")
            self._show_lstm_state(state)
        else:
            state = self.lstm_action.configure(
                enabled=False,
                contract_path=str(getattr(self.config, "lstm_contract_path", "") or ""),
                category_names=list(getattr(self.config, "category_names", []) or []),
                yolo_model_path=str(getattr(self.config, "yolo_model_path", "") or ""),
            )
            self._show_lstm_state(state)
        if hasattr(self, "worker"):
            self.worker.set_action_session(session)

    def _load_action_session(self, config=None, processor=None):
        from lstm_action.bundle import load_runtime_bundle
        from lstm_action.stream import TimelineSession

        config = config or self.config
        processor = processor or self.processor
        path = str(getattr(config, "lstm_bundle_path", "") or "").strip()
        if not path:
            return None, "顺序模式需要动作模型包，不能按 YOLO 放行"
        try:
            bundle_path = Path(path)
            if not bundle_path.is_absolute():
                bundle_path = self._app_base_dir / bundle_path
            actual_processor = getattr(processor, "_inner", processor)
            actual_yolo_path = getattr(actual_processor, "_model_path", None) or getattr(actual_processor, "model_path", None)
            yolo_path = Path(actual_yolo_path or str(config.yolo_model_path))
            if not yolo_path.is_absolute():
                yolo_path = self._app_base_dir / yolo_path
            class_names = processor.get_class_names() if processor is not None and hasattr(processor, "get_class_names") else []
            class_names = list(class_names.values()) if isinstance(class_names, dict) else list(class_names)
            bundle = load_runtime_bundle(
                bundle_path,
                category_names=list(config.category_names or []),
                yolo_path=str(yolo_path),
                yolo_conf=float(config.yolo_conf_threshold),
                yolo_class_names=class_names or None,
                step_counts=list(config.get_runtime_category_counts()),
            )
        except ValueError as exc:
            return None, str(exc)
        return TimelineSession(
            bundle,
            confirm_seconds=float(getattr(config, "action_switch_confirm_seconds", 0.5)),
        ), ""

    def _show_lstm_state(self, state) -> None:
        if not hasattr(self, "_lstm_status_label"):
            return
        if not state.loaded:
            self._lstm_status_label.setText(f"时序动作：{state.reason}")
            return
        if self._uses_lstm_sequence():
            folder = self._loaded_action_folder_name()
            if folder:
                self._lstm_status_label.setText(f"时序动作：已加载 {folder}")
            else:
                self._lstm_status_label.setText("时序动作：连续模型已加载")
            return
        passed = "、".join(state.passed_categories) if state.passed_categories else "无"
        self._lstm_status_label.setText(f"时序动作：{state.reason}，已 Pass {passed}")

    def _show_action_progress(self, report: dict) -> None:
        if not hasattr(self, "_lstm_status_label"):
            return
        phase = str(report.get("phase") or "")
        action = str(report.get("action") or "")
        seen = "、".join(report.get("seen_parts") or []) or "无"
        missing = "、".join(report.get("missing_parts") or []) or "无"
        if phase:
            self._lstm_status_label.setText(f"时序动作：{phase} {action}，已见 {seen}，缺少 {missing}".strip())

    def _update_lstm_action(self, data: dict, detections, image_size) -> None:
        if not bool(getattr(self.config, "lstm_action_enabled", False)):
            return
        if not image_size:
            return
        width, height = int(image_size[0]), int(image_size[1])
        rows = [{"label": d.label, "conf": d.conf, "box": d.box} for d in detections]
        hands = []
        for item in data.get("hand_landmarks") or []:
            if isinstance(item, dict):
                hands.append(item.get("points") or [])
        state = self.lstm_action.update(
            rows,
            hands,
            width,
            height,
            float(self.config.yolo_conf_threshold),
        )
        self._show_lstm_state(state)

    @Slot()
    def _on_close_clicked(self) -> None:
        self.close()

    def closeEvent(self, event) -> None:
        if self._model_loading:
            # 提交还在进行，此时拆相机会和提交交错；等加载结束再关。
            self._close_after_model_load = True
            event.ignore()
            return
        self._close_camera_session(clear_visuals=False)
        self.flush_production_stats()
        event.accept()
