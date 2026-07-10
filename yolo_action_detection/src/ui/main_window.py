"""YOLO OBB 动作检测主界面（科技风双栏布局）。

左侧实时检测主窗口，右侧业务面板：KPI 行 + 大尺寸步骤检测区 + 核心操作区。
步骤卡片按配置有效步骤动态生成，NG 锁步可视化，按钮状态与真实运行态同步。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QStackedWidget, QFrame, QScrollArea, QSizePolicy,
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
from detection_logging.fail_evidence import FailEvidenceContext, FailEvidenceSaver
from detection_logging.production_stats import ProductionStatsManager
from pcb_inspection.models import PcbInspectionConfig, PcbResult
from pcb_inspection.engine import MultiPcbInspectionEngine
from step_sequence.step_sequence_engine import StepSequenceEngine, RoundResult, StepStatus
from ui.runtime_ui_tokens import (
    FONT_TITLE,
    PAGE_BG,
    PAGE_MARGIN,
    PANEL_BG,
    PANEL_BG_DARK,
    STROKE_MAIN,
    TEXT_ACCENT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    VIEWPORT_BG,
    frame_style,
    scroll_area_style,
    text_style,
)
from ui.widgets.native_panels import (
    KpiRow,
    StepCard,
    main_button_style,
)
from yolo_runtime.yolo_result_models import DetectionOverlayState, ObbDetection


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
    ) -> None:
        super().__init__()
        self.config = config
        self.state = state
        self.processor = processor
        self.step_engine = step_engine
        self._processor_factory = processor_factory
        self._app_base_dir = (
            Path(app_base_dir)
            if app_base_dir is not None
            else Path(__file__).resolve().parents[2]
        )
        self._runtime_model_signature = self._model_runtime_signature(config)
        self._last_model_reload_error = ""

        self.setWindowTitle("双翼科技视觉行为引导系统")
        self.setMinimumSize(1200, 760)
        self.setStyleSheet(f"QMainWindow {{ background-color: {PAGE_BG}; }}")

        self.worker = CameraWorker(config, state)
        self._display_timer = QTimer(self)
        self._display_timer.timeout.connect(self._update_display)

        # 生产统计（延后到 evidence_dir 之后初始化）
        self._stats_manager: Optional[ProductionStatsManager] = None
        # 计数去重键：同一轮同一结果只计一次
        self._last_counted_pass_round = -1
        self._last_counted_ng_key: Optional[tuple] = None
        self._round_start_time = 0.0
        # PCB 多板检查引擎（按需创建）
        self._pcb_engine: Optional[MultiPcbInspectionEngine] = None
        self._init_pcb_engine()
        # PASS 文案展示定时器；只负责清理 UI 文案，不阻塞下一轮检测。
        self._round_pass_timer = QTimer(self)
        self._round_pass_timer.setSingleShot(True)
        self._round_pass_timer.timeout.connect(self._on_round_pass_settled)

        # 步骤卡片缓存：index -> StepCard
        self._step_cards: List[StepCard] = []
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
            enabled=getattr(config, "sound_feedback_enabled", True),
            resource_dir=resource_dir,
        )
        self._fail_evidence_saver = fail_evidence_saver or FailEvidenceSaver(
            enabled=getattr(config, "fail_evidence_enabled", True),
            base_dir=evidence_dir,
        )
        self._stats_manager = ProductionStatsManager(
            records_dir=evidence_dir.parent / "production_records",
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
        self._sync_button_states()

    def _init_pcb_engine(self) -> None:
        """根据配置创建或销毁 PCB 检查引擎。"""
        if getattr(self.config, "pcb_inspection_enabled", False):
            pcb_config = PcbInspectionConfig.from_config(self.config)
            self._pcb_engine = MultiPcbInspectionEngine(pcb_config)
        else:
            self._pcb_engine = None

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN)
        outer.setSpacing(12)

        self._stacked = QStackedWidget()
        outer.addWidget(self._stacked)

        # 主页面
        self._main_page = QWidget()
        self._main_page.setStyleSheet("background: transparent;")
        main_layout = QVBoxLayout(self._main_page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(12)

        # 顶部标题栏
        header = QFrame()
        header.setStyleSheet(frame_style(PANEL_BG_DARK, border=STROKE_MAIN, radius=10))
        header.setFixedHeight(52)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 6, 16, 6)
        self._title_label = QLabel("双翼科技视觉行为引导系统")
        self._title_label.setFont(QFont("Microsoft YaHei UI", FONT_TITLE, QFont.Bold))
        self._title_label.setStyleSheet(text_style(TEXT_PRIMARY, size=FONT_TITLE, weight=800))
        self._status_label = QLabel("READY")
        self._status_label.setStyleSheet(text_style(TEXT_ACCENT, size=14, weight=700))
        self._result_label = QLabel("")
        self._result_label.setFont(QFont("Microsoft YaHei UI", 14, QFont.Bold))
        self._result_label.setMinimumWidth(80)
        self._result_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._result_label.setVisible(False)
        self._btn_config = QPushButton("配置")
        self._btn_config.setFixedHeight(34)
        self._btn_config.setMinimumWidth(86)
        self._btn_config.setStyleSheet(
            f"QPushButton {{ background-color: #19345F; border: 1px solid {STROKE_MAIN}; "
            f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: 700; padding: 4px 12px; "
            f"border-radius: 8px; }}"
            f"QPushButton:hover {{ background-color: #21416E; border: 1px solid #35D7FF; }}"
        )
        self._btn_close = QPushButton("关闭")
        self._btn_close.setFixedHeight(34)
        self._btn_close.setMinimumWidth(86)
        self._btn_close.setStyleSheet(
            "QPushButton { background-color: #8E2230; border: 1px solid #FF6A75; "
            "color: #FFFFFF; font-size: 14px; font-weight: 700; padding: 4px 12px; "
            "border-radius: 8px; }"
            "QPushButton:hover { background-color: #A82838; border: 1px solid #FF8A92; }"
        )
        header_layout.addWidget(self._title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self._status_label)
        header_layout.addWidget(self._result_label)
        header_layout.addWidget(self._btn_config)
        header_layout.addWidget(self._btn_close)
        main_layout.addWidget(header)

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
        mid_layout.addWidget(video_frame, 7)

        # 右侧业务面板
        right_panel = QFrame()
        right_panel.setStyleSheet(frame_style(PANEL_BG, border=STROKE_MAIN, radius=12))
        right_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(12)

        # 右上：KPI 单行
        self._kpi_row = KpiRow()
        right_layout.addWidget(self._kpi_row)

        # 右中：步骤检测区（滚动）
        steps_title = QLabel("步骤检测")
        steps_title.setStyleSheet(text_style(TEXT_ACCENT, size=15, weight=700))
        right_layout.addWidget(steps_title)

        self._steps_scroll = QScrollArea()
        self._steps_scroll.setWidgetResizable(True)
        self._steps_scroll.setStyleSheet(scroll_area_style())
        self._steps_container = QWidget()
        self._steps_container.setStyleSheet("background: transparent;")
        self._steps_layout = QVBoxLayout(self._steps_container)
        self._steps_layout.setContentsMargins(0, 0, 4, 0)
        self._steps_layout.setSpacing(10)
        self._steps_layout.addStretch(1)
        self._steps_scroll.setWidget(self._steps_container)
        right_layout.addWidget(self._steps_scroll, 1)

        # 右下：核心操作区
        ctrl_layout = QGridLayout()
        ctrl_layout.setContentsMargins(0, 4, 0, 0)
        ctrl_layout.setHorizontalSpacing(10)
        ctrl_layout.setVerticalSpacing(8)

        self._btn_camera = QPushButton("打开相机")
        self._btn_camera.setStyleSheet(main_button_style("primary"))
        self._btn_detect = QPushButton("开始检测")
        self._btn_detect.setStyleSheet(main_button_style("primary"))

        ctrl_layout.addWidget(self._btn_camera, 0, 0)
        ctrl_layout.addWidget(self._btn_detect, 0, 1)
        right_layout.addLayout(ctrl_layout)

        mid_layout.addWidget(right_panel, 4)
        main_layout.addLayout(mid_layout, 1)

        self._stacked.addWidget(self._main_page)

    def _connect_signals(self) -> None:
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.overlay_state_updated.connect(self._on_overlay_updated)
        self.worker.pipeline_stats_updated.connect(self._on_stats_updated)
        self.worker.camera_params_updated.connect(self._on_camera_params)

        self._btn_camera.clicked.connect(self._on_toggle_camera)
        self._btn_detect.clicked.connect(self._on_toggle_detect)
        self._btn_config.clicked.connect(self._on_config_clicked)
        self._btn_close.clicked.connect(self._on_close_clicked)

    # ------------------------------------------------------------------
    # 动态步骤卡片
    # ------------------------------------------------------------------
    def _effective_step_names(self) -> List[str]:
        """配置中的有效（非空）步骤名称，保留顺序。"""
        names = getattr(self.config, "category_names", []) or []
        return [n.strip() for n in names if n and n.strip()]

    def _refresh_steps(self) -> None:
        """根据配置有效步骤重建步骤卡片。"""
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
        for i, name in enumerate(names):
            card = StepCard(i + 1, name)
            self._step_cards.append(card)
            # 插在 stretch 之前
            self._steps_layout.insertWidget(self._steps_layout.count() - 1, card)

        # 无有效步骤时给出提示
        if not self._step_cards:
            hint = QLabel("未配置有效步骤，请进入配置页设置步骤类别")
            hint.setStyleSheet(text_style(TEXT_SECONDARY, size=13, weight=500))
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._steps_layout.insertWidget(self._steps_layout.count() - 1, hint)
            self._steps_hint = hint

        self._update_step_display(self.step_engine.get_state() if self.step_engine else None)
        self._last_step_focus_key = None

    # ------------------------------------------------------------------
    # 显示与状态同步
    # ------------------------------------------------------------------
    def _update_display(self) -> None:
        # 成熟 YOLO 实时项目的关键策略是“追最新帧、丢旧帧”：
        # 主画面底图始终来自最新预览帧，推理结果只作为有时效的覆盖层。
        frame = self.worker.get_latest_preview_frame()
        if frame is None:
            return
        overlay = self.worker.get_latest_overlay()
        if self._overlay_is_fresh(overlay):
            frame = self._draw_overlay(frame, overlay)
        self._show_frame(frame)

    def _overlay_is_fresh(self, overlay) -> bool:
        """判断 overlay 是否还能叠加到最新预览帧上。"""
        if not overlay or not getattr(overlay, "detections", None):
            return False
        timestamp = float(getattr(overlay, "timestamp", 0.0) or 0.0)
        if timestamp <= 0:
            return True
        max_age_ms = float(getattr(self.config, "display_result_max_age_ms", 250) or 0)
        if max_age_ms <= 0:
            return True
        return (time.time() - timestamp) * 1000 <= max_age_ms

    def _draw_overlay(self, frame: np.ndarray, overlay) -> np.ndarray:
        if not overlay or not overlay.detections:
            return frame
        show_conf = self.config.show_confidence_overlay
        show_tid = self.config.show_track_id_overlay
        for det in overlay.detections:
            if len(det.polygon) < 4:
                continue
            pts = np.array(det.polygon, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 128), 2)
            label_parts = [det.label]
            if show_conf:
                label_parts.append(f"{det.conf:.2f}")
            if show_tid and det.track_id is not None:
                label_parts.append(f"#{det.track_id}")
            label = " ".join(label_parts)
            tx, ty = int(det.polygon[0][0]), int(det.polygon[0][1]) - 5
            cv2.putText(frame, label, (tx, max(15, ty)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1)

        frame = self._draw_tip_and_holes(frame, overlay)

        # PCB 模式：在画面上绘制 PCB ID、元器件归属标注和连续 FAIL 帧
        if self._pcb_engine is not None:
            frame = self._draw_pcb_overlay(frame, overlay)

        return frame

    def _draw_pcb_overlay(self, frame: np.ndarray, overlay) -> np.ndarray:
        """在画面上绘制 PCB ID、已识别元器件、连续 FAIL 帧和最终结果。"""
        if not self._pcb_engine:
            return frame
        pcb_class = getattr(self.config, "pcb_class_name", "pcb")
        comp_names = [n for n in getattr(self.config, "pcb_component_class_names", []) if n]

        # 找到画面中的 PCB 检测
        for det in overlay.detections:
            if det.label != pcb_class or det.track_id is None:
                continue
            tid = det.track_id
            state = self._pcb_engine.pcb_states.get(tid)
            if state is None:
                continue

            # 在 PCB OBB 中心上方绘制 PCB ID 和状态
            cx, cy = int(det.center[0]), int(det.center[1])
            label_y = cy - 30

            # 状态颜色
            if state.result.value == "pass":
                color = (0, 200, 0)   # 绿色
            elif state.result.value == "fail":
                color = (0, 0, 255)   # 红色
            else:
                color = (255, 200, 0) # 橙色

            # PCB ID
            cv2.putText(frame, f"PCB #{tid}", (cx - 40, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 结果标签
            if state.result.value == "pass":
                cv2.putText(frame, "PASS", (cx - 25, label_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            elif state.result.value == "fail":
                missing = "/".join(state.missing_classes[:3])
                cv2.putText(frame, f"NG 缺:{missing}", (cx - 60, label_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                # 显示连续 FAIL 帧数和已识别槽位
                identified = sum(
                    1 for s in state.last_slot_states.values() if s.present
                )
                total = len(comp_names)
                cv2.putText(frame, f"{identified}/{total} FAIL:{state.consecutive_fail}",
                            (cx - 50, label_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 在每个已识别元器件旁标注其所属 PCB ID
            for comp_name in comp_names:
                for cd in overlay.detections:
                    if cd.label != comp_name:
                        continue
                    # 简单标注：在元器件旁边写上 PCB ID
                    comp_cx = int(cd.center[0])
                    comp_cy = int(cd.center[1])
                    cv2.putText(frame, f"#{tid}", (comp_cx + 15, comp_cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return frame

    def _draw_tip_and_holes(self, frame: np.ndarray, overlay) -> np.ndarray:
        """绘制枪头点、洞口中心、内外圈和距离。"""
        from calibration.models import CalibrationTransform, HoleDefinition
        cal_data = getattr(self.config, "calibration_points", [])
        holes_data = getattr(self.config, "holes", [])
        tool_class = self.config.tool_class_name

        if not cal_data or not holes_data:
            return frame

        calibrator = CalibrationTransform.from_dict({"points": cal_data})
        if not calibrator.is_valid:
            return frame

        holes = [HoleDefinition.from_dict(h) for h in holes_data]

        obb_dets = []
        for d in overlay.detections:
            obb_dets.append(ObbDetection(
                class_id=d.class_id, label=d.label, conf=d.conf,
                track_id=d.track_id, polygon=d.polygon, box=d.box, center=d.center,
            ))
        tip, _ = extract_tip_point(obb_dets, tool_class, calibrator, overlay.source_frame_id)

        for hole in holes:
            if not hole.enabled:
                continue
            cx, cy = int(hole.center_px[0]), int(hole.center_px[1])
            inner_r = self._mm_to_pixel_radius(hole.inner_radius_mm, calibrator)
            cv2.circle(frame, (cx, cy), max(1, inner_r), (0, 255, 0), 2)
            outer_r = self._mm_to_pixel_radius(hole.outer_radius_mm, calibrator)
            cv2.circle(frame, (cx, cy), max(1, outer_r), (0, 255, 255), 1)
            cv2.putText(frame, hole.name, (cx - 10, cy - max(1, outer_r) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

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
                            cv2.putText(frame, f"{dist:.1f}mm",
                                        (tx + 10, ty - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                            cv2.line(frame, (tx, ty),
                                     (int(hole.center_px[0]), int(hole.center_px[1])),
                                     (0, 255, 255), 1)
                            break

        return frame

    def _mm_to_pixel_radius(self, radius_mm: float, calibrator) -> int:
        if not calibrator.is_valid:
            return int(radius_mm)
        origin_px = calibrator.mm_to_pixel(0.0, 0.0)
        edge_px = calibrator.mm_to_pixel(radius_mm, 0.0)
        return int(((edge_px[0] - origin_px[0]) ** 2 + (edge_px[1] - origin_px[1]) ** 2) ** 0.5)

    def _show_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self._video_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._video_label.setPixmap(scaled)

    # ------------------------------------------------------------------
    # 步骤状态 → 卡片状态映射
    # ------------------------------------------------------------------
    def _update_step_display(self, state) -> None:
        """根据步骤引擎状态更新步骤卡片与 KPI。"""
        if not self._step_cards:
            return

        terminal_ng_step_idx = -1
        if state is not None and state.round_result == RoundResult.ACTION_NG:
            terminal_ng_step_idx = state.action_ng_step
        has_unresolved_ng = terminal_ng_step_idx >= 0
        ng_step_idx = terminal_ng_step_idx

        current_idx = state.current_step_index if state else -1

        for card_i, card in enumerate(self._step_cards):
            # card_i 对应有效步骤序号；引擎 steps 可能比卡片多（含空步骤），
            # 需要把卡片序号映射到引擎 steps 中的对应索引。
            engine_idx = self._card_to_engine_index(card_i)
            step_state = StepStatus.WAITING
            if state is not None and 0 <= engine_idx < len(state.steps):
                step_state = state.steps[engine_idx].status

            # 状态映射
            if step_state == StepStatus.PASS:
                card.set_step_state("pass")
            elif engine_idx == ng_step_idx:
                card.set_step_state("ng")
            elif has_unresolved_ng and engine_idx > ng_step_idx:
                # NG 之后的步骤锁定
                card.set_step_state("locked")
            elif engine_idx == current_idx:
                card.set_step_state("active")
            elif engine_idx < current_idx:
                # 已经过的步骤应已 PASS（防御性）
                card.set_step_state("pass")
            else:
                card.set_step_state("waiting")

            # 显示数量进度
            if state is not None and 0 <= engine_idx < len(state.steps):
                ss = state.steps[engine_idx]
                if ss.required_count > 1 and step_state == StepStatus.WAITING:
                    card.set_quantity_progress(ss.current_count, ss.required_count)

        # KPI 与结果标签
        if state is not None:
            if state.round_result == RoundResult.PASS:
                if state.round_id != self._last_counted_pass_round:
                    self._last_counted_pass_round = state.round_id
                    self._stats_manager.record_pass()
                    self._play_pass_feedback()
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
                self._update_kpi()

        self._sync_step_focus_scroll(state)

    def _start_next_round_after_pass(self) -> None:
        if not self.step_engine:
            return
        self.step_engine.start_round()
        self._round_start_time = time.time()
        self._last_counted_ng_key = None
        self._sticky_ng_step_idx = -1
        self._last_step_focus_key = None
        self._round_pass_timer.start(2000)
        state = self.step_engine.get_state()
        self._update_step_display(state)
        self._sync_step_focus_scroll(state, reason="pass-next-round", force=True)

    def _start_next_round_after_ng(self) -> None:
        if not self.step_engine:
            return
        self._round_pass_timer.stop()
        self.step_engine.start_round(require_first_step_rearm=True)
        self._round_start_time = time.time()
        self._last_counted_pass_round = -1
        self._last_counted_ng_key = None
        self._sticky_ng_step_idx = -1
        self._last_step_focus_key = None
        state = self.step_engine.get_state()
        self._update_step_display(state)
        self._sync_step_focus_scroll(state, reason="ng-next-round", force=True)

    def _play_pass_feedback(self) -> None:
        if not getattr(self.config, "sound_feedback_enabled", True):
            return
        try:
            self._sound_feedback.play_pass()
        except Exception:
            pass

    def _handle_action_ng_feedback(self, state) -> None:
        if getattr(self.config, "sound_feedback_enabled", True):
            try:
                self._sound_feedback.play_fail()
            except Exception:
                pass
        if getattr(self.config, "fail_evidence_enabled", True):
            self._save_fail_evidence(state)

    def _save_fail_evidence(self, state) -> None:
        try:
            frame, frame_id = self._get_feedback_frame()
            data = self._last_overlay_data or {}
            source_frame_id = int(data.get("source_frame_id") or frame_id or 0)
            step_idx = int(getattr(state, "action_ng_step", -1))
            names = getattr(self.config, "category_names", []) or []
            step_name = names[step_idx] if 0 <= step_idx < len(names) else ""
            context = FailEvidenceContext(
                round_id=int(getattr(state, "round_id", 0)),
                action_ng_step=step_idx,
                step_name=step_name,
                source_frame_id=source_frame_id,
                detections=list(data.get("detections", []) or []),
                model_path=str(data.get("model_path", "")),
                timestamp=float(data.get("timestamp") or time.time()),
            )
            self._fail_evidence_saver.save_fail_event(frame, context)
        except Exception:
            pass

    def _handle_pcb_ng_feedback(self, result, frame_id: int) -> None:
        """PCB FAIL 时播放声音和保存证据。"""
        if getattr(self.config, "sound_feedback_enabled", True):
            try:
                self._sound_feedback.play_fail()
            except Exception:
                pass
        if getattr(self.config, "fail_evidence_enabled", True):
            try:
                frame, _ = self._get_feedback_frame()
                data = self._last_overlay_data or {}
                context = FailEvidenceContext(
                    round_id=int(result.track_id),
                    action_ng_step=-1,
                    step_name=f"PCB #{result.track_id} 缺: {'/'.join(result.missing_classes)}",
                    source_frame_id=int(frame_id),
                    detections=list(data.get("detections", []) or []),
                    model_path=str(data.get("model_path", "")),
                    timestamp=float(result.timestamp or time.time()),
                )
                self._fail_evidence_saver.save_fail_event(frame, context)
            except Exception:
                pass

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
        self._kpi_row.set_value("output", str(batch.total))
        self._kpi_row.set_value("yield", f"{batch.yield_rate:.1f}%")
        self._kpi_row.set_value("ok", str(batch.ok))
        self._kpi_row.set_value("ng", str(batch.ng))

    def _on_round_pass_settled(self) -> None:
        """PASS 文案展示结束；不改变步骤状态机。"""
        if self._result_label.text() == "PASS":
            self._clear_result_label()
            self._status_label.setText("检测中")
            self._status_label.setStyleSheet(text_style(TEXT_ACCENT, size=14, weight=700))

    def _set_result_label(self, text: str, color: str) -> None:
        """显示顶部 PASS/NG 结果，避免无结果时保留空白占位。"""
        self._result_label.setText(text)
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
        if not can_detect and not infer_on:
            if not cam_on:
                self._status_label.setText("请先打开相机")
            elif self.processor is None:
                self._status_label.setText("处理器未加载")
            self._status_label.setStyleSheet(text_style(TEXT_SECONDARY, size=14, weight=500))

    # ------------------------------------------------------------------
    # 信号槽
    # ------------------------------------------------------------------
    @Slot(QImage)
    def _on_frame_ready(self, qimg: QImage) -> None:
        pass

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._status_label.setText(f"错误: {msg}")
        self._status_label.setStyleSheet("color: #F44336; font-weight: bold;")

    @Slot(dict)
    def _on_overlay_updated(self, data: dict) -> None:
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
        if self._pcb_engine is not None:
            # PCB 多板检查模式
            image_size = data.get("image_size")
            pcb_results = self._pcb_engine.update(obb_dets, image_size=image_size)
            self._handle_pcb_results(pcb_results, data.get("source_frame_id", 0))
        elif self.step_engine:
            try:
                state = self.step_engine.update(obb_dets, frame_id=data.get("source_frame_id", 0))
            except TypeError:
                state = self.step_engine.update(obb_dets)
            self._update_step_display(state)

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
        )
        self._record_coordinate_log(overlay, state)
        self._update_display()

    def _handle_pcb_results(self, results: list, frame_id: int) -> None:
        """处理 PCB 检查结果，接入声音/证据/统计。"""
        for result in results:
            if result.result == PcbResult.PASS:
                self._stats_manager.record_pass()
                self._play_pass_feedback()
                self._set_result_label("PASS", "#4CAF50")
                self._status_label.setText(f"PCB #{result.track_id} PASS")
            elif result.result == PcbResult.FAIL:
                self._stats_manager.record_ng()
                self._handle_pcb_ng_feedback(result, frame_id)
                self._set_result_label("NG", "#F44336")
                self._status_label.setText(
                    f"PCB #{result.track_id} NG 缺: {'/'.join(result.missing_classes)}"
                )
            self._update_kpi()

    @Slot(dict)
    def _on_stats_updated(self, stats: dict) -> None:
        self._last_pipeline_stats = dict(stats)

    @Slot(dict)
    def _on_camera_params(self, params: dict) -> None:
        self._last_camera_params = dict(params)

    def _record_coordinate_log(self, overlay, step_state) -> None:
        """把当前推理帧写入坐标诊断日志（底层能力，不在主界面显示状态）。"""
        if not self._coordinate_logger.active:
            return
        frame, _frame_id = self.worker.get_display_frame()
        image_size = None
        if frame is not None:
            h, w = frame.shape[:2]
            image_size = [w, h]
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
            # 关闭前先停止检测
            if self.state.inference_on:
                self.worker.stop_inference()
            self.worker.close_camera()
            self._display_timer.stop()
            self._status_label.setText("READY")
            self._status_label.setStyleSheet(text_style(TEXT_ACCENT, size=14, weight=700))
        else:
            if self.worker.open_camera():
                self._display_timer.start(33)
                self._status_label.setText("预览中")
                self._status_label.setStyleSheet(text_style(TEXT_ACCENT, size=14, weight=700))
            else:
                self._status_label.setText("相机打开失败")
                self._status_label.setStyleSheet("color: #F44336; font-weight: bold;")
        self._sync_button_states()

    @Slot()
    def _on_toggle_detect(self) -> None:
        if self.state.inference_on:
            self.worker.stop_inference()
            self._round_pass_timer.stop()
            self._status_label.setText("已停止")
            self._status_label.setStyleSheet(text_style(TEXT_SECONDARY, size=14, weight=500))
        else:
            if not self.state.camera_on:
                self._status_label.setText("请先打开相机")
                self._status_label.setStyleSheet("color: #FF9800; font-weight: bold;")
                self._sync_button_states()
                return
            if self.processor is None:
                self._status_label.setText("处理器未加载，无法检测")
                self._status_label.setStyleSheet("color: #F44336; font-weight: bold;")
                self._sync_button_states()
                return
            if self.step_engine:
                self.step_engine.start_round()
                self._round_start_time = time.time()
                self._round_id_reset()
                self._last_counted_pass_round = -1
                self._last_counted_ng_key = None
                self._update_step_display(self.step_engine.get_state())
            self.worker.start_inference()
            self._status_label.setText("检测中")
            self._status_label.setStyleSheet(text_style(TEXT_ACCENT, size=14, weight=700))
            self._clear_result_label()
        self._sync_button_states()

    def _round_id_reset(self) -> None:
        """开始检测时重置本轮去重键（round_id 由 start_round 自增）。"""
        self._last_counted_pass_round = -1

    @Slot()
    def _on_config_clicked(self) -> None:
        if self._config_page is None:
            from ui.config_page import ConfigPage
            self._config_page = ConfigPage(self.config, stats_manager=self._stats_manager)
            self._config_page.back_clicked.connect(self._on_config_back)
            self._config_page.stats_changed.connect(self._on_stats_changed)
            self._config_page.config_saved.connect(self._on_config_saved)
            self._stacked.addWidget(self._config_page)
        self._config_page.refresh_stats_display()
        self._stacked.setCurrentWidget(self._config_page)
        self._status_label.setText("配置页")
        self._status_label.setStyleSheet("color: #9C27B0; font-weight: bold;")

    def _on_config_back(self) -> None:
        """从配置页返回主界面：刷新动态步骤卡片和 PCB 引擎。"""
        self._stacked.setCurrentWidget(self._main_page)
        self._rebuild_step_engine()
        self._init_pcb_engine()
        self._refresh_steps()
        self._sync_button_states()
        self._update_kpi()
        self._status_label.setText("READY" if not self.state.camera_on else "预览中")
        self._status_label.setStyleSheet(text_style(TEXT_ACCENT, size=14, weight=700))

    def _on_stats_changed(self) -> None:
        """配置页归零后同步主界面 KPI。"""
        self._update_kpi()
        if self._config_page is not None:
            self._config_page.refresh_stats_display()

    def _on_config_saved(self) -> None:
        """配置页保存后立即应用运行时配置。"""
        self._apply_feedback_config()
        model_ok = self._apply_runtime_model_config()
        self._rebuild_step_engine()
        self._refresh_steps()
        self._sync_button_states()
        self._update_kpi()
        if not model_ok:
            self._status_label.setText(self._last_model_reload_error or "模型加载失败")
            self._status_label.setStyleSheet("color: #F44336; font-weight: bold;")
        elif not self.state.inference_on:
            self._status_label.setText("配置已应用")
            self._status_label.setStyleSheet(text_style(TEXT_ACCENT, size=14, weight=700))

    def _apply_feedback_config(self) -> None:
        """同步保存后可立即生效的反馈配置。"""
        self._sound_feedback.enabled = bool(getattr(self.config, "sound_feedback_enabled", True))
        self._fail_evidence_saver.enabled = bool(getattr(self.config, "fail_evidence_enabled", True))

    @staticmethod
    def _model_runtime_signature(config) -> tuple:
        """影响推理处理器实例的配置签名。"""
        return (
            str(getattr(config, "yolo_model_path", "")),
            float(getattr(config, "yolo_conf_threshold", 0.0)),
            float(getattr(config, "yolo_iou_threshold", 0.0)),
            str(getattr(config, "ultralytics_device", "")),
            bool(getattr(config, "ultralytics_half", False)),
            str(getattr(config, "ultralytics_tracker", "")),
            int(getattr(config, "ultralytics_max_det", 0)),
            bool(getattr(config, "ultralytics_track_persist", True)),
        )

    def _create_processor_from_config(self):
        """创建新处理器。测试可注入 factory，生产路径复用 main.create_yolo_processor。"""
        if self._processor_factory is not None:
            return self._processor_factory(self.config, app_base_dir=self._app_base_dir)
        from main import create_yolo_processor

        return create_yolo_processor(self.config, app_base_dir=self._app_base_dir)

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
        self._status_label.setStyleSheet(text_style(TEXT_SECONDARY, size=14, weight=700))

        try:
            new_processor = self._create_processor_from_config()
        except Exception as exc:
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

        if was_inference_on and self.state.camera_on:
            self.worker.start_inference()
            self._status_label.setText("检测中")
        else:
            self._status_label.setText("模型已加载")
        self._status_label.setStyleSheet(text_style(TEXT_ACCENT, size=14, weight=700))
        return True

    def _rebuild_step_engine(self) -> None:
        """根据最新配置重建步骤引擎，保证 UI 卡片和判定顺序一致。"""
        if self.step_engine is None:
            return
        self.step_engine = StepSequenceEngine(
            step_class_names=self.config.category_names,
            step_counts=getattr(self.config, "category_counts", None),
            enter_stable_frames=self.config.action_pass_stable_frames,
            out_of_order_frames=self.config.action_ng_stable_frames,
            leave_stable_frames=getattr(self.config, "action_leave_stable_frames", 4),
            order_constraint_enabled=self.config.action_order_constraint_enabled,
        )
        self._sticky_ng_step_idx = -1
        self._last_step_focus_key = None

    @Slot()
    def _on_close_clicked(self) -> None:
        self.close()

    def closeEvent(self, event) -> None:
        self._display_timer.stop()
        self._round_pass_timer.stop()
        if self._coordinate_logger.active:
            self._coordinate_logger.stop_session()
        self.worker.close_camera()
        event.accept()
