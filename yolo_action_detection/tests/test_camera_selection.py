from __future__ import annotations

import ctypes
import json
import os
import sys
import threading
import time
from types import SimpleNamespace
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import ConfigManager
from app_state import AppState
from camera import mvsdk_camera as mvsdk_camera_module
from camera.camera_worker import CameraWorker
from camera.mvsdk_camera import MvSdkCamera, MvSdkDevice
from ui.config_page import ConfigPage
from ui.main_window import MainWindow
from yolo_runtime.yolo_result_models import DetectionOverlayState, ObbDetection
from step_sequence.step_sequence_engine import StepSequenceEngine
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QMessageBox


class FakeDeviceInfo:
    def __init__(self, friendly_name: str, port_type: str, sn: str) -> None:
        self._friendly_name = friendly_name
        self._port_type = port_type
        self._sn = sn

    def GetFriendlyName(self):
        return self._friendly_name

    def GetPortType(self):
        return self._port_type

    def GetSn(self):
        return self._sn


class FakeMvSdk:
    CAMERA_MEDIA_TYPE_MONO8 = 1
    CAMERA_MEDIA_TYPE_BGR8 = 3
    CAMERA_STATUS_TIME_OUT = -12
    c_ubyte = ctypes.c_ubyte

    class _IspCapacity:
        bMonoSensor = 0

    class _ResolutionRange:
        iWidthMax = 16
        iHeightMax = 8

    class _Capability:
        def __init__(self) -> None:
            self.sIspCapacity = FakeMvSdk._IspCapacity()
            self.sResolutionRange = FakeMvSdk._ResolutionRange()

    def __init__(self, devices) -> None:
        self.devices = list(devices)
        self.initialized_device = None
        self.uninitialized_handles = []
        self.freed_buffers = []

    def CameraEnumerateDevice(self):
        return list(self.devices)

    def CameraInit(self, device, _load_mode, _team):
        self.initialized_device = device
        return 41

    def CameraGetCapability(self, _handle):
        return self._Capability()

    def CameraSetIspOutFormat(self, _handle, _format):
        return 0

    def CameraSetTriggerMode(self, _handle, _mode):
        return 0

    def CameraPlay(self, _handle):
        return 0

    def CameraAlignMalloc(self, _size, _alignment):
        return 73

    def CameraUnInit(self, handle):
        self.uninitialized_handles.append(handle)

    def CameraAlignFree(self, buffer):
        self.freed_buffers.append(buffer)


class FakeSdkException(Exception):
    def __init__(self, error_code: int, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message


class BlockingReadMvSdk(FakeMvSdk):
    def __init__(self, devices) -> None:
        super().__init__(devices)
        self.read_started = threading.Event()
        self.release_read = threading.Event()
        self.read_finished = threading.Event()
        self.uninit_before_read_finished = False

    def CameraGetImageBuffer(self, _handle, _timeout_ms):
        self.read_started.set()
        self.release_read.wait(timeout=2.0)
        self.read_finished.set()
        raise FakeSdkException(self.CAMERA_STATUS_TIME_OUT, "timeout")

    def CameraUnInit(self, handle):
        if not self.read_finished.is_set():
            self.uninit_before_read_finished = True
        super().CameraUnInit(handle)


class FrameMvSdk(FakeMvSdk):
    class _FrameHead:
        iWidth = 16
        iHeight = 8
        uBytes = 16 * 8 * 3

    def __init__(self, devices) -> None:
        super().__init__(devices)
        self._allocated_buffer = None

    def CameraAlignMalloc(self, size, _alignment):
        self._allocated_buffer = ctypes.create_string_buffer(size)
        return ctypes.addressof(self._allocated_buffer)

    def CameraGetImageBuffer(self, _handle, _timeout_ms):
        time.sleep(0.002)
        return 91, self._FrameHead()

    def CameraImageProcess(self, _handle, _raw, output, frame_head):
        ctypes.memset(output, 127, frame_head.uBytes)

    def CameraReleaseImageBuffer(self, _handle, _raw):
        return 0

    def CameraFlipFrameBuffer(self, _buffer, _frame_head, _direction):
        return 0


class EchoProcessor:
    def process_frame(self, _frame, *, source_frame_id):
        return DetectionOverlayState(
            source_frame_id=source_frame_id,
            model_path="camera-test.onnx",
            latency_ms=3.0,
        )

    def get_class_names(self):
        return {}


def wait_until(predicate, timeout_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_selected_mvsdk_camera_sn_round_trips_without_device_catalog(tmp_path):
    config_path = tmp_path / "config.json"
    config = ConfigManager()

    assert config.mvsdk_camera_sn == ""

    config.mvsdk_camera_sn = "SN-B"
    config.save(config_path)

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["mvsdk_camera_sn"] == "SN-B"
    assert "mvsdk_devices" not in saved
    assert "mvsdk_device_index" not in saved

    loaded = ConfigManager()
    loaded.load(config_path)
    assert loaded.mvsdk_camera_sn == "SN-B"


def test_mvsdk_camera_opens_exact_sn_when_friendly_names_match(monkeypatch):
    first = FakeDeviceInfo("MV-Camera", "USB3", "SN-A")
    selected = FakeDeviceInfo("MV-Camera", "USB3", "SN-B")
    sdk = FakeMvSdk([selected, first])
    monkeypatch.setattr(mvsdk_camera_module, "get_mvsdk", lambda: sdk)

    assert [device.sn for device in MvSdkCamera.enumerate_devices()] == ["SN-B", "SN-A"]
    sdk.devices = [first, selected]
    camera = MvSdkCamera()

    assert camera.open(sn="SN-B") is True
    assert sdk.initialized_device is selected
    assert camera.device is not None
    assert camera.device.sn == "SN-B"
    assert str(camera.device) == "MV-Camera | USB3 | SN: SN-B"

    camera.close()


def test_mvsdk_camera_missing_sn_does_not_fall_back_to_first_device(monkeypatch):
    first = FakeDeviceInfo("MV-Camera", "USB3", "SN-A")
    sdk = FakeMvSdk([first])
    monkeypatch.setattr(mvsdk_camera_module, "get_mvsdk", lambda: sdk)

    camera = MvSdkCamera()

    assert camera.open(sn="SN-MISSING") is False
    assert sdk.initialized_device is None
    assert "SN-MISSING" in camera.last_error
    assert "不在线" in camera.last_error


def test_mvsdk_camera_reports_sdk_error_code_when_device_is_occupied(monkeypatch):
    selected = FakeDeviceInfo("MV-Camera", "USB3", "SN-B")
    sdk = FakeMvSdk([selected])

    def raise_device_occupied(*_args):
        raise FakeSdkException(-45, "device occupied")

    sdk.CameraInit = raise_device_occupied
    monkeypatch.setattr(mvsdk_camera_module, "get_mvsdk", lambda: sdk)

    camera = MvSdkCamera()

    assert camera.open(sn="SN-B") is False
    assert "SN-B" in camera.last_error
    assert "SDK 错误码 -45" in camera.last_error
    assert "device occupied" in camera.last_error


def test_camera_worker_requires_an_explicit_camera_selection(monkeypatch):
    first = FakeDeviceInfo("MV-Camera", "USB3", "SN-A")
    sdk = FakeMvSdk([first])
    monkeypatch.setattr(mvsdk_camera_module, "get_mvsdk", lambda: sdk)
    config = ConfigManager()
    state = AppState()
    worker = CameraWorker(config, state)
    errors = []
    worker.error_occurred.connect(errors.append)

    opened = worker.open_camera()
    try:
        assert opened is False
        assert state.camera_on is False
        assert sdk.initialized_device is None
        assert errors and "选择" in errors[-1]
    finally:
        worker.close_camera()


def test_camera_worker_releases_sdk_only_after_blocking_read_finishes(monkeypatch):
    selected = FakeDeviceInfo("MV-Camera", "USB3", "SN-B")
    sdk = BlockingReadMvSdk([selected])
    monkeypatch.setattr(mvsdk_camera_module, "get_mvsdk", lambda: sdk)
    config = ConfigManager()
    config.mvsdk_camera_sn = "SN-B"
    worker = CameraWorker(config, AppState())

    assert worker.open_camera() is True
    assert worker.active_device is not None
    assert worker.active_device.sn == "SN-B"
    assert sdk.read_started.wait(timeout=1.0)

    release_timer = threading.Timer(0.05, sdk.release_read.set)
    release_timer.start()
    try:
        worker.close_camera()
    finally:
        release_timer.cancel()
        sdk.release_read.set()

    assert sdk.read_finished.is_set()
    assert sdk.uninit_before_read_finished is False


def test_camera_worker_close_clears_all_frames_and_runtime_overlay(monkeypatch):
    selected = FakeDeviceInfo("MV-Camera", "USB3", "SN-B")
    sdk = FrameMvSdk([selected])
    monkeypatch.setattr(mvsdk_camera_module, "get_mvsdk", lambda: sdk)
    config = ConfigManager()
    config.mvsdk_camera_sn = "SN-B"
    worker = CameraWorker(config, AppState())
    worker.set_frame_processor(EchoProcessor())

    assert worker.open_camera() is True
    worker.start_inference()
    assert wait_until(lambda: worker.get_display_frame()[0] is not None)
    assert worker.get_latest_preview_frame() is not None
    assert worker.get_latest_overlay().source_frame_id > 0

    worker.close_camera()

    assert worker.get_latest_preview_frame() is None
    assert worker.get_display_frame() == (None, 0)
    assert worker.get_latest_overlay() == DetectionOverlayState()
    assert worker.get_pipeline_stats()["infer_latency_ms"] == 0.0


def test_camera_config_lists_devices_and_saves_selected_sn(
    qapp, tmp_path, monkeypatch
):
    config = ConfigManager()
    config._config_path = str(tmp_path / "config.json")
    config.mvsdk_camera_sn = "SN-A"
    page = ConfigPage(config)
    devices = [
        MvSdkDevice(0, "Left", "USB3", "SN-A"),
        MvSdkDevice(1, "Right", "GigE", "SN-B"),
    ]
    refresh_requests = []
    page.camera_refresh_requested.connect(lambda: refresh_requests.append(True))
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: None)

    page.set_camera_devices(devices)

    assert page._camera_device_input.count() == 3
    assert page._camera_device_input.itemText(1) == "Left | USB3 | SN: SN-A"
    assert page._camera_device_input.itemData(2) == "SN-B"
    assert page._camera_device_input.currentData() == "SN-A"

    page._camera_device_input.setCurrentIndex(2)
    assert config.mvsdk_camera_sn == "SN-A"
    page._camera_refresh_btn.click()
    assert refresh_requests == [True]

    page._on_save()

    saved = json.loads(Path(config.get_config_path()).read_text(encoding="utf-8"))
    assert saved["mvsdk_camera_sn"] == "SN-B"
    assert saved["mvsdk_friendly_name"] == "Right"


def test_camera_config_keeps_offline_saved_sn_unselected(qapp):
    config = ConfigManager()
    config.mvsdk_camera_sn = "SN-OFFLINE"
    page = ConfigPage(config)

    page.set_camera_devices([MvSdkDevice(0, "Online", "USB3", "SN-ONLINE")])

    assert page._camera_device_input.currentIndex() == 0
    assert page._camera_device_input.currentData() == ""
    assert "SN-OFFLINE" in page._camera_device_status.text()
    assert "不在线" in page._camera_device_status.text()
    assert config.mvsdk_camera_sn == "SN-OFFLINE"


def test_camera_config_migrates_only_unique_legacy_friendly_name(qapp):
    config = ConfigManager()
    config.mvsdk_friendly_name = "Legacy"
    page = ConfigPage(config)

    page.set_camera_devices(
        [
            MvSdkDevice(0, "Legacy", "USB3", "SN-A"),
            MvSdkDevice(1, "Other", "USB3", "SN-B"),
        ]
    )
    assert page._camera_device_input.currentData() == "SN-A"
    assert config.mvsdk_camera_sn == ""

    ambiguous_config = ConfigManager()
    ambiguous_config.mvsdk_friendly_name = "Legacy"
    ambiguous_page = ConfigPage(ambiguous_config)
    ambiguous_page.set_camera_devices(
        [
            MvSdkDevice(0, "Legacy", "USB3", "SN-A"),
            MvSdkDevice(1, "Legacy", "USB3", "SN-B"),
        ]
    )
    assert ambiguous_page._camera_device_input.currentIndex() == 0
    assert "按 SN 选择" in ambiguous_page._camera_device_status.text()


def test_camera_config_rejects_device_without_sn(qapp, tmp_path, monkeypatch):
    config = ConfigManager()
    config._config_path = str(tmp_path / "config.json")
    page = ConfigPage(config)
    warnings = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )

    page.set_camera_devices([MvSdkDevice(0, "Unknown", "USB3", "")])
    item = page._camera_device_input.model().item(1)
    assert item is not None and item.isEnabled() is False

    page._camera_device_input.setCurrentIndex(1)
    page._on_save()

    assert warnings and "缺少 SN" in warnings[-1][1]
    assert config.mvsdk_camera_sn == ""
    assert not Path(config.get_config_path()).exists()


def test_main_window_enumerates_devices_at_startup_and_injects_config_page(
    qapp, tmp_path, monkeypatch
):
    devices = [MvSdkDevice(0, "Startup", "USB3", "SN-START")]
    calls = []

    def enumerate_devices():
        calls.append(True)
        return list(devices)

    monkeypatch.setattr(CameraWorker, "enumerate_devices", staticmethod(enumerate_devices))
    config = ConfigManager()
    config._config_path = str(tmp_path / "config.json")
    config.mvsdk_camera_sn = "SN-START"
    processor = EchoProcessor()
    window = MainWindow(
        config,
        AppState(),
        processor=processor,
        step_engine=StepSequenceEngine(config.category_names),
        processor_factory=lambda *_args, **_kwargs: processor,
        evidence_base_dir=tmp_path / "evidence",
    )

    try:
        assert calls == [True]
        window._on_config_clicked()
        assert window._config_page._camera_device_input.currentData() == "SN-START"
        assert window._config_page._camera_device_input.count() == 2
    finally:
        window.close()


def test_config_save_switches_running_camera_by_sn_and_keeps_stats(
    qapp, tmp_path, monkeypatch
):
    config = ConfigManager()
    config._config_path = str(tmp_path / "config.json")
    config.mvsdk_camera_sn = "SN-OLD"
    processor = EchoProcessor()
    state = AppState()
    window = MainWindow(
        config,
        state,
        processor=processor,
        step_engine=StepSequenceEngine(config.category_names),
        processor_factory=lambda *_args, **_kwargs: processor,
        evidence_base_dir=tmp_path / "evidence",
    )
    calls = []
    window.worker._mvsdk_camera = SimpleNamespace(
        device=MvSdkDevice(0, "Old", "USB3", "SN-OLD")
    )
    state.set_camera_on(True)
    state.set_inference_on(True)
    window._stats_manager.record_pass()

    def stop_inference():
        calls.append("stop_inference")
        state.set_inference_on(False)

    def close_camera():
        calls.append("close_camera")
        state.set_camera_on(False)
        window.worker._mvsdk_camera = None

    def open_camera():
        calls.append(f"open_camera:{config.mvsdk_camera_sn}")
        window.worker._mvsdk_camera = SimpleNamespace(
            device=MvSdkDevice(1, "New", "USB3", config.mvsdk_camera_sn)
        )
        state.set_camera_on(True)
        return True

    monkeypatch.setattr(window.worker, "stop_inference", stop_inference)
    monkeypatch.setattr(window.worker, "close_camera", close_camera)
    monkeypatch.setattr(window.worker, "open_camera", open_camera)

    config.mvsdk_camera_sn = "SN-NEW"
    window._on_config_saved()

    assert calls == ["stop_inference", "close_camera", "open_camera:SN-NEW"]
    assert state.camera_on is True
    assert state.inference_on is False
    assert window._display_timer.isActive()
    assert window._status_label.text() == "预览中"
    assert window._stats_manager.batch.total == 1
    assert window._stats_manager.batch.ok == 1


def test_camera_switch_clears_visual_and_inspection_session(
    qapp, tmp_path, monkeypatch
):
    config = ConfigManager()
    config._config_path = str(tmp_path / "config.json")
    config.mvsdk_camera_sn = "SN-OLD"
    config.category_names = ["parent", "child", "", "", "", ""]
    config.category_counts = [1, 1, 1, 1, 1, 1]
    config.first_category_region_check_enabled = True
    processor = EchoProcessor()
    state = AppState()
    window = MainWindow(
        config,
        state,
        processor=processor,
        step_engine=StepSequenceEngine(config.category_names),
        processor_factory=lambda *_args, **_kwargs: processor,
        evidence_base_dir=tmp_path / "evidence",
    )
    parent = ObbDetection(
        class_id=0,
        label="parent",
        conf=0.95,
        track_id=7,
        polygon=[(20, 20), (80, 20), (80, 80), (20, 80)],
        box=(20, 20, 80, 80),
        center=(50, 50),
    )
    old_pcb_engine = window._pcb_engine
    old_pcb_engine.update([parent], image_size=(100, 100))
    assert old_pcb_engine.pcb_states

    window._video_label.setPixmap(QPixmap(16, 16))
    window._latest_runtime_overlay = DetectionOverlayState(
        source_frame_id=12,
        detections=[parent],
    )
    window._last_overlay_data = {"detections": [parent.to_dict()]}
    window._set_result_label("NG", "#F44336")
    window._set_notice("旧相机异常")
    window._step_cards[0].setVisible(True)
    window._step_cards[0].set_step_state("ng")

    window.worker._mvsdk_camera = SimpleNamespace(
        device=MvSdkDevice(0, "Old", "USB3", "SN-OLD")
    )
    state.set_camera_on(True)

    def close_camera():
        state.set_camera_on(False)
        window.worker._mvsdk_camera = None

    def open_camera():
        window.worker._mvsdk_camera = SimpleNamespace(
            device=MvSdkDevice(1, "New", "USB3", config.mvsdk_camera_sn)
        )
        state.set_camera_on(True)
        return True

    monkeypatch.setattr(window.worker, "close_camera", close_camera)
    monkeypatch.setattr(window.worker, "open_camera", open_camera)

    config.mvsdk_camera_sn = "SN-NEW"
    window._on_config_saved()

    pixmap = window._video_label.pixmap()
    assert pixmap is None or pixmap.isNull()
    assert window._latest_runtime_overlay == DetectionOverlayState()
    assert window._last_overlay_data == {}
    assert all(not card.isVisibleTo(window._steps_container) for card in window._step_cards)
    assert window._pcb_engine is not old_pcb_engine
    assert window._pcb_engine.pcb_states == {}
    assert window._notice_label.text() == ""
    assert window._notice_label.isVisible() is False
    assert window._result_label.text() == ""
    assert window._result_label.isVisible() is False


@pytest.mark.parametrize(
    ("camera_on", "active_sn"),
    [(False, None), (True, "SN-SAME")],
    ids=["closed-camera", "same-sn"],
)
def test_config_save_does_not_open_closed_camera_or_restart_same_sn(
    qapp, tmp_path, monkeypatch, camera_on, active_sn
):
    config = ConfigManager()
    config._config_path = str(tmp_path / f"{camera_on}.json")
    config.mvsdk_camera_sn = "SN-SAME" if active_sn else "SN-NEW"
    processor = EchoProcessor()
    state = AppState()
    window = MainWindow(
        config,
        state,
        processor=processor,
        step_engine=StepSequenceEngine(config.category_names),
        processor_factory=lambda *_args, **_kwargs: processor,
        evidence_base_dir=tmp_path / "evidence",
    )
    if active_sn:
        window.worker._mvsdk_camera = SimpleNamespace(
            device=MvSdkDevice(0, "Current", "USB3", active_sn)
        )
    state.set_camera_on(camera_on)
    calls = []
    monkeypatch.setattr(window.worker, "close_camera", lambda: calls.append("close"))
    monkeypatch.setattr(window.worker, "open_camera", lambda: calls.append("open"))

    window._on_config_saved()

    assert calls == []
    assert state.camera_on is camera_on


def test_failed_camera_switch_stays_closed_without_fallback(
    qapp, tmp_path, monkeypatch
):
    config = ConfigManager()
    config._config_path = str(tmp_path / "config.json")
    config.mvsdk_camera_sn = "SN-OLD"
    processor = EchoProcessor()
    state = AppState()
    window = MainWindow(
        config,
        state,
        processor=processor,
        step_engine=StepSequenceEngine(config.category_names),
        processor_factory=lambda *_args, **_kwargs: processor,
        evidence_base_dir=tmp_path / "evidence",
    )
    window.worker._mvsdk_camera = SimpleNamespace(
        device=MvSdkDevice(0, "Old", "USB3", "SN-OLD")
    )
    state.set_camera_on(True)
    state.set_inference_on(True)
    calls = []

    def stop_inference():
        calls.append("stop_inference")
        state.set_inference_on(False)

    def close_camera():
        calls.append("close_camera")
        state.set_camera_on(False)
        window.worker._mvsdk_camera = None

    def fail_open():
        calls.append(f"open_camera:{config.mvsdk_camera_sn}")
        return False

    monkeypatch.setattr(window.worker, "stop_inference", stop_inference)
    monkeypatch.setattr(window.worker, "close_camera", close_camera)
    monkeypatch.setattr(window.worker, "open_camera", fail_open)

    config.mvsdk_camera_sn = "SN-OFFLINE"
    window._on_config_saved()

    assert calls == ["stop_inference", "close_camera", "open_camera:SN-OFFLINE"]
    assert state.camera_on is False
    assert state.inference_on is False
    assert window.worker.active_device is None
    assert window._display_timer.isActive() is False
    assert "SN-OFFLINE" in window._status_label.text()
    assert "切换失败" in window._status_label.text()
