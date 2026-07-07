from __future__ import annotations

import platform
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from camera.mv_mvsdk import get_mvsdk, get_mvsdk_load_error


@dataclass(frozen=True)
class MvSdkDevice:
    index: int
    friendly_name: str
    port_type: str
    sn: str

    def __str__(self) -> str:
        if self.port_type:
            return f"{self.friendly_name} ({self.port_type})"
        return self.friendly_name


class MvSdkCamera:
    def __init__(
        self,
        *,
        exposure_us: int = 30 * 1000,
        auto_exposure: bool = False,
        trigger_mode: int = 0,
    ) -> None:
        self._exposure_us = int(exposure_us)
        self._auto_exposure = bool(auto_exposure)
        self._trigger_mode = int(trigger_mode)

        self._mvsdk = get_mvsdk()
        self._dev: Optional[MvSdkDevice] = None

        self._h_camera: int = 0
        self._cap = None
        self._mono = False
        self._frame_buffer = 0
        self._frame_buffer_size = 0

        self._lock = threading.Lock()

    @staticmethod
    def enumerate_devices() -> list[MvSdkDevice]:
        mvsdk = get_mvsdk()
        if mvsdk is None:
            return []

        devices = []
        dev_list = mvsdk.CameraEnumerateDevice()
        for i, dev in enumerate(dev_list):
            try:
                friendly = str(dev.GetFriendlyName())
            except Exception:
                friendly = str(i)
            try:
                port = str(dev.GetPortType())
            except Exception:
                port = ""
            try:
                sn = str(dev.GetSn())
            except Exception:
                sn = ""
            devices.append(MvSdkDevice(index=int(i), friendly_name=friendly, port_type=port, sn=sn))
        return devices

    @staticmethod
    def is_available() -> bool:
        return get_mvsdk() is not None

    @staticmethod
    def load_error() -> Optional[str]:
        return get_mvsdk_load_error()

    def is_opened(self) -> bool:
        return self._h_camera > 0

    @property
    def device(self) -> Optional[MvSdkDevice]:
        return self._dev

    def open(self, *, friendly_name: str, parameter_mode: str = "preserve", parameter_group: int = 0, parameter_file: str = "") -> bool:
        mvsdk = self._mvsdk
        if mvsdk is None:
            return False

        self.close()

        dev_list = mvsdk.CameraEnumerateDevice()
        if not dev_list:
            return False

        dev_info = None
        dev_idx = 0
        for i, dev in enumerate(dev_list):
            try:
                if str(dev.GetFriendlyName()) == str(friendly_name):
                    dev_info = dev
                    dev_idx = i
                    break
            except Exception:
                continue
        if dev_info is None:
            dev_info = dev_list[0]
            dev_idx = 0

        try:
            h_camera = mvsdk.CameraInit(dev_info, -1, -1)
        except Exception:
            return False

        try:
            cap = mvsdk.CameraGetCapability(h_camera)

            mono = bool(getattr(cap.sIspCapacity, "bMonoSensor", 0) != 0)
            if mono:
                mvsdk.CameraSetIspOutFormat(h_camera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
            else:
                mvsdk.CameraSetIspOutFormat(h_camera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

            mvsdk.CameraSetTriggerMode(h_camera, int(self._trigger_mode))

            # 相机参数模式：不默认覆盖曝光
            if parameter_mode == "load_group":
                try:
                    mvsdk.CameraLoadParameter(h_camera, int(parameter_group))
                except Exception:
                    pass
            elif parameter_mode == "load_file" and parameter_file:
                try:
                    mvsdk.CameraReadParameterFromFile(h_camera, parameter_file)
                except Exception:
                    pass
            elif parameter_mode == "manual":
                if self._auto_exposure:
                    mvsdk.CameraSetAeState(h_camera, 1)
                else:
                    mvsdk.CameraSetAeState(h_camera, 0)
                    mvsdk.CameraSetExposureTime(h_camera, int(self._exposure_us))
            # preserve 模式：不主动写入任何参数

            mvsdk.CameraPlay(h_camera)

            width_max = int(getattr(cap.sResolutionRange, "iWidthMax", 640))
            height_max = int(getattr(cap.sResolutionRange, "iHeightMax", 480))
            frame_buffer_size = width_max * height_max * (1 if mono else 3)
            frame_buffer = mvsdk.CameraAlignMalloc(int(frame_buffer_size), 16)

            try:
                friendly = str(dev_info.GetFriendlyName())
            except Exception:
                friendly = str(dev_idx)
            try:
                port = str(dev_info.GetPortType())
            except Exception:
                port = ""
            try:
                sn = str(dev_info.GetSn())
            except Exception:
                sn = ""

            self._h_camera = int(h_camera)
            self._cap = cap
            self._mono = bool(mono)
            self._frame_buffer = int(frame_buffer)
            self._frame_buffer_size = int(frame_buffer_size)
            self._dev = MvSdkDevice(index=int(dev_idx), friendly_name=friendly, port_type=port, sn=sn)

            return True
        except Exception:
            try:
                mvsdk.CameraUnInit(h_camera)
            except Exception:
                pass
            self._h_camera = 0
            return False

    def close(self) -> None:
        mvsdk = self._mvsdk
        if mvsdk is None:
            return

        with self._lock:
            h = int(self._h_camera)
            buf = int(self._frame_buffer)
            self._h_camera = 0
            self._cap = None
            self._mono = False
            self._frame_buffer = 0
            self._frame_buffer_size = 0
            self._dev = None

        if h > 0:
            try:
                mvsdk.CameraUnInit(h)
            except Exception:
                pass

        if buf:
            try:
                mvsdk.CameraAlignFree(buf)
            except Exception:
                pass

    def read(self, *, timeout_ms: int = 200) -> Optional[np.ndarray]:
        mvsdk = self._mvsdk
        if mvsdk is None:
            return None

        with self._lock:
            h = int(self._h_camera)
            buf = int(self._frame_buffer)
            mono = bool(self._mono)

        if h <= 0 or buf <= 0:
            return None

        try:
            p_raw, frame_head = mvsdk.CameraGetImageBuffer(h, int(timeout_ms))
        except Exception as e:
            try:
                if getattr(e, "error_code", None) == getattr(mvsdk, "CAMERA_STATUS_TIME_OUT", -12):
                    return None
            except Exception:
                pass
            return None

        try:
            mvsdk.CameraImageProcess(h, p_raw, buf, frame_head)
        finally:
            try:
                mvsdk.CameraReleaseImageBuffer(h, p_raw)
            except Exception:
                pass

        try:
            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(buf, frame_head, 1)
        except Exception:
            pass

        try:
            ubytes = int(getattr(frame_head, "uBytes", 0))
            width = int(getattr(frame_head, "iWidth", 0))
            height = int(getattr(frame_head, "iHeight", 0))
        except Exception:
            return None

        if ubytes <= 0 or width <= 0 or height <= 0:
            return None

        frame_data = (mvsdk.c_ubyte * ubytes).from_address(buf)
        frame = np.frombuffer(frame_data, dtype=np.uint8)
        channels = 1 if mono else 3
        try:
            frame = frame.reshape((height, width, channels))
        except Exception:
            return None

        if channels == 1:
            gray = frame[:, :, 0]
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return frame.copy()

    def read_params(self) -> dict:
        """读取当前相机参数快照。"""
        from camera.camera_params import read_camera_params
        mvsdk = self._mvsdk
        if mvsdk is None or self._h_camera <= 0:
            return {"read_ok": False, "error": "相机未打开"}
        snap = read_camera_params(self._h_camera, mvsdk)
        return snap.to_dict()


class MvSdkCapture:
    def __init__(self, camera: MvSdkCamera) -> None:
        self._camera = camera

    def isOpened(self) -> bool:
        return self._camera.is_opened()

    def read(self):
        frame = self._camera.read()
        if frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        self._camera.close()
