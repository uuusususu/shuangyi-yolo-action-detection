"""MvSDK 相机参数读取与控制。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CameraParamSnapshot:
    ae_enabled: bool = False
    exposure_us: int = 0
    gain: float = 0.0
    gamma: float = 0.0
    contrast: int = 0
    brightness: int = 0
    read_ok: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return {"ae_enabled": self.ae_enabled, "exposure_us": self.exposure_us,
                "gain": self.gain, "gamma": self.gamma, "contrast": self.contrast,
                "brightness": self.brightness, "read_ok": self.read_ok, "error": self.error}

    def display_text(self) -> str:
        if not self.read_ok:
            return f"参数读取失败: {self.error}" if self.error else "参数未读取"
        ae = "自动" if self.ae_enabled else "手动"
        return f"AE={ae} 曝光={self.exposure_us}us 增益={self.gain:.1f} Gamma={self.gamma:.1f}"


def read_camera_params(h_camera: int, mvsdk) -> CameraParamSnapshot:
    snap = CameraParamSnapshot()
    try:
        snap.ae_enabled = bool(mvsdk.CameraGetAeState(h_camera) == 1)
    except Exception:
        pass
    try:
        snap.exposure_us = int(mvsdk.CameraGetExposureTime(h_camera))
    except Exception:
        pass
    try:
        snap.gain = float(mvsdk.CameraGetGain(h_camera))
    except Exception:
        pass
    try:
        snap.gamma = float(mvsdk.CameraGetGamma(h_camera))
    except Exception:
        pass
    try:
        snap.contrast = int(mvsdk.CameraGetContrast(h_camera))
    except Exception:
        pass
    try:
        snap.brightness = int(mvsdk.CameraGetBrightness(h_camera))
    except Exception:
        pass
    snap.read_ok = True
    return snap


def apply_camera_params(h_camera: int, mvsdk, *, ae=None, exposure_us=None, gain=None, gamma=None, contrast=None, brightness=None) -> CameraParamSnapshot:
    try:
        if ae is not None:
            mvsdk.CameraSetAeState(h_camera, 1 if ae else 0)
        if exposure_us is not None and not ae:
            mvsdk.CameraSetExposureTime(h_camera, int(exposure_us))
        if gain is not None:
            mvsdk.CameraSetGain(h_camera, int(gain))
        if gamma is not None:
            mvsdk.CameraSetGamma(h_camera, int(gamma))
        if contrast is not None:
            mvsdk.CameraSetContrast(h_camera, int(contrast))
        if brightness is not None:
            mvsdk.CameraSetBrightness(h_camera, int(brightness))
    except Exception as e:
        return CameraParamSnapshot(error=f"参数写入失败: {e}")
    return read_camera_params(h_camera, mvsdk)


def load_camera_parameter_group(h_camera: int, mvsdk, group: int) -> Optional[str]:
    try:
        mvsdk.CameraLoadParameter(h_camera, group)
        return None
    except Exception as e:
        return f"参数组加载失败: {e}"


def load_camera_parameter_file(h_camera: int, mvsdk, path: str) -> Optional[str]:
    try:
        mvsdk.CameraReadParameterFromFile(h_camera, path)
        return None
    except Exception as e:
        return f"参数文件加载失败: {e}"
