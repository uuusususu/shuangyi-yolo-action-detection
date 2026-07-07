#coding=utf-8
"""通用相机管理器模块

基于 OpenCV VideoCapture 封装，提供：
- 相机枚举
- 采集线程 + 单槽缓存（低延迟）
- 自动重连
- 黑白/彩色自适应
"""
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class CameraInfo:
    """相机信息"""
    index: int
    backend: int

    def __str__(self) -> str:
        return f"相机 {self.index}"


class CvCameraManager:
    """OpenCV 相机管理器

    Features:
        - 枚举索引相机
        - 采集线程 + latest frame 低延迟缓存
        - 失败检测与自动重连
        - 上下文管理器支持
    """

    def __init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._opened_index: Optional[int] = None
        self._backend: int = cv2.CAP_DSHOW

        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_lock = threading.Lock()

        self._fail_count = 0
        self._reconnect_count = 0

        self._width = 640
        self._height = 480
        self._fps = 30

    @staticmethod
    def enumerate_cameras(
        max_index: int = 10,
        backend: int = cv2.CAP_DSHOW
    ) -> List[CameraInfo]:
        """枚举可用相机

        Args:
            max_index: 最大枚举索引
            backend: OpenCV 后端（Windows 推荐 CAP_DSHOW）

        Returns:
            可用相机列表
        """
        cameras: List[CameraInfo] = []
        for i in range(int(max_index) + 1):
            cap = cv2.VideoCapture(i, backend)
            if cap is not None and cap.isOpened():
                cameras.append(CameraInfo(index=i, backend=backend))
            if cap is not None:
                cap.release()
        return cameras

    def open(
        self,
        index: int,
        *,
        backend: int = cv2.CAP_DSHOW,
        width: int = 640,
        height: int = 480,
        fps: int = 30
    ) -> bool:
        """打开相机

        Args:
            index: 相机索引
            backend: OpenCV 后端
            width: 分辨率宽度
            height: 分辨率高度
            fps: 帧率

        Returns:
            是否成功
        """
        self.close()

        self._backend = int(backend)
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)

        cap = cv2.VideoCapture(int(index), self._backend)
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))
        cap.set(cv2.CAP_PROP_FPS, float(self._fps))

        self._cap = cap
        self._opened_index = int(index)
        self._fail_count = 0
        return True

    def start(self) -> None:
        """启动采集线程"""
        if self._cap is None or not self._cap.isOpened():
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止采集线程"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def close(self) -> None:
        """关闭相机并释放资源"""
        self.stop()
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._opened_index = None
        with self._latest_lock:
            self._latest_frame = None

    def is_opened(self) -> bool:
        """相机是否已打开"""
        return self._cap is not None and self._cap.isOpened()

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """获取最新帧（线程安全）

        Returns:
            BGR 格式图像，无帧时返回 None
        """
        with self._latest_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    @property
    def fail_count(self) -> int:
        """当前连续失败次数"""
        return self._fail_count

    @property
    def reconnect_count(self) -> int:
        """重连次数"""
        return self._reconnect_count

    def _reconnect(self) -> None:
        """自动重连"""
        self._reconnect_count += 1
        idx = self._opened_index
        if idx is None:
            return
        if self._cap is not None:
            self._cap.release()
        self._cap = None

        time.sleep(0.2)
        self.open(
            idx,
            backend=self._backend,
            width=self._width,
            height=self._height,
            fps=self._fps
        )

    def _loop(self) -> None:
        """采集循环（在独立线程中运行）"""
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                self._fail_count += 1
            else:
                ok, frame = self._cap.read()
                if ok and frame is not None and frame.size > 0:
                    # 灰度图自动转 BGR
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    with self._latest_lock:
                        self._latest_frame = frame
                    self._fail_count = 0
                else:
                    self._fail_count += 1

            # 连续 30 次失败触发重连
            if self._fail_count >= 30:
                self._fail_count = 0
                self._reconnect()

            time.sleep(0.001)

    def __enter__(self) -> "CvCameraManager":
        return self

    def __exit__(self, *args) -> None:
        self.close()
