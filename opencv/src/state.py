"""应用状态管理模块"""
from dataclasses import dataclass, field
from typing import Callable, List


@dataclass
class AppState:
    """应用状态管理器"""
    
    camera_on: bool = False
    inference_on: bool = False
    current_fps: float = 0.0
    last_error: str = ""
    
    _listeners: List[Callable] = field(default_factory=list, repr=False)
    
    def add_listener(self, callback: Callable) -> None:
        """添加状态变化监听器"""
        if callback not in self._listeners:
            self._listeners.append(callback)
    
    def remove_listener(self, callback: Callable) -> None:
        """移除状态变化监听器"""
        if callback in self._listeners:
            self._listeners.remove(callback)
    
    def _notify_listeners(self, state_name: str, value) -> None:
        """通知所有监听器状态变化"""
        for listener in self._listeners:
            try:
                listener(state_name, value)
            except Exception as e:
                print(f"状态监听器执行失败: {e}")
    
    def set_camera_on(self, value: bool) -> None:
        """设置相机状态
        
        如果关闭相机，推理状态也会自动关闭
        """
        old_value = self.camera_on
        self.camera_on = value
        
        if old_value != value:
            self._notify_listeners("camera_on", value)
        
        # 关闭相机时自动关闭推理
        if not value and self.inference_on:
            self.set_inference_on(False)
    
    def set_inference_on(self, value: bool) -> bool:
        """设置推理状态
        
        Args:
            value: 目标状态
            
        Returns:
            是否设置成功。如果相机未开启，无法开启推理。
        """
        # 相机未开启时不能开始推理
        if value and not self.camera_on:
            self.last_error = "请先开启相机"
            return False
        
        old_value = self.inference_on
        self.inference_on = value
        
        if old_value != value:
            self._notify_listeners("inference_on", value)
        
        return True
    
    def set_fps(self, value: float) -> None:
        """设置当前帧率"""
        self.current_fps = value
        self._notify_listeners("current_fps", value)
    
    def set_error(self, message: str) -> None:
        """设置错误信息"""
        self.last_error = message
        self._notify_listeners("last_error", message)
    
    def clear_error(self) -> None:
        """清除错误信息"""
        self.last_error = ""
    
    def reset(self) -> None:
        """重置所有状态"""
        self.camera_on = False
        self.inference_on = False
        self.current_fps = 0.0
        self.last_error = ""
        self._notify_listeners("reset", None)
