import time
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Dict


@dataclass
class GestureTriggerCycle:
    """通用手势触发器
    
    支持多种手势的状态机管理，每种手势独立跟踪状态。
    """
    name: str
    n_on: int = 4
    hold_seconds: float = 2.0
    time_provider: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        self._state = "idle"
        self._on_count = 0
        self._hold_until = 0.0

    def reset(self) -> List[Tuple[str, bool]]:
        events: List[Tuple[str, bool]] = []
        if self._state == "holding":
            events.append((self.name, False))
        self._state = "idle"
        self._on_count = 0
        self._hold_until = 0.0
        return events

    def update(self, raw_detected: bool) -> List[Tuple[str, bool]]:
        now = self.time_provider()
        events: List[Tuple[str, bool]] = []

        if self._state == "idle":
            if raw_detected:
                self._on_count += 1
                if self._on_count >= max(1, int(self.n_on)):
                    self._state = "holding"
                    self._on_count = 0
                    self._hold_until = now + float(self.hold_seconds)
                    events.append((self.name, True))
            else:
                self._on_count = 0
            return events

        if self._state == "holding":
            if now >= self._hold_until:
                self._state = "locked"
                events.append((self.name, False))
            return events

        if self._state == "locked":
            if not raw_detected:
                self._state = "idle"
                self._on_count = 0
            return events

        self._state = "idle"
        self._on_count = 0
        self._hold_until = 0.0
        return events


@dataclass
class EventGestureTriggerCycle:
    name: str
    n_on: int = 1
    hold_seconds: float = 2.0
    time_provider: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        self._state = "idle"
        self._on_count = 0
        self._hold_until = 0.0

    def reset(self) -> List[Tuple[str, bool]]:
        events: List[Tuple[str, bool]] = []
        if self._state == "holding":
            events.append((self.name, False))
        self._state = "idle"
        self._on_count = 0
        self._hold_until = 0.0
        return events

    def update(self, *, trigger_pulse: bool, blocked: bool) -> List[Tuple[str, bool]]:
        now = self.time_provider()
        events: List[Tuple[str, bool]] = []

        if self._state == "idle":
            if trigger_pulse:
                self._on_count += 1
                if self._on_count >= max(1, int(self.n_on)):
                    self._state = "holding"
                    self._on_count = 0
                    self._hold_until = now + float(self.hold_seconds)
                    events.append((self.name, True))
            else:
                self._on_count = 0
            return events

        if self._state == "holding":
            if now >= self._hold_until:
                self._state = "locked"
                events.append((self.name, False))
            return events

        if self._state == "locked":
            if not blocked:
                self._state = "idle"
                self._on_count = 0
            return events

        self._state = "idle"
        self._on_count = 0
        self._hold_until = 0.0
        return events


@dataclass
class DebouncedGestureState:
    name: str
    n_on: int = 3
    n_off: int = 6

    def __post_init__(self) -> None:
        self._active = False
        self._on_count = 0
        self._off_count = 0

    def reset(self) -> List[Tuple[str, bool]]:
        events: List[Tuple[str, bool]] = []
        if self._active:
            events.append((self.name, False))
        self._active = False
        self._on_count = 0
        self._off_count = 0
        return events

    def update(self, raw_detected: bool) -> List[Tuple[str, bool]]:
        events: List[Tuple[str, bool]] = []

        if raw_detected:
            self._off_count = 0
            if not self._active:
                self._on_count += 1
                if self._on_count >= max(1, int(self.n_on)):
                    self._active = True
                    self._on_count = 0
                    events.append((self.name, True))
            return events

        self._on_count = 0
        if self._active:
            self._off_count += 1
            if self._off_count >= max(1, int(self.n_off)):
                self._active = False
                self._off_count = 0
                events.append((self.name, False))
        return events



@dataclass
class GestureSequenceTracker:
    """手势序列跟踪器
    
    跟踪手势序列完成状态：✌ → 握住 → 撕纸
    当三个手势按顺序全部完成后，触发完成回调。
    """
    on_sequence_complete: Callable[[], None] = None
    sequence: List[str] = field(default_factory=lambda: ["✌", "握住", "撕纸"])
    
    def __post_init__(self) -> None:
        self._current_index = 0  # 当前等待的手势索引
    
    def set_sequence(self, sequence: List[str]) -> None:
        """设置手势序列"""
        self.sequence = list(sequence)
    
    def reset(self) -> None:
        """重置序列状态"""
        self._current_index = 0
    
    def get_current_step(self) -> int:
        """获取当前步骤索引（0-2）"""
        return self._current_index
    
    def get_expected_gesture(self) -> str:
        """获取当前期望的手势名称"""
        if self._current_index < len(self.sequence):
            return self.sequence[self._current_index]
        return ""
    
    def is_gesture_expected(self, gesture_name: str) -> bool:
        """判断手势是否为当前期望的手势"""
        return gesture_name == self.get_expected_gesture()
    
    def is_gesture_completed(self, gesture_name: str) -> bool:
        """判断手势是否已完成（在当前期望之前）"""
        if gesture_name not in self.sequence:
            return False
        gesture_index = self.sequence.index(gesture_name)
        return gesture_index < self._current_index
    
    def on_gesture_triggered(self, gesture_name: str) -> bool:
        """处理手势触发事件
        
        Args:
            gesture_name: 触发的手势名称
            
        Returns:
            True 如果序列完成，False 否则
        """
        if self._current_index >= len(self.sequence):
            return False
        
        expected = self.sequence[self._current_index]
        
        if gesture_name == expected:
            self._current_index += 1
            
            # 检查是否完成整个序列
            if self._current_index >= len(self.sequence):
                if self.on_sequence_complete is not None:
                    self.on_sequence_complete()
                self.reset()
                return True
        
        return False
