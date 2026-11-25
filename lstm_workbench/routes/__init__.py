"""HTTP 路由模块化分层：媒体流、零件工作区、智能标注、YOLO 训练/验证、动作时序训练与复盘。"""
from .action import ActionRoutesMixin
from .media import MediaRoutesMixin
from .parts import PartsRoutesMixin
from .smart_label import SmartLabelRoutesMixin
from .yolo import YoloRoutesMixin

__all__ = [
    "ActionRoutesMixin",
    "MediaRoutesMixin",
    "PartsRoutesMixin",
    "SmartLabelRoutesMixin",
    "YoloRoutesMixin",
]
