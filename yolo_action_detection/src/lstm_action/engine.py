"""用现有类别和置信度，由 LSTM 单独判过。"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lstm_workbench.features import FEATURE_CODE_VERSION, build_feature_row, feature_size
from lstm_workbench.model_registry import file_sha256


@dataclass
class LstmActionState:
    loaded: bool = False
    reason: str = ""
    passed_categories: list[str] = field(default_factory=list)
    active_category: str = ""


class LstmActionEngine:
    def __init__(self) -> None:
        self._enabled = False
        self._model = None
        self._class_ids: list[str] = []
        self._part_names: list[str] = []
        self._window = 48
        self._buffer: list[list[float]] = []
        self._passed: list[str] = []
        self.state = LstmActionState(reason="未启用")

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def configure(
        self,
        *,
        enabled: bool,
        contract_path: str,
        category_names: list[str],
        yolo_model_path: str,
    ) -> LstmActionState:
        self._buffer.clear()
        self._passed = []
        self._model = None
        self._enabled = bool(enabled)
        names = [str(name).strip() for name in category_names if str(name).strip()]
        if not self._enabled:
            self.state = LstmActionState(reason="未启用")
            return self.state
        if not str(contract_path).strip():
            self.state = LstmActionState(reason="未加载：没有契约路径")
            return self.state
        path = Path(contract_path)
        if path.is_dir():
            path = path / "contract.json"
        weight = path.parent / "best_lstm.pt"
        if not path.is_file() or not weight.is_file():
            self.state = LstmActionState(reason="未加载：找不到契约或权重")
            return self.state
        contract = json.loads(path.read_text(encoding="utf-8"))
        network = str(contract.get("network") or "lstm")
        if contract.get("train_unit") == "timeline" or network != "lstm":
            self.state = LstmActionState(reason="未加载：这份模型要按整条连续推理，工位还没接入")
            return self.state
        expected = ["idle", *names]
        if list(contract.get("class_ids") or []) != expected:
            self.state = LstmActionState(reason="未加载：类别顺序不一致")
            return self.state
        part_names = [str(name) for name in contract.get("part_class_names") or []]
        if int(contract.get("input_size") or 0) != feature_size(len(part_names)):
            self.state = LstmActionState(reason="未加载：特征长度不一致")
            return self.state
        if int(contract.get("hidden_size") or 0) != 128 or int(contract.get("num_layers") or 0) != 2:
            self.state = LstmActionState(reason="未加载：网络结构不一致")
            return self.state
        if str(contract.get("feature_code_version") or "") != FEATURE_CODE_VERSION:
            self.state = LstmActionState(reason="未加载：特征代码版本不一致")
            return self.state
        recorded = str(contract.get("yolo_sha256") or "")
        if recorded and yolo_model_path and Path(yolo_model_path).is_file():
            if file_sha256(Path(yolo_model_path)) != recorded:
                self.state = LstmActionState(reason="未加载：YOLO 指纹不一致")
                return self.state
        import torch
        from lstm_workbench.model import ActionLSTM

        model = ActionLSTM(feature_size(len(part_names)), len(expected))
        model.load_state_dict(torch.load(weight, map_location="cpu"))
        model.eval()
        self._model = model
        self._class_ids = expected
        self._part_names = part_names
        self._window = int(contract.get("window") or 48)
        self.state = LstmActionState(loaded=True, reason="已加载")
        return self.state

    def update(
        self,
        detections: list[dict],
        hands: list[list[tuple]],
        frame_w: int,
        frame_h: int,
        conf_threshold: float,
    ) -> LstmActionState:
        if not self._enabled or self._model is None:
            return self.state
        padded = []
        for hand in hands:
            points = []
            for point in hand:
                z = float(point[2]) if len(point) > 2 else 0.0
                points.append((float(point[0]), float(point[1]), z))
            padded.append(points)
        row = build_feature_row(detections, padded, frame_w, frame_h, self._part_names, conf_threshold)
        self._buffer.append(row)
        if len(self._buffer) > self._window:
            self._buffer = self._buffer[-self._window :]
        if len(self._buffer) < self._window:
            self.state = LstmActionState(loaded=True, reason="攒帧中", passed_categories=list(self._passed))
            return self.state
        import torch

        batch = torch.tensor([self._buffer], dtype=torch.float32)
        with torch.no_grad():
            probs = torch.softmax(self._model(batch), dim=1)[0]
        index = int(torch.argmax(probs).item())
        name = self._class_ids[index]
        score = float(probs[index].item())
        active = ""
        if name != "idle" and name in self._class_ids[1:] and score >= conf_threshold:
            active = name
            if name not in self._passed:
                self._passed.append(name)
        self.state = LstmActionState(
            loaded=True,
            reason="已加载",
            passed_categories=list(self._passed),
            active_category=active,
        )
        return self.state
