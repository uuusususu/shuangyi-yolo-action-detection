"""时序使能不改顺序检测。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parents[1] / "src"
for item in (str(ROOT), str(SRC)):
    if item not in sys.path:
        sys.path.insert(0, item)

from config import ConfigManager
from lstm_workbench.model import ActionLSTM
from lstm_action.engine import LstmActionEngine
from step_sequence.step_sequence_engine import StepSequenceEngine, StepStatus


class _Det:
    def __init__(self, label: str):
        self.label = label


def _contract(folder: Path, class_ids: list[str]) -> Path:
    folder.mkdir(parents=True)
    model = ActionLSTM(131, len(class_ids))
    torch.save(model.state_dict(), folder / "best_lstm.pt")
    payload = {
        "class_ids": class_ids,
        "part_class_names": ["part"],
        "input_size": 131,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "feature_code_version": "1",
        "window": 48,
        "yolo_sha256": "",
    }
    path = folder / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_old_config_keeps_lstm_disabled(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"config_schema_version": 4, "category_names": ["包装"]}), encoding="utf-8")
    config = ConfigManager()
    config.load(path)
    assert config.lstm_action_enabled is False
    assert config.lstm_contract_path == ""
    assert config.config_schema_version == 4


def test_disabled_keeps_step_count_pass_and_skips_weights(tmp_path: Path):
    contract = _contract(tmp_path / "run", ["idle", "包装"])
    steps = StepSequenceEngine(step_class_names=["包装"], step_counts=[1], enter_stable_frames=1)
    lstm = LstmActionEngine()
    lstm.configure(enabled=False, contract_path=str(contract), category_names=["包装"], yolo_model_path="")
    state = steps.update([_Det("包装")])
    assert state.steps[0].status == StepStatus.PASS
    assert lstm.model_loaded is False


def test_class_order_mismatch_does_not_pass_and_step_still_updates(tmp_path: Path):
    contract = _contract(tmp_path / "run", ["idle", "其他"])
    steps = StepSequenceEngine(step_class_names=["包装"], step_counts=[1], enter_stable_frames=1)
    lstm = LstmActionEngine()
    loaded = lstm.configure(enabled=True, contract_path=str(contract), category_names=["包装"], yolo_model_path="")
    state = steps.update([_Det("包装")])
    assert loaded.loaded is False
    assert loaded.passed_categories == []
    assert state.steps[0].status == StepStatus.PASS
    assert lstm.model_loaded is False


def test_timeline_and_mstcn_contracts_are_refused(tmp_path: Path):
    folder = tmp_path / "run"
    contract = _contract(folder, ["idle", "包装"])
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["train_unit"] = "timeline"
    payload["network"] = "lstm"
    contract.write_text(json.dumps(payload), encoding="utf-8")
    engine = LstmActionEngine()
    loaded = engine.configure(
        enabled=True, contract_path=str(contract), category_names=["包装"], yolo_model_path=""
    )
    assert loaded.loaded is False
    assert loaded.reason == "未加载：这份模型要按整条连续推理，工位还没接入"

    payload["train_unit"] = ""
    payload["network"] = "mstcn"
    contract.write_text(json.dumps(payload), encoding="utf-8")
    loaded = engine.configure(
        enabled=True, contract_path=str(contract), category_names=["包装"], yolo_model_path=""
    )
    assert loaded.loaded is False
    assert "还没接入" in loaded.reason


def test_old_lstm_contract_still_loads(tmp_path: Path):
    contract = _contract(tmp_path / "old", ["idle", "包装"])
    engine = LstmActionEngine()
    loaded = engine.configure(
        enabled=True, contract_path=str(contract), category_names=["包装"], yolo_model_path=""
    )
    assert loaded.loaded is True


def test_matching_contract_passes_only_lstm_state(tmp_path: Path):
    contract = _contract(tmp_path / "run", ["idle", "包装"])
    steps = StepSequenceEngine(step_class_names=["包装"], step_counts=[1], enter_stable_frames=1)
    before = steps.get_state()
    lstm = LstmActionEngine()
    loaded = lstm.configure(enabled=True, contract_path=str(contract), category_names=["包装"], yolo_model_path="")
    assert loaded.loaded is True

    def always_pack(batch):
        logits = torch.zeros(batch.shape[0], 2)
        logits[:, 1] = 8
        return logits

    lstm._model.forward = always_pack
    detection = {"label": "part", "conf": 0.9, "box": (0, 0, 10, 10)}
    last = None
    for _ in range(48):
        last = lstm.update([detection], [], 100, 100, 0.7)
    assert last is not None
    assert last.passed_categories == ["包装"]
    assert steps.get_state().steps[0].status == before.steps[0].status
