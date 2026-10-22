# Sequential Fail Retry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 顺序模式增加可选的「类别/数量失败重试」：第 1 步到位后，当前步连续不对的每一帧计 1 次，满配置次数则正式 NG。

**Architecture:** 判定只改 `StepSequenceEngine.update()`。配置两项新字段默认关闭。主界面按引擎状态显示次数并区分 NG 文案。证据 metadata 多记实际/要求数量和次数。ROI 与形状匹配不改。

**Tech Stack:** Python 3.11、PySide6、pytest、现有 `ConfigManager` / `StepSequenceEngine` / `FailEvidenceSaver`。

**Spec:** `docs/superpowers/specs/2026-08-24-sequential-fail-retry-design.md`

**Run tests from:** `yolo_action_detection`，解释器 `..\.venv\Scripts\python.exe`

---

## File map

| File | Responsibility |
| --- | --- |
| `yolo_action_detection/src/config.py` | `action_fail_retry_enabled`、`action_fail_retry_frames` 默认值与校验 |
| `yolo_action_detection/src/step_sequence/step_sequence_engine.py` | 计次、清零、数量/漏检 NG、状态带出次数 |
| `yolo_action_detection/src/main.py` | `create_step_engine` 传入新参数 |
| `yolo_action_detection/src/ui/config_page.py` | 动作判定区开关与次数框 |
| `yolo_action_detection/src/ui/widgets/native_panels.py` | 卡片显示 `3 / 10` 重试进度 |
| `yolo_action_detection/src/ui/main_window.py` | 热重载签名、卡片可见性、NG 文案 |
| `yolo_action_detection/src/detection_logging/fail_evidence.py` | metadata 增加数量与次数 |
| `yolo_action_detection/tests/test_fail_retry.py` | 引擎判定单测 |
| `yolo_action_detection/tests/test_step_quantity.py` | 开关关闭时旧行为不被破坏 |
| `yolo_action_detection/tests/test_runtime_model_configuration.py` | 配置往返与引擎重建 |
| `yolo_action_detection/tests/test_ui_redesign.py` | 卡片可见性与次数显示 |
| `yolo_action_detection/tests/test_fail_evidence_retry.py` | 证据 metadata |

不新建独立判定模块。次数逻辑放在现有 `update()` 里，错序判定之前。

---

### Task 1: 配置字段

**Files:**
- Modify: `yolo_action_detection/src/config.py`
- Test: `yolo_action_detection/tests/test_fail_retry.py`

- [ ] **Step 1: Write the failing test**

Create `yolo_action_detection/tests/test_fail_retry.py` with this header and first tests:

```python
"""顺序模式类别/数量失败重试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import ConfigManager
from step_sequence.step_sequence_engine import StepSequenceEngine, RoundResult, StepStatus
from yolo_runtime.yolo_result_models import ObbDetection


def _det(label: str) -> ObbDetection:
    return ObbDetection(
        class_id=0, label=label, conf=0.9, track_id=1,
        polygon=[(10, 10), (50, 10), (50, 50), (10, 50)],
        box=(10, 10, 50, 50), center=(30, 30),
    )


def _engine(**kwargs) -> StepSequenceEngine:
    params = dict(
        step_class_names=["散热片", "导热垫"],
        step_counts=[1, 3],
        enter_stable_frames=1,
        leave_stable_frames=4,
        out_of_order_frames=10,
        order_constraint_enabled=True,
        fail_retry_enabled=True,
        fail_retry_frames=10,
    )
    params.update(kwargs)
    eng = StepSequenceEngine(**params)
    eng.start_round()
    return eng


def test_config_defaults_disable_fail_retry():
    config = ConfigManager()
    assert config.action_fail_retry_enabled is False
    assert config.action_fail_retry_frames == 10


def test_config_validate_clamps_fail_retry_frames(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "config_schema_version": 4,
        "model_task": "ultralytics_obb",
        "action_fail_retry_enabled": True,
        "action_fail_retry_frames": 0,
    }), encoding="utf-8")
    config = ConfigManager()
    config.load(path)
    assert config.action_fail_retry_enabled is True
    assert config.action_fail_retry_frames == 1

    path.write_text(json.dumps({
        "config_schema_version": 4,
        "model_task": "ultralytics_obb",
        "action_fail_retry_frames": 999,
    }), encoding="utf-8")
    config.load(path)
    assert config.action_fail_retry_frames == 100
    assert config.action_fail_retry_enabled is False
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```
..\.venv\Scripts\python.exe -m pytest tests/test_fail_retry.py::test_config_defaults_disable_fail_retry tests/test_fail_retry.py::test_config_validate_clamps_fail_retry_frames -v
```

Expected: FAIL，`ConfigManager` 没有 `action_fail_retry_enabled`。

- [ ] **Step 3: Write minimal implementation**

In `yolo_action_detection/src/config.py`, after `action_order_constraint_enabled: bool = True` add:

```python
    action_fail_retry_enabled: bool = False
    action_fail_retry_frames: int = 10
```

In `validate()`, after `self.action_ng_stable_frames = max(1, int(self.action_ng_stable_frames))` add:

```python
        self.action_fail_retry_enabled = _coerce_bool(self.action_fail_retry_enabled)
        self.action_fail_retry_frames = max(1, min(100, int(self.action_fail_retry_frames)))
```

Do not bump `CURRENT_CONFIG_SCHEMA_VERSION`. `load()` already copies unknown matching attributes; missing keys keep dataclass defaults.

- [ ] **Step 4: Run test to verify it passes**

Run the same two tests. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add yolo_action_detection/src/config.py yolo_action_detection/tests/test_fail_retry.py
git commit -m "feat: add sequential fail-retry config fields"
```

Only commit if the user asked to commit this session; otherwise leave staged/unstaged and continue.

---

### Task 2: 引擎 — 开关关闭与空工位

**Files:**
- Modify: `yolo_action_detection/src/step_sequence/step_sequence_engine.py`
- Test: `yolo_action_detection/tests/test_fail_retry.py`

- [ ] **Step 1: Write the failing tests**

Append to `test_fail_retry.py`:

```python
def test_disabled_wrong_quantity_does_not_ng():
    eng = _engine(fail_retry_enabled=False)
    for _ in range(20):
        state = eng.update([_det("导热垫"), _det("导热垫")])
    assert state.round_result == RoundResult.IN_PROGRESS
    assert state.steps[0].status == StepStatus.WAITING
    assert state.steps[0].fail_retry_count == 0


def test_first_step_zero_does_not_count():
    eng = _engine()
    for _ in range(20):
        state = eng.update([])
    assert state.round_result == RoundResult.IN_PROGRESS
    assert state.current_step_index == 0
    assert state.steps[0].fail_retry_count == 0
    assert state.fail_retry_count == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```
..\.venv\Scripts\python.exe -m pytest tests/test_fail_retry.py::test_disabled_wrong_quantity_does_not_ng tests/test_fail_retry.py::test_first_step_zero_does_not_count -v
```

Expected: FAIL（`StepSequenceEngine` 不接受 `fail_retry_enabled`，`StepState` 无 `fail_retry_count`）。

- [ ] **Step 3: Add constructor and state fields, keep update() behavior identical when disabled**

In `StepState` add:

```python
    fail_retry_count: int = 0
```

In `StepSequenceState` add:

```python
    fail_retry_enabled: bool = False
    fail_retry_frames: int = 10
    fail_retry_count: int = 0
```

In `StepSequenceEngine.__init__` add parameters after `order_constraint_enabled`:

```python
        fail_retry_enabled: bool = False,
        fail_retry_frames: int = 10,
```

Store:

```python
        self._fail_retry_enabled = bool(fail_retry_enabled)
        self._fail_retry_frames = max(1, min(100, int(fail_retry_frames)))
```

In `start_round`, inside the configured-step reset loop, add `step.fail_retry_count = 0`.

In `get_state()` pass:

```python
            fail_retry_enabled=self._fail_retry_enabled,
            fail_retry_frames=self._fail_retry_frames,
            fail_retry_count=(
                self._steps[self._current_step_idx].fail_retry_count
                if 0 <= self._current_step_idx < len(self._steps)
                else 0
            ),
```

Do not change `update()` yet. Disabled tests should pass because wrong quantity still does not NG and `fail_retry_count` stays 0.

- [ ] **Step 4: Run the two tests**

Expected: PASS.

- [ ] **Step 5: Commit** if the user asked.

```bash
git add yolo_action_detection/src/step_sequence/step_sequence_engine.py yolo_action_detection/tests/test_fail_retry.py
git commit -m "feat: thread fail-retry state through step engine"
```

---

### Task 3: 引擎 — 计次与数量/漏检 NG

**Files:**
- Modify: `yolo_action_detection/src/step_sequence/step_sequence_engine.py`
- Test: `yolo_action_detection/tests/test_fail_retry.py`

- [ ] **Step 1: Write the failing tests**

Append:

```python
def test_second_step_missing_ten_frames_ng():
    eng = _engine()
    assert eng.update([_det("散热片")]).steps[0].status == StepStatus.PASS
    for i in range(9):
        state = eng.update([])
        assert state.round_result == RoundResult.IN_PROGRESS, i
        assert state.current_step_index == 1
        assert state.fail_retry_count == i + 1
    state = eng.update([])
    assert state.round_result == RoundResult.ACTION_NG
    assert state.action_ng_step == 1
    assert state.steps[1].status == StepStatus.NG
    assert "导热垫" in state.action_ng_reason
    assert "实际0/要求3" in state.action_ng_reason
    assert "连续10帧" in state.action_ng_reason


def test_ninth_wrong_frame_still_in_progress():
    eng = _engine()
    eng.update([_det("散热片")])
    for _ in range(9):
        state = eng.update([_det("导热垫"), _det("导热垫")])
    assert state.round_result == RoundResult.IN_PROGRESS
    assert state.fail_retry_count == 9


def test_tenth_too_many_ng():
    eng = _engine()
    eng.update([_det("散热片")])
    four = [_det("导热垫") for _ in range(4)]
    for _ in range(9):
        eng.update(four)
    state = eng.update(four)
    assert state.round_result == RoundResult.ACTION_NG
    assert "实际4/要求3" in state.action_ng_reason


def test_correct_on_tenth_frame_passes_and_resets():
    eng = _engine()
    eng.update([_det("散热片")])
    two = [_det("导热垫"), _det("导热垫")]
    for _ in range(9):
        eng.update(two)
    state = eng.update([_det("导热垫") for _ in range(3)])
    assert state.round_result == RoundResult.PASS
    assert state.steps[1].status == StepStatus.PASS
    assert state.steps[1].fail_retry_count == 0


def test_first_step_wrong_quantity_ng_stays_on_first():
    eng = _engine()
    two = [_det("散热片"), _det("散热片")]
    for _ in range(9):
        state = eng.update(two)
        assert state.current_step_index == 0
        assert state.round_result == RoundResult.IN_PROGRESS
    state = eng.update(two)
    assert state.round_result == RoundResult.ACTION_NG
    assert state.action_ng_step == 0
    assert state.current_step_index == 0
    assert "实际2/要求1" in state.action_ng_reason
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```
..\.venv\Scripts\python.exe -m pytest tests/test_fail_retry.py::test_second_step_missing_ten_frames_ng tests/test_fail_retry.py::test_ninth_wrong_frame_still_in_progress tests/test_fail_retry.py::test_tenth_too_many_ng tests/test_fail_retry.py::test_correct_on_tenth_frame_passes_and_resets tests/test_fail_retry.py::test_first_step_wrong_quantity_ng_stays_on_first -v
```

Expected: FAIL（`update()` 还不计次）。

- [ ] **Step 3: Implement counting in `update()`**

Add these methods on `StepSequenceEngine`:

```python
    def _quantity_ng_reason(self, step: StepState) -> str:
        return (
            f"步骤{step.index + 1}({step.class_name})连续{self._fail_retry_frames}帧"
            f"类别未检出或数量不符（实际{step.current_count}/要求{step.required_count}）"
        )

    def _should_count_fail_retry(self, step: StepState, current_count: int) -> bool:
        if not self._fail_retry_enabled:
            return False
        first_idx = self._first_configured_step_index()
        if step.index == first_idx:
            return current_count != 0 and current_count != step.required_count
        return current_count != step.required_count

    def _mark_quantity_ng(self, step: StepState) -> None:
        reason = self._quantity_ng_reason(step)
        self._round_result = RoundResult.ACTION_NG
        self._action_ng_step = step.index
        self._action_ng_reason = reason
        step.status = StepStatus.NG
        step.ng_at = time.time()
        step.ng_reason = reason
        self._round_completed_at = time.time()
```

In `update()`, after setting `current_step_hit` and the existing PASS / leave_stable branch, **before** the order-NG block, add:

```python
        if current_step_hit:
            expected_step.fail_retry_count = 0
        elif self._should_count_fail_retry(expected_step, current_count):
            expected_step.fail_retry_count += 1
            if expected_step.fail_retry_count >= self._fail_retry_frames:
                self._mark_quantity_ng(expected_step)
                return self.get_state()
```

Put the `fail_retry_count = 0` assignment inside the existing `if current_step_hit:` block (first line of that block). Put the `elif` increment after the existing `else:` leave_stable block, still before order NG.

When quantity NG fires, return immediately so the same frame does not also raise order NG.

- [ ] **Step 4: Run the five tests plus Task 2 tests**

Run:

```
..\.venv\Scripts\python.exe -m pytest tests/test_fail_retry.py tests/test_step_quantity.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit** if asked.

```bash
git add yolo_action_detection/src/step_sequence/step_sequence_engine.py yolo_action_detection/tests/test_fail_retry.py
git commit -m "feat: count sequential quantity misses toward NG"
```

---

### Task 4: 引擎 — 错序优先顺序

**Files:**
- Modify: `yolo_action_detection/src/step_sequence/step_sequence_engine.py` only if tests require
- Test: `yolo_action_detection/tests/test_fail_retry.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_quantity_ng_wins_over_order_ng_same_frame():
    """当前步已满 10 帧不对时，即使后面步骤刚好凑齐，也只报数量/漏检。"""
    eng = _engine(out_of_order_frames=1)
    eng.update([_det("散热片")])
    for _ in range(9):
        eng.update([])
    state = eng.update([_det("C")])  # 后面类别，本步仍 0/3
    # 本步第 10 帧不对，应数量 NG，不得变成「检测到步骤C」
    assert state.round_result == RoundResult.ACTION_NG
    assert state.action_ng_step == 1
    assert "数量不符" in state.action_ng_reason or "未检出" in state.action_ng_reason


def test_order_ng_still_works_before_retry_exhausted():
    eng = StepSequenceEngine(
        step_class_names=["A", "B", "C"],
        step_counts=[1, 1, 1],
        enter_stable_frames=1,
        out_of_order_frames=1,
        order_constraint_enabled=True,
        fail_retry_enabled=True,
        fail_retry_frames=10,
    )
    eng.start_round()
    eng.update([_det("A")])
    state = eng.update([_det("C")])
    assert state.round_result == RoundResult.ACTION_NG
    assert "检测到步骤3" in state.action_ng_reason
```

Fix the first test: step names are 散热片/导热垫 only, so later-step order NG cannot fire on `"C"`. Use three steps:

```python
def test_quantity_ng_wins_over_order_ng_same_frame():
    eng = _engine(
        step_class_names=["散热片", "导热垫", "标签"],
        step_counts=[1, 3, 1],
        out_of_order_frames=1,
    )
    eng.update([_det("散热片")])
    for _ in range(9):
        eng.update([])
    state = eng.update([_det("标签")])
    assert state.round_result == RoundResult.ACTION_NG
    assert state.action_ng_step == 1
    assert "导热垫" in state.action_ng_reason
    assert "检测到步骤" not in state.action_ng_reason
```

- [ ] **Step 2: Run tests**

Expected: first test PASS if Task 3 returns before order NG; second test PASS with existing order logic.

- [ ] **Step 3: If second test fails, do not change order-NG conditions.** Only ensure quantity NG returns before the order block. Re-run.

- [ ] **Step 4: Commit** if asked.

---

### Task 5: 接线 — create_step_engine 与热重载

**Files:**
- Modify: `yolo_action_detection/src/main.py`
- Modify: `yolo_action_detection/src/ui/main_window.py`
- Test: `yolo_action_detection/tests/test_runtime_model_configuration.py`

- [ ] **Step 1: Write the failing test**

In `test_runtime_model_configuration.py` add:

```python
def test_rebuild_step_engine_passes_fail_retry_settings(qapp, tmp_path):
    config = _config(tmp_path)
    config.roi_inspection_enabled = False
    config.action_fail_retry_enabled = True
    config.action_fail_retry_frames = 7
    window = _window(
        config,
        lambda runtime_config, **_: DummyProcessor(runtime_config.yolo_model_path),
        DummyProcessor("old"),
    )
    window._rebuild_step_engine()
    assert window.step_engine._fail_retry_enabled is True
    assert window.step_engine._fail_retry_frames == 7
    signature = window._step_engine_config_signature(config)
    assert True in signature or 7 in signature
```

Use `_window` the same way other tests in that file do.

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL（重建引擎未传新参数，签名不含这两项）。

- [ ] **Step 3: Wire parameters**

`main.py` `create_step_engine`:

```python
    return StepSequenceEngine(
        step_class_names=config.category_names,
        step_counts=config.get_runtime_category_counts(),
        enter_stable_frames=config.action_pass_stable_frames,
        out_of_order_frames=config.action_ng_stable_frames,
        leave_stable_frames=config.action_leave_stable_frames,
        order_constraint_enabled=config.action_order_constraint_enabled,
        fail_retry_enabled=getattr(config, "action_fail_retry_enabled", False),
        fail_retry_frames=getattr(config, "action_fail_retry_frames", 10),
    )
```

`MainWindow._step_engine_config_signature` append:

```python
            bool(getattr(config, "action_fail_retry_enabled", False)),
            int(getattr(config, "action_fail_retry_frames", 10)),
```

`MainWindow._rebuild_step_engine` pass the same two kwargs as `create_step_engine`.

- [ ] **Step 4: Run test**

Expected: PASS.

- [ ] **Step 5: Commit** if asked.

---

### Task 6: 配置页

**Files:**
- Modify: `yolo_action_detection/src/ui/config_page.py`
- Test: `yolo_action_detection/tests/test_runtime_model_configuration.py`

- [ ] **Step 1: Write the failing test**

```python
def test_config_page_round_trips_fail_retry(qapp, tmp_path, monkeypatch):
    config = _config(tmp_path)
    page = ConfigPage(config)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: None)

    assert page._fail_retry_cb.isChecked() is False
    assert page._fail_retry_frames_input.isEnabled() is False
    assert page._fail_retry_frames_input.value() == 10

    page._fail_retry_cb.setChecked(True)
    assert page._fail_retry_frames_input.isEnabled() is True
    page._fail_retry_frames_input.setValue(8)
    page._on_save()

    saved = json.loads(Path(config.get_config_path()).read_text(encoding="utf-8"))
    assert saved["action_fail_retry_enabled"] is True
    assert saved["action_fail_retry_frames"] == 8
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL，没有 `_fail_retry_cb`。

- [ ] **Step 3: Add controls**

In `_build_action_section`, after the order constraint `SwitchRow`:

```python
        retry_row = SwitchRow(
            "类别/数量失败重试",
            "第 1 步到位后，当前步类别未检出或数量不符时按帧累计，满次数报 NG。",
        )
        self._fail_retry_cb = retry_row.checkbox
        self._fail_retry_cb.stateChanged.connect(self._sync_fail_retry_inputs_enabled)
        group.content_layout.addWidget(retry_row)
        self._fail_retry_frames_input = QSpinBox()
        self._fail_retry_frames_input.setRange(1, 100)
        self._add_form_grid(group, [
            (FormField("失败重试次数", self._fail_retry_frames_input, "连续不对的推理帧上限，默认 10。"), False),
        ])
        self._sync_fail_retry_inputs_enabled()
```

Add:

```python
    def _sync_fail_retry_inputs_enabled(self) -> None:
        enabled = bool(self._fail_retry_cb.isChecked())
        self._fail_retry_frames_input.setEnabled(enabled)
```

In `_load_values` after order checkbox:

```python
        self._fail_retry_cb.setChecked(getattr(c, "action_fail_retry_enabled", False))
        self._fail_retry_frames_input.setValue(int(getattr(c, "action_fail_retry_frames", 10)))
        self._sync_fail_retry_inputs_enabled()
```

In `_on_save` after `action_order_constraint_enabled`:

```python
        c.action_fail_retry_enabled = self._fail_retry_cb.isChecked()
        c.action_fail_retry_frames = self._fail_retry_frames_input.value()
```

- [ ] **Step 4: Run test**

Expected: PASS.

- [ ] **Step 5: Commit** if asked.

---

### Task 7: 识别卡片次数与可见性

**Files:**
- Modify: `yolo_action_detection/src/ui/widgets/native_panels.py`
- Modify: `yolo_action_detection/src/ui/main_window.py`
- Test: `yolo_action_detection/tests/test_ui_redesign.py`

- [ ] **Step 1: Write the failing tests**

In `test_ui_redesign.py`:

```python
def test_recognition_item_shows_fail_retry_progress(qapp):
    item = RecognitionListItem(2, "导热垫", required_count=3)
    item.set_fail_retry_progress(3, 10, visible=True)
    assert item.fail_retry_text() == "3 / 10"
    item.set_fail_retry_progress(0, 10, visible=False)
    assert item.fail_retry_text() == ""


def test_fail_retry_keeps_second_step_visible_at_zero(qapp):
    config = ConfigManager()
    config.roi_inspection_enabled = False
    config.action_fail_retry_enabled = True
    config.action_fail_retry_frames = 10
    config.category_names = ["散热片", "导热垫", "", "", "", ""]
    config.category_counts = [1, 3, 1, 1, 1, 1]
    window = _window(config)
    window.step_engine = StepSequenceEngine(
        step_class_names=config.category_names,
        step_counts=config.get_runtime_category_counts(),
        enter_stable_frames=1,
        fail_retry_enabled=True,
        fail_retry_frames=10,
    )
    window.step_engine.start_round()
    first = window.step_engine.update([_detection("散热片")])
    window._update_step_display(first)
    second = window.step_engine.update([])
    window._update_step_display(second)
    assert window._step_cards[1].isHidden() is False
    assert window._step_cards[1].fail_retry_text() == "1 / 10"
```

Use the existing `_detection` / `_window` helpers in that file. If `_detection` is named differently, match the file.

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL，没有 `set_fail_retry_progress`。

- [ ] **Step 3: Implement card API**

In `RecognitionListItem.__init__`, after `_quantity_label`:

```python
        self._retry_label = QLabel("", self)
        self._retry_label.setMinimumWidth(70)
        self._retry_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._retry_label.setVisible(False)
        layout.addWidget(self._retry_label)
```

Add:

```python
    def set_fail_retry_progress(self, current: int, limit: int, *, visible: bool) -> None:
        if not visible:
            self._retry_label.setVisible(False)
            self._retry_label.setText("")
            return
        self._retry_label.setText(f"{max(0, int(current))} / {max(1, int(limit))}")
        self._retry_label.setVisible(True)

    def fail_retry_text(self) -> str:
        if not self._retry_label.isVisible():
            return ""
        return self._retry_label.text()
```

Style `_retry_label` like `_quantity_label` inside `set_step_state` (same border/background).

In `MainWindow._update_step_display`, replace the `should_show` block:

```python
                retry_enabled = bool(getattr(state, "fail_retry_enabled", False))
                retry_started = (
                    retry_enabled
                    and engine_idx == current_idx
                    and (
                        engine_idx != self._card_to_engine_index(0)
                        or ss.fail_retry_count > 0
                        or ss.current_count > 0
                    )
                )
                # 第 2 步及以后一旦成为当前步即算已开始计次（含 0 个）。
                if retry_enabled and engine_idx == current_idx:
                    first_engine = self._card_to_engine_index(0)
                    retry_started = engine_idx != first_engine or ss.fail_retry_count > 0 or ss.current_count > 0
                card.set_fail_retry_progress(
                    int(getattr(state, "fail_retry_count", 0) if engine_idx == current_idx else ss.fail_retry_count),
                    int(getattr(state, "fail_retry_frames", 10)),
                    visible=retry_started and step_state not in (StepStatus.PASS,),
                )
                should_show = (
                    ss.current_count > 0
                    or step_state == StepStatus.PASS
                    or engine_idx == ng_step_idx
                    or retry_started
                )
```

Simplify `retry_started` to exactly the spec:

- 开关打开且该卡是当前等待步，并且（不是第 1 个已配置步，或第 1 步 `fail_retry_count > 0` 或 `current_count > 0`）

```python
                first_configured = next(
                    (i for i, step in enumerate(state.steps) if step.configured),
                    -1,
                )
                retry_started = bool(
                    getattr(state, "fail_retry_enabled", False)
                    and engine_idx == current_idx
                    and (
                        engine_idx != first_configured
                        or ss.fail_retry_count > 0
                        or ss.current_count > 0
                    )
                )
```

For NG notice, replace the hardcoded 顺序异常 when reason is quantity:

```python
            elif state.round_result == RoundResult.ACTION_NG:
                notice = state.action_ng_reason or "检测到顺序异常，请按配置顺序重新执行当前步骤。"
                if "数量不符" in notice or "未检出" in notice:
                    self._set_notice(notice)
                else:
                    self._set_notice("检测到顺序异常，请按配置顺序重新执行当前步骤。")
```

- [ ] **Step 4: Run tests**

```
..\.venv\Scripts\python.exe -m pytest tests/test_ui_redesign.py::test_recognition_item_shows_fail_retry_progress tests/test_ui_redesign.py::test_fail_retry_keeps_second_step_visible_at_zero tests/test_ui_redesign.py::test_sequence_waiting_items_stay_hidden_until_detected -q
```

Expected: PASS。最后一项确认开关默认关时第 2 步仍隐藏。

- [ ] **Step 5: Commit** if asked.

---

### Task 8: 证据 metadata

**Files:**
- Modify: `yolo_action_detection/src/detection_logging/fail_evidence.py`
- Modify: `yolo_action_detection/src/ui/main_window.py`
- Test: `yolo_action_detection/tests/test_fail_evidence_retry.py`

- [ ] **Step 1: Write the failing test**

Create `yolo_action_detection/tests/test_fail_evidence_retry.py`:

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detection_logging.fail_evidence import FailEvidenceContext, FailEvidenceSaver


def test_quantity_ng_metadata_includes_counts(tmp_path):
    saver = FailEvidenceSaver(enabled=True, base_dir=tmp_path)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    context = FailEvidenceContext(
        round_id=3,
        action_ng_step=1,
        step_name="导热垫",
        source_frame_id=9,
        actual_count=2,
        required_count=3,
        fail_retry_count=10,
        ng_reason="步骤2(导热垫)连续10帧类别未检出或数量不符（实际2/要求3）",
    )
    folder = saver.save_fail_event(frame, context)
    assert folder is not None
    payload = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert payload["actual_count"] == 2
    assert payload["required_count"] == 3
    assert payload["fail_retry_count"] == 10
    assert "数量不符" in payload["ng_reason"]
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL，`FailEvidenceContext` 无这些字段。

- [ ] **Step 3: Implement**

Add optional fields to `FailEvidenceContext`:

```python
    actual_count: int | None = None
    required_count: int | None = None
    fail_retry_count: int | None = None
    ng_reason: str = ""
```

In `_metadata`, after detections:

```python
        if context.actual_count is not None:
            metadata["actual_count"] = int(context.actual_count)
        if context.required_count is not None:
            metadata["required_count"] = int(context.required_count)
        if context.fail_retry_count is not None:
            metadata["fail_retry_count"] = int(context.fail_retry_count)
        if context.ng_reason:
            metadata["ng_reason"] = str(context.ng_reason)
```

In `MainWindow._save_fail_evidence`, after resolving `step_idx`:

```python
            ng_step = None
            if state is not None and 0 <= step_idx < len(getattr(state, "steps", []) or []):
                ng_step = state.steps[step_idx]
            context = FailEvidenceContext(
                round_id=int(getattr(state, "round_id", 0)),
                action_ng_step=step_idx,
                step_name=step_name,
                source_frame_id=source_frame_id,
                detections=list(data.get("detections", []) or []),
                model_path=str(data.get("model_path", "")),
                timestamp=float(data.get("timestamp") or time.time()),
                actual_count=None if ng_step is None else int(ng_step.current_count),
                required_count=None if ng_step is None else int(ng_step.required_count),
                fail_retry_count=None if ng_step is None else int(ng_step.fail_retry_count),
                ng_reason=str(getattr(state, "action_ng_reason", "") or ""),
            )
```

Directory naming unchanged. Existing ROI evidence tests must still pass because new fields default empty and are omitted when `None`/empty.

- [ ] **Step 4: Run tests**

```
..\.venv\Scripts\python.exe -m pytest tests/test_fail_evidence_retry.py tests/test_roi_runtime_integration.py -q --tb=line
```

If the ROI file is slow, run only evidence-related tests in it. Expected: PASS.

- [ ] **Step 5: Commit** if asked.

---

### Task 9: 全量回归

- [ ] **Step 1: Run sequential and config suites**

```
..\.venv\Scripts\python.exe -m pytest tests/test_fail_retry.py tests/test_step_quantity.py tests/test_fail_evidence_retry.py tests/test_ui_redesign.py tests/test_runtime_model_configuration.py tests/test_result_sound_playback.py -q
```

Expected: PASS.

- [ ] **Step 2: Run compile**

```
..\.venv\Scripts\python.exe -m compileall src
```

Expected: exit 0.

- [ ] **Step 3: Mark spec status**

In `docs/superpowers/specs/2026-08-24-sequential-fail-retry-design.md` change `状态：待实现` to `状态：已实现` only after Task 9 Step 1 passes.

---

## Self-review

**Spec coverage**

| Spec | Task |
| --- | --- |
| 配置字段默认关、1–100、不升 schema | Task 1 |
| 第 1 步 0 不计次 | Task 2 |
| 开关关不 NG | Task 2 |
| 第 2 步 0/偏少/偏多满 10 帧 NG、原因字符串 | Task 3 |
| 第 9 帧不对仍进行、第 10 帧对则 PASS 清零 | Task 3 |
| 第 1 步数量不对在第 1 步 NG | Task 3 |
| 同一帧数量 NG 优先于错序 | Task 4 |
| 错序在未满额时仍可用 | Task 4 |
| create_step_engine / 签名 / 重建 | Task 5 |
| 配置页开关与次数 | Task 6 |
| 卡片显示与 0 个仍可见、关时仍隐藏 | Task 7 |
| NG 文案不套「顺序异常」 | Task 7 |
| 证据 metadata | Task 8 |
| 现有数量/声音回归 | Task 9 |

**Placeholders:** none. Task 7 names `_detection` — implementer must match `test_ui_redesign.py` 现有 helper（该文件已有 `_detection` 或等价函数，以文件为准）。

**Types:** `fail_retry_enabled: bool`、`fail_retry_frames: int`、`fail_retry_count: int` 在引擎、状态、配置、证据中一致。
