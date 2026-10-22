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
