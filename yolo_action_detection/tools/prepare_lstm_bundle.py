"""把已经训好的连续 LSTM 复制成产线模型包。不重训，也不覆盖原来的 run。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
ROOT = Path(__file__).resolve().parents[2]
for item in (str(ROOT), str(SRC)):
    if item not in sys.path:
        sys.path.insert(0, item)

from lstm_action.bundle import prepare_runtime_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="生成连续 LSTM 动作模型包")
    parser.add_argument("source", type=Path, help="训练轮次目录，里面有 best_lstm.pt 和 contract.json")
    parser.add_argument("dest", type=Path, help="新的部署目录，不能已经存在")
    parser.add_argument("--mapping", type=Path, help="已确认的 ID 到名称 JSON，例如 {\"idle\": \"空闲\", \"c17\": \"包装\"}")
    args = parser.parse_args()
    mapping = None
    if args.mapping is not None:
        mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    prepare_runtime_bundle(args.source, args.dest, mapping)
    print(f"已写成 {args.dest}")


if __name__ == "__main__":
    main()
