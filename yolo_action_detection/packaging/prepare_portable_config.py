from __future__ import annotations

import argparse
import json
from pathlib import Path


def prepare_portable_config(
    source_path: str | Path,
    destination_path: str | Path,
    bundled_model_dir: str | Path,
) -> Path:
    source = Path(source_path)
    destination = Path(destination_path)
    model_dir = Path(bundled_model_dir)

    payload = json.loads(source.read_text(encoding="utf-8"))
    model_name = Path(str(payload.get("yolo_model_path", ""))).name
    if not model_name:
        raise ValueError("yolo_model_path is empty")
    if not (model_dir / model_name).is_file():
        raise FileNotFoundError(f"Bundled model not found: {model_dir / model_name}")

    payload["yolo_model_path"] = f"config/{model_name}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("bundled_model_dir")
    args = parser.parse_args()
    prepare_portable_config(args.source, args.destination, args.bundled_model_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
