from __future__ import annotations

import argparse
import json
from pathlib import Path


CURRENT_CONFIG_SCHEMA_VERSION = 4
LEGACY_REGION_KEYS = {
    "first_category_region_check_enabled",
    "pcb_inspection_enabled",
    "pcb_class_name",
    "pcb_component_class_names",
    "pcb_fail_stable_frames",
    "pcb_round_interval_seconds",
    "pcb_assignment_margin_ratio",
    "roi_template_min_matches",
}


def prepare_portable_config(
    source_path: str | Path,
    destination_path: str | Path,
    bundled_model_dir: str | Path,
) -> Path:
    source = Path(source_path)
    destination = Path(destination_path)
    model_dir = Path(bundled_model_dir)

    payload = json.loads(source.read_text(encoding="utf-8"))
    version = payload.get("config_schema_version")
    if type(version) is not int or version != CURRENT_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported config_schema_version: {version!r}; "
            f"expected {CURRENT_CONFIG_SCHEMA_VERSION}"
        )
    legacy_keys = sorted(LEGACY_REGION_KEYS.intersection(payload))
    if legacy_keys:
        raise ValueError(f"Legacy region fields are not allowed: {', '.join(legacy_keys)}")
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
