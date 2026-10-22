"""连续 LSTM 的部署模型包。复制已有权重，不重训，也不猜测动作名称。"""
from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from lstm_workbench.features import FEATURE_CODE_VERSION, feature_size, hands_in_training, safe_feature_std
from lstm_workbench.model_registry import file_sha256

SUPPORTED_DECODE_VERSION = 1


@dataclass
class RuntimeBundle:
    directory: Path
    contract: dict
    manifest: dict
    class_ids: list[str]
    names_by_id: dict[str, str]
    ids_by_name: dict[str, str]
    part_names: list[str]
    required_parts: dict[str, list[str]]
    time_base_fps: float
    min_action_seconds: float
    feature_mean: list[float]
    feature_std: list[float]
    yolo_conf_threshold: float
    hands_in_training: bool
    weights_sha256: str
    contract_sha256: str
    model: object = field(repr=False)


def uses_lstm_sequence(config) -> bool:
    """ROI 关闭且步骤顺序开启时，步骤只由连续 LSTM 动作结算。"""
    return (
        not bool(getattr(config, "roi_inspection_enabled", False))
        and bool(getattr(config, "action_order_constraint_enabled", False))
    )


def prepare_runtime_bundle(
    source: Path,
    dest: Path,
    mapping: dict[str, str] | None = None,
) -> Path:
    """把某一轮权重复制到新目录，并写上已确认的名称映射。不覆盖源目录。"""
    source = Path(source)
    dest = Path(dest)
    contract_path = source / "contract.json"
    weight_path = source / "best_lstm.pt"
    if not contract_path.is_file() or not weight_path.is_file():
        raise ValueError("源目录里没有 contract.json 或 best_lstm.pt")
    if dest.exists():
        raise ValueError("部署目录已存在，不覆盖已有模型包")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    class_ids = [str(item) for item in contract.get("class_ids") or []]
    if not class_ids or "idle" not in class_ids:
        raise ValueError("契约没有空闲类别，不能做部署包")
    names = _mapping_from_source(source, class_ids, mapping)
    dest.mkdir(parents=True)
    shutil.copy2(weight_path, dest / "best_lstm.pt")
    shutil.copy2(contract_path, dest / "contract.json")
    manifest = {
        "manifest_version": 1,
        "action_classes": [{"id": class_id, "name": names[class_id]} for class_id in class_ids],
        "contract_sha256": file_sha256(dest / "contract.json"),
        "weights_sha256": file_sha256(dest / "best_lstm.pt"),
        "mapping_source": "confirmed",
    }
    (dest / "runtime_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return dest


def bundle_directory_for_model_file(selected: str | Path) -> Path:
    """选中训练文件夹，或文件夹里的 best_lstm.pt。有手部、没手部都能加载。"""
    path = Path(selected)
    directory = path.parent if path.is_file() else path
    missing = [
        name for name in ("best_lstm.pt", "contract.json", "runtime_manifest.json")
        if not (directory / name).is_file()
    ]
    if missing:
        raise ValueError("这个文件夹还要有：" + "、".join(missing))
    return directory


def load_runtime_bundle(
    directory: Path,
    *,
    category_names: list[str],
    yolo_path: str = "",
    yolo_conf: float | None = None,
    hands_sha: str | None = None,
    hands_complexity: int | None = None,
    yolo_class_names: list[str] | None = None,
    step_counts: list[int] | None = None,
) -> RuntimeBundle:
    """校验部署包。失败时抛出带具体原因的 ValueError，不加载权重。"""
    directory = Path(directory)
    contract_path = directory / "contract.json"
    weight_path = directory / "best_lstm.pt"
    manifest_path = directory / "runtime_manifest.json"
    if not contract_path.is_file() or not weight_path.is_file() or not manifest_path.is_file():
        raise ValueError("动作模型包不完整：需要 best_lstm.pt、contract.json 和 runtime_manifest.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _check_manifest(manifest, contract, contract_path, weight_path)
    _check_contract_shape(contract)
    names_by_id, ids_by_name = _names(manifest, contract)
    _check_categories(category_names, ids_by_name, names_by_id)
    _check_counts(category_names, step_counts)
    part_names = [str(name) for name in contract.get("part_class_names") or []]
    required = _check_required_parts(contract, part_names)
    if yolo_class_names is not None and list(yolo_class_names) != part_names:
        raise ValueError("当前零件类别顺序和契约不一致")
    recorded_yolo = str(contract.get("yolo_sha256") or "")
    if not recorded_yolo:
        raise ValueError("契约没有 YOLO 指纹")
    if yolo_path:
        if not Path(yolo_path).is_file():
            raise ValueError(f"当前 YOLO 模型不存在：{yolo_path}")
        if file_sha256(Path(yolo_path)) != recorded_yolo:
            raise ValueError("当前 YOLO 模型和契约不是同一个文件")
    trained_conf = float(contract["yolo_conf_threshold"])
    if yolo_conf is not None and abs(float(yolo_conf) - trained_conf) > 1e-6:
        raise ValueError("当前置信度阈值和训练契约不同")
    _check_hands(contract, hands_sha, hands_complexity)
    model = _load_weights(contract, weight_path, len(contract["class_ids"]))
    return RuntimeBundle(
        directory=directory,
        contract=contract,
        manifest=manifest,
        class_ids=[str(item) for item in contract["class_ids"]],
        names_by_id=names_by_id,
        ids_by_name=ids_by_name,
        part_names=part_names,
        required_parts=required,
        time_base_fps=float(contract["time_base_fps"]),
        min_action_seconds=float(contract["min_action_seconds"]),
        feature_mean=[float(item) for item in contract["feature_mean"]],
        feature_std=[float(item) for item in safe_feature_std(contract["feature_std"])],
        yolo_conf_threshold=trained_conf,
        hands_in_training=hands_in_training(contract),
        weights_sha256=str(manifest["weights_sha256"]),
        contract_sha256=str(manifest["contract_sha256"]),
        model=model,
    )


def _mapping_from_source(source: Path, class_ids: list[str], mapping: dict[str, str] | None) -> dict[str, str]:
    if mapping is not None:
        names = {str(key): str(value).strip() for key, value in mapping.items()}
    else:
        path = source / "runtime_manifest.json"
        if not path.is_file():
            raise ValueError("缺少已确认的动作名称映射，不能按配置顺序猜测")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        names = {
            str(item.get("id")): str(item.get("name") or "").strip()
            for item in manifest.get("action_classes") or []
        }
    missing = [class_id for class_id in class_ids if not names.get(class_id)]
    if missing:
        raise ValueError(f"名称映射缺少 {missing[0]}")
    values = [names[class_id] for class_id in class_ids]
    if len(values) != len(set(values)):
        raise ValueError("动作名称有重复")
    if names.get("idle") in {names[class_id] for class_id in class_ids if class_id != "idle"}:
        raise ValueError("空闲名称不能和动作重名")
    return {class_id: names[class_id] for class_id in class_ids}


def _check_manifest(manifest: dict, contract: dict, contract_path: Path, weight_path: Path) -> None:
    if int(manifest.get("manifest_version") or 0) != 1:
        raise ValueError("部署清单版本不受支持")
    if file_sha256(contract_path) != str(manifest.get("contract_sha256") or ""):
        raise ValueError("契约文件和清单指纹不一致")
    if file_sha256(weight_path) != str(manifest.get("weights_sha256") or ""):
        raise ValueError("权重文件和清单指纹不一致")
    class_ids = [str(item) for item in contract.get("class_ids") or []]
    listed = [str(item.get("id")) for item in manifest.get("action_classes") or []]
    if listed != class_ids:
        raise ValueError("部署清单没有按契约顺序列出全部动作")


def _check_contract_shape(contract: dict) -> None:
    network = str(contract.get("network") or "")
    if network != "lstm" or contract.get("train_unit") != "timeline":
        raise ValueError("只支持连续训练的 LSTM，不支持 MS-TCN 或旧滑窗模型")
    if int(contract.get("decode_version") or 0) != SUPPORTED_DECODE_VERSION:
        raise ValueError("解码版本不受支持")
    if str(contract.get("feature_code_version") or "") != FEATURE_CODE_VERSION:
        raise ValueError("特征代码版本不一致")
    class_ids = [str(item) for item in contract.get("class_ids") or []]
    part_names = [str(name) for name in contract.get("part_class_names") or []]
    width = feature_size(len(part_names), hands=hands_in_training(contract))
    if int(contract.get("input_size") or 0) != width:
        raise ValueError("特征长度和零件类别数量不一致")
    if int(contract.get("hidden_size") or 0) != 128 or int(contract.get("num_layers") or 0) != 2:
        raise ValueError("网络结构和连续 LSTM 不一致")
    mean = contract.get("feature_mean") or []
    std = contract.get("feature_std") or []
    if len(mean) != width or len(std) != width:
        raise ValueError("标准化向量长度和特征长度不一致")
    if not all(math.isfinite(float(item)) for item in list(mean) + list(std)):
        raise ValueError("标准化向量含有无效数字")
    if float(contract.get("time_base_fps") or 0) <= 0:
        raise ValueError("时间基必须大于 0")
    if float(contract.get("min_action_seconds") or 0) <= 0:
        raise ValueError("最短动作时长必须大于 0")
    if "idle" not in class_ids:
        raise ValueError("契约没有空闲类别")


def _names(manifest: dict, contract: dict) -> tuple[dict[str, str], dict[str, str]]:
    class_ids = [str(item) for item in contract["class_ids"]]
    names_by_id: dict[str, str] = {}
    for item in manifest.get("action_classes") or []:
        names_by_id[str(item.get("id"))] = str(item.get("name") or "").strip()
    if [class_id for class_id in class_ids if not names_by_id.get(class_id)]:
        raise ValueError("名称映射没有覆盖契约里的全部类别")
    if len(set(names_by_id.values())) != len(names_by_id):
        raise ValueError("动作名称有重复")
    ids_by_name = {name: class_id for class_id, name in names_by_id.items()}
    return names_by_id, ids_by_name


def _check_categories(category_names: list[str], ids_by_name: dict[str, str], names_by_id: dict[str, str]) -> None:
    configured = [str(name).strip() for name in category_names if str(name).strip()]
    if len(configured) != len(set(configured)):
        raise ValueError("配置的动作名称有重复")
    idle_name = names_by_id.get("idle", "空闲")
    model_names = {name for name in ids_by_name if name != idle_name}
    if set(configured) != model_names:
        missing = sorted(model_names - set(configured))
        extra = sorted(set(configured) - model_names)
        detail = ""
        if extra:
            detail = f"配置多了模型没有的 {extra[0]}"
        elif missing:
            detail = f"配置缺少模型里的 {missing[0]}"
        raise ValueError(detail or "配置动作和模型动作不一致")


def _check_counts(category_names: list[str], step_counts: list[int] | None) -> None:
    if step_counts is None:
        return
    names = [str(name).strip() for name in category_names]
    for index, name in enumerate(names):
        if not name:
            continue
        count = step_counts[index] if index < len(step_counts) else 1
        if int(count) != 1:
            raise ValueError(f"动作步骤「{name}」的数量必须为 1，不能把数量当成重复次数")


def _check_required_parts(contract: dict, part_names: list[str]) -> dict[str, list[str]]:
    raw = contract.get("required_parts")
    if not isinstance(raw, dict):
        raise ValueError("契约没有 required_parts")
    known = set(part_names)
    checked: dict[str, list[str]] = {}
    for class_id in contract["class_ids"]:
        class_id = str(class_id)
        if class_id == "idle":
            continue
        parts = [str(name) for name in raw.get(class_id) or []]
        if not parts:
            raise ValueError(f"动作 {class_id} 没有零件要求，不能按纯动作直接放行")
        unknown = [name for name in parts if name not in known]
        if unknown:
            raise ValueError(f"动作 {class_id} 要求了不存在的零件 {unknown[0]}")
        checked[class_id] = parts
    return checked


def _check_hands(contract: dict, hands_sha: str | None, hands_complexity: int | None) -> None:
    if not hands_in_training(contract):
        return
    recorded = str(contract.get("hands_sha256") or "")
    if not recorded:
        raise ValueError("契约没有手部模型指纹")
    if hands_sha is None or hands_complexity is None:
        from lstm_workbench.model_registry import probe_hands

        status = probe_hands()
        if not status.get("ok"):
            raise ValueError(status.get("error") or "手部模型还不能核对指纹")
        if hands_sha is None:
            hands_sha = str(status.get("sha256") or "")
        if hands_complexity is None and status.get("model_complexity") is not None:
            hands_complexity = int(status["model_complexity"])
    if str(hands_sha) != recorded:
        raise ValueError("当前手部模型和契约不是同一个文件")
    recorded_complexity = contract.get("hands_model_complexity")
    if recorded_complexity is not None and hands_complexity is not None:
        if int(recorded_complexity) != int(hands_complexity):
            raise ValueError("手部模型复杂度和契约不一致")


def _load_weights(contract: dict, weight_path: Path, class_count: int):
    import torch
    from lstm_workbench.model import ActionLSTM

    model = ActionLSTM(
        int(contract["input_size"]),
        class_count,
        hidden_size=int(contract.get("hidden_size") or 128),
        num_layers=int(contract.get("num_layers") or 2),
        dropout=float(contract.get("dropout") or 0.0),
    )
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to("cpu")
    model.eval()
    return model
