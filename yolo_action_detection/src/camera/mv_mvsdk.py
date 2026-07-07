import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional, Tuple


_MVSDK_MODULE: Optional[ModuleType] = None
_MVSDK_LOAD_ERROR: Optional[str] = None


def _load_mvsdk_module() -> Tuple[Optional[ModuleType], Optional[str]]:
    try:
        source_base_dir = Path(__file__).resolve().parents[2]
        candidate_base_dirs = [source_base_dir]
        if getattr(sys, "frozen", False):
            candidate_base_dirs.insert(0, Path(getattr(sys, "_MEIPASS", source_base_dir)))
            candidate_base_dirs.insert(0, Path(sys.executable).resolve().parent)

        mvsdk_path = None
        for base_dir in candidate_base_dirs:
            candidate = base_dir / "python_demo" / "mvsdk.py"
            if candidate.exists():
                mvsdk_path = candidate
                break
        if mvsdk_path is None:
            searched = ", ".join(str(base / "python_demo" / "mvsdk.py") for base in candidate_base_dirs)
            return None, f"无法加载 mvsdk 模块，已搜索: {searched}"

        spec = importlib.util.spec_from_file_location("_bundled_mvsdk", str(mvsdk_path))
        if spec is None or spec.loader is None:
            return None, f"无法加载 mvsdk 模块: {mvsdk_path}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, None
    except Exception as e:
        return None, str(e)


def get_mvsdk() -> Optional[ModuleType]:
    global _MVSDK_MODULE
    global _MVSDK_LOAD_ERROR
    if _MVSDK_MODULE is not None or _MVSDK_LOAD_ERROR is not None:
        return _MVSDK_MODULE
    module, err = _load_mvsdk_module()
    _MVSDK_MODULE = module
    _MVSDK_LOAD_ERROR = err
    return _MVSDK_MODULE


def get_mvsdk_load_error() -> Optional[str]:
    if _MVSDK_MODULE is None and _MVSDK_LOAD_ERROR is None:
        get_mvsdk()
    return _MVSDK_LOAD_ERROR


def is_mvsdk_available() -> bool:
    return get_mvsdk() is not None
