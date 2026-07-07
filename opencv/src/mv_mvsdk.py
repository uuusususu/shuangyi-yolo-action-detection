import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Optional, Tuple


_MVSDK_MODULE: Optional[ModuleType] = None
_MVSDK_LOAD_ERROR: Optional[str] = None


def _load_mvsdk_module() -> Tuple[Optional[ModuleType], Optional[str]]:
    try:
        base_dir = Path(__file__).resolve().parents[1]
        mvsdk_path = base_dir / "python_demo" / "mvsdk.py"
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
