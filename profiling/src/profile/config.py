from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, Union

from .layout import DATASET_ROOT, INPUT_ROOT, OUTPUT_ROOT, PROFILE_ROOT, STORAGE_ROOT

# Canonical config lives in profiling/configs. Keep a legacy fallback to avoid breaking older callers.
CONFIG_DIR = PROFILE_ROOT.parents[1] / "configs"
LEGACY_CONFIG_DIR = PROFILE_ROOT / "config"
PATH_CANDIDATES = (
    CONFIG_DIR / "paths.yaml",
    LEGACY_CONFIG_DIR / "paths.yaml",
)


def _select_paths_file() -> Path:
    for path in PATH_CANDIDATES:
        if path.exists():
            return path
    searched = ", ".join(str(p) for p in PATH_CANDIDATES)
    raise FileNotFoundError(f"Missing path configuration (searched: {searched})")


PATHS_FILE = _select_paths_file()

_CONTEXT = {
    "input_root": str(INPUT_ROOT),
    "output_root": str(OUTPUT_ROOT),
    "data_root": str(DATASET_ROOT),
    "storage_root": str(STORAGE_ROOT),
}


def _resolve(value: Any) -> Any:
    if isinstance(value, str):
        return Path(value.format(**_CONTEXT))
    if isinstance(value, dict):
        return {k: _resolve(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v) for v in value]
    return value


class _Namespace(dict):
    def __getattr__(self, item: str) -> Union["_Namespace", Path]:
        if item not in self:
            raise AttributeError(item)
        value = self[item]
        if isinstance(value, dict):
            value = _Namespace(value)
            self[item] = value
        return value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _load_paths() -> _Namespace:
    if not PATHS_FILE.exists():
        raise FileNotFoundError(f"Missing path configuration: {PATHS_FILE}")
    if PATHS_FILE == PATH_CANDIDATES[1]:
        print(f"[WARN] Using legacy path config at {PATHS_FILE}; prefer {PATH_CANDIDATES[0]}")
    data = yaml.safe_load(PATHS_FILE.read_text()) or {}
    resolved: Dict[str, Any] = _resolve(data)
    return _Namespace(resolved)


PATHS = _load_paths()


def _namespace_to_dict(ns: Union[_Namespace, Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in dict(ns).items():
        if isinstance(value, _Namespace):
            result[key] = _namespace_to_dict(value)
        elif isinstance(value, dict):
            result[key] = _namespace_to_dict(value)  # pragma: no cover - defensive
        elif isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def paths_as_dict() -> Dict[str, Any]:
    """Return a plain dict view of PATHS with string paths."""
    return _namespace_to_dict(PATHS)


__all__ = ["PATHS", "paths_as_dict"]
