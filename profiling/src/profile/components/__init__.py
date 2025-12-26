from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf

from profile import component_output_root, component_raw_root
from profile.config import paths_as_dict

from .batching import BatchDimension, BatchingConfig
from .common import ComponentSpec

_CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs" / "components"
_PATHS_GROUP = "paths"
_PATHS_CONFIG_NAME = "defaults"
_PATHS_REGISTERED = False


def _register_paths_config() -> None:
    global _PATHS_REGISTERED
    if _PATHS_REGISTERED:
        return
    cs = ConfigStore.instance()
    cs.store(group=_PATHS_GROUP, name=_PATHS_CONFIG_NAME, node=paths_as_dict(), package="paths")
    _PATHS_REGISTERED = True


def _register_resolvers() -> None:
    if not OmegaConf.has_resolver("component_output_root"):
        OmegaConf.register_new_resolver(
            "component_output_root",
            lambda name: str(component_output_root(name)),
            use_cache=True,
        )
    if not OmegaConf.has_resolver("component_raw_root"):
        OmegaConf.register_new_resolver(
            "component_raw_root",
            lambda name: str(component_raw_root(name)),
            use_cache=True,
        )


def _iter_component_configs() -> Iterable[Path]:
    if not _CONFIG_ROOT.exists():
        raise FileNotFoundError(f"Component config directory missing: {_CONFIG_ROOT}")
    return sorted(path for path in _CONFIG_ROOT.glob("*.yaml") if path.is_file())


def list_component_names() -> List[str]:
    return [path.stem for path in _iter_component_configs()]


def load_component_spec(name: str) -> ComponentSpec:
    """Instantiate a component spec by name without relying on a global registry."""
    cfg_path = _CONFIG_ROOT / f"{name}.yaml"
    if not cfg_path.exists():
        raise KeyError(f"Unknown component '{name}'")
    cfg = OmegaConf.load(cfg_path)
    context = OmegaConf.create({"paths": paths_as_dict()})
    wrapper = OmegaConf.create({"spec": cfg})
    merged = OmegaConf.merge(context, wrapper)
    return instantiate(merged.spec, _convert_="all")


_register_resolvers()
_register_paths_config()

__all__ = [
    "BatchDimension",
    "BatchingConfig",
    "ComponentSpec",
    "load_component_spec",
    "list_component_names",
]
