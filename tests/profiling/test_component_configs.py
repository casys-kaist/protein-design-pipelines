from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
from omegaconf import OmegaConf

from profile.components.batching import BatchDimension, BatchingConfig


def _strip_target(data: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in data.items() if key != "_target_"}


@pytest.mark.parametrize("path", sorted(Path("profiling/configs/components").glob("*.yaml")))
def test_component_batching_config_schema(path: Path) -> None:
    cfg = OmegaConf.load(path)
    batching = cfg.get("batching")
    if batching is None:
        return

    data = OmegaConf.to_container(batching, resolve=False)
    assert isinstance(data, dict), f"{path} batching must be a mapping"

    data = _strip_target(data)
    assert "container" not in data, f"{path} batching must not contain container settings"

    for key in ("input", "output"):
        if key not in data:
            continue
        raw = data[key]
        if raw is None:
            continue
        assert isinstance(raw, dict), f"{path} batching.{key} must be a mapping"
        data[key] = BatchDimension(**_strip_target(raw))

    BatchingConfig(**data)
