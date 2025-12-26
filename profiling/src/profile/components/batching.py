from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BatchDimension:
    """Descriptor for a batch-related variable (input or output)."""

    key: Optional[str] = None
    default: Optional[int] = None
    description: Optional[str] = None


@dataclass
class BatchingConfig:
    """Declarative batch metadata for a component."""

    input: Optional[BatchDimension] = None
    output: Optional[BatchDimension] = None


__all__ = ["BatchDimension", "BatchingConfig"]
