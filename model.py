from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from network_requirement import ParallelismConfig
    from operators.builders import PatternEntry


@dataclass(frozen=True)
class ArchitectureConfig:
    """Marker base class for per-architecture config (LlamaConfig, V3Config)."""


# build_pattern signature: (architecture, phase, parallelism) -> list[PatternEntry]
# phase is one of {"prefill", "decode", "training"}.
BuildPatternFn = Callable[
    ["ArchitectureConfig", str, "ParallelismConfig"],
    "list[PatternEntry]",
]


@dataclass(frozen=True)
class Model:
    name: str
    role: str
    architecture: ArchitectureConfig
    build_pattern: BuildPatternFn
