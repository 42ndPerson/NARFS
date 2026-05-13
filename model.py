from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from network_requirement import ParallelismConfig
    from operators.pattern import PatternEntry


@dataclass(frozen=True)
class ArchitectureConfig:
    """Marker base class for per-architecture config (LlamaConfig, V3Config)."""


@dataclass(frozen=True)
class Model(ABC):
    """A model is a structural description of an architecture, parameterized
    by an `ArchitectureConfig`. Subclasses (e.g. `LlamaModel`, `V3Model`)
    implement `forward_sequence` to produce the operator pattern for a given
    execution phase.

    `name` and `role` identify a particular model *instance* in an RLHF
    pipeline (actor / critic / reward model); the same `LlamaModel` subclass
    can be instantiated multiple times under different roles.
    """

    name: str
    role: str
    architecture: ArchitectureConfig

    @abstractmethod
    def forward_sequence(
        self, phase: str, parallelism: "ParallelismConfig"
    ) -> "list[PatternEntry]":
        """Operator pattern for a single forward pass under `phase`.

        `phase` is one of {"prefill", "decode", "training"}. The returned
        sequence carries operators with full-model shapes (per-rank division
        is handled by the operator's `parallelism_role` at compute time)
        plus per-position NetworkRequirements that drive collective ops.
        """
        ...
