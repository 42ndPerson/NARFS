from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from operators.pattern import PatternEntry


@dataclass(frozen=True)
class ArchitectureConfig:
    """Marker base class for per-architecture config (LlamaConfig, V3Config)."""


@dataclass(frozen=True)
class Model(ABC):
    """A model is a structural description of an architecture, parameterized
    by an `ArchitectureConfig`. Subclasses (e.g. `LlamaModel`, `V3Model`)
    implement `forward_sequence` to produce the labeled operator pattern
    for a given execution phase.

    `name` and `role` identify a particular model *instance* in an RLHF
    pipeline (actor / critic / reward model); the same `LlamaModel` subclass
    can be instantiated multiple times under different roles.

    `forward_sequence` is parallelism-agnostic. It emits the architectural
    structure as a list of `LabeledOp`s wrapped in `LayerRepeat`s where
    appropriate. Sharding decisions are owned by the `ParallelismScheme`
    that consumes the labeled pattern.
    """

    name: str
    role: str
    architecture: ArchitectureConfig

    @abstractmethod
    def forward_sequence(self, phase: str) -> "list[PatternEntry]":
        """Labeled operator pattern for a single forward pass under `phase`.

        `phase` is one of {"prefill", "decode", "training"}. The returned
        sequence carries `LabeledOp` entries (and `LayerRepeat` wrappers)
        where each operator is tagged with a canonical label that the
        strategy uses to decide sharding and collective emission.
        """
        ...
