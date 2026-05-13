from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Optional, Union

from hardware import HardwareSpec
from network_requirement import NetworkRequirement
from operator_base import Operator
from workload_context import WorkloadContext


@dataclass(frozen=True)
class LabeledOp:
    """An operator emitted by an Architecture's `forward_sequence`, tagged
    with a semantic label that the Strategy uses to decide sharding and
    collective emission.

    Labels are the vocabulary between Architecture and Strategy. The
    architecture emits structural operators with canonical labels (e.g.
    "qkv", "attn_out", "ffn_down", "lm_head"); the strategy looks them up
    in its label-to-decision table.
    """
    label: str
    operator: Operator


class OperatorWithReqs(NamedTuple):
    """A parallelism-pure operator paired with the strategy's per-rank
    decoration. Produced by the Strategy after walking the LabeledOp
    pattern.

    - `operator`: the parallelism-agnostic operator (full-model shapes).
    - `requirements`: collectives that overlap with this op's kernel time.
    - `compute_time_fn(ctx, hw)`: per-rank kernel time. The strategy bakes
      the role-aware division (typically `op.kernel_time(ctx, hw) / tp`).
    - `per_rank_parameter_bytes`: per-rank weight footprint after the
      scheme's role (used by DP-AR sizing and Adam optimizer sizing).
    """

    operator: Operator
    requirements: list[NetworkRequirement]
    compute_time_fn: Callable[[WorkloadContext, HardwareSpec], float]
    per_rank_parameter_bytes: int = 0


@dataclass(frozen=True)
class LayerRepeat:
    """Repeated layer block. `members` may be either `LabeledOp`s (as
    emitted by the architecture) or `OperatorWithReqs` tuples (as produced
    by the strategy after attaching requirements). The expand helpers
    handle both shapes.
    """

    count: int
    members: list = field(default_factory=list)

    def expand(self) -> list:
        return list(self.members) * self.count


# Pattern entries either come direct from the architecture (LabeledOp) or
# from the strategy post-decoration (OperatorWithReqs); plus LayerRepeat
# wrappers around either.
PatternEntry = Union[LabeledOp, OperatorWithReqs, LayerRepeat]


@dataclass(frozen=True)
class LayerTag:
    """Layer affinity tag attached to an expanded operator.

    `layer_index` is set for operators that came from a `LayerRepeat`
    (sequential across all repeats in the pattern). Operators outside any
    LayerRepeat — embeddings, final norm, LM head — get `layer_index=None`
    and `anchor` distinguishes pattern position relative to the repeats:

    - `anchor="first"` → appears before every LayerRepeat (e.g. embedding).
      Layer-aware partitioner pins these to the first PP stage.
    - `anchor="last"`  → appears after every LayerRepeat (e.g. final norm,
      LM head). Pinned to the last PP stage.
    - `anchor=None`    → none of the above (defensive default).
    """

    layer_index: Optional[int] = None
    anchor: Optional[str] = None


def expand_tagged_pattern(pattern: list[PatternEntry]) -> list:
    """Expand a pattern, flattening LayerRepeats. Returns the same entry
    types as the input (LabeledOp or OperatorWithReqs)."""
    out = []
    for item in pattern:
        if isinstance(item, LayerRepeat):
            out.extend(item.expand())
        else:
            out.append(item)
    return out


def expand_pattern_with_layers(
    pattern: list[PatternEntry],
) -> list[tuple[OperatorWithReqs, LayerTag]]:
    """Expand a pattern of `OperatorWithReqs` (and LayerRepeats of them),
    attaching a `LayerTag` to each operator.

    Layer indexing scheme: LayerRepeats consume a sequential range of layer
    indices starting from 0. With `[Repeat(3, dense), Repeat(58, moe)]`,
    dense iterations get layer indices 0..2 and MoE iterations get 3..60.
    All operators within the same iteration share the same layer index.

    Anchor classification: top-level (non-LayerRepeat) entries before any
    LayerRepeat get `anchor="first"`; entries after the last LayerRepeat
    get `anchor="last"`; entries between LayerRepeats (rare) default to
    "first".
    """
    repeat_positions = [i for i, item in enumerate(pattern) if isinstance(item, LayerRepeat)]
    first_repeat = repeat_positions[0] if repeat_positions else -1
    last_repeat = repeat_positions[-1] if repeat_positions else -1

    out: list[tuple[OperatorWithReqs, LayerTag]] = []
    next_layer_idx = 0
    for pos, item in enumerate(pattern):
        if isinstance(item, LayerRepeat):
            for layer_iter in range(item.count):
                idx = next_layer_idx + layer_iter
                tag = LayerTag(layer_index=idx, anchor=None)
                for member in item.members:
                    assert not isinstance(member, LabeledOp), (
                        "expand_pattern_with_layers expects strategy-decorated "
                        "OperatorWithReqs entries, not raw LabeledOps"
                    )
                    out.append((member, tag))
            next_layer_idx += item.count
        else:
            assert not isinstance(item, LabeledOp), (
                "expand_pattern_with_layers expects strategy-decorated "
                "OperatorWithReqs entries, not raw LabeledOps"
            )
            if first_repeat < 0 or pos < first_repeat:
                anchor = "first"
            elif pos > last_repeat:
                anchor = "last"
            else:
                anchor = "first"
            out.append((item, LayerTag(layer_index=None, anchor=anchor)))
    return out


def split_pattern(
    expanded: list,
) -> tuple[list[Operator], list[list[NetworkRequirement]]]:
    """Split parallel lists out of a fully-expanded pattern.

    Polymorphic over entry type: `OperatorWithReqs` tuples unpack into
    (op, reqs); `LabeledOp` entries yield (operator, empty reqs). Used by
    tests inspecting both labeled (architecture-output) and decorated
    (strategy-output) patterns.
    """
    ops: list[Operator] = []
    reqs: list[list[NetworkRequirement]] = []
    for item in expanded:
        if isinstance(item, LabeledOp):
            ops.append(item.operator)
            reqs.append([])
        else:
            ops.append(item.operator)
            reqs.append(item.requirements)
    return ops, reqs


def split_pattern_with_layers(
    expanded: list[tuple[OperatorWithReqs, LayerTag]],
) -> tuple[list[Operator], list[list[NetworkRequirement]], list[LayerTag]]:
    """Split parallel lists from `expand_pattern_with_layers` output."""
    ops: list[Operator] = [t[0].operator for t in expanded]
    reqs: list[list[NetworkRequirement]] = [t[0].requirements for t in expanded]
    tags: list[LayerTag] = [t[1] for t in expanded]
    return ops, reqs, tags
