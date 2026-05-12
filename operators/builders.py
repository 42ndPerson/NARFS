from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union

from network_requirement import NetworkRequirement
from operator_base import Operator


# An operator paired with the network requirements it emits at this position.
# `tuple` matches the signature shape the brief specifies for layer builders.
OperatorWithReqs = tuple[Operator, list[NetworkRequirement]]


@dataclass(frozen=True)
class LayerRepeat:
    """Repeated block of (operator, requirements) tuples.

    Counterpart to operators.composite.Repeat but for the "with-requirements"
    pattern carried through ModelMapping.
    """

    count: int
    members: list[OperatorWithReqs] = field(default_factory=list)

    def expand(self) -> list[OperatorWithReqs]:
        return list(self.members) * self.count


PatternEntry = Union[OperatorWithReqs, LayerRepeat]


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


def expand_tagged_pattern(pattern: list[PatternEntry]) -> list[OperatorWithReqs]:
    out: list[OperatorWithReqs] = []
    for item in pattern:
        if isinstance(item, LayerRepeat):
            out.extend(item.expand())
        else:
            out.append(item)
    return out


def expand_pattern_with_layers(
    pattern: list[PatternEntry],
) -> list[tuple[OperatorWithReqs, LayerTag]]:
    """Expand a pattern, attaching a `LayerTag` to each operator.

    Layer indexing scheme: LayerRepeats consume a sequential range of layer
    indices starting from 0. With `[Repeat(3, dense), Repeat(58, moe)]`,
    dense iterations get layer indices 0..2 and MoE iterations get 3..60.
    All operators within the same iteration share the same layer index.

    Anchor classification: top-level (non-LayerRepeat) entries before any
    LayerRepeat get `anchor="first"`; entries after the last LayerRepeat
    get `anchor="last"`; entries between LayerRepeats (rare) default to
    "first" but the layer-aware partitioner treats those defensively.
    """
    # Locate the index of the first and last LayerRepeat to classify anchors.
    repeat_positions = [i for i, item in enumerate(pattern) if isinstance(item, LayerRepeat)]
    first_repeat = repeat_positions[0] if repeat_positions else None
    last_repeat = repeat_positions[-1] if repeat_positions else None

    out: list[tuple[OperatorWithReqs, LayerTag]] = []
    next_layer_idx = 0
    for pos, item in enumerate(pattern):
        if isinstance(item, LayerRepeat):
            for layer_iter in range(item.count):
                idx = next_layer_idx + layer_iter
                tag = LayerTag(layer_index=idx, anchor=None)
                for op_with_reqs in item.members:
                    out.append((op_with_reqs, tag))
            next_layer_idx += item.count
        else:
            if first_repeat is None or pos < first_repeat:
                anchor = "first"
            elif pos > last_repeat:
                anchor = "last"
            else:
                # Between repeats — defensive default. None of our shipped
                # builders produce this case.
                anchor = "first"
            out.append((item, LayerTag(layer_index=None, anchor=anchor)))
    return out


def split_pattern(
    expanded: list[OperatorWithReqs],
) -> tuple[list[Operator], list[list[NetworkRequirement]]]:
    """Split parallel lists out of a fully-expanded tagged pattern."""
    ops: list[Operator] = [t[0] for t in expanded]
    reqs: list[list[NetworkRequirement]] = [t[1] for t in expanded]
    return ops, reqs


def split_pattern_with_layers(
    expanded: list[tuple[OperatorWithReqs, LayerTag]],
) -> tuple[list[Operator], list[list[NetworkRequirement]], list[LayerTag]]:
    """Split parallel lists from `expand_pattern_with_layers` output."""
    ops: list[Operator] = [t[0][0] for t in expanded]
    reqs: list[list[NetworkRequirement]] = [t[0][1] for t in expanded]
    tags: list[LayerTag] = [t[1] for t in expanded]
    return ops, reqs, tags
