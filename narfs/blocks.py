"""L1 node types: the structure that lowerings emit and the evaluator times.

The core has eight node types and no knowledge of models, frameworks or
parallelism:

    leaves:      Compute, Collective, Transfer, Work
    combinators: Seq, Par, Repeat, OnDevices

Leaves carry physical work: real shapes, real bytes, and handles for device
sets. A handle is an opaque, hashable key; the layout layer (L2) binds it to
device arrays for one candidate, and only the L0 models read the bound arrays.

Nodes compare and hash by identity, so the evaluator can memoize a node that a
tree uses twice (for example a stage block in a pipeline's fill and in its
steady state).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

COLLECTIVE_KINDS = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")


def _tuple(node, field: str) -> None:
    value = getattr(node, field)
    if not isinstance(value, tuple):
        object.__setattr__(node, field, tuple(value))


# ------------------------------------------------------------------ leaves ---

@dataclass(frozen=True, eq=False)
class Compute:
    """GPU work on the devices of the enclosing scope, costed by the compute
    model on that scope's device class. `work` is an L0 work item (a kernel)."""
    work: Any
    name: str = ""


@dataclass(frozen=True, eq=False)
class Collective:
    """One collective on every instance of a communication group at once.

    `bytes` follows the nccl-tests convention, per rank: the buffer of an
    all-reduce, the output of an all-gather, the input of a reduce-scatter, and
    the total sent by one rank in an all-to-all. `lib` names the communication
    library profile that runs it."""
    kind: str
    comm: Any
    bytes: float
    lib: str = "nccl"
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind not in COLLECTIVE_KINDS:
            raise ValueError(f"unknown collective kind {self.kind!r}; expected one of {COLLECTIVE_KINDS}")
        if self.bytes < 0:
            raise ValueError("collective bytes must be >= 0")


@dataclass(frozen=True, eq=False)
class Transfer:
    """Point-to-point sends over bound (src, dst, weight) pairs. Each pair moves
    weight x `bytes`. `reverse` sends dst -> src, as for backward activations."""
    pairs: Any
    bytes: float = 1.0
    reverse: bool = False
    name: str = ""

    def __post_init__(self) -> None:
        if self.bytes < 0:
            raise ValueError("transfer bytes must be >= 0")


@dataclass(frozen=True, eq=False)
class Work:
    """`amount` on each bound resource: worker-seconds on a CPU pool, or an L0
    work item such as one engine host step."""
    resource: Any
    amount: Any
    name: str = ""


# -------------------------------------------------------------- combinators ---

@dataclass(frozen=True, eq=False)
class Seq:
    """Children one after another: t = sum of t, D = sum of D."""
    children: tuple
    name: str = ""

    def __post_init__(self) -> None:
        _tuple(self, "children")


@dataclass(frozen=True, eq=False)
class Par:
    """Children at the same time: the fluid bound,
    t = max(max of t, max over r of the summed D[r] / C[r])."""
    children: tuple
    name: str = ""

    def __post_init__(self) -> None:
        _tuple(self, "children")


@dataclass(frozen=True, eq=False)
class Repeat:
    """`n` identical, independent repetitions: t = n x t, D = n x D. A real n
    below 1 is a time slice of the body."""
    n: float
    body: Any
    name: str = ""

    def __post_init__(self) -> None:
        if not self.n >= 0:
            raise ValueError(f"Repeat count must be >= 0, got {self.n}")


@dataclass(frozen=True, eq=False)
class OnDevices:
    """Runs `body` on a bound device set. It sets the device class for the
    Compute leaves inside, and the devices are busy for the whole scope. A
    scope nested in another must use a subset of its devices."""
    devset: Any
    body: Any
    name: str = ""


Node = Union[Compute, Collective, Transfer, Work, Seq, Par, Repeat, OnDevices]
LEAVES = (Compute, Collective, Transfer, Work)


def label(node) -> str:
    """The name shown by `explain`: the node's name, else its type."""
    return node.name or type(node).__name__


def walk(node):
    """Every node of a tree once, parents before children (shared nodes once)."""
    seen: set[int] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        yield n
        if isinstance(n, (Seq, Par)):
            stack.extend(reversed(n.children))
        elif isinstance(n, (Repeat, OnDevices)):
            stack.append(n.body)
