"""Block types: the structure that expansion emits and the evaluator times.

Structure is small frozen objects. `Repeat` keeps trees small, so a tree is
cheap to rebuild for every candidate. Blocks hold no device arrays: a
collective names a mesh axis, and the evaluator gets the device groups from
the placement of the enclosing `Call`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union


# ------------------------------------------------------------------ leaves ---

@dataclass(frozen=True)
class Op:
    """One operator on a local (per-rank) shape.

    kind: "gemm", "attention", "elementwise" or "memory".
    shape: (M, N, K) for a GEMM, used for the efficiency tier; () otherwise.
    """
    kind: str
    flops: float
    bytes: float
    shape: tuple = ()
    kernels: int = 1
    count: int = 1
    name: str = ""


@dataclass(frozen=True)
class Collective:
    """One collective on a mesh axis of the enclosing call, as a lockstep
    burst over every instance of that axis. `size` is bytes per rank."""
    kind: str          # "allreduce", "allgather", "reducescatter", "alltoall"
    axis: str | tuple  # "tp", "dp", "ep", "cp", or a tuple such as ("dp", "ep")
                       # (pp transfers use StageP2P)
    size: float
    count: int = 1
    name: str = ""


@dataclass(frozen=True)
class StageP2P:
    """Activation transfers between neighboring PP stages of the enclosing
    call, for every (dp, ep, cp, tp) coordinate. `size` is bytes per rank."""
    size: float
    count: int = 1
    name: str = "pp p2p"


@dataclass(frozen=True)
class Stream:
    """A steady stream of `size` bytes per device of the enclosing call
    (for example weight offload over PCIe), plus a fixed start-up time."""
    size: float
    kind: str = "pcie"
    fill: float = 0.0
    name: str = ""


@dataclass(frozen=True)
class Transfers:
    """Fixed point-to-point transfers between device IDs (used for
    resharding between calls). Built per candidate from two layouts."""
    src: tuple
    dst: tuple
    size: tuple
    name: str = "reshard"


# -------------------------------------------------------------- composites ---

@dataclass(frozen=True)
class Seq:
    children: tuple
    name: str = ""


@dataclass(frozen=True)
class Par:
    children: tuple
    name: str = ""


@dataclass(frozen=True)
class Repeat:
    """`n` identical, independent repetitions of `body`: same shapes, same
    devices, same context, no overlap across the boundary."""
    n: int
    body: "Block"
    name: str = ""


@dataclass(frozen=True)
class OnStage:
    """Evaluate `body` as work of one PP stage: its collectives use only
    that stage's devices."""
    stage: int
    body: "Block"
    name: str = ""


@dataclass(frozen=True)
class Scale:
    """A fraction of `body`: time and demand times `factor`. Used to cut a
    block into equal chunks for overlap."""
    factor: float
    body: "Block"
    name: str = ""


@dataclass(frozen=True)
class Pipeline:
    """Pipeline over PP stages. `stages[s]` is the work of stage s for one
    microbatch (forward, or forward + backward for 1F1B).

    T = sum_s t_s / v + (nmb - 1/v) * slot, where slot is the steady-state
    time of one microbatch with every stage active (a fluid Parallel over the
    stages). v = 1 is 1F1B; v > 1 is the interleaved schedule.
    """
    stages: tuple
    nmb: int
    p2p: "Block | None" = None
    virtual: int = 1
    name: str = "pipeline"


@dataclass(frozen=True)
class FlowShop2:
    """Two-stage chunk pipeline: chunk i of `b` can start only after chunk i
    of `a` ends. T = max_k (a_1 + ... + a_k + b_k + ... + b_K).

    count_a=False means the `a` work is already counted elsewhere, so only
    the exposed part of `b` is returned. `buffer` limits how far `a` can run
    ahead of `b` (None means unlimited)."""
    a: tuple
    b: tuple
    count_a: bool = True
    buffer: int | None = None
    name: str = "flow shop"


@dataclass(frozen=True)
class AsyncPipe:
    """Two-stage pipeline across RL iterations (generation of step k+1 runs
    while step k trains). Returns the steady time per iteration."""
    producer: "Block"
    consumer: "Block"
    name: str = "async"


@dataclass(frozen=True)
class GenBatch:
    """Generation by a continuous-batching engine, as a closed form over
    decode steps. All quantities are for ONE replica (one dp x ep index):
    expansion splits the requests over the replicas, so the evaluator needs
    no degrees.

    lengths / probs: the response length distribution.
    comm: collectives per decode step, each sized for ONE sequence.
    """
    replica_requests: int        # requests served by one replica
    prompt_len: float
    lengths: tuple
    probs: tuple
    weight_bytes: float          # per rank, read once per step
    flops_per_token: float       # per rank
    kv_bytes_per_token: float    # per rank, per context token
    kernels_per_step: int
    max_seqs: int
    kv_capacity_tokens: float    # per replica
    comm: tuple = ()
    prefill: "Block | None" = None
    name: str = "generate"


@dataclass(frozen=True)
class Call:
    """One RLHF model call on its device group. Sets the placement used by
    the collectives inside it, and marks its devices busy for its duration."""
    name: str
    body: "Block"


Block = Union[Op, Collective, StageP2P, Stream, Transfers, Seq, Par, Repeat, Scale, OnStage,
              Pipeline, FlowShop2, AsyncPipe, GenBatch, Call]
