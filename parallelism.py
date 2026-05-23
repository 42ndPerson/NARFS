"""Parallelism-axis vocabulary shared across the simulator.

Three kinds of thing live here, all describing *parallelism layout* rather
than any one consumer's use of it:

  - `ParallelismConfig` — the (tp, pp, dp, ep) degree tuple for a mapping.
  - `ParallelismGroup` — which axis a collective / shard set spans.
  - `ParallelismRole` / `TPMode` / `EPMode` — how a `ParallelismScheme`
    decides an individual operator shards. Post-Phase-2c operators are
    parallelism-pure; the role lives in the scheme's decoration table, not
    on the operator.

DP and PP are absent from `ParallelismRole`: DP doesn't divide per-op
compute (it only adds a gradient AllReduce per layer template at backward
time), and PP is a placement decision made in `ModelMapping`, not a
per-operator role.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model import Model


@dataclass(frozen=True)
class Shard:
    """A single rank in a (dp, pp, tp, ep) parallelism mesh for one model."""
    model: "Model"
    dp: int  # data parallel rank
    pp: int  # pipeline parallel rank
    tp: int  # tensor parallel rank
    ep: int = 0  # expert parallel rank (0 = no EP / degenerate single-rank EP group)


@dataclass(frozen=True)
class ParallelismConfig:
    tp: int = 1
    pp: int = 1
    dp: int = 1
    ep: int = 1


class ParallelismGroup(Enum):
    """The parallelism axis a collective or shard set spans."""
    TP = "tp"
    EP = "ep"
    PP = "pp"
    DP = "dp"


class TPMode(Enum):
    REPLICATED = auto()        # no TP split
    COLUMN_PARALLEL = auto()   # split along output dim N (e.g. QKV, gate, up)
    ROW_PARALLEL = auto()      # split along input dim K (e.g. O, down) — emits AR
    VOCAB_PARALLEL = auto()    # split along vocab dim (embedding, LM head)
    HEADS_PARALLEL = auto()    # attention: split heads (n_q, n_kv)


class EPMode(Enum):
    REPLICATED = auto()        # no EP split
    EXPERT_PARALLEL = auto()   # routed-expert sharding (V3 MoE)


@dataclass(frozen=True)
class ParallelismRole:
    """How a `ParallelismScheme` shards one operator under TP and EP.

    Drives both per-rank cost scaling (compute / memory divide along the
    appropriate dim) and emitted collectives (ROW_PARALLEL under tp>1 fires
    a TP-AR; EXPERT_PARALLEL fires EP all-to-all dispatch/combine).
    """
    tp: TPMode = TPMode.REPLICATED
    ep: EPMode = EPMode.REPLICATED


# Convenient constant for the default REPLICATED-on-REPLICATED role.
REPLICATED = ParallelismRole()
