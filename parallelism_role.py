"""Per-operator parallelism roles.

An `Operator` carries a `ParallelismRole` declaring how it shards under TP
and EP. The role drives both:

  - per-rank cost scaling (compute_flops / memory_bytes divide along the
    appropriate dim under each axis), and
  - emitted collectives (e.g. ROW_PARALLEL Gemm under tp>1 fires a TP-AR
    after itself; EXPERT_PARALLEL ops fire EP-AllToAll dispatch/combine).

DP and PP are intentionally absent: DP doesn't divide compute (only adds a
gradient AR per layer template at backward time), and PP is a placement
decision made in ModelMapping, not a per-operator role.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto


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
    tp: TPMode = TPMode.REPLICATED
    ep: EPMode = EPMode.REPLICATED


# Convenient constant for the default REPLICATED-on-REPLICATED role.
REPLICATED = ParallelismRole()
