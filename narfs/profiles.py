"""Framework and engine profiles: facts about how a framework really runs the
work. A profile chooses the structure a lowering emits (blocking or
overlapped communication, which collectives exist, the schedule) and holds
the settings the memory model reads. It never holds a fitted factor.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from narfs.parallel.rules import MEGATRON, ShardingRules

GiB = 2**30


@dataclass(frozen=True)
class TrainProfile:
    """A Megatron-style training framework."""
    name: str = "megatron"
    rules: ShardingRules = MEGATRON
    lib: str = "nccl"
    tp_comm: str = "blocking"              # "blocking", or "overlapped" (TP communication overlap)
    tp_chunks: int = 4
    sequence_parallel: bool = False
    pp_virtual: int = 1                    # 1 = 1F1B, > 1 = interleaved
    scatter_gather_pp: bool = True         # pipeline sends split over tp, then all-gathered
    dp_overlap: bool = True                # DP gradient sync overlaps the last backward
    dp_bucket_params: float = 40e6         # Megatron-Core DDP: bucket = max(40M, 1M x dp) parameters
    grad_dtype: str = "fp32"               # gradient reduction dtype
    zero_stage: int = 0                    # 0, or 1 (distributed optimizer)
    recompute: str = "selective"           # "none", "selective" or "full"
    vocab_parallel_loss: bool = True
    reserve: float = 4 * GiB               # unmodeled memory (fragmentation, NCCL buffers, workspaces);
    reserve_measured: bool = False         # read it from the framework's memory report

    def __post_init__(self) -> None:
        if self.tp_comm not in ("blocking", "overlapped"):
            raise ValueError(f"tp_comm must be 'blocking' or 'overlapped', got {self.tp_comm!r}")
        if self.recompute not in ("none", "selective", "full"):
            raise ValueError(f"unknown recompute policy {self.recompute!r}")
        if self.zero_stage not in (0, 1):
            raise ValueError("zero_stage must be 0 or 1 (ZeRO-2/3 and FSDP are not modeled yet)")
        if self.pp_virtual < 1:
            raise ValueError("pp_virtual must be >= 1")

    def with_(self, **changes) -> "TrainProfile":
        return replace(self, **changes)


MEGATRON_DEFAULT = TrainProfile()
MEGATRON_OVERLAP = TrainProfile("megatron-overlap", tp_comm="overlapped", sequence_parallel=True)
MEGATRON_DISTOPT = TrainProfile("megatron-distopt", zero_stage=1)
