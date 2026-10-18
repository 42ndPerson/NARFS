"""Framework profiles: facts about how a framework really runs the work.

A profile chooses the STRUCTURE that expansion emits (blocking or overlapped
communication, which collectives exist) and the settings that the memory
model and GenBatch use. It never holds a fitted factor.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class FrameworkProfile:
    name: str
    # training
    tp_comm: str = "blocking"          # "blocking" or "overlapped"
    tp_chunks: int = 4                 # chunks for overlapped TP communication
    sequence_parallel: bool = False    # RS + AG instead of AR
    dp_overlap: bool = True            # DP gradient sync overlaps the last backward
    dp_bucket_bytes: float = 50e6
    grad_dtype_bytes: int = 4
    zero_stage: int = 0                # 0, 1 (distributed optimizer) or 2
    recompute: str = "selective"       # "none", "selective" or "full"
    pp_virtual: int = 1                # 1 = 1F1B, > 1 = interleaved
    vocab_parallel_loss: bool = True   # no all-gather of logits
    # generation engine
    gen_max_seqs: int = 256
    gen_kv_fraction: float = 0.9       # share of free HBM for the KV cache
    gen_kernels_per_layer: int = 2     # with CUDA graphs, launches per layer are few
    # colocation
    offload_idle_weights: bool = False

    def __post_init__(self) -> None:
        if self.tp_comm not in ("blocking", "overlapped"):
            raise ValueError(f"tp_comm must be 'blocking' or 'overlapped', got {self.tp_comm!r}")
        if self.recompute not in ("none", "selective", "full"):
            raise ValueError(f"unknown recompute policy {self.recompute!r}")
        if self.zero_stage not in (0, 1, 2):
            raise ValueError("zero_stage must be 0, 1 or 2 (ZeRO-3 / FSDP is not implemented yet)")

    def with_(self, **changes) -> "FrameworkProfile":
        return replace(self, **changes)


MEGATRON_DEFAULT = FrameworkProfile("megatron-default")
MEGATRON_ASYNC_TP = FrameworkProfile("megatron-async-tp", tp_comm="overlapped", sequence_parallel=True)
MEGATRON_DISTOPT = FrameworkProfile("megatron-distopt", zero_stage=1)
VLLM_DEFAULT = FrameworkProfile("vllm-default")

PROFILES = {p.name: p for p in (MEGATRON_DEFAULT, MEGATRON_ASYNC_TP, MEGATRON_DISTOPT, VLLM_DEFAULT)}
