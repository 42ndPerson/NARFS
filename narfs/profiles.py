"""Framework and engine profiles: facts about how a framework really runs the
work. A profile chooses the structure a lowering emits (blocking or
overlapped communication, which collectives exist, the schedule) and holds
the settings the memory model reads. It never holds a fitted factor.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

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

# inference engines shard the expert FFN over tp when experts are not spread over ep
VLLM_RULES = MEGATRON.extend("vllm", expert_ffn="tp")


@dataclass(frozen=True)
class EngineProfile:
    """A continuous-batching inference engine (vLLM-style). Engine policies
    are profile facts; the generation lowering reads them."""
    name: str = "vllm"
    version: str = ""
    rules: ShardingRules = VLLM_RULES
    lib: str = "vllm"                      # communication library profile (its all-reduce paths)
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192     # token budget per step
    capture_sizes: tuple = (1, 2, 4) + tuple(range(8, 513, 8))
    gpu_memory_utilization: float = 0.9
    host_step: tuple = (1.0e-3, 5.0e-6)    # host seconds per step: a + b x batch
    async_scheduling: bool = False
    prefix_caching: bool = True
    cascade_attention: bool = False
    reserve: float = 3.0e9                 # non-torch memory + CUDA graph pool, from the startup log
    flags: tuple = ()                      # assumed (unmeasured) values

    def padded(self, batch: float) -> float:
        """The CUDA-graph size a batch runs at: the smallest capture size of at
        least the batch (eager beyond the largest)."""
        for s in self.capture_sizes:
            if s >= batch:
                return float(s)
        return float(math.ceil(batch))

    def with_(self, **changes) -> "EngineProfile":
        flags = tuple(f for f in self.flags if f not in changes)
        return replace(self, flags=flags, **changes)


_ENGINE_FIELDS = {"max_num_seqs": "max_num_seqs", "max_num_batched_tokens": "max_num_batched_tokens",
                  "capture_sizes": "capture_sizes", "gpu_memory_utilization": "gpu_memory_utilization",
                  "host_step_s": "host_step", "async_scheduling": "async_scheduling",
                  "prefix_caching": "prefix_caching", "cascade_attention": "cascade_attention",
                  "reserve_bytes": "reserve"}


def load_engine(name_or_path, nccl: str = "nccl"):
    """(EngineProfile, EngineLibrary) from an engine data file in narfs/data."""
    from narfs.hardware import DATA
    from narfs.network.comm import EngineLibrary, EnginePath
    path = Path(name_or_path)
    if not path.suffix:
        path = DATA / f"{name_or_path}.json"
    d = json.loads(path.read_text())
    if d.get("kind") != "engine":
        raise ValueError(f"{path} is not an engine profile")
    vals, flags = {}, []
    for key, entry in d["values"].items():
        if key == "allreduce_paths":
            continue
        v = entry["value"]
        vals[_ENGINE_FIELDS[key]] = tuple(v) if isinstance(v, list) else v
        if entry.get("kind") == "assumed":
            flags.append(_ENGINE_FIELDS[key])
    ar = d["values"]["allreduce_paths"]
    if ar.get("latency_kind") == "assumed":
        flags.append("allreduce_latency")
    paths = {int(n): tuple(EnginePath(float(b), p, float(lat), bool(inc)) for b, p, lat, inc in rows)
             for n, rows in ar["value"].items()}
    prof = EngineProfile(name=d["name"], version=d.get("version", ""), lib=d["name"], flags=tuple(flags), **vals)
    lib = EngineLibrary(d["name"], paths, nccl=nccl, source=ar.get("source", ""))
    return prof, lib
