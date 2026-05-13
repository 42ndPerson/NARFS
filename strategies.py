"""Parallelism strategies — the layer that owns *how* a model is sharded
onto hardware. The core simulator (CombinedOp, ModelMapping, simulator.py,
RLHFTimeline) is intentionally unaware of parallelism semantics; a strategy
is what bridges the parallelism-free architecture (`Model.forward_sequence`)
to the resolved simulation plan (`ModelMapping`).

A strategy carries a `ParallelismConfig` and exposes `build_mm(...)` to
produce a `ModelMapping` from an ordered device pool. The strategy owns
its own placement rule — caller passes an ordered list of devices, strategy
walks the canonical shard-coordinate order and pairs each coord with a
device. Concrete strategies (`MegatronStrategy`, future `FSDPStrategy`,
`HybridFlowStrategy`) implement scheme-specific sharding decisions.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

from mesh import Mesh
from model import Model
from model_mapping import ModelMapping
from network_requirement import ParallelismConfig
from shard import Shard
from topo import Device
from workload_context import Workload


class ParallelismStrategy(ABC):
    """A strategy owns the parallelism / sharding / collective-emission logic
    for one scheme. `build_mm(...)` is the single entry point that produces
    a fully-resolved `ModelMapping` ready for simulation.
    """

    parallelism: ParallelismConfig

    @abstractmethod
    def build_mm(
        self,
        model: Model,
        workload: Workload,
        devices: list[Device],
        name: str = "",
        role: str = "",
    ) -> ModelMapping:
        ...


@dataclass(frozen=True)
class MegatronStrategy(ParallelismStrategy):
    """Canonical Megatron-style sharding.

    Sharding scheme (applied to transformer architectures):
      - QKV, gate, up, LM head:   column-parallel (output dim split)
      - O, down, shared-expert-down: row-parallel (input dim split, TP-AR after)
      - Attention, RoPE:          heads-parallel
      - Routed MoE experts:       EP-sharded (replica axis)
      - Embedding, RMSNorm, SwiGLU: replicated (or follow upstream sharding)
      - DP gradient AR per layer template in TRAINING when dp > 1
      - PP activation P2P per microbatch when pp > 1

    Placement convention: when iterating `(dp, pp, ep, tp)`, TP varies
    innermost so adjacent devices in the input list become adjacent TP
    ranks — matching the typical intra-host NVLink/NVSwitch layout where
    TP collectives benefit most from device proximity. EP varies next,
    then PP, then DP. Caller is responsible for ordering devices to match
    topology (devices on the same host first, then next host, etc.).
    """

    parallelism: ParallelismConfig

    def build_mm(
        self,
        model: Model,
        workload: Workload,
        devices: list[Device],
        name: str = "",
        role: str = "",
    ) -> ModelMapping:
        del name, role  # reserved for RLHF-identity work
        expected = (
            self.parallelism.dp
            * self.parallelism.pp
            * self.parallelism.ep
            * self.parallelism.tp
        )
        if len(devices) != expected:
            # Mention each axis by name so misconfigured runs surface the offending degree.
            raise ValueError(
                f"MegatronStrategy.build_mm: requires {expected} devices for "
                f"DP={self.parallelism.dp} PP={self.parallelism.pp} "
                f"EP={self.parallelism.ep} TP={self.parallelism.tp}, "
                f"got {len(devices)}"
            )

        s2d: dict[Shard, Device] = {}
        idx = 0
        # TP innermost so consecutive devices become TP-adjacent ranks.
        for d in range(self.parallelism.dp):
            for p in range(self.parallelism.pp):
                for e in range(self.parallelism.ep):
                    for t in range(self.parallelism.tp):
                        shard = Shard(model=model, dp=d, pp=p, tp=t, ep=e)
                        s2d[shard] = devices[idx]
                        idx += 1
        d2s = {dev: shard for shard, dev in s2d.items()}
        mesh = Mesh(devices=list(devices))
        return ModelMapping(
            model=model,
            mesh=mesh,
            workload=workload,
            parallelism=self.parallelism,
            shards_to_devices=s2d,
            device_to_shard=d2s,
        )
