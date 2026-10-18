"""Cost of one operator on a local shape on a device class.

Three sources, in order: a measured table entry (if loaded), then the
roofline with the device's efficiency factors. Results are pure functions of
(operator, device spec), so a cache can be added later without other changes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from narfs.blocks import Op
from narfs.hardware import DeviceSpec


@dataclass(frozen=True)
class CostTable:
    """Measured operator times, for example from a profiling run.

    Entries are keyed by (device name, op kind, shape). For a GEMM that is not
    in the table, the nearest entry with the same (N, K) is scaled by M.
    """

    entries: dict = field(default_factory=dict, hash=False, compare=False)

    @classmethod
    def load(cls, path: str | Path) -> "CostTable":
        rows = json.loads(Path(path).read_text())
        return cls({(r["device"], r["kind"], tuple(r["shape"])): float(r["seconds"]) for r in rows})

    def lookup(self, device: str, op: Op) -> float | None:
        hit = self.entries.get((device, op.kind, op.shape))
        if hit is not None or op.kind != "gemm" or len(op.shape) != 3:
            return hit
        m, n, k = op.shape
        best = None
        for (dev, kind, shape), seconds in self.entries.items():
            if dev == device and kind == "gemm" and shape[1:] == (n, k):
                if best is None or abs(shape[0] - m) < abs(best[0] - m):
                    best = (shape[0], seconds)
        if best is None:
            return None
        return best[1] * m / best[0]


@dataclass(frozen=True)
class CostModel:
    table: CostTable | None = None

    def op_time(self, op: Op, spec: DeviceSpec) -> float:
        """Time for one execution of `op` (its `count` is applied by the caller)."""
        if self.table is not None:
            hit = self.table.lookup(spec.name, op)
            if hit is not None:
                return hit
        compute_eff, memory_eff = spec.compute_eff, spec.memory_eff
        if op.kind == "gemm":
            m = op.shape[0] if op.shape else float("inf")
            if m < 16:
                compute_eff, memory_eff = spec.gemv_eff, spec.gemv_memory_eff
            elif op.shape and min(op.shape) < 1024:
                compute_eff = spec.small_gemm_eff
        t = max(op.flops / (spec.peak_flops * compute_eff), op.bytes / (spec.hbm_bw * memory_eff))
        return t + op.kernels * spec.launch_s
