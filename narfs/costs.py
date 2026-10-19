"""L0: the compute cost model and the host cost model.

Compute cost is a roofline with measured ceilings and wave quantization. It
has no size tiers and no fitted efficiency:

    t = max(F_padded / (eta_gemm x P_peak x u_wave),  bytes / (eta_hbm x B_hbm))

    F_padded  FLOPs of the tiles, padded to the tile shape (one shape, 128x256)
    u_wave    work units / (waves x SMs): the share of SMs busy, over all waves.
              A work unit is a tile, or for skinny outputs that cannot fill the
              SMs, one split of a tile's K loop (split-K, as cuBLAS does).

A wgmma instruction reaches 96% of peak in isolation; the shortfall of real
GEMMs comes from sustained clocks, waves and tile pipeline overhead, which is
what eta_gemm (one ceiling per dtype) and u_wave model. Measured tables
(`CostTable`) override the formula where they exist.

The host model gives the time of host work items, such as one engine step's
scheduling and launch work (a + b x batch, measured per engine version).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from narfs.hardware import DTYPE_BYTES, GpuProfile

KERNEL_KINDS = ("gemm", "attention", "elementwise", "op")
SPLIT_K_BLOCK = 64          # K depth of one Hopper BF16 main-loop stage (128x256x64 tiles)


@dataclass(frozen=True)
class Kernel:
    """One GPU kernel on local (per-rank) shapes.

    gemm:        shape = (m, n, k); flops and bytes follow from it
    attention:   flops, bytes, and `tiles` (one CTA per head and query block)
    elementwise: memory-bound vector work
    op:          any other kernel; `tensor` says whether its FLOPs use tensor cores
    """
    kind: str
    flops: float
    bytes: float
    shape: tuple = ()
    dtype: str = "bf16"
    tiles: float = 0.0
    tensor: bool = False
    groups: float = 1.0                # a grouped GEMM: `groups` GEMMs of `shape` in one kernel
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KERNEL_KINDS:
            raise ValueError(f"unknown kernel kind {self.kind!r}")
        if self.flops < 0 or self.bytes < 0:
            raise ValueError("kernel flops and bytes must be >= 0")


def gemm(m: float, n: float, k: float, dtype: str = "bf16", name: str = "", out_dtype: str | None = None,
         groups: float = 1.0) -> Kernel:
    """C[m, n] = A[m, k] x B[k, n]: reads A and B once and writes C. With
    `groups`, a grouped GEMM (for example one per local expert) in one kernel."""
    b = DTYPE_BYTES[dtype]
    bo = DTYPE_BYTES[out_dtype or ("bf16" if dtype == "fp8" else dtype)]
    return Kernel("gemm", 2.0 * m * n * k * groups, groups * (b * (m * k + k * n) + bo * m * n), (m, n, k), dtype,
                  groups=groups, name=name)


def elementwise(flops: float, bytes_: float, name: str = "") -> Kernel:
    return Kernel("elementwise", float(flops), float(bytes_), name=name)


@dataclass(frozen=True)
class CostTable:
    """Measured kernel times, keyed by (GPU name, kind, shape, dtype)."""
    entries: dict

    @classmethod
    def load(cls, path: str | Path) -> "CostTable":
        rows = json.loads(Path(path).read_text())
        return cls({(r["gpu"], r["kind"], tuple(r["shape"]), r.get("dtype", "bf16")): float(r["seconds"])
                    for r in rows})

    def lookup(self, gpu: str, w: Kernel) -> float | None:
        return self.entries.get((gpu, w.kind, tuple(w.shape), w.dtype))


def waves(tiles: float, sms: int) -> tuple[int, float]:
    """(waves, u_wave) for `tiles` independent tiles on `sms` SMs."""
    if tiles <= 0:
        return 0, 1.0
    w = math.ceil(tiles / sms)
    return w, tiles / (w * sms)


class ComputeModel:
    def __init__(self, gpus: dict, table: CostTable | None = None):
        self.gpus = dict(gpus)
        self.table = table
        self._cache: dict = {}

    def gpu(self, device_class: str) -> GpuProfile:
        try:
            return self.gpus[device_class]
        except KeyError:
            raise KeyError(f"no GPU profile for device class {device_class!r}") from None

    def slowest(self, classes) -> str:
        """The class that gates SPMD work on a mixed device set."""
        return min(classes, key=lambda c: self.gpu(c).peak("bf16"))

    def terms(self, w: Kernel, device_class: str) -> tuple[float, float, int, float]:
        """(compute seconds, memory seconds, waves, u_wave)."""
        g = self.gpu(device_class)
        if w.kind == "gemm":
            m, n, k = w.shape
            tm, tn = g.tile
            tiles = math.ceil(w.groups * math.ceil(m / tm) * math.ceil(n / tn))
            # skinny outputs cannot fill the SMs: the K loop is split across SMs (split-K);
            # the fp32 partial sums are small next to the weights and mostly stay in L2
            split = max(1, min(g.sms // tiles, math.ceil(k / SPLIT_K_BLOCK))) if 0 < tiles < g.sms else 1
            nw, u = waves(tiles * split, g.sms)
            padded = 2.0 * w.groups * math.ceil(m / tm) * tm * math.ceil(n / tn) * tn * k
            compute = padded / (g.eta(w.dtype) * g.peak(w.dtype) * u) if tiles else 0.0
        elif w.kind == "attention" or (w.kind == "op" and w.tensor):
            nw, u = waves(w.tiles, g.sms) if w.tiles else (0, 1.0)
            compute = w.flops / (g.eta(w.dtype) * g.peak(w.dtype) * u)
        else:
            nw, u = 0, 1.0
            compute = w.flops / g.vector_flops
        memory = w.bytes / (g.eta_hbm * g.hbm_bw)
        return compute, memory, nw, u

    def time(self, w: Kernel, device_class: str) -> float:
        key = (w, device_class)
        t = self._cache.get(key)
        if t is None:
            hit = self.table.lookup(device_class, w) if self.table is not None else None
            if hit is not None:
                t = hit
            else:
                compute, memory, _, _ = self.terms(w, device_class)
                t = max(compute, memory)
            if len(self._cache) > 100_000:
                self._cache.clear()
            self._cache[key] = t
        return t

    def describe(self, w: Kernel, device_class: str) -> str:
        if self.table is not None and self.table.lookup(device_class, w) is not None:
            return f"{w.kind} {w.name}: measured table"
        compute, memory, nw, u = self.terms(w, device_class)
        shape = "x".join(f"{int(s)}" for s in w.shape) if w.shape else ""
        if compute >= memory:
            what = f"compute-bound, {nw} wave{'s' if nw != 1 else ''} ({u:.0%} of SMs)" if nw else "compute-bound"
        else:
            what = "memory-bound (HBM)"
        return f"{w.kind} {shape} {w.dtype}: {what}".replace("  ", " ")


@dataclass(frozen=True)
class HostStep:
    """One engine step's host work (scheduling, input preparation, launches)."""
    engine: str
    batch: float


class HostModel:
    def __init__(self, steps: dict | None = None):
        self.steps = dict(steps or {})            # engine -> (a seconds, b seconds per sequence)

    def seconds(self, item) -> float:
        if isinstance(item, HostStep):
            try:
                a, b = self.steps[item.engine]
            except KeyError:
                raise KeyError(f"no host step model for engine {item.engine!r}") from None
            return a + b * item.batch
        raise TypeError(f"no host cost for {type(item).__name__}")
