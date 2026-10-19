"""GPU profiles: the physical constants of one GPU type, with provenance.

Every value names a physical thing and has a source. Values of kind
"assumed" are defaults with no trustworthy measurement; they are listed in
`flags` so that explain can report them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
DTYPE_BYTES = {"fp32": 4, "tf32": 4, "bf16": 2, "fp16": 2, "fp8": 1}


@dataclass(frozen=True)
class GpuProfile:
    name: str
    sms: int
    clock_hz: float
    tensor_fpc: dict                   # dtype -> dense tensor FLOPs per clock per SM
    vector_flops: float                # FP32 non-tensor FLOP/s
    hbm_bw: float                      # bytes/s
    hbm_bytes: float
    eta_gemm: dict                     # dtype -> GEMM ceiling (fraction of peak)
    eta_hbm: float                     # achievable HBM rate / peak
    tile: tuple = (128, 256)           # GEMM tile (M, N)
    nvlink_rate: float = 375e9         # payload bytes/s per direction
    nvls_level: float = 0.82
    nic_rate: float = 49e9
    pcie_rate: float = 55.4e9
    launch_eager_s: float = 3e-6
    graph_node_s: float = 0.5e-6
    flags: tuple = ()                  # names of assumed (unmeasured) values
    sources: dict = field(default_factory=dict, compare=False, hash=False, repr=False)

    def peak(self, dtype: str) -> float:
        """Dense tensor FLOP/s for a dtype."""
        try:
            return self.sms * self.tensor_fpc[dtype] * self.clock_hz
        except KeyError:
            raise KeyError(f"{self.name} has no tensor rate for dtype {dtype!r}") from None

    def eta(self, dtype: str) -> float:
        return self.eta_gemm.get(dtype, self.eta_gemm["bf16"])

    def with_(self, **values) -> "GpuProfile":
        """A copy with measured values replacing defaults (their flags are cleared)."""
        from dataclasses import replace
        flags = tuple(f for f in self.flags if f not in values)
        return replace(self, flags=flags, **values)


_FIELDS = {"sms": "sms", "clock_hz": "clock_hz", "tensor_flops_per_clock_per_sm": "tensor_fpc",
           "vector_flops": "vector_flops", "hbm_bytes_per_s": "hbm_bw", "hbm_capacity": "hbm_bytes",
           "eta_gemm": "eta_gemm", "eta_hbm": "eta_hbm", "tile": "tile", "nvlink_rate": "nvlink_rate",
           "nvls_level": "nvls_level", "nic_rate": "nic_rate", "pcie_rate": "pcie_rate",
           "launch_eager_s": "launch_eager_s", "graph_node_s": "graph_node_s"}


def load_gpu(name_or_path: str | Path) -> GpuProfile:
    """A GPU profile from a data file: a path, or a name in narfs/data."""
    path = Path(name_or_path)
    if not path.suffix:
        path = DATA / f"{name_or_path}.json"
    d = json.loads(path.read_text())
    if d.get("kind") != "gpu":
        raise ValueError(f"{path} is not a GPU profile")
    vals, sources, flags = {}, {}, []
    for key, entry in d["values"].items():
        name = _FIELDS[key]
        v = entry["value"]
        vals[name] = tuple(v) if isinstance(v, list) else v
        sources[name] = entry.get("source", "")
        if entry.get("kind") == "assumed":
            flags.append(name)
    return GpuProfile(name=d["name"], flags=tuple(flags), sources=sources, **vals)


H100_SXM = load_gpu("h100-sxm")
