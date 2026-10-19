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


def data_path(name_or_path: str | Path) -> Path:
    """A data file: a path, or a name in narfs/data."""
    path = Path(name_or_path)
    return DATA / f"{name_or_path}.json" if not path.suffix else path


def read_values(d: dict, fields: dict, base: dict, flags, skip: tuple = ()) -> tuple[dict, dict, list]:
    """(values, sources, flags) of a profile file over its base: a value the
    file gives replaces the base value and its flag; "assumed" values are flagged."""
    vals, sources, flags = dict(base), {}, list(flags)
    for key, entry in d["values"].items():
        if key in skip:
            continue
        if key not in fields:
            raise ValueError(f"{d.get('name', 'profile')}: unknown value {key!r}; known: {sorted(fields)}")
        name = fields[key]
        v = entry["value"]
        vals[name] = tuple(v) if isinstance(v, list) else v
        sources[name] = entry.get("source", "")
        if name in flags:
            flags.remove(name)
        if entry.get("kind") == "assumed":
            flags.append(name)
    return vals, sources, flags


def load_gpu(name_or_path: str | Path) -> GpuProfile:
    """A GPU profile from a data file. A file with "base" names another
    profile: the values it leaves out come from the base, with their flags, so
    a cluster measurement can give only what it measured."""
    path = data_path(name_or_path)
    d = json.loads(path.read_text())
    if d.get("kind") != "gpu":
        raise ValueError(f"{path} is not a GPU profile")
    base = load_gpu(d["base"]) if "base" in d else None
    inherited = {f: getattr(base, f) for f in _FIELDS.values()} if base else {}
    vals, sources, flags = read_values(d, _FIELDS, inherited, base.flags if base else [])
    missing = sorted(set(_FIELDS.values()) - set(vals))
    if missing:
        raise ValueError(f"{path}: no value for {missing}")
    if base:
        sources = {**base.sources, **sources}
    name = d.get("name") or (base.name if base else path.stem)
    return GpuProfile(name=name, flags=tuple(flags), sources=sources, **vals)


H100_SXM = load_gpu("h100-sxm")
