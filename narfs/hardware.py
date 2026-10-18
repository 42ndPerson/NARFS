"""Device specifications (one entry per device class)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceSpec:
    """Peak rates and default efficiency factors for one GPU type.

    The efficiency factors are roofline defaults. Calibrated cost tables
    (narfs.costs) replace them when measurements are available.
    """

    name: str
    peak_flops: float            # dense bf16 FLOP/s
    hbm_bw: float                # bytes/s
    hbm_bytes: float             # capacity in bytes
    compute_eff: float = 0.7     # large GEMMs
    small_gemm_eff: float = 0.4  # min(M, N, K) < 1024
    gemv_eff: float = 0.3        # M < 16
    memory_eff: float = 0.8
    gemv_memory_eff: float = 0.65
    launch_s: float = 5e-6       # fixed cost per kernel launch (alpha)
    pcie_bw: float = 25e9        # host <-> device, bytes/s per direction


A100_80GB = DeviceSpec("A100-80GB", peak_flops=312e12, hbm_bw=2.0e12, hbm_bytes=80e9, launch_s=8e-6)
H100_80GB = DeviceSpec("H100-80GB", peak_flops=989e12, hbm_bw=3.35e12, hbm_bytes=80e9, launch_s=5e-6, pcie_bw=50e9)
