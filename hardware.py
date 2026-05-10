from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareSpec:
    peak_compute_flops: float           # FP16 TFLOPS, stored as raw FLOPS not T
    peak_memory_bandwidth: float        # HBM bytes/sec
    ridge_flops_per_byte: float         # roofline ridge: AI above this is compute-bound
    compute_efficiency: float = 0.7     # achieved fraction of peak for large GEMMs
    memory_efficiency: float = 0.8      # achieved fraction of peak HBM BW
    small_gemm_efficiency: float = 0.4  # used when any GEMM dim < 1024
    # GEMV-shaped GEMMs (M < 16, typical for batch=1 decode) have very
    # different efficiency characteristics from batched GEMMs — typically much
    # lower than even small_gemm_efficiency. Splitting these out lets decode
    # predictions reflect the steeper utilization drop without affecting
    # small-but-non-GEMV cases.
    gemv_efficiency: float = 0.3
    # Achievable HBM BW for GEMV-shaped GEMMs is also lower than streaming
    # bulk reads; typical 0.6-0.7 vs 0.8 for large kernels.
    gemv_memory_efficiency: float = 0.65
