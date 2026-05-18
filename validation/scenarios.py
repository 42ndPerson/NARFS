"""Reference scenarios for validation against published latency numbers.

Each scenario specifies:
  - the model architecture and its build_pattern
  - parallelism (tp/pp/dp/ep)
  - hardware spec
  - workload + workload context
  - reference latency in ms
  - tolerance (% of reference)
  - which metric in SimulationResult to compare ("total" / "decode_step")

Tolerances default to 50% but a few scenarios use 100% where the brief
explicitly calls out a "sanity bound" rather than a tight target.
"""
from __future__ import annotations
from dataclasses import dataclass

from hardware import HardwareSpec
from model import ArchitectureConfig
from parallelism import ParallelismConfig
from model import Model
from operators.llama_builder import LlamaModel
from operators.llama_config import LlamaConfig
from operators.v3_block import V3Model
from operators.v3_config import V3Config
from workload_context import Workload, WorkloadContext


# ---- hardware -----------------------------------------------------------------

A100_80GB = HardwareSpec(
    peak_compute_flops=312e12,
    peak_memory_bandwidth=2.0e12,
    # default efficiency factors: 0.7 / 0.8 / 0.4
    # ~10 ms per decode token: representative vLLM-class framework +
    # kernel-launch overhead on A100 (300 kernels × ~30 μs/launch plus
    # framework dispatch / paged-attention block management). Tune per
    # measured deployment.
    decode_overhead_per_token_ms=10.0,
)

H100_80GB = HardwareSpec(
    peak_compute_flops=989e12,
    peak_memory_bandwidth=3.35e12,
    # H100 has lower per-kernel launch overhead than A100 (~5-10 μs);
    # set per scenario when modeling decode workloads.
    decode_overhead_per_token_ms=5.0,
)


# ---- architectures ------------------------------------------------------------

LLAMA2_7B = LlamaConfig(
    h=4096,
    n_layers=32,
    n_q=32,
    n_kv=32,  # MHA, not GQA
    head_size=128,
    m=11008,
    rope_dim=128,
    vocab=32000,
    flash_block_size=64,
    dtype_bytes=2,
)

LLAMA3_8B = LlamaConfig(
    h=4096,
    n_layers=32,
    n_q=32,
    n_kv=8,
    head_size=128,
    m=14336,
    rope_dim=128,
    vocab=128_000,
    flash_block_size=64,
    dtype_bytes=2,
)

LLAMA3_70B = LlamaConfig(
    h=8192,
    n_layers=80,
    n_q=64,
    n_kv=8,
    head_size=128,
    m=28672,
    rope_dim=128,
    vocab=128_000,
    flash_block_size=64,
    dtype_bytes=2,
)

DEEPSEEK_V3 = V3Config()  # the existing defaults are V3 spec


# ---- scenario type ------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    description: str
    architecture: ArchitectureConfig
    model_class: type[Model]
    parallelism: ParallelismConfig
    hardware: HardwareSpec
    workload: Workload
    workload_ctx: WorkloadContext
    reference_latency_ms: float
    tolerance_percent: float = 50.0
    metric: str = "total"  # or "decode_step"
    spine_bandwidth_gbps: float = 4800.0  # NVLink-class shared spine; tuned per-scenario

    # Topology selector. Defaults to the single-spine fabric, which only the
    # single-GPU scenarios use (they have no inter-device traffic, so the
    # fabric model is moot for them); the multi-GPU scenarios pick a
    # realistic fabric explicitly.
    #   "shared_spine"  — every host pair shares one Link (default).
    #   "nvswitch"      — every host pair has its own dedicated Link
    #                     (full-bisection NVLink/NVSwitch model).
    #   "multinode"     — NVSwitch within a node, one shared IB Link per
    #                     (node_a, node_b) pair; uses
    #                     `hosts_per_node` and `inter_node_bandwidth_gbps`.
    topology_kind: str = "shared_spine"
    hosts_per_node: int = 8
    inter_node_bandwidth_gbps: float = 400.0  # 8x100Gb IB ≈ 400 GB/s

    # Per-GPU NVLink aggregate egress / ingress bandwidth. Set per scenario
    # to the GPU's full NVLink-port aggregate (A100 ≈ 2400 Gbps, H100 ≈
    # 3600 Gbps per direction). The topology builders feed this into
    # `Path`'s `src_device_bandwidth` / `dst_device_bandwidth` so per-GPU
    # contention shows up: every path leaving a GPU shares one phantom
    # self-loop `Link`, and concurrent transfers from that GPU divide the
    # cap. Irrelevant for single-GPU scenarios; the default leaves it at
    # a generic NVLink-class value.
    gpu_egress_bandwidth_gbps: float = 4800.0

    # Per-GPU NIC bandwidth, used for the attach links that bridge a GPU
    # to a shared (int-endpointed) inter-node IB Link. For 400 Gb IB
    # (DGX-class H100), ~400 Gbps per direction. Note: due to how Path
    # connectivity is structured, this caps per-(src-GPU, dst-node)
    # bandwidth rather than per-GPU NIC aggregate across destination
    # nodes — modeling true per-GPU NIC aggregate would require a
    # device-self-loop attach at a non-endpoint position, which Path's
    # current validation disallows.
    nic_bandwidth_gbps: float = 400.0

    # Per-attach-link latency in ms. Applied to each attach Link bridging
    # `Device → shared int-endpointed Link → Device` (so the path picks
    # up twice this value when traversing a cross-node IB path). Models
    # NIC + PCIe traversal on each end; the IB fabric's own latency
    # lives on the shared Link itself. Default 0 matches the current
    # latency-free baseline; raise per scenario to model realistic
    # small-message costs.
    nic_latency_ms: float = 0.0


# ---- scenario instances -------------------------------------------------------


SCENARIOS: list[Scenario] = [
    Scenario(
        name="llama2_7b_prefill_a100",
        description="Llama-2-7B prefill, batch=1, prompt=512, single A100-80GB. Reference ~50ms (vLLM).",
        architecture=LLAMA2_7B,
        model_class=LlamaModel,
        parallelism=ParallelismConfig(tp=1, pp=1, dp=1, ep=1),
        hardware=A100_80GB,
        workload=Workload.PREPARATION,
        workload_ctx=WorkloadContext(
            workload_type=Workload.PREPARATION,
            batch_size=1, microbatch_size=1,
            prompt_len=512, response_len=0, num_microbatches=1,
        ),
        reference_latency_ms=50.0,
        tolerance_percent=50.0,
        metric="total",
        spine_bandwidth_gbps=4800.0,
    ),
    Scenario(
        name="llama2_7b_decode_a100",
        description="Llama-2-7B decode, batch=1, single A100-80GB. Reference ~25ms/token (vLLM).",
        architecture=LLAMA2_7B,
        model_class=LlamaModel,
        parallelism=ParallelismConfig(tp=1, pp=1, dp=1, ep=1),
        hardware=A100_80GB,
        workload=Workload.GENERATION,
        # response_len=1 → decode_time == one-step latency
        workload_ctx=WorkloadContext(
            workload_type=Workload.GENERATION,
            batch_size=1, microbatch_size=1,
            prompt_len=512, response_len=1, num_microbatches=1,
        ),
        reference_latency_ms=25.0,
        tolerance_percent=50.0,
        metric="decode_step",
        spine_bandwidth_gbps=4800.0,
    ),
    Scenario(
        name="llama3_70b_prefill_8xa100_tp8",
        description=(
            "Llama-3-70B prefill, batch=1, prompt=2048, 8xA100-80GB TP=8 on a "
            "full-bisection NVSwitch fabric. Reference ~250ms."
        ),
        architecture=LLAMA3_70B,
        model_class=LlamaModel,
        parallelism=ParallelismConfig(tp=8, pp=1, dp=1, ep=1),
        hardware=A100_80GB,
        workload=Workload.PREPARATION,
        workload_ctx=WorkloadContext(
            workload_type=Workload.PREPARATION,
            batch_size=1, microbatch_size=1,
            prompt_len=2048, response_len=0, num_microbatches=1,
        ),
        reference_latency_ms=250.0,
        tolerance_percent=50.0,
        metric="total",
        spine_bandwidth_gbps=2400.0,  # A100 NVLink: per-pair ~300 GB/s
        topology_kind="nvswitch",
        gpu_egress_bandwidth_gbps=2400.0,  # A100 NVLink aggregate egress
    ),
    Scenario(
        name="llama3_8b_training_8xa100_tp2_pp2_dp2",
        description=(
            "Llama-3-8B training step, batch=64, microbatch=4, prompt=2048, "
            "8xA100-80GB TP=2 PP=2 DP=2 on an NVSwitch fabric. "
            "Reference ~3-5s (Megatron-class). Sanity bound."
        ),
        architecture=LLAMA3_8B,
        model_class=LlamaModel,
        parallelism=ParallelismConfig(tp=2, pp=2, dp=2, ep=1),
        hardware=A100_80GB,
        workload=Workload.TRAINING,
        # Per-DP-rank: global_batch / dp = 64/2 = 32. microbatch_size=4
        # implies num_microbatches=8 (the pipeline-loop multiplier).
        # The WorkloadContext invariant batch_size = microbatch_size × num_microbatches
        # is enforced. Operators size their per-invocation work off
        # microbatch_size × prompt_len.
        workload_ctx=WorkloadContext(
            workload_type=Workload.TRAINING,
            batch_size=32, microbatch_size=4,
            prompt_len=2048, response_len=0, num_microbatches=8,
        ),
        reference_latency_ms=4000.0,  # midpoint of 3-5s
        tolerance_percent=100.0,  # explicit "sanity bound, not tight target"
        metric="total",
        spine_bandwidth_gbps=2400.0,  # A100 NVLink: per-pair ~300 GB/s
        topology_kind="nvswitch",
        gpu_egress_bandwidth_gbps=2400.0,  # A100 NVLink aggregate egress
    ),
    Scenario(
        name="deepseek_v3_prefill_32xh100_ep8_tp4",
        description=(
            "DeepSeek-V3 prefill, batch=1, prompt=4096, 32xH100-80GB EP=8 TP=4 "
            "PP=1 across 4 nodes × 8 GPUs: NVSwitch within node, IB between. "
            "TP shards attention 4-way (a TP group stays intra-node); EP=8 "
            "spans nodes. Reference ~500ms (looser bound)."
        ),
        architecture=DEEPSEEK_V3,
        model_class=V3Model,
        parallelism=ParallelismConfig(tp=4, pp=1, dp=1, ep=8),
        hardware=H100_80GB,
        workload=Workload.PREPARATION,
        workload_ctx=WorkloadContext(
            workload_type=Workload.PREPARATION,
            batch_size=1, microbatch_size=1,
            prompt_len=4096, response_len=0, num_microbatches=1,
        ),
        reference_latency_ms=500.0,
        tolerance_percent=100.0,  # V3 numbers are less stable
        metric="total",
        # H100 NVLink: per-pair ~450 GB/s ≈ 3600 Gbps (intra-node).
        spine_bandwidth_gbps=3600.0,
        topology_kind="multinode",
        hosts_per_node=8,
        # 8x400Gb IB per node ≈ 400 GB/s shared across the (node_a, node_b)
        # pair, then divided by the cross-node connections through the same
        # IB pool.
        inter_node_bandwidth_gbps=400.0 * 8,  # 8 lanes per direction
        gpu_egress_bandwidth_gbps=3600.0,  # H100 NVLink aggregate egress
    ),
]
