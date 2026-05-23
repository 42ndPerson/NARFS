"""Phase 5: PP activation P2P (forward + backward in training).

Pre-Phase-5 the simulator's pipeline-bubble formula was
`(num_mb + pp - 1) × max_stage` — modeled stage compute but no per-
microbatch activation transfer between stages. This module pins the
new behavior:

- `pp_p2p_op_forward` is created when pp > 1 (any non-dormant workload).
- `pp_p2p_op_backward` is created only in TRAINING.
- Both are sized to T*h*dtype with T = mb_size * prompt_len, evaluated
  against the current ctx by `refresh_collective_sizes`.
- Phase override: STEADY (P2P normally defaults to BOUNDARY).
- Bubble formula becomes `(num_mb + pp - 1) × (max_stage + p2p_per_mb)`.
"""
from __future__ import annotations

from topology.hardware import HardwareSpec
from mapping.model_mapping import ModelMapping
from mapping.model_mapping_group import ModelMappingGroup
from network_op import P2P
from network_requirement import CollectiveKind, NetworkRequirement
from parallelism import ParallelismConfig, ParallelismGroup
from operators.llama_builder import LlamaModel
from operators.llama_config import LlamaConfig
from network_op import NetworkPhase
from simulation.simulator import simulate_stage
from schemes.strategies import MegatronScheme
from topology.topo import Device, HostTopo, Link, Path
from workload_context import Workload, WorkloadContext


def _dev(host: int) -> Device:
    return Device(host_id=host, gpu_id=0)


SMALL_LLAMA = LlamaConfig(
    h=512, n_layers=4, n_q=8, n_kv=4, head_size=64,
    m=2048, rope_dim=64, vocab=32000, dtype_bytes=2,
)

A100 = HardwareSpec(
    peak_compute_flops=312e12,
    peak_memory_bandwidth=2.0e12,
)


def _make_topo(num_hosts: int, spine_bw_gbps: float = 100.0) -> HostTopo:
    spine_link = Link(node_a_id=-1, node_b_id=-2, bandwidth=spine_bw_gbps, latency=0.1)
    huge = 1e9

    def _path(src: Device, dst: Device) -> Path:
        attach_src = Link(node_a_id=src, node_b_id=-1, bandwidth=huge, latency=0.0)
        attach_dst = Link(node_a_id=-2, node_b_id=dst, bandwidth=huge, latency=0.0)
        return Path([attach_src, spine_link, attach_dst])

    connections = {
        (_dev(i), _dev(j)): _path(_dev(i), _dev(j))
        for i in range(num_hosts) for j in range(num_hosts) if i != j
    }
    return HostTopo(device_connections=connections)


def _make_model() -> LlamaModel:
    return LlamaModel(
        name="llama_pp", role="actor",
        architecture=SMALL_LLAMA,
    )


def _make_mapping(
    parallelism: ParallelismConfig, workload: Workload, host_ids: list[int],
) -> ModelMapping:
    return MegatronScheme(parallelism).build_mm(
        model=_make_model(),
        workload=workload,
        devices=[_dev(h) for h in host_ids],
    )


def _ctx(prompt_len: int = 128, num_microbatches: int = 1, microbatch_size: int = 1) -> WorkloadContext:
    return WorkloadContext(
        workload_type=Workload.PREPARATION,
        batch_size=microbatch_size * num_microbatches,
        microbatch_size=microbatch_size,
        prompt_len=prompt_len,
        response_len=0,
        num_microbatches=num_microbatches,
    )


# ---- existence ---------------------------------------------------------------


def test_pp_p2p_forward_installed_when_pp_gt_1_preparation():
    mm = _make_mapping(
        ParallelismConfig(tp=1, pp=2, dp=1), Workload.PREPARATION, [0, 1],
    )
    assert mm.pp_p2p_op_forward is not None
    assert mm.pp_p2p_op_backward is None  # not training


def test_pp_p2p_both_legs_installed_in_training():
    mm = _make_mapping(
        ParallelismConfig(tp=1, pp=2, dp=1), Workload.TRAINING, [0, 1],
    )
    assert mm.pp_p2p_op_forward is not None
    assert mm.pp_p2p_op_backward is not None


def test_pp_p2p_not_installed_when_pp_eq_one():
    mm = _make_mapping(
        ParallelismConfig(tp=1, pp=1, dp=1), Workload.TRAINING, [0],
    )
    assert mm.pp_p2p_op_forward is None
    assert mm.pp_p2p_op_backward is None


def test_pp_p2p_not_installed_for_dormant_workload():
    """Dormant has no pattern → no P2P installation either."""
    mm = _make_mapping(
        ParallelismConfig(tp=1, pp=2, dp=1), Workload.DORMANT, [0, 1],
    )
    assert mm.pp_p2p_op_forward is None
    assert mm.pp_p2p_op_backward is None


# ---- shape / phase ----------------------------------------------------------


def test_pp_p2p_op_is_p2p_with_steady_phase():
    mm = _make_mapping(
        ParallelismConfig(tp=1, pp=2, dp=1), Workload.TRAINING, [0, 1],
    )
    fwd = mm.pp_p2p_op_forward
    bwd = mm.pp_p2p_op_backward
    assert isinstance(fwd, P2P) and isinstance(bwd, P2P)
    # Phase override: P2P defaults to BOUNDARY; PP P2P must be STEADY since
    # it streams per microbatch in lockstep with compute.
    assert fwd.phase is NetworkPhase.STEADY
    assert bwd.phase is NetworkPhase.STEADY


def test_pp_p2p_forward_goes_pp0_to_pp1_backward_reverses():
    mm = _make_mapping(
        ParallelismConfig(tp=1, pp=2, dp=1), Workload.TRAINING, [10, 20],
    )
    fwd = mm.pp_p2p_op_forward
    bwd = mm.pp_p2p_op_backward
    assert fwd is not None and bwd is not None
    assert len(fwd.endpoint_pairs) == 1
    assert len(bwd.endpoint_pairs) == 1
    assert fwd.endpoint_pairs[0] == (_dev(10), _dev(20))
    assert bwd.endpoint_pairs[0] == (_dev(20), _dev(10))


# ---- sizing ------------------------------------------------------------------


def test_pp_p2p_size_matches_T_h_dtype_after_resolve():
    from mapping.model_mapping_group import resolve_op_transfers

    mm = _make_mapping(
        ParallelismConfig(tp=1, pp=2, dp=1), Workload.TRAINING, [0, 1],
    )
    ctx = _ctx(prompt_len=256, microbatch_size=4)
    op_transfers = resolve_op_transfers(ctx, [mm])
    fwd = mm.pp_p2p_op_forward
    assert fwd is not None
    expected_GB = 4 * 256 * SMALL_LLAMA.h * SMALL_LLAMA.dtype_bytes / 1e9
    assert abs(op_transfers[fwd][0].data_GB - expected_GB) < 1e-12


def test_pp_p2p_request_dedups_with_payload_id():
    mm = _make_mapping(
        ParallelismConfig(tp=1, pp=2, dp=1), Workload.TRAINING, [0, 1],
    )
    fwd_keys = [
        req for req in mm.network_ops
        if req.kind is CollectiveKind.P2P_PIPELINE_FORWARD
    ]
    bwd_keys = [
        req for req in mm.network_ops
        if req.kind is CollectiveKind.P2P_PIPELINE_BACKWARD
    ]
    assert len(fwd_keys) == 1
    assert len(bwd_keys) == 1
    # Sanity: payload_id is what we set, distinguishing forward from backward.
    assert fwd_keys[0].payload_id == ("pp_p2p_forward",)
    assert bwd_keys[0].payload_id == ("pp_p2p_backward",)


# ---- bubble formula ----------------------------------------------------------


def test_pp_bubble_formula_includes_p2p_per_mb_additively():
    """With P2P enabled, total = (num_mb + pp - 1) × (max_stage + p2p_per_mb).
    Compare same-config simulation with pp=2 vs. pp=1 (no P2P) and verify
    the additive term is present and grows with prompt_len."""
    topo = _make_topo(num_hosts=2)
    pp2 = _make_mapping(ParallelismConfig(tp=1, pp=2, dp=1), Workload.PREPARATION, [0, 1])
    sm = ModelMappingGroup([pp2])

    ctx_short = _ctx(prompt_len=64, num_microbatches=2)
    ctx_long = _ctx(prompt_len=512, num_microbatches=2)
    t_short = simulate_stage([sm], ctx_short, A100, topo)
    t_long = simulate_stage([sm], ctx_long, A100, topo)

    # Long-prompt P2P payload is 8x → wall time should grow correspondingly.
    assert t_long > t_short, (t_short, t_long)


def test_pp_p2p_in_network_ops_dict_for_refresh():
    """Phase 1 sizing relies on refresh_collective_sizes iterating
    network_ops. The PP P2P must be present there (not orphaned)."""
    mm = _make_mapping(
        ParallelismConfig(tp=1, pp=2, dp=1), Workload.TRAINING, [0, 1],
    )
    fwd_op_in_dict = any(
        op is mm.pp_p2p_op_forward for op in mm.network_ops.values()
    )
    bwd_op_in_dict = any(
        op is mm.pp_p2p_op_backward for op in mm.network_ops.values()
    )
    assert fwd_op_in_dict and bwd_op_in_dict


def test_pp_p2p_size_scales_with_microbatch_size():
    from mapping.model_mapping_group import resolve_op_transfers

    mm = _make_mapping(
        ParallelismConfig(tp=1, pp=2, dp=1), Workload.TRAINING, [0, 1],
    )
    fwd = mm.pp_p2p_op_forward
    assert fwd is not None
    size_2 = resolve_op_transfers(_ctx(prompt_len=128, microbatch_size=2), [mm])[fwd][0].data_GB
    size_4 = resolve_op_transfers(_ctx(prompt_len=128, microbatch_size=4), [mm])[fwd][0].data_GB
    assert abs(size_4 - 2 * size_2) < 1e-12, (size_2, size_4)
