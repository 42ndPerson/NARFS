"""Tests for AutoMappingBridge.

Avoids importing the auto_mapping package at runtime — the bridge
duck-types its inputs, so the tests construct lightweight stand-ins with
the same attribute shapes as `verl.utils.topology.Topology`,
`HostSpec`, `auto_mapping.assignment.GroupAssignment`, and
`auto_mapping.solver.Workload`.

Coverage:
- Cached HostTopo construction from a multi-host, multi-block Topology.
- TP-within-host shard mapping (single-host submesh).
- PP-across-hosts shard mapping (full-block submesh).
- simulate_per_model wires through to SubmeshMapping.simulate_isolated
  (positive total time, no exceptions on a small Llama).
- simulate_iteration wires through to simulate_rlhf_iteration with
  multiple stage types and groups, returning total_time > 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from bridge import AutoMappingBridge
from hardware import HardwareSpec
from operators.llama_builder import build_llama_pattern
from operators.llama_config import LlamaConfig


# ---- stand-in input types ----------------------------------------------------


@dataclass
class _HostSpec:
    host_id: str
    block_id: str
    num_gpus: int
    gpu_type: str = "test"


@dataclass
class _Topology:
    """Duck-types verl.utils.topology.Topology."""
    hosts: list[_HostSpec]
    intra_host_bw: float
    intra_block_bw: float
    inter_block_bw: float
    block_pair_bw: dict[tuple[str, str], float] = field(default_factory=dict)

    def num_hosts(self) -> int:
        return len(self.hosts)

    def gpus_per_host(self) -> int:
        return self.hosts[0].num_gpus

    def hosts_by_block(self) -> dict[str, list[_HostSpec]]:
        out: dict[str, list[_HostSpec]] = {}
        for h in self.hosts:
            out.setdefault(h.block_id, []).append(h)
        return out

    def bandwidth_between(self, h1: _HostSpec, h2: _HostSpec) -> float:
        if h1.host_id == h2.host_id:
            return self.intra_host_bw
        if h1.block_id == h2.block_id:
            return self.intra_block_bw
        return self.inter_block_bw


@dataclass
class _GroupAssignment:
    """Duck-types auto_mapping.assignment.GroupAssignment."""
    group_index: int
    submesh_shape: tuple[int, int]
    host_ids: list[str]
    gpus_per_host: list[int]
    block_ids: list[str]


@dataclass
class _AutoWorkload:
    """Duck-types auto_mapping.solver.Workload."""
    d_in: int
    d_out: int
    compute_type: str


@dataclass
class _DataflowGraph:
    """Duck-types auto_mapping.solver.DataflowGraph (only the read API the bridge uses)."""
    stages: list[list[int]]

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    def roles_in_stage(self, stage_idx: int) -> list[int]:
        return self.stages[stage_idx]


# ---- shared fixtures ---------------------------------------------------------


SMALL_LLAMA = LlamaConfig(
    h=512, n_layers=4, n_q=8, n_kv=4, head_size=64,
    m=2048, rope_dim=64, vocab=32000, dtype_bytes=2,
)
A100 = HardwareSpec(
    peak_compute_flops=312e12,
    peak_memory_bandwidth=2.0e12,
    ridge_flops_per_byte=156.0,
)


def _two_block_topology(hosts_per_block: int = 2, gpus_per_host: int = 4) -> _Topology:
    """4 hosts across 2 blocks, gpus_per_host each."""
    hosts: list[_HostSpec] = []
    for b in range(2):
        for h in range(hosts_per_block):
            hosts.append(_HostSpec(
                host_id=f"host_{b}_{h}",
                block_id=f"block_{b}",
                num_gpus=gpus_per_host,
            ))
    return _Topology(
        hosts=hosts,
        intra_host_bw=600.0,   # NVSwitch-class
        intra_block_bw=200.0,  # IB intra-rack
        inter_block_bw=50.0,   # cross-rack spine
    )


def _make_bridge(topology: _Topology, model_ids: list[int]) -> AutoMappingBridge:
    return AutoMappingBridge(
        topology=topology,
        hardware=A100,
        model_architectures={mid: SMALL_LLAMA for mid in model_ids},
        model_build_patterns={mid: build_llama_pattern for mid in model_ids},
    )


# ---- tests -------------------------------------------------------------------


def test_bridge_caches_host_topo_built_from_topology():
    topo = _two_block_topology()
    bridge = _make_bridge(topo, model_ids=[0])

    # The cached HostTopo should have 4 hosts → 12 directed (i, j) connections.
    ht = bridge.host_topo
    assert ht.intra_host_bandwidth == 600.0
    n = len(topo.hosts)
    assert len(ht.hosts_connections) == n * (n - 1)

    # Bandwidth tiers preserved: intra-block pair has 200 Gbps link, inter-block has 50.
    h0_int = bridge.host_id_int("host_0_0")
    h0_2_int = bridge.host_id_int("host_0_1")
    h1_int = bridge.host_id_int("host_1_0")
    intra_block_path = ht.hosts_connections[(h0_int, h0_2_int)]
    inter_block_path = ht.hosts_connections[(h0_int, h1_int)]
    assert intra_block_path.bandwidth == 200.0
    assert inter_block_path.bandwidth == 50.0


def test_bridge_repeated_construction_does_not_rebuild_topo():
    topo = _two_block_topology()
    bridge = _make_bridge(topo, model_ids=[0])
    first = bridge.host_topo
    second = bridge.host_topo
    assert first is second  # cached, same object


def test_simulate_per_model_routes_to_simulate_isolated_single_host_submesh():
    # 1-host submesh (TP=4 within one host).
    topo = _two_block_topology(hosts_per_block=2, gpus_per_host=4)
    bridge = _make_bridge(topo, model_ids=[0])

    assignment = _GroupAssignment(
        group_index=0,
        submesh_shape=(1, 4),
        host_ids=["host_0_0"],
        gpus_per_host=[4],
        block_ids=["block_0"],
    )

    device_mesh = SimpleNamespace(assignment=assignment)

    w = _AutoWorkload(d_in=64, d_out=0, compute_type="inference")
    # parallelism_plan = (P, T, D) = (1, 4, 1)
    t = bridge.simulate_per_model((1, 4, 1), 0, w, device_mesh)
    assert t > 0.0


def test_simulate_per_model_pp_across_hosts_full_block_submesh():
    # (h=2, w=4) full-block submesh: PP=2 across hosts, TP=4 within each.
    topo = _two_block_topology(hosts_per_block=2, gpus_per_host=4)
    bridge = _make_bridge(topo, model_ids=[0])

    assignment = _GroupAssignment(
        group_index=0,
        submesh_shape=(2, 4),
        host_ids=["host_0_0", "host_0_1"],
        gpus_per_host=[4, 4],
        block_ids=["block_0", "block_0"],
    )

    device_mesh = SimpleNamespace(assignment=assignment)

    w = _AutoWorkload(d_in=64, d_out=0, compute_type="training")
    t = bridge.simulate_per_model((2, 4, 1), 0, w, device_mesh)
    assert t > 0.0


def test_pp_across_hosts_co_locates_tp_within_host():
    """Verify the shard placement: each PP rank gets a whole host's worth of TP shards."""
    topo = _two_block_topology(hosts_per_block=2, gpus_per_host=4)
    bridge = _make_bridge(topo, model_ids=[0])

    assignment = _GroupAssignment(
        group_index=0, submesh_shape=(2, 4),
        host_ids=["host_0_0", "host_0_1"],
        gpus_per_host=[4, 4],
        block_ids=["block_0", "block_0"],
    )

    # Build the mesh the bridge would build, then check shard mapping shape.
    mesh = bridge._mesh_for_assignment(assignment)
    assert mesh.num_devices_per_host == 4
    assert mesh.host_ids == [bridge.host_id_int("host_0_0"), bridge.host_id_int("host_0_1")]

    from network_requirement import ParallelismConfig
    from model import Model

    parallelism = ParallelismConfig(tp=4, pp=2, dp=1, ep=1)
    model = Model(
        name="probe", role="actor",
        architecture=SMALL_LLAMA, build_pattern=build_llama_pattern,
    )
    s2h, _ = bridge._assign_shards(model, parallelism, mesh, assignment)

    # All TP ranks for pp=0 should share host_0_0; for pp=1, host_0_1.
    h0 = bridge.host_id_int("host_0_0")
    h1 = bridge.host_id_int("host_0_1")
    pp0_hosts = {s2h[s] for s in s2h if s.pp == 0}
    pp1_hosts = {s2h[s] for s in s2h if s.pp == 1}
    assert pp0_hosts == {h0}, f"PP rank 0 should be on host_0_0, got {pp0_hosts}"
    assert pp1_hosts == {h1}, f"PP rank 1 should be on host_0_1, got {pp1_hosts}"


def test_simulate_iteration_runs_full_rlhf_timeline():
    """Three models in three stage types → three simulator stages, total > 0."""
    topo = _two_block_topology(hosts_per_block=2, gpus_per_host=4)
    bridge = _make_bridge(topo, model_ids=[0, 1, 2])

    g = [(0,), (1,), (2,)]  # three placement groups, one model each
    assignments = [
        _GroupAssignment(
            group_index=0, submesh_shape=(1, 4),
            host_ids=["host_0_0"], gpus_per_host=[4],
            block_ids=["block_0"],
        ),
        _GroupAssignment(
            group_index=1, submesh_shape=(1, 4),
            host_ids=["host_0_1"], gpus_per_host=[4],
            block_ids=["block_0"],
        ),
        _GroupAssignment(
            group_index=2, submesh_shape=(1, 4),
            host_ids=["host_1_0"], gpus_per_host=[4],
            block_ids=["block_1"],
        ),
    ]
    workloads = {
        0: _AutoWorkload(d_in=128, d_out=64, compute_type="generation"),
        1: _AutoWorkload(d_in=128, d_out=0, compute_type="inference"),
        2: _AutoWorkload(d_in=128, d_out=0, compute_type="training"),
    }
    # auto_parallel returns (cost, (P, T, D)); bridge unpacks defensively.
    l_parallel = {
        0: (0.1, (1, 4, 1)),
        1: (0.1, (1, 4, 1)),
        2: (0.1, (1, 4, 1)),
    }

    total = bridge.simulate_iteration(g, l_parallel, workloads, assignments)
    assert total > 0.0


def test_simulate_iteration_groups_models_by_stage_type():
    """Two models with the same compute_type should land in the same stage."""
    topo = _two_block_topology()
    bridge = _make_bridge(topo, model_ids=[0, 1])

    g = [(0, 1)]  # single placement group with two co-located models
    assignments = [
        _GroupAssignment(
            group_index=0, submesh_shape=(1, 4),
            host_ids=["host_0_0"], gpus_per_host=[4],
            block_ids=["block_0"],
        ),
    ]
    workloads = {
        0: _AutoWorkload(d_in=64, d_out=0, compute_type="training"),
        1: _AutoWorkload(d_in=64, d_out=0, compute_type="training"),
    }
    l_parallel = {0: (1, 4, 1), 1: (1, 4, 1)}  # bare (P,T,D) form

    timeline = bridge._build_timeline(g, l_parallel, workloads, assignments)
    # Only one stage type (training) is populated → 1 stage with 1 submesh.
    assert len(timeline.stages) == 1
    assert len(timeline.stages[0]) == 1
    assert len(timeline.stages[0][0].model_mappings) == 2


def test_unknown_compute_type_raises():
    topo = _two_block_topology()
    bridge = _make_bridge(topo, model_ids=[0])
    assignment = _GroupAssignment(
        group_index=0, submesh_shape=(1, 4),
        host_ids=["host_0_0"], gpus_per_host=[4],
        block_ids=["block_0"],
    )
    workloads = {0: _AutoWorkload(d_in=64, d_out=0, compute_type="bogus")}
    with pytest.raises(ValueError, match="compute_type"):
        bridge.simulate_iteration([(0,)], {0: (1, 4, 1)}, workloads, [assignment])


def test_missing_architecture_falls_back_to_qwen_hardwire():
    """Roles without an explicit architecture fall back to the hardwired
    Qwen2.5-0.5B-Instruct config (TODO in bridge.py). Pre-hardwire the bridge
    raised; this test pins the new behavior so the hardwire's removal is
    deliberate when it lands.
    """
    topo = _two_block_topology()
    # Bridge constructed with NO architecture for model 99 (and an empty
    # model_architectures dict to confirm the hardwire fires even when
    # nothing is supplied).
    bridge = AutoMappingBridge(
        topology=topo,
        hardware=A100,
        # model_architectures / model_build_patterns intentionally omitted
        # to exercise the Qwen hardwire end-to-end.
    )
    assignment = _GroupAssignment(
        group_index=0, submesh_shape=(1, 2),
        host_ids=["host_0_0"], gpus_per_host=[2],
        block_ids=["block_0"],
    )
    w = _AutoWorkload(d_in=64, d_out=0, compute_type="training")
    device_mesh = SimpleNamespace(assignment=assignment)

    # Qwen2.5-0.5B has n_q=14, n_kv=2 — so the only valid TP degrees are 1 and 2
    # (the only divisors of 2 that also divide 14). Use TP=2, PP=1, DP=1.
    t = bridge.simulate_per_model((1, 2, 1), 99, w, device_mesh)
    assert t > 0.0


def test_bridge_returns_inf_on_infeasible_parallelism():
    """auto_parallel iterates many TP/PP combos; some violate architecture
    divisibility constraints (e.g. Qwen2.5-0.5B has n_kv=2, so TP=4 fails the
    `n_kv % tp == 0` check inside build_llama_layer). The bridge swallows
    the resulting ValueError and returns inf so the search can skip the
    infeasible point and keep going.
    """
    topo = _two_block_topology(hosts_per_block=2, gpus_per_host=4)
    # Default Qwen hardwire (no model_architectures supplied).
    bridge = AutoMappingBridge(topology=topo, hardware=A100)

    assignment = _GroupAssignment(
        group_index=0, submesh_shape=(1, 4),
        host_ids=["host_0_0"], gpus_per_host=[4],
        block_ids=["block_0"],
    )
    w = _AutoWorkload(d_in=64, d_out=0, compute_type="training")
    device_mesh = SimpleNamespace(assignment=assignment)

    # Qwen2.5-0.5B has n_kv=2 → TP=4 is infeasible.
    cost = bridge.simulate_per_model((1, 4, 1), 0, w, device_mesh)
    assert cost == float("inf"), f"expected inf for infeasible TP=4, got {cost}"

    # The variadic dispatcher (the actual auto_parallel call path) must also
    # surface inf, not raise.
    cost_via_dispatch = bridge.simulate(
        (1, 4, 1), 0, w, device_mesh, assignment=assignment
    )
    assert cost_via_dispatch == float("inf")

    # A feasible TP=2 still produces a finite cost — the inf path is
    # narrowly scoped to the actually-infeasible combos.
    cost_ok = bridge.simulate_per_model((1, 2, 1), 0, w, device_mesh)
    assert 0.0 < cost_ok < float("inf")


def test_explicit_architecture_overrides_qwen_hardwire():
    """Caller-supplied model_architectures takes precedence over the hardwire."""
    topo = _two_block_topology()
    bridge = _make_bridge(topo, model_ids=[42])  # SMALL_LLAMA for model 42

    assignment = _GroupAssignment(
        group_index=0, submesh_shape=(1, 4),
        host_ids=["host_0_0"], gpus_per_host=[4],
        block_ids=["block_0"],
    )
    w = _AutoWorkload(d_in=64, d_out=0, compute_type="training")
    device_mesh = SimpleNamespace(assignment=assignment)

    # If the hardwire were taking precedence, TP=4 would fail Qwen's
    # n_kv=2 divisibility check. SMALL_LLAMA has n_kv=4, so TP=4 succeeds.
    t = bridge.simulate_per_model((1, 4, 1), 42, w, device_mesh)
    assert t > 0.0


def test_missing_assignment_on_device_mesh_raises():
    topo = _two_block_topology()
    bridge = _make_bridge(topo, model_ids=[0])
    w = _AutoWorkload(d_in=64, d_out=0, compute_type="training")

    device_mesh_no_assignment = SimpleNamespace()  # no assignment attribute

    with pytest.raises(RuntimeError, match="no GroupAssignment"):
        bridge.simulate_per_model((1, 4, 1), 0, w, device_mesh_no_assignment)


def test_simulate_per_model_explicit_assignment_kwarg():
    """The new preferred call form: pass assignment explicitly, bypass
    device_mesh.assignment entirely. This is what auto_parallel does after
    the integration."""
    topo = _two_block_topology(hosts_per_block=2, gpus_per_host=4)
    bridge = _make_bridge(topo, model_ids=[0])

    assignment = _GroupAssignment(
        group_index=0, submesh_shape=(1, 4),
        host_ids=["host_0_0"], gpus_per_host=[4],
        block_ids=["block_0"],
    )
    bare_device_mesh = SimpleNamespace()  # no assignment attribute
    w = _AutoWorkload(d_in=64, d_out=0, compute_type="training")

    t = bridge.simulate_per_model((1, 4, 1), 0, w, bare_device_mesh, assignment=assignment)
    assert t > 0.0


def test_dispatcher_routes_simulate_kwargs_to_simulate_per_model():
    """`bridge.simulate(parallelism, l, W, device_mesh, assignment=...)` is the
    shape `simulators.simulate(...)` forwards from auto_parallel. Verify the
    kwarg is honored end-to-end."""
    topo = _two_block_topology()
    bridge = _make_bridge(topo, model_ids=[0])

    assignment = _GroupAssignment(
        group_index=0, submesh_shape=(1, 4),
        host_ids=["host_0_0"], gpus_per_host=[4],
        block_ids=["block_0"],
    )
    bare_device_mesh = SimpleNamespace()
    w = _AutoWorkload(d_in=64, d_out=0, compute_type="training")

    t = bridge.simulate((1, 4, 1), 0, w, bare_device_mesh, assignment=assignment)
    assert t > 0.0


def test_simulate_iteration_with_dataflow_graph_uses_dag_stages():
    """When a DataflowGraph is passed, stage construction follows
    `dag.stages` rather than the compute_type-string fallback. A model in two
    stages (e.g. an actor that rolls out and trains) should appear in both."""
    topo = _two_block_topology()
    bridge = _make_bridge(topo, model_ids=[0, 1])

    assignments = [
        _GroupAssignment(
            group_index=0, submesh_shape=(1, 4),
            host_ids=["host_0_0"], gpus_per_host=[4],
            block_ids=["block_0"],
        ),
        _GroupAssignment(
            group_index=1, submesh_shape=(1, 4),
            host_ids=["host_0_1"], gpus_per_host=[4],
            block_ids=["block_0"],
        ),
    ]
    g = [(0,), (1,)]
    workloads = {
        0: _AutoWorkload(d_in=64, d_out=8, compute_type="generation"),
        1: _AutoWorkload(d_in=64, d_out=0, compute_type="training"),
    }
    l_parallel = {0: (1, 4, 1), 1: (1, 4, 1)}
    # Three stages: rollout (model 0), preparation (BOTH), training (model 1).
    # Model 0 contributes in stage 0 and stage 1; model 1 contributes in
    # stage 1 and stage 2. The compute_type fallback can't express this.
    dag = _DataflowGraph(stages=[[0], [0, 1], [1]])

    timeline = bridge._build_timeline(g, l_parallel, workloads, assignments, dataflow_graph=dag)
    assert len(timeline.stages) == 3

    # Stage 0: only model 0's submesh
    assert len(timeline.stages[0]) == 1
    assert len(timeline.stages[0][0].model_mappings) == 1
    # Stage 1: both submeshes (one for each group with a model in this stage)
    assert len(timeline.stages[1]) == 2
    # Stage 2: only model 1's submesh
    assert len(timeline.stages[2]) == 1
    assert len(timeline.stages[2][0].model_mappings) == 1


def test_simulate_iteration_with_dag_returns_positive_total():
    """End-to-end with DataflowGraph: bridge.simulate_iteration should run."""
    topo = _two_block_topology()
    bridge = _make_bridge(topo, model_ids=[0])

    assignments = [
        _GroupAssignment(
            group_index=0, submesh_shape=(1, 4),
            host_ids=["host_0_0"], gpus_per_host=[4],
            block_ids=["block_0"],
        ),
    ]
    g = [(0,)]
    workloads = {0: _AutoWorkload(d_in=128, d_out=0, compute_type="training")}
    l_parallel = {0: (1, 4, 1)}
    dag = _DataflowGraph(stages=[[0]])

    total = bridge.simulate_iteration(g, l_parallel, workloads, assignments, dataflow_graph=dag)
    assert total > 0.0
