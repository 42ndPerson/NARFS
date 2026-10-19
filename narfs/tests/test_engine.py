"""Variation helpers and the generation engine: the tail against an exact
Monte Carlo of the FIFO scheduler, segment convergence, and engine policies."""
import heapq

import numpy as np
import pytest

from narfs.blocks import Collective, Compute, OnDevices, Par, Work
from narfs.costs import ComputeModel, HostModel
from narfs.engine import lower_generate, slots
from narfs.evaluate import Evaluator, evaluate
from narfs.hardware import DATA, H100_SXM
from narfs.model.architecture import LLAMA3_70B, LLAMA3_8B
from narfs.network.comm import load_nccl
from narfs.network.presets import hgx_h100
from narfs.parallel.binding import Binder
from narfs.parallel.mesh import Degrees, map_devices
from narfs.physics import Physics
from narfs.profiles import load_engine
from narfs.tests.helpers import ToyLeaves
from narfs.tests.walk import of_type
from narfs.variation import Dist, exponential_tail_max, trajectory, varying_repeat

LOGNORMAL = Dist.lognormal(5.6, 0.8, 4096)
EPROF, VLLM = load_engine("vllm-573f2f9-h100")


def test_dist_moments_and_cap():
    d = Dist.histogram([100, 200, 400], [0.5, 0.25, 0.25])
    assert d.mean == pytest.approx(200.0)
    assert d.second == pytest.approx(0.5 * 1e4 + 0.25 * 4e4 + 0.25 * 1.6e5)
    assert d.expected_max(1) == pytest.approx(d.mean)
    assert d.expected_max(10_000) == pytest.approx(400.0)
    capped = Dist.histogram([100, 5000], [0.98, 0.02], cap=1024)
    assert capped.cap == 1024 and capped.expected_max(2048) == pytest.approx(1024.0, rel=1e-6)


def test_fitted_tails_and_the_closed_form():
    rng = np.random.default_rng(3)
    s = np.minimum(rng.lognormal(5.6, 0.8, 1000), 4096)
    e = Dist.from_samples(s, 4096, tail="exp")
    assert e.info["tail"] == "exp" and e.info["xi"] == 0.0 and e.info["p_u"] == pytest.approx(0.1, abs=0.01)
    closed = exponential_tail_max(e.info["u"], e.info["beta"], e.info["p_u"], 4096, 2048)
    assert e.expected_max(2048) == pytest.approx(closed, rel=0.005)
    g = Dist.from_samples(s, 4096)                       # 1,000 observations: auto picks the GPD
    assert g.info["tail"] == "gpd" and g.info["xi"] > 0
    assert sum(g.pmf) == pytest.approx(1.0)
    assert Dist.from_samples(s[:50], 4096).info["tail"] == "none"     # too few exceedances


def _fifo_end(lengths, cap):
    free = [0] * min(cap, len(lengths))
    heapq.heapify(free)
    end = 0
    for L in lengths:
        s = heapq.heappop(free)
        heapq.heappush(free, s + L)
        end = max(end, s + L)
    return end


@pytest.mark.parametrize("slots_, expected", [(256, 3817), (64, 4426)])
def test_tail_matches_the_exact_fifo_monte_carlo(slots_, expected):
    rng = np.random.default_rng(1)
    L = np.arange(1, 4097)
    mc = []
    for _ in range(60):
        x = rng.choice(L, size=2048, p=LOGNORMAL.pmf)
        mc.append(max(_fifo_end(x[r::8], slots_) for r in range(8)))
    mc = float(np.mean(mc))
    assert mc == pytest.approx(expected, rel=0.03)             # the design's Monte Carlo values
    assert trajectory(LOGNORMAL, 2048, slots_, 8).end == pytest.approx(mc, rel=0.025)


def test_sixteen_segments_match_the_per_step_sum():
    toy = ToyLeaves({"gpu0": 1.0})
    bind = {"g": ["gpu0"]}
    # a step that grows with batch and context, as decode does
    body = lambda s: Compute(1e-3 * (1 + 0.004 * s.active + 2e-6 * s.active * (512 + s.age)))
    for slots_ in (256, 64):
        coarse = evaluate(OnDevices("g", varying_repeat(LOGNORMAL, 2048, slots_, 8, body, 16)), toy, bind).lo
        fine = evaluate(OnDevices("g", varying_repeat(LOGNORMAL, 2048, slots_, 8, body, 512)), toy, bind).lo
        assert coarse == pytest.approx(fine, rel=0.002)


def test_active_count_never_exceeds_the_slots():
    t = trajectory(LOGNORMAL, 2048, 64, 8)
    assert max(s.active for s in t.segments) <= 64
    assert min(s.active for s in t.segments) >= 1
    assert sum(s.steps for s in t.segments) == pytest.approx(t.end)


def test_slots_follow_kv_capacity_and_prefix_sharing():
    # per slot: 512 prompt tokens / 8 samples + E[L^2] / (2 E[L]) generated tokens
    per = 512 / 8 + LOGNORMAL.second / (2 * LOGNORMAL.mean)
    assert slots(EPROF, 8, 512, LOGNORMAL, 100 * per) == 100
    assert slots(EPROF, 8, 512, LOGNORMAL, 1e12) == EPROF.max_num_seqs
    assert slots(EPROF.with_(prefix_caching=False), 8, 512, LOGNORMAL, 100 * per) < 100
    assert EPROF.padded(1) == 1 and EPROF.padded(9) == 16 and EPROF.padded(600) == 600


def physics(nodes=1):
    topo = hgx_h100(nodes)
    phys = Physics(topo, {"nccl": load_nccl(DATA / "nccl-2.32.3.json"), "vllm": VLLM},
                   ComputeModel({"h100-sxm": H100_SXM}), HostModel({"vllm": EPROF.host_step}))
    return topo, phys


def test_engine_structure_and_policies():
    tree = lower_generate("g", LLAMA3_70B, Degrees(tp=8), EPROF, 64, 8, 512, LOGNORMAL)
    assert {c.lib for c in of_type(tree, Collective)} == {"vllm"}
    assert of_type(tree, Work) and not of_type(tree, Par)                    # synchronous host scheduling
    overlapped = lower_generate("g", LLAMA3_70B, Degrees(tp=8), EPROF.with_(async_scheduling=True), 64, 8, 512,
                                LOGNORMAL)
    assert of_type(overlapped, Par)
    assert "decode steps" in tree.body.name
    with pytest.raises(ValueError, match="pp > 1"):
        lower_generate("g", LLAMA3_8B, Degrees(pp=2), EPROF, 64, 8, 512, LOGNORMAL)


def test_engine_time_by_replicas_and_flags():
    topo, phys = physics(1)
    def run(deg):
        tree = lower_generate("g", LLAMA3_8B, deg, EPROF, 256, 8, 512, LOGNORMAL)
        b = Binder({"g": map_devices(range(deg.size), deg)}, host_ids=lambda x: topo.rid("host", x))
        return Evaluator(phys, b)(tree).lo
    one, eight = run(Degrees()), run(Degrees(dp=8))
    assert 2 < one / eight < 8                     # 8 replicas share the requests; the tail does not shrink 8x
    assert {"host_step", "reserve", "allreduce_latency"} <= set(EPROF.flags)


def test_a_measured_engine_profile_keeps_the_base_paths(tmp_path):
    f = tmp_path / "vllm-cluster-a.json"
    f.write_text('{"kind": "engine", "base": "vllm-573f2f9-h100", "values": {'
                 '"host_step_s": {"value": [0.0008, 4e-06], "kind": "measured", "source": "decode sweep"}}}')
    prof, lib = load_engine(f)
    assert prof.host_step == (0.0008, 4e-06) and "host_step" not in prof.flags
    assert {"reserve", "allreduce_latency"} <= set(prof.flags)
    assert prof.name == EPROF.name and prof.version == EPROF.version and lib.paths == VLLM.paths
