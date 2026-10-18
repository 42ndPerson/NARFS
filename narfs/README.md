# narfs — RLHF placement simulator (rebuild)

A new simulator that sits beside the original code (`topology/`, `operators/`,
`schemes/`, `mapping/`, `simulation/`). Nothing in the original code changes,
and the original tests still pass.

The design goal: an optimizer can evaluate one **placement candidate** in a
few milliseconds, at 1,600 GPUs, and still see link contention, irregular
layouts and comm/compute overlap.

## How to run

```bash
pip install -r narfs/requirements.txt
python -m pytest narfs/tests          # from the repository root
```

```python
from narfs.tests.cases import ppo_case       # a 128-GPU PPO example
prob, cand, sim = ppo_case(nodes=16)
r = sim.evaluate(cand)                        # None if not feasible
print(r.time, sim.bounds(cand))               # fluid lower bound and upper bound
for row in sim.explain(cand)[-10:]:           # (block, name, seconds, hot resource)
    print(row)

from narfs.optimize import anneal
best = anneal(sim, cand, steps=200)
```

## Structure

The levels, from the outside in. The optimizer searches levels 1–3; all four
are in the hot loop.

| Level | What it decides | Code |
|---|---|---|
| 1. Calls → device groups | which RLHF calls share which devices | `rlhf.py` (`Candidate`, `CallPlan`) |
| 2. Degrees + sharding rules | local shapes per rank | `parallel/rules.py` (`ShardingRules`, `local_dims`) |
| 3. Device mapping | mesh coordinate → device ID (any list, irregular allowed) | `parallel/mesh.py` (`Placement`, `map_devices`) |
| 4. Expansion | a small block tree per call | `expand.py`, `model/library.py`, `profiles.py` |

Expansion does not depend on level 3. A collective names a mesh axis, and the
evaluator gets the device groups from the placement of the enclosing `Call`.

### Blocks (`blocks.py`) and their closed forms (`evaluate.py`)

Every block returns a `Summary(t, demand, mem_peak)`. `demand` is a vector
over the global resource index (bytes on links, busy seconds on `gpu`).

| Block | Time (fluid model, `lo`) | Upper bound (`hi`) |
|---|---|---|
| `Op` | roofline with GEMV / small-GEMM tiers, or a measured table entry | same |
| `Collective`, `StageP2P`, `Transfers`, `Stream` | `max(demand / cap)` + steps · α | same |
| `Seq` | Σ tᵢ | same |
| `Par` | `max(max tᵢ, max_r Σ Dᵢ(r) / C(r))` | Σ tᵢ |
| `Repeat(n, body)` | n · t (identical, independent repetitions only) | same |
| `Pipeline` | `Σ t_s / v + (nmb − 1/v) · slot`, slot = fluid Par over stages | slot = max t_s + shared bound |
| `FlowShop2(a, b)` | `max_k (a₁ + … + a_k + b_k + … + b_K)`; buffer recurrence; `count_a=False` gives only the exposed part of b | Σ a + Σ b |
| `AsyncPipe` | `max(T_p, T_c, shared bound)` | T_p + T_c |
| `GenBatch` | continuous batching: saturated phase + tail over the expected longest response | same |
| `Call` | body; marks its devices busy | same |

Overlap lives only in structure (`Par`, `FlowShop2`, `Pipeline`, `AsyncPipe`).
The framework profile chooses the structure (for example blocking TP
all-reduce, or a `FlowShop2` of GEMM chunks and reduce-scatter chunks). There
are no fitted overlap factors.

### Network (`network/`)

- `topology.py`: a rail-optimized cluster as integer arrays. Directed
  resources with explicit IDs: `nvl_tx`, `nvl_rx`, `nic_tx`, `nic_rx`, `pcie`,
  `gpu`, `leaf_up`, `leaf_down`, `spine`. `route()` is arithmetic; no paths are
  stored.
- `kernels.py`: the **only** array-style code. Each kernel handles every
  instance of one collective class in one `bincount`, so cost grows with the
  member count, not with device pairs.

### Memory (`memory.py`)

Model state (weights, gradients, optimizer state) is counted once per model
per device. The largest transient of the calls on a device is added to it.
Generation gets its KV capacity from the free HBM.

## Measured speed

Build + feasibility check + evaluation of one PPO candidate (actor 70B,
critic and reward 8B), on the development container. These are
measurements of this code on one machine, not a promise:

| GPUs | ms per candidate |
|---|---|
| 128 | ≈ 10 |
| 512 | ≈ 12 |
| 1,600 | ≈ 17 |

`tests/test_system.py::test_hot_loop_budget` fails above 50 ms at 512 GPUs.

**The simulated times are not validated against real runs.** No profiling
data was used. Treat absolute times as uncalibrated; the tests check
structure, hand-computed closed forms and invariants, not accuracy.

## Deviations from the plan

- `Summary` has one memory field, `mem_peak`, not `mem_resident` +
  `mem_transient`. Model state is added once per model by the memory model,
  so blocks carry only the transient peak.
- A per-evaluation memo of collective results (`EvalContext._collectives`)
  was added. It holds results only inside one evaluation. There is no cache
  across candidates and no delta evaluation yet.
- EP ranks hold different data (Megatron-Core and DP + EP engines): the data
  replicas are dp × ep. Non-expert gradients sync over the `("dp", "ep")`
  groups, and routed-expert gradients over `dp`. A `Collective` can name a
  tuple of axes for this.

## Known limitations

- Generation requires pp = 1.
- ZeRO-3 / FSDP is not implemented (`zero_stage` 0, 1, 2 only).
- Routed experts are not TP-sharded; they are replicated over the tp ranks.
- The PP stage split is even by layer count, not balanced by cost.
- The `hi` bound is "no overlap", not a strict worst case. A Pipeline keeps
  its schedule in `hi`.
- C_eff (efficiency for links shared by flows of different sizes) is not
  modeled; the fluid model assumes perfect sharing.
- Import adapter only for Hugging Face `config.json` (Llama, Mixtral/Qwen MoE,
  DeepSeek-V3 MLA). There are no Megatron or vLLM config adapters.
- No profiling script. `CostTable.load` reads measured rows if you have them.
- The DP sync window is the backward of stage 0 only.

## Layout

```
narfs/
  hardware.py        DeviceSpec and presets
  network/           topology.py, kernels.py (array code), presets.py
  blocks.py          block types
  costs.py           operator cost: table, then roofline
  evaluate.py        closed forms, bounds, explain
  parallel/          mesh.py (Degrees, Placement), rules.py (ShardingRules)
  model/             architecture.py, library.py (components), hf_import.py
  profiles.py        framework profiles
  expand.py          call → block tree
  memory.py          per-device memory, KV capacity
  rlhf.py            Problem, Candidate, feasibility, resharding, iteration tree
  optimize.py        Simulator, moves, simulated annealing
  tests/
```
