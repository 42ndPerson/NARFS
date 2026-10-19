# narfs

An analytical simulator for RLHF placement. You give it one **candidate**: for
each model call of an RL iteration, a device list, parallel degrees, an axis
order and a micro-batch size. It gives back the time of one iteration, the
time of each phase, and whether the candidate fits in memory (with a reason
if it does not). It holds no search code: an external optimizer calls it in a
loop.

It is deterministic (no sampling, no clock). In this repository's tests, one
candidate takes about 3 ms at 16 GPUs and 15 to 20 ms at 512 GPUs (Llama-3-70B
GRPO, new mapping, same degrees). A test fails above 50 ms at 512 GPUs.

## Run

```bash
pip install -r narfs/requirements.txt
python -m pytest narfs/tests          # from the repository root
```

## Example

```python
import narfs

lengths = narfs.Dist.lognormal(5.6, 0.8, 4096)        # response lengths; or Dist.histogram, Dist.from_samples
policy, ref = narfs.Model("policy", narfs.LLAMA3_8B), narfs.Model("ref", narfs.LLAMA3_8B)
flow = narfs.grpo(policy, ref, lengths, prompts=64, group=8, minibatches=4, profile=narfs.MEGATRON_DISTOPT)
sim = narfs.Simulator(narfs.Problem(narfs.hgx_h100(2), flow))

train = narfs.Degrees(dp=2, tp=4)
cand = narfs.colocated([(8, ["generate"]), (8, ["ref_logprob", "old_logprob", "policy_train"])],
                       {"generate": narfs.Degrees(dp=8), "ref_logprob": train, "old_logprob": train,
                        "policy_train": train})
r = sim.evaluate(cand)
r.feasible, r.time        # True, 33.25 s
r.phases                  # {"grpo": 33.25, "generate": 20.6, "score": 5.38, "ref_logprob": 2.69,
                          #  "old_logprob": 2.69, "verifier": 0.05, "policy_train": 6.94, "sync policy": 0.33}
print(sim.explain(cand, max_depth=1))
```

```
iteration: 33.245 s (lo 33.245 s, hi 33.289 s)

grpo            33245.141 ms   33289.493 ms  largest part: generate (62%)       gpu[gpu 0] 62%
  generate      20597.462 ms   20597.462 ms  generate: 3062 decode steps, 64 slots x 8 replicas
  score          5379.173 ms    5379.173 ms  shared resource: gpu[gpu 8]        gpu[gpu 8] 100%
  policy_train   6940.694 ms    6985.046 ms  minibatches                        gpu[gpu 8] 100%
  sync policy     327.812 ms     327.812 ms  point-to-point sends               nic_tx[gpu 8] 100%

assumed (unmeasured) values:
  h100-sxm: eta_hbm
  call generate (vllm): host_step
  ...
```

Each `explain` row gives the fluid time `lo`, the upper bound `hi`, the term
that set the time (a child, or a shared resource), and the most loaded
resource with its utilization.

## The API

| Name | What it is |
| --- | --- |
| `Problem(topology, flow, gpus=..., libraries=..., cost_table=None)` | what stays fixed: the cluster, the flow (models, profiles, workload), the GPU profiles by device class and the communication libraries |
| `Simulator(problem)` | checks the problem (errors name the node) and builds the caches |
| `sim.evaluate(candidate) -> Result` | never raises for an infeasible candidate |
| `sim.explain(candidate, max_depth=5) -> str` | bound-by rows and the assumed constants |
| `sim.call_names` | the calls a candidate must plan |
| `CallPlan(devices, degrees, order=AXES, micro_batch=1)` | the plan of one call; `order` lists the mesh axes from slowest to fastest varying |
| `Candidate.of({call: CallPlan})`, `colocated(groups, degrees)` | a candidate; `colocated` gives consecutive device ranges to groups of calls |

`Result` fields: `feasible`, `reason` (for example `"call old_logprob has no
placement"` or `"out of memory on device 8: 84.0 GB needed, 80.0 GB available
(calls: generate, ref_logprob, old_logprob, policy_train)"`), `time`, `lo`,
`hi` (seconds per iteration), `phases` (seconds per flow node, in flow
order), `memory_peak` and `memory_headroom` (bytes per device) and
`kv_capacity` (KV tokens per engine replica). `time` is the fluid estimate
`lo`: the lowerings emit `Par` only where the framework really overlaps work.
`hi` shows how much the estimate relies on perfect sharing.

The objective (iteration time, tokens per second, cost) belongs to the
optimizer.

## Flows

An RL algorithm is a short flow that reads like its pseudocode. Calls have
names; the candidate assigns devices to them.

```python
def grpo(policy, ref, lengths, prompts=512, group=8):
    roll = generate(policy, prompts, group, lengths)
    score = par(forward(ref, roll, "ref_logprob"), forward(policy, roll, "old_logprob"),
                cpu("verifier", roll, seconds_each=0.05, workers=512), name="score")
    update = train(policy, roll, minibatches=4, name="policy_train")
    return seq(roll, score, update, sync(policy, to=roll), name="grpo")
```

| Flow node | Lowers to |
| --- | --- |
| `generate(model, prompts, samples, response, prompt=512, engine=VLLM_H100)` | the engine lowering, in decode-step space, on the call's devices |
| `forward(model, data)`, `train(model, data, minibatches)` | the schedule lowering (1F1B or interleaved), forward only or with backward, the DP gradient sync and the optimizer |
| `cpu(pool, items, seconds_each, workers)` | `Work` on a named CPU pool: worker-seconds against its workers |
| `sync(model, to=call)` | the reshard `Transfer` of the model's weights from its train call's layout to the call's layout |
| `seq`, `par` | `Seq`, `Par` |
| `lagged(producer, consumer)` | `Par`: the one-step-lag steady state (generation of step k+1 runs during learning of step k) |

`narfs.rlhf` has three reference flows: `grpo`, `async_grpo` and `ppo`.

## Layers

| Layer | Modules | Knows |
| --- | --- | --- |
| L0 physics | `costs.py`, `hardware.py`, `network/`, `physics.py` | compute cost (roofline with waves and split-K), the tiered router, the communication library model (NCCL and engine all-reduce paths), host cost |
| L1 core | `blocks.py`, `evaluate.py` | eight node types, the fluid bound and the upper bound, memo, explain |
| L2 layout | `parallel/` | degrees, device mapping, handles, the binder, layout transfers (stage boundaries and weight reshards) |
| L3 lowerings | `model/`, `expand.py`, `engine.py`, `variation.py` | components, schedules, overlap, the generation engine, length variation |
| L4 problem | `rlhf.py`, `profiles.py`, `sim.py`, `optimize.py` | flows, framework and engine profiles, the simulator API, the candidate encoding |
| memory | `memory.py` | per-device bytes, feasibility, the engine's KV capacity |

Only the evaluator turns structure into time, and only the layout layer reads
the device mapping. For one candidate:

1. The memory model checks feasibility and computes each engine's KV capacity.
2. Each call's program is lowered, or reused from the cache when its degrees,
   micro-batch and KV capacity are unchanged.
3. The flow becomes core structure, and the binder binds every handle to
   device arrays.
4. The evaluator walks the tree; the L0 models give each leaf's cost.

### Core rules

Every node evaluates to `(lo, hi)` and the resource demand of each:

| Node | Rule |
| --- | --- |
| `Compute(work)` | the compute cost model, on the device class of the enclosing scope |
| `Collective(kind, group, bytes)` | the communication library model: time and demand |
| `Transfer(pairs, bytes)` | routes, then one bincount; time = max over resources of demand / capacity, plus latency |
| `Work(resource, amount)` | amount / capacity on a CPU pool or host thread |
| `Seq` | times and demands add |
| `Repeat(n, body)` | times and demands scale by n (a real n below 1 is a time slice) |
| `Par` | `lo` = max(longest child, max over resources of summed demand / capacity): the fluid bound. `hi`: children in one device scope add; across scopes, longest child + max over resources of (sum − largest) |
| `OnDevices(devices, body)` | sets the device class; the devices are busy for the body's time |

The cost of a leaf depends on what the operation is (kind, size, group,
library), never on why it exists.

## System profiles

Constants are data files in `narfs/data`, and every value has a kind
(`specification`, `theory`, `measured`, `configuration` or `assumed`) and a
source:

| File | Holds |
| --- | --- |
| `h100-sxm.json` | SMs, clock, tensor rate per dtype, HBM, GEMM and HBM ceilings, NVLink and NVLS rates, NIC and PCIe rates, launch costs |
| `nccl-2.32.3.json` | NCCL's latency tables and protocol limits, from its source code |
| `vllm-573f2f9-h100.json` | vLLM's scheduler settings, host time per step, memory reserve and all-reduce paths by size, from its source code |

A cluster measurement is a file with `"base"`: it gives only what it measured,
and the rest comes from the base with its flags.

```json
{"kind": "gpu", "name": "h100-sxm", "base": "h100-sxm",
 "values": {"eta_hbm": {"value": 0.91, "kind": "measured", "source": "triad kernel, cluster A"}}}
```

Values of kind `assumed` show in `explain`. Today these are: the achievable
HBM rate (`eta_hbm` 0.85), the engine host time per step (1 ms + 5 µs ×
batch), the engine memory reserve (3 GB), the latency of vLLM's own
all-reduce paths (6, 10 and 12 µs), and the training memory reserve (4 GiB).

## Models

`Architecture` lists named dimensions and the components of each layer.
`llama(...)` builds dense Llama-style models, and `from_hf_config` reads a
Hugging Face `config.json`. Components (`embed`, `head`, `gqa`, `mla`,
`dense`, `moe`, `mamba2`) are registered by name in `narfs.model.library`;
a new component gives its forward structure, its parameters by sharding class
and its bytes per token. Parameter counts match Hugging Face for Llama-3-8B,
Llama-3-70B and Mixtral-8x7B.

## Limits

- Generation with pp > 1 is not modeled. ZeRO-2/3 and FSDP are not modeled
  (`zero_stage` is 0 or 1).
- Flows are series-parallel. `lagged` is a fixed one-step lag. Streaming
  asynchronous RL (AReaL, PipelineRL, partial rollouts) and multi-turn
  rollouts are not modeled yet; they need a steady-state engine mode, not a
  change to the core or the API (see the TODO in `rlhf.py`).
- `sync` costs the bytes moved; weight conversion inside the engine is not
  modeled.

## Tests

| File | Checks |
| --- | --- |
| `test_evaluate.py` | the core rules on hand-computed trees |
| `test_kernels.py` | bytes per resource of each collective pattern, by hand; NCCL path choice |
| `test_costs.py` | wave counts and ceilings by hand; no cliff across sizes; profile files |
| `test_parallel.py` | group arrays, mappings, stage boundaries and reshard bytes |
| `test_model.py` | parameter counts against Hugging Face configs; component structure |
| `test_expand.py` | 1F1B, interleaved and flow-shop closed forms |
| `test_engine.py` | the generation tail against an exact FIFO Monte Carlo; segment convergence; engine policies |
| `test_memory.py` | bytes per term; feasibility; KV capacity |
| `test_system.py` | flows, phases, weight sync, determinism, the performance budget, infeasibility reasons |
