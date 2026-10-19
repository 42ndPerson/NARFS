"""NARFS: an analytical RLHF placement simulator.

Five layers own five kinds of knowledge:

    L0  physics: compute cost, the tiered network router, the communication
        library model and host cost (narfs.costs, narfs.network)
    L1  the core evaluator: eight node types, the fluid bound, memo and
        explain (narfs.blocks, narfs.evaluate)
    L2  the layout layer: degrees, device mapping, handles and layout
        transfers (narfs.parallel)
    L3  lowerings: components, schedules, overlap, variation and the
        generation engine (narfs.model, narfs.expand, narfs.engine)
    L4  the problem: RL flows, models, profiles and the workload
        (narfs.rlhf, narfs.profiles)

Only the evaluator turns structure into time, and only the layout layer reads
the device mapping. The simulator API (narfs.sim) and the candidate encoding
(narfs.optimize) are the public entry points; the names below are the ones a
user of the simulator needs.
"""
from narfs.hardware import H100_SXM, GpuProfile, load_gpu
from narfs.model.architecture import DEEPSEEK_V3, LLAMA3_8B, LLAMA3_70B, MIXTRAL_8X7B, Architecture, llama
from narfs.model.hf_import import from_hf_config
from narfs.network.presets import hgx_h100, rail_optimized
from narfs.network.topology import Tier, Topology
from narfs.optimize import CallPlan, Candidate, colocated
from narfs.parallel.mesh import AXES, Degrees
from narfs.profiles import (MEGATRON_DEFAULT, MEGATRON_DISTOPT, MEGATRON_OVERLAP, VLLM_H100, EngineProfile,
                            TrainProfile, load_engine)
from narfs.rlhf import Model, async_grpo, cpu, forward, generate, grpo, lagged, par, ppo, seq, sync, train
from narfs.sim import Problem, Result, Simulator
from narfs.variation import Dist

__all__ = [
    "Problem", "Result", "Simulator",
    "CallPlan", "Candidate", "colocated", "AXES", "Degrees",
    "Model", "generate", "forward", "train", "cpu", "sync", "seq", "par", "lagged", "grpo", "async_grpo", "ppo",
    "Dist",
    "Architecture", "llama", "from_hf_config", "LLAMA3_8B", "LLAMA3_70B", "MIXTRAL_8X7B", "DEEPSEEK_V3",
    "Topology", "Tier", "hgx_h100", "rail_optimized",
    "GpuProfile", "H100_SXM", "load_gpu",
    "TrainProfile", "EngineProfile", "MEGATRON_DEFAULT", "MEGATRON_OVERLAP", "MEGATRON_DISTOPT", "VLLM_H100",
    "load_engine",
]
