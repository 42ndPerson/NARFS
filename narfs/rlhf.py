"""L4: RL flows.

An RL algorithm is a short flow the user writes. It reads like the
algorithm's pseudocode, in the style of HybridFlow's single-controller
programs. A flow names its calls; a candidate assigns devices to them, and
the simulator lowers each call to core structure. Users never see blocks,
handles or placements, and errors name the call.

    generate   -> the engine lowering on the call's devices
    forward    -> the schedule lowering, forward only (logprobs, values, rewards)
    train      -> the schedule lowering with backward, gradient sync and optimizer
    cpu        -> Work on a CPU pool: worker-seconds against its workers
    sync       -> the reshard transfer of a model's weights to a call's layout
    seq, par   -> Seq, Par
    lagged     -> Par: the fixed-lag steady state (generation of step k+1
                  runs during learning of step k)

TODO (version 2): streaming asynchronous RL (AReaL, PipelineRL, partial
rollouts) needs a steady-state engine mode, an update cost per weight
version and a staleness rule. These are additions here and in the engine
lowering; the core and the simulator API stay as they are.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from narfs.expand import Batch
from narfs.model.architecture import Architecture
from narfs.profiles import MEGATRON_DEFAULT, VLLM_H100, EngineProfile, TrainProfile
from narfs.variation import Dist


@dataclass(frozen=True)
class Model:
    name: str
    arch: Architecture


@dataclass(frozen=True, eq=False)
class Rollout:
    """The data a generate call produces: prompts x samples sequences."""
    call: str
    sequences: int
    prompt: float
    response: Dist

    @property
    def batch(self) -> Batch:
        p, r = self.prompt, self.response
        mean = p + r.mean
        sq = (p * p + 2 * p * r.mean + r.second) / mean
        return Batch(self.sequences, mean, sq, r.std)


class Flow:
    """A node of a flow. Every node has a `name`; errors and phases use it."""


@dataclass(frozen=True, eq=False)
class Call(Flow):
    name: str
    kind: str                          # "generate", "forward" or "train"
    model: Model
    profile: object
    data: Rollout | None = None        # the rollout a forward or train call reads
    prompts: int = 0                   # generate: requests are prompts x samples
    samples: int = 1
    prompt: float = 0.0
    response: Dist | None = None
    minibatches: int = 1

    @property
    def output(self) -> Rollout:
        if self.kind != "generate":
            raise TypeError(f"call {self.name} is not a generate call")
        return Rollout(self.name, self.prompts * self.samples, self.prompt, self.response)


@dataclass(frozen=True, eq=False)
class Cpu(Flow):
    name: str
    pool: str
    worker_seconds: float
    workers: int


@dataclass(frozen=True, eq=False)
class Sync(Flow):
    name: str
    model: Model
    to: str                            # the call whose layout receives the weights


@dataclass(frozen=True, eq=False)
class Composite(Flow):
    name: str
    kind: str                          # "seq", "par" or "lagged"
    children: tuple = field(default=())


def _data(data) -> Rollout:
    if isinstance(data, Call):
        return data.output
    if isinstance(data, Rollout):
        return data
    raise TypeError("a forward or train call reads a rollout: pass a generate call or its .output")


def generate(model: Model, prompts: int, samples: int = 1, response: Dist | None = None, prompt: float = 512.0,
             engine: EngineProfile | None = None, name: str = "generate") -> Call:
    if response is None:
        raise ValueError(f"call {name}: generate needs a response length distribution")
    return Call(name, "generate", model, engine or VLLM_H100, prompts=prompts, samples=samples,
                prompt=float(prompt), response=response)


def forward(model: Model, data, name: str | None = None, profile: TrainProfile = MEGATRON_DEFAULT) -> Call:
    return Call(name or f"{model.name}_forward", "forward", model, profile, _data(data))


def train(model: Model, data, minibatches: int = 1, name: str | None = None,
          profile: TrainProfile = MEGATRON_DEFAULT) -> Call:
    return Call(name or f"{model.name}_train", "train", model, profile, _data(data), minibatches=minibatches)


def cpu(pool: str, items, seconds_each: float, workers: int, name: str | None = None) -> Cpu:
    """`items` CPU tasks of `seconds_each` on a pool of `workers` (for example a
    verifier per rollout); `items` may be a generate call (one task per sequence)."""
    n = items.output.sequences if isinstance(items, Call) else int(items)
    return Cpu(name or pool, pool, n * float(seconds_each), int(workers))


def sync(model: Model, to: Call, name: str | None = None) -> Sync:
    return Sync(name or f"sync {model.name}", model, to.name)


def seq(*nodes, name: str = "seq") -> Composite:
    return Composite(name, "seq", tuple(nodes))


def par(*nodes, name: str = "par") -> Composite:
    return Composite(name, "par", tuple(nodes))


def lagged(producer, consumer, lag: int = 1, name: str = "lagged") -> Composite:
    if lag != 1:
        raise ValueError("only a lag of 1 is modeled")
    return Composite(name, "lagged", (producer, consumer))


# ------------------------------------------------------------ reference flows ---

def grpo(policy: Model, ref: Model, lengths: Dist, prompts: int = 512, group: int = 8, prompt: float = 512.0,
         verifier_s: float = 0.05, workers: int = 512, minibatches: int = 4,
         engine: EngineProfile | None = None, profile: TrainProfile = MEGATRON_DEFAULT):
    roll = generate(policy, prompts, group, lengths, prompt, engine)
    score = par(forward(ref, roll, "ref_logprob", profile), forward(policy, roll, "old_logprob", profile),
                cpu("verifier", roll, verifier_s, workers), name="score")
    update = train(policy, roll, minibatches, "policy_train", profile)
    return seq(roll, score, update, sync(policy, roll), name="grpo")


def async_grpo(policy: Model, ref: Model, lengths: Dist, prompts: int = 512, group: int = 8, prompt: float = 512.0,
               minibatches: int = 4, engine: EngineProfile | None = None,
               profile: TrainProfile = MEGATRON_DEFAULT):
    roll = generate(policy, prompts, group, lengths, prompt, engine)
    learn = seq(par(forward(ref, roll, "ref_logprob", profile), forward(policy, roll, "old_logprob", profile)),
                train(policy, roll, minibatches, "policy_train", profile), sync(policy, roll), name="learn")
    return lagged(roll, learn, name="async grpo")


def ppo(actor: Model, critic: Model, ref: Model, reward: Model, lengths: Dist, prompts: int = 512,
        prompt: float = 512.0, minibatches: int = 4, engine: EngineProfile | None = None,
        profile: TrainProfile = MEGATRON_DEFAULT):
    roll = generate(actor, prompts, 1, lengths, prompt, engine)
    score = par(forward(actor, roll, "actor_logprob", profile), forward(ref, roll, "ref_logprob", profile),
                forward(critic, roll, "critic_value", profile), forward(reward, roll, "reward", profile), name="score")
    update = par(train(actor, roll, minibatches, "actor_train", profile),
                 train(critic, roll, minibatches, "critic_train", profile), name="update")
    return seq(roll, score, update, sync(actor, roll), name="ppo")
