"""L3: the component library.

Each component type is one registered class. Its methods see a `Ctx`: local
(per-rank) shapes from the sharding rules, handles for the call's groups, and
the framework's facts (blocking or overlapped TP communication, sequence
parallelism). They never see devices. A component author writes shapes, not
FLOPs: `c.gemm` computes FLOPs and bytes, and `c.tp_after` applies the
profile's choice of communication.

    forward(c)                one layer over c.tokens tokens
    step(c, mix)              one engine step for a token mix
    params(c)                 per-rank parameters, by sharding class
    act_bytes_per_token(c)    saved activations (for the memory model)
    kv_bytes_per_token(c)     KV cache per context token
    state_bytes(c)            recurrent state per sequence, constant in context

Backward is derived by per-op rules (`backward`): a GEMM gives a dgrad and a
wgrad GEMM, attention its FlashAttention backward, and each collective its
adjoint. A component overrides `backward` only when the rules do not fit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from narfs import costs
from narfs.blocks import Collective, Compute, OnDevices, Par, Repeat, Seq, Transfer, Work
from narfs.hardware import DTYPE_BYTES
from narfs.model.architecture import Architecture
from narfs.parallel.mesh import Degrees
from narfs.parallel.rules import ShardingRules, local_dims
from narfs.variation import critical_instance

REGISTRY: dict = {}


def component(name: str):
    """Register a component class under `name`."""
    def register(cls):
        cls.kind = name
        REGISTRY[name] = cls()
        return cls
    return register


def get(name: str):
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"no component registered as {name!r}; known: {sorted(REGISTRY)}") from None


@dataclass(frozen=True)
class Params:
    """Per-rank parameter counts by sharding class."""
    tp: float = 0.0                    # sharded over tp (full = tp x this)
    rep: float = 0.0                   # replicated over tp and ep
    ep: float = 0.0                    # routed experts, sharded over ep (full = ep x this)

    def __add__(self, o: "Params") -> "Params":
        return Params(self.tp + o.tp, self.rep + o.rep, self.ep + o.ep)

    def __mul__(self, n: float) -> "Params":
        return Params(self.tp * n, self.rep * n, self.ep * n)

    @property
    def non_expert(self) -> float:
        return self.tp + self.rep

    @property
    def total(self) -> float:
        return self.tp + self.rep + self.ep


@dataclass(frozen=True)
class TokenMix:
    """One engine step's tokens: `batch` decoding sequences at mean context
    `context` (GEMMs run on `padded` rows, a captured CUDA-graph size), plus
    `prefill` prompt tokens whose mean attended context is `prefill_context`."""
    batch: float
    padded: float
    context: float
    prefill: float = 0.0
    prefill_context: float = 0.0

    @property
    def rows(self) -> float:
        return self.padded + self.prefill


@dataclass(frozen=True, eq=False)
class Ctx:
    arch: Architecture
    local: dict
    degrees: Degrees
    tokens: float = 0.0                # tokens per rank in one pass, after the cp split
    seq: float = 0.0                   # mean sequence length
    seq_sq: float = 0.0                # E[s^2] / E[s]: the length-biased sequence length (0: seq)
    tp: Any = None                     # handles of the call's groups in this stage
    cp: Any = None
    ep: Any = None
    lib: str = "nccl"                  # communication library profile
    tp_comm: str = "blocking"          # "blocking" or "overlapped"
    tp_chunks: int = 4
    sp: bool = False                   # sequence parallelism: RS + AG instead of AR
    vocab_parallel_loss: bool = True

    # ------------------------------------------------------------- shapes ---
    def dim(self, name: str, default: int | None = None) -> int:
        return self.arch.dim(name, default)

    def local_dim(self, name: str, default: int | None = None) -> int:
        if name in self.local:
            return self.local[name]
        if default is not None:
            return default
        raise KeyError(f"{self.arch.name} has no dimension {name!r}")

    @property
    def dt(self) -> int:
        return DTYPE_BYTES[self.arch.dtype]

    @property
    def tp_deg(self) -> int:
        return self.degrees.tp

    @property
    def h_sp(self) -> float:
        """Hidden size per token outside the TP region, per rank (split by SP)."""
        return self.dim("hidden") / (self.tp_deg if self.sp else 1)

    def causal_context(self) -> float:
        """Mean attended context of a token in causal attention over whole
        sequences: half the length-biased sequence length."""
        return (self.seq_sq or self.seq) / 2.0

    # ------------------------------------------------------------- leaves ---
    def gemm(self, m: float, n: float, k: float, name: str, groups: float = 1.0) -> Compute:
        return Compute(costs.gemm(m, n, k, self.arch.dtype, name, groups=groups), name)

    def elementwise(self, flops: float, bytes_: float, name: str) -> Compute:
        return Compute(costs.elementwise(flops, bytes_, name), name)

    def op(self, kind: str, flops: float, bytes_: float, name: str, tensor: bool = False) -> Compute:
        return Compute(costs.Kernel("op", float(flops), float(bytes_), dtype=self.arch.dtype, tensor=tensor,
                                    name=name or kind), name or kind)

    def norm(self, tokens: float, width: float, name: str) -> Compute:
        return self.elementwise(5.0 * tokens * width, 2.0 * tokens * width * self.dt, name)

    def attention(self, q: float, ctx: float, heads: float, d_qk: float, d_v: float, kv_heads: float,
                  name: str = "attention") -> Compute:
        """FlashAttention over `q` query tokens with mean attended context `ctx`:
        K and V stream once per 128-query block."""
        blocks = max(1.0, q / 128.0)
        flops = 2.0 * q * ctx * heads * (d_qk + d_v)
        bytes_ = self.dt * (q * heads * (d_qk + d_v) + blocks * ctx * kv_heads * (d_qk + d_v))
        return Compute(costs.Kernel("attention", flops, bytes_, dtype=self.arch.dtype, tiles=heads * blocks,
                                    name=name), name)

    def decode_attention(self, batch: float, ctx: float, heads: float, d_qk: float, d_v: float,
                         kv_bytes: float, tiles: float, name: str = "decode attention") -> Compute:
        """Paged decode attention: each sequence reads its whole KV cache."""
        flops = 2.0 * batch * ctx * heads * (d_qk + d_v)
        bytes_ = batch * ctx * kv_bytes + self.dt * batch * heads * (d_qk + d_v)
        return Compute(costs.Kernel("attention", flops, bytes_, dtype=self.arch.dtype, tiles=tiles, name=name), name)

    # ------------------------------------------------------ communication ---
    def _coll(self, kind: str, handle, nbytes: float, name: str) -> Collective:
        return Collective(kind, handle, float(nbytes), self.lib, name)

    def tp_after(self, op: Compute, nbytes: float) -> tuple:
        """An op whose output is reduced over TP: blocking, or chunked and
        overlapped with the op, as the framework profile says."""
        if self.tp_deg == 1:
            return (op,)
        kind = "reduce_scatter" if self.sp else "all_reduce"
        if self.tp_comm == "overlapped":
            k = self.tp_chunks
            a = _chunk(op, k)
            b = self._coll(kind, self.tp, nbytes / k, f"tp {kind} chunk")
            return (Seq((a, Repeat(k - 1, Par((a, b))), b), name=f"{op.name} + tp {kind}"),)
        return (op, self._coll(kind, self.tp, nbytes, f"tp {kind}"))

    def tp_before(self, nbytes: float) -> tuple:
        """With sequence parallelism, a TP region starts with an all-gather."""
        if self.tp_deg > 1 and self.sp:
            return (self._coll("all_gather", self.tp, nbytes, "tp all_gather"),)
        return ()

    def tp_reduce(self, nbytes: float, name: str = "tp all_reduce") -> tuple:
        return (self._coll("all_reduce", self.tp, nbytes, name),) if self.tp_deg > 1 else ()

    def ep_all_to_all(self, nbytes: float, name: str) -> tuple:
        return (self._coll("all_to_all", self.ep, nbytes, name),) if self.degrees.ep > 1 else ()

    def cp_all_gather(self, nbytes: float, name: str) -> tuple:
        return (self._coll("all_gather", self.cp, nbytes, name),) if self.degrees.cp > 1 else ()


def _chunk(op: Compute, k: int) -> Compute:
    """One of k chunks of an op: a smaller GEMM, not a fraction of one."""
    w = op.work
    if w.kind == "gemm":
        m, n, kk = w.shape
        return Compute(costs.gemm(m / k, n, kk, w.dtype, w.name, groups=w.groups), op.name + " chunk")
    return Compute(costs.Kernel(w.kind, w.flops / k, w.bytes / k, w.shape, w.dtype, w.tiles / k, w.tensor,
                                w.groups, w.name), op.name + " chunk")


def make_ctx(arch: Architecture, degrees: Degrees, rules: ShardingRules, **kw) -> Ctx:
    return Ctx(arch, local_dims(arch, degrees, rules), degrees, **kw)


# ------------------------------------------------------------- backward ---

_ADJOINT = {"all_reduce": "all_reduce", "all_gather": "reduce_scatter", "reduce_scatter": "all_gather",
            "all_to_all": "all_to_all"}


def _bwd_kernel(w: costs.Kernel, name: str):
    if w.kind == "gemm":
        m, n, k = w.shape
        return Seq((Compute(costs.gemm(m, k, n, w.dtype, name + " dgrad", groups=w.groups), name + " dgrad"),
                    Compute(costs.gemm(k, n, m, w.dtype, name + " wgrad", groups=w.groups), name + " wgrad")),
                   name=name + " bwd")
    scale = 2.5 if w.kind == "attention" else 2.0               # FlashAttention backward recomputes S = QK^T
    return Compute(costs.Kernel(w.kind, scale * w.flops, 2.0 * w.bytes, w.shape, w.dtype, w.tiles, w.tensor,
                                w.groups, w.name), name + " bwd")


def backward(node, memo: dict | None = None):
    """The backward of a forward block, by per-op rules, in reverse order.
    Shared forward nodes give shared backward nodes."""
    memo = {} if memo is None else memo
    hit = memo.get(id(node))
    if hit is not None:
        return hit[1]
    match node:
        case Compute():
            out = _bwd_kernel(node.work, node.name or node.work.name)
        case Collective():
            out = Collective(_ADJOINT[node.kind], node.comm, node.bytes, node.lib, (node.name or node.kind) + " bwd")
        case Transfer():
            out = Transfer(node.pairs, node.bytes, not node.reverse, (node.name or "send") + " bwd")
        case Seq():
            out = Seq(tuple(backward(c, memo) for c in reversed(node.children)), name=node.name + " bwd" if node.name else "")
        case Par():
            out = Par(tuple(backward(c, memo) for c in node.children), name=node.name + " bwd" if node.name else "")
        case Repeat():
            out = Repeat(node.n, backward(node.body, memo), name=node.name + " bwd" if node.name else "")
        case OnDevices():
            out = OnDevices(node.devset, backward(node.body, memo), name=node.name + " bwd" if node.name else "")
        case Work():
            out = node
        case _:
            raise TypeError(f"no backward rule for {type(node).__name__}")
    memo[id(node)] = (node, out)
    return out


# ----------------------------------------------------------- components ---

class Component:
    kind = ""
    dims: tuple = ()

    def forward(self, c: Ctx):
        raise NotImplementedError

    def step(self, c: Ctx, mix: TokenMix):
        raise NotImplementedError

    def backward(self, c: Ctx, fwd):
        return backward(fwd)

    def params(self, c: Ctx) -> Params:
        raise NotImplementedError

    def act_bytes_per_token(self, c: Ctx) -> float:
        return 0.0

    def kv_bytes_per_token(self, c: Ctx) -> float:
        return 0.0

    def state_bytes(self, c: Ctx) -> float:
        return 0.0


@component("embed")
class Embedding(Component):
    """Vocab-parallel input embedding: each TP rank looks up its vocab shard,
    and the partial rows are reduced over TP."""
    dims = ("hidden", "vocab")

    def _body(self, c: Ctx, t: float):
        h = c.dim("hidden")
        lookup = c.elementwise(0.0, 2.0 * t * h * c.dt, "embedding lookup")
        if c.tp_deg == 1:
            return Seq((lookup,), name="embed")
        kind = "reduce_scatter" if c.sp else "all_reduce"
        return Seq((lookup, c._coll(kind, c.tp, t * h * c.dt, f"embedding {kind}")), name="embed")

    def forward(self, c: Ctx):
        return self._body(c, c.tokens)

    def step(self, c: Ctx, mix: TokenMix):
        return self._body(c, mix.rows)

    def params(self, c: Ctx) -> Params:
        return Params(tp=c.local_dim("vocab") * c.dim("hidden"))


@component("head")
class Head(Component):
    """Final norm, LM head and loss (or log-probabilities). The loss is
    vocab-parallel: three tiny all-reduces instead of gathering the logits."""
    dims = ("hidden", "vocab")

    def forward(self, c: Ctx):
        t, h, v = c.tokens, c.dim("hidden"), c.local_dim("vocab")
        parts = [c.norm(t / (c.tp_deg if c.sp else 1), h, "final norm"), *c.tp_before(t * h * c.dt),
                 c.gemm(t, v, h, "lm head"), c.elementwise(5.0 * t * v, 2.0 * t * v * c.dt, "loss")]
        if c.tp_deg > 1:
            if c.vocab_parallel_loss:
                parts += [c._coll("all_reduce", c.tp, 3 * t * 4.0, "vocab-parallel loss all_reduce")]
            else:
                parts += [c._coll("all_gather", c.tp, t * c.dim("vocab") * c.dt, "logits all_gather")]
        return Seq(tuple(parts), name="head")

    def step(self, c: Ctx, mix: TokenMix):
        b, h, v = mix.padded, c.dim("hidden"), c.local_dim("vocab")
        parts = [c.norm(b, h, "final norm"), c.gemm(b, v, h, "lm head")]
        if c.tp_deg > 1:
            parts.append(c._coll("all_gather", c.tp, b * c.dim("vocab") * c.dt, "logits all_gather"))
        parts.append(c.elementwise(5.0 * b * c.dim("vocab"), 4.0 * b * c.dim("vocab"), "sampling"))
        return Seq(tuple(parts), name="head")

    def params(self, c: Ctx) -> Params:
        head = 0.0 if c.arch.tied_embeddings else c.local_dim("vocab") * c.dim("hidden")
        return Params(tp=head, rep=c.dim("hidden"))


@component("gqa")
class GQA(Component):
    """Grouped-query (or multi-head) attention with RoPE."""
    dims = ("hidden", "heads", "kv_heads", "head_dim")

    def forward(self, c: Ctx):
        t, h, dt = c.tokens, c.dim("hidden"), c.dt
        hq, hkv, d = c.local_dim("heads"), c.local_dim("kv_heads"), c.dim("head_dim")
        parts = [c.norm(t / (c.tp_deg if c.sp else 1), h, "attn norm"), *c.tp_before(t * h * dt),
                 c.gemm(t, (hq + 2 * hkv) * d, h, "qkv"),
                 c.elementwise(3.0 * t * (hq + hkv) * d, 2.0 * t * (hq + hkv) * d * dt, "rope"),
                 *c.cp_all_gather(c.degrees.cp * 2 * t * hkv * d * dt, "cp kv all_gather"),
                 c.attention(t, c.causal_context(), hq, d, d, hkv),
                 *c.tp_after(c.gemm(t, h, hq * d, "attn out"), t * h * dt)]
        return Seq(tuple(parts), name="gqa")

    def step(self, c: Ctx, mix: TokenMix):
        r, h, dt = mix.rows, c.dim("hidden"), c.dt
        hq, hkv, d = c.local_dim("heads"), c.local_dim("kv_heads"), c.dim("head_dim")
        parts = [c.norm(r, h, "attn norm"), c.gemm(r, (hq + 2 * hkv) * d, h, "qkv"),
                 c.elementwise(3.0 * r * (hq + hkv) * d, 2.0 * r * (hq + hkv) * d * dt, "rope"),
                 c.decode_attention(mix.batch, mix.context, hq, d, d, self.kv_bytes_per_token(c), mix.batch * hkv)]
        if mix.prefill:
            parts.append(c.attention(mix.prefill, mix.prefill_context, hq, d, d, hkv, "prefill attention"))
        parts += c.tp_after(c.gemm(r, h, hq * d, "attn out"), r * h * dt)
        return Seq(tuple(parts), name="gqa")

    def params(self, c: Ctx) -> Params:
        h, d = c.dim("hidden"), c.dim("head_dim")
        hq, hkv = c.local_dim("heads"), c.local_dim("kv_heads")
        return Params(tp=h * (hq + 2 * hkv) * d + hq * d * h, rep=h)

    def act_bytes_per_token(self, c: Ctx) -> float:
        hq, hkv, d = c.local_dim("heads"), c.local_dim("kv_heads"), c.dim("head_dim")
        return c.dt * (2 * c.h_sp + (hq + 2 * hkv) * d + hq * d)

    def kv_bytes_per_token(self, c: Ctx) -> float:
        return 2.0 * c.local_dim("kv_heads") * c.dim("head_dim") * c.dt


@component("mla")
class MLA(Component):
    """Multi-head latent attention (DeepSeek-V2/V3): low-rank Q and KV, a
    shared RoPE key, and a latent KV cache replicated over TP. Decode uses
    the absorbed form, attending over the latent."""
    dims = ("hidden", "heads", "head_dim", "v_head_dim", "q_lora", "kv_lora", "rope_dim")

    def _shapes(self, c: Ctx):
        return (c.dim("hidden"), c.local_dim("heads"), c.dim("head_dim"), c.dim("rope_dim"), c.dim("v_head_dim"),
                c.dim("q_lora"), c.dim("kv_lora"))

    def forward(self, c: Ctx):
        t, dt = c.tokens, c.dt
        h, hq, d, dr, dv, ql, kl = self._shapes(c)
        parts = [c.norm(t / (c.tp_deg if c.sp else 1), h, "attn norm"), *c.tp_before(t * h * dt),
                 c.gemm(t, ql, h, "q down"), c.norm(t, ql, "q norm"), c.gemm(t, hq * (d + dr), ql, "q up"),
                 c.gemm(t, kl + dr, h, "kv down"), c.norm(t, kl, "kv norm"), c.gemm(t, hq * (d + dv), kl, "kv up"),
                 c.elementwise(3.0 * t * (hq + 1) * dr, 2.0 * t * (hq + 1) * dr * dt, "rope"),
                 *c.cp_all_gather(c.degrees.cp * t * (kl + dr) * dt, "cp latent all_gather"),
                 c.attention(t, c.causal_context(), hq, d + dr, dv, hq),
                 *c.tp_after(c.gemm(t, h, hq * dv, "attn out"), t * h * dt)]
        return Seq(tuple(parts), name="mla")

    def step(self, c: Ctx, mix: TokenMix):
        r, b, dt = mix.rows, mix.batch, c.dt
        h, hq, d, dr, dv, ql, kl = self._shapes(c)
        parts = [c.norm(r, h, "attn norm"), c.gemm(r, ql, h, "q down"), c.gemm(r, hq * (d + dr), ql, "q up"),
                 c.gemm(r, kl + dr, h, "kv down"),
                 c.gemm(mix.padded, kl, d, "absorb k", groups=hq),
                 c.decode_attention(b, mix.context, hq, kl + dr, kl, self.kv_bytes_per_token(c),
                                    b * math.ceil(hq / 64), "latent decode attention"),
                 c.gemm(mix.padded, dv, kl, "absorb v", groups=hq)]
        if mix.prefill:
            p = mix.prefill
            parts += [c.gemm(p, hq * (d + dv), kl, "kv up"),
                      c.attention(p, mix.prefill_context, hq, d + dr, dv, hq, "prefill attention")]
        parts += c.tp_after(c.gemm(r, h, hq * dv, "attn out"), r * h * dt)
        return Seq(tuple(parts), name="mla")

    def params(self, c: Ctx) -> Params:
        h, hq, d, dr, dv, ql, kl = self._shapes(c)
        return Params(tp=ql * hq * (d + dr) + kl * hq * (d + dv) + hq * dv * h,
                      rep=h * ql + ql + h * (kl + dr) + kl + h)

    def act_bytes_per_token(self, c: Ctx) -> float:
        h, hq, d, dr, dv, ql, kl = self._shapes(c)
        return c.dt * (2 * c.h_sp + ql + hq * (d + dr) + (kl + dr) + hq * (d + dv) + hq * dv)

    def kv_bytes_per_token(self, c: Ctx) -> float:
        return (c.dim("kv_lora") + c.dim("rope_dim")) * c.dt


@component("dense")
class DenseFFN(Component):
    """SwiGLU feed-forward block."""
    dims = ("hidden", "ffn")

    def _body(self, c: Ctx, t: float, f: float, sp_tokens: float, name: str = "dense", norm: bool = True):
        h, dt = c.dim("hidden"), c.dt
        parts = [c.norm(sp_tokens, h, "ffn norm")] if norm else []
        parts += [*c.tp_before(t * h * dt),
                  c.gemm(t, 2 * f, h, "gate+up"), c.elementwise(4.0 * t * f, 3.0 * t * f * dt, "swiglu"),
                  *c.tp_after(c.gemm(t, h, f, "down"), t * h * dt)]
        return Seq(tuple(parts), name=name)

    def forward(self, c: Ctx):
        return self._body(c, c.tokens, c.local_dim("ffn"), c.tokens / (c.tp_deg if c.sp else 1))

    def step(self, c: Ctx, mix: TokenMix):
        return self._body(c, mix.rows, c.local_dim("ffn"), mix.rows)

    def params(self, c: Ctx) -> Params:
        h = c.dim("hidden")
        return Params(tp=3.0 * h * c.local_dim("ffn"), rep=h)

    def act_bytes_per_token(self, c: Ctx) -> float:
        return c.dt * (2 * c.h_sp + 3 * c.local_dim("ffn"))


@component("moe")
class MoE(Component):
    """Router, dispatch, routed experts, combine, and shared experts.

    Routed experts are sharded over ep. If the rules also shard `expert_ffn`
    over tp (as inference engines do), every TP rank sees every token and the
    output is all-reduced over TP; otherwise each TP rank routes its 1/tp of the
    tokens (sequence-parallel dispatch, as Megatron does). The busiest EP rank
    sets the time: its token-expert rows come from routing statistics
    (`critical_instance`), not from a fixed factor."""
    dims = ("hidden", "experts", "top_k", "expert_ffn")

    def _rows(self, c: Ctx, tokens: float):
        """(tokens entering the router per rank, rows on the busiest EP rank)."""
        tp_experts = c.local_dim("expert_ffn") < c.dim("expert_ffn")
        tl = tokens if (tp_experts or c.tp_deg == 1) else tokens / c.tp_deg
        ep, k = c.degrees.ep, c.dim("top_k")
        mean = tl * k
        std = math.sqrt(ep * tl * k * (1.0 / ep) * (1.0 - 1.0 / ep)) if ep > 1 else 0.0
        return tl, critical_instance(mean, std, ep), tp_experts

    def _body(self, c: Ctx, tokens: float, name: str = "moe", active: float | None = None):
        h, dt = c.dim("hidden"), c.dt
        e_all, k, ef = c.dim("experts"), c.dim("top_k"), c.local_dim("expert_ffn")
        el = c.local_dim("experts")
        tl, rows, tp_experts = self._rows(c, tokens)
        groups = el if active is None else active
        per = rows / max(groups, 1e-9)
        parts = [c.norm(tl, h, "moe norm"), c.gemm(tl, e_all, h, "router"),
                 *c.ep_all_to_all(tl * k * h * dt, "ep dispatch"),
                 c.gemm(per, 2 * ef, h, "experts gate+up", groups=groups),
                 c.elementwise(4.0 * rows * ef, 3.0 * rows * ef * dt, "experts swiglu"),
                 c.gemm(per, h, ef, "experts down", groups=groups),
                 *c.ep_all_to_all(tl * k * h * dt, "ep combine")]
        if tp_experts:
            parts += c.tp_reduce(tokens * h * dt, "moe tp all_reduce")
        shared = c.dim("shared_experts", 0)
        if shared:
            f = shared * c.dim("expert_ffn") // c.tp_deg
            t_s = tokens / (c.tp_deg if c.sp else 1)
            parts.append(DenseFFN()._body(c, tokens, f, t_s, "shared expert", norm=False))
        return Seq(tuple(parts), name=name)

    def forward(self, c: Ctx):
        return self._body(c, c.tokens)

    def step(self, c: Ctx, mix: TokenMix):
        # few tokens per step: only the experts that receive a token read their weights
        _, rows, _ = self._rows(c, mix.rows)
        picks = rows * c.degrees.ep
        el = c.local_dim("experts")
        active = el * (1.0 - (1.0 - 1.0 / c.dim("experts")) ** picks)
        return self._body(c, mix.rows, active=max(active, 1.0))

    def params(self, c: Ctx) -> Params:
        h = c.dim("hidden")
        shared = c.dim("shared_experts", 0) * c.dim("expert_ffn") // c.tp_deg
        return Params(tp=3.0 * h * shared, rep=h * c.dim("experts") + h,
                      ep=c.local_dim("experts") * 3.0 * h * c.local_dim("expert_ffn"))

    def act_bytes_per_token(self, c: Ctx) -> float:
        k, ef = c.dim("top_k"), c.local_dim("expert_ffn")
        shared = c.dim("shared_experts", 0) * c.dim("expert_ffn") / c.tp_deg
        return c.dt * (2 * c.h_sp + k * (c.dim("hidden") + 3 * ef) + 3 * shared)


@component("mamba2")
class Mamba2(Component):
    """Mamba-2 mixer: input projection, causal conv, chunked state-space scan
    (SSD), gated norm and output projection. Its state is constant in context."""
    dims = ("hidden", "ssm_inner", "ssm_heads", "ssm_state", "ssm_groups")

    def _shapes(self, c: Ctx):
        return (c.dim("hidden"), c.local_dim("ssm_inner"), c.local_dim("ssm_heads"), c.dim("ssm_state"),
                c.local_dim("ssm_groups"), c.dim("conv_kernel", 4))

    def forward(self, c: Ctx):
        t, dt = c.tokens, c.dt
        h, di, nh, n, g, kc = self._shapes(c)
        xbc = di + 2 * g * n
        parts = [c.norm(t / (c.tp_deg if c.sp else 1), h, "mixer norm"), *c.tp_before(t * h * dt),
                 c.gemm(t, 2 * di + 2 * g * n + nh, h, "in_proj"),
                 c.elementwise(2.0 * kc * t * xbc, 2.0 * t * xbc * dt, "conv1d"),
                 c.op("scan", 6.0 * t * di * n, 4.0 * t * di * dt, "ssd scan", tensor=True),
                 c.norm(t, di, "gated norm"),
                 *c.tp_after(c.gemm(t, h, di, "out_proj"), t * h * dt)]
        return Seq(tuple(parts), name="mamba2")

    def step(self, c: Ctx, mix: TokenMix):
        r, b, dt = mix.rows, mix.batch, c.dt
        h, di, nh, n, g, kc = self._shapes(c)
        xbc = di + 2 * g * n
        parts = [c.norm(r, h, "mixer norm"), c.gemm(r, 2 * di + 2 * g * n + nh, h, "in_proj"),
                 c.elementwise(2.0 * kc * r * xbc, 2.0 * r * xbc * dt, "conv1d"),
                 c.op("state update", 6.0 * b * di * n, 2.0 * b * self.state_bytes(c), "ssm state update")]
        if mix.prefill:
            parts.append(c.op("scan", 6.0 * mix.prefill * di * n, 4.0 * mix.prefill * di * dt, "prefill scan",
                              tensor=True))
        parts += [c.norm(r, di, "gated norm"), *c.tp_after(c.gemm(r, h, di, "out_proj"), r * h * dt)]
        return Seq(tuple(parts), name="mamba2")

    def params(self, c: Ctx) -> Params:
        h, di, nh, n, g, kc = self._shapes(c)
        xbc = di + 2 * g * n
        return Params(tp=h * (2 * di + 2 * g * n + nh) + di * h + xbc * kc + 3 * nh + di, rep=h)

    def act_bytes_per_token(self, c: Ctx) -> float:
        h, di, nh, n, g, kc = self._shapes(c)
        return c.dt * (2 * c.h_sp + (2 * di + 2 * g * n + nh) + (di + 2 * g * n) + 2 * di)

    def state_bytes(self, c: Ctx) -> float:
        h, di, nh, n, g, kc = self._shapes(c)
        return di * n * 4.0 + (di + 2 * g * n) * (kc - 1) * c.dt


# ----------------------------------------------------------- whole model ---

def layer_params(c: Ctx, layer: tuple) -> Params:
    total = Params()
    for name in layer:
        total = total + get(name).params(c)
    return total


def model_params(arch: Architecture, rules: ShardingRules | None = None) -> float:
    """Total parameters of the full (unsharded) model."""
    from narfs.parallel.rules import MEGATRON
    c = make_ctx(arch, Degrees(), rules or MEGATRON)
    total = get("embed").params(c) + get("head").params(c)
    for layer in arch.layers:
        total = total + layer_params(c, layer)
    return total.total


def check_architecture(arch: Architecture) -> None:
    """Every component is registered and finds its dimensions."""
    for name in sorted(arch.components() | {"embed", "head"}):
        comp = get(name)
        missing = [d for d in comp.dims if not arch.has(d)]
        if missing:
            raise ValueError(f"{arch.name}: component {name!r} needs dimensions {missing}")
