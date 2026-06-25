"""
Kimi Linear host scaffolding for a Gated DeltaNet-2 token mixer (Flax NNX).

This file provides the pieces your gdn2_layer / gdn2_core files deliberately omit,
so you can run "GDN-2 inside Kimi Linear" rather than the GDN-2 paper's own
[GDN-2, MLP, SWA, MLP] hybrid cell:

  * GatedRMSNorm   — CORRECTED output stage. Kimi Linear Eq. 10 uses a
                     low-rank SIGMOID output gate:  Sigmoid(W↑g W↓g x) ⊙ RMSNorm(O).
                     Your gdn2_layer used RMSNorm(O) · SiLU(full-rank gate), which is
                     the GDN-2 paper's choice; Kimi Linear's ablation found sigmoid
                     beats swish, and uses low-rank for matched parameter count.
  * MoE            — reference channel mixer (shared expert + top-k routed experts).
  * MLAttentionNoPE— reference global layer, NoPE (positional info delegated to GDN-2).
  * KimiLinearBlock— pre-norm + residual around a token mixer, then pre-norm + residual
                     around the channel mixer.
  * KimiLinearBackbone — interleaves GDN-2 and MLA blocks at the Kimi Linear 3:1 ratio,
                     with the first layer kept dense (no MoE), as in the paper.

MoE and MLA here are CORRECT-BUT-COMPACT references meant to make the backbone run
and to fix the block structure. For real training, swap in your verified MLA and a
grouped-GEMM / dispatched MoE — they are labelled below.

Assumes your core module is importable as `gdn2_core` and your layer as `gdn2_layer`.
See `patch_note()` at the bottom for the 3-line change to wire GatedRMSNorm into your
existing GatedDeltaNet2.
"""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp

F32 = jnp.float32


# --------------------------------------------------------------------------- #
#  Small building blocks
# --------------------------------------------------------------------------- #
class RMSNorm(nnx.Module):
    """Plain RMSNorm used for the pre-norms around mixer / channel-mixer."""

    def __init__(self, dim: int, *, eps: float = 1e-5, rngs: nnx.Rngs):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((dim,)))

    def __call__(self, x):
        xf = x.astype(F32)
        rms = jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
        return (xf * rms).astype(x.dtype) * self.weight.value


class LowRankLinear(nnx.Module):
    """y = W_up(W_down x). The W↑(W↓·) factorization Kimi Linear uses for its
    output gate and decay, kept at rank = head dim for parameter parity."""

    def __init__(
        self,
        in_features: int,
        rank: int,
        out_features: int,
        *,
        use_bias: bool = False,
        rngs: nnx.Rngs,
    ):
        self.down = nnx.Linear(in_features, rank, use_bias=False, rngs=rngs)
        self.up = nnx.Linear(rank, out_features, use_bias=use_bias, rngs=rngs)

    def __call__(self, x):
        return self.up(self.down(x))


# --------------------------------------------------------------------------- #
#  CORRECTED output stage  (the GatedRMSNorm you asked for)
# --------------------------------------------------------------------------- #
class GatedRMSNorm(nnx.Module):
    """Head-wise RMSNorm of the recurrent output, gated by a LOW-RANK SIGMOID gate.

    Implements Kimi Linear Eq. 10's output stage:

        Sigmoid(W↑g W↓g x) ⊙ RMSNorm(O)

    Two corrections vs the GDN-2 paper block used in your gdn2_layer:
      (1) sigmoid, not SiLU/swish  — Kimi Linear's ablation found the swish output
          gate (GDN's choice) performs substantially worse than sigmoid, and they
          adopt sigmoid across all experiments including their GDN hybrid baseline.
      (2) low-rank gate (W↑ W↓), rank = head dim — Kimi Linear factorizes the gate
          "to ensure a fair parameter comparison" against the full-attention baseline.

    The gate is produced INSIDE the norm from the block input x, so the call site in
    your layer collapses to `O = self.o_norm(O_heads, x)` (no separate gate_proj).
    """

    def __init__(
        self,
        head_dim: int,
        d_model: int,
        inner_dim: int,
        gate_rank: int,
        *,
        eps: float = 1e-5,
        rngs: nnx.Rngs,
    ):
        self.eps = eps
        self.head_dim = head_dim  # dv, the axis RMSNorm normalizes over (head-wise)
        self.inner_dim = inner_dim  # Hv * dv, the full token-mixer output width
        self.weight = nnx.Param(jnp.ones((head_dim,)))
        self.gate = LowRankLinear(
            d_model, gate_rank, inner_dim, use_bias=False, rngs=rngs
        )

    def __call__(self, O_heads, x):
        """O_heads: [B, L, Hv, dv]   x: [B, L, d_model]  ->  [B, L, Hv*dv]."""
        B, L, Hv, dv = O_heads.shape
        O = O_heads.astype(F32)
        rms = jax.lax.rsqrt(jnp.mean(O * O, axis=-1, keepdims=True) + self.eps)
        O = O * rms * self.weight.value  # head-wise RMSNorm
        g = jax.nn.sigmoid(self.gate(x).astype(F32))  # low-rank SIGMOID gate
        g = g.reshape(B, L, Hv, dv)
        return (O * g).reshape(B, L, Hv * dv)


# --------------------------------------------------------------------------- #
#  Channel mixers  (reference)
# --------------------------------------------------------------------------- #
class SwiGLU(nnx.Module):
    """SwiGLU FFN: W_down( SiLU(W_gate x) ⊙ W_up x ). Dense first-layer channel mixer
    and the shape used for each MoE expert."""

    def __init__(self, d_model: int, d_ff: int, *, rngs: nnx.Rngs):
        self.w_gate = nnx.Linear(d_model, d_ff, use_bias=False, rngs=rngs)
        self.w_up = nnx.Linear(d_model, d_ff, use_bias=False, rngs=rngs)
        self.w_down = nnx.Linear(d_ff, d_model, use_bias=False, rngs=rngs)

    def __call__(self, x):
        return self.w_down(jax.nn.silu(self.w_gate(x)) * self.w_up(x))


class MoE(nnx.Module):
    """Reference MoE channel mixer: 1 shared expert + top-k routed experts.

    NOTE: this computes ALL routed experts densely and masks — correct but O(E) in
    FLOPs. Fine for unit tests / small configs; replace with a dispatched grouped-GEMM
    MoE for training. Kimi Linear's 48B config uses 256 routed / 8 active / 1 shared.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_routed: int = 8,
        n_shared: int = 1,
        top_k: int = 2,
        *,
        rngs: nnx.Rngs,
    ):
        self.n_routed = n_routed
        self.top_k = top_k
        self.router = nnx.Linear(d_model, n_routed, use_bias=False, rngs=rngs)

        # Stacked routed-expert weights: [E, in, out] / [E, out, in].
        kg, ku, kd = jax.random.split(rngs.params(), 3)
        s = d_model**-0.5
        self.we_gate = nnx.Param(jax.random.normal(kg, (n_routed, d_model, d_ff)) * s)
        self.we_up = nnx.Param(jax.random.normal(ku, (n_routed, d_model, d_ff)) * s)
        self.we_down = nnx.Param(
            jax.random.normal(kd, (n_routed, d_ff, d_model)) * (d_ff**-0.5)
        )

        self.shared = SwiGLU(d_model, d_ff * n_shared, rngs=rngs)

    def __call__(self, x):
        B, L, d = x.shape
        logits = self.router(x)  # [B,L,E]
        topv, topi = jax.lax.top_k(logits, self.top_k)  # [B,L,k]
        gates = jax.nn.softmax(topv, axis=-1)  # over selected
        oh = jax.nn.one_hot(topi, self.n_routed, dtype=F32)  # [B,L,k,E]
        full = (oh * gates[..., None]).sum(2)  # [B,L,E] sparse weights

        # Dense expert compute (reference): apply every expert, weight by `full`.
        g = jnp.einsum("bld,edf->blef", x, self.we_gate.value)
        u = jnp.einsum("bld,edf->blef", x, self.we_up.value)
        a = jax.nn.silu(g) * u  # [B,L,E,d_ff]
        eo = jnp.einsum("blef,efd->bled", a, self.we_down.value)  # [B,L,E,d]
        routed = jnp.einsum("ble,bled->bld", full, eo)  # weighted combine

        return routed + self.shared(x)


# --------------------------------------------------------------------------- #
#  Global layer  (reference NoPE-MLA)
# --------------------------------------------------------------------------- #
class MLAttentionNoPE(nnx.Module):
    """Compact NoPE Multi-head Latent Attention used as the 1-in-4 global layer.

    Low-rank q and kv latents (the MLA compression), causal softmax, NO positional
    encoding — the position/recency signal is delegated entirely to the GDN-2 layers,
    exactly as Kimi Linear delegates it to KDA. Returns (out, None) so it shares the
    token-mixer interface with GDN-2 (which returns (out, state)).

    This is a structural reference: it omits the decoupled-RoPE dimensions and the
    KV-cache machinery of a production MLA. Swap in your verified MLA for real runs;
    just keep it NoPE.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 16,
        head_dim: int = 128,
        q_lora: int = 512,
        kv_lora: int = 256,
        *,
        rngs: nnx.Rngs,
    ):
        self.H = n_heads
        self.dh = head_dim
        self.scale = head_dim**-0.5
        self.wq = LowRankLinear(d_model, q_lora, n_heads * head_dim, rngs=rngs)
        self.wkv_down = nnx.Linear(d_model, kv_lora, use_bias=False, rngs=rngs)
        self.wk_up = nnx.Linear(kv_lora, n_heads * head_dim, use_bias=False, rngs=rngs)
        self.wv_up = nnx.Linear(kv_lora, n_heads * head_dim, use_bias=False, rngs=rngs)
        self.wo = nnx.Linear(n_heads * head_dim, d_model, use_bias=False, rngs=rngs)

    def __call__(self, x, state=None):
        B, L, _ = x.shape
        split = lambda t: t.reshape(B, L, self.H, self.dh).transpose(0, 2, 1, 3)
        q = split(self.wq(x))  # [B,H,L,dh]
        c = self.wkv_down(x)
        k = split(self.wk_up(c))
        v = split(self.wv_up(c))

        scores = jnp.einsum("bhld,bhmd->bhlm", q, k) * self.scale  # [B,H,L,L]
        causal = jnp.tril(jnp.ones((L, L), bool))
        scores = jnp.where(causal[None, None], scores, -jnp.inf)
        attn = jax.nn.softmax(scores.astype(F32), axis=-1).astype(x.dtype)
        o = jnp.einsum("bhlm,bhmd->bhld", attn, v)
        o = o.transpose(0, 2, 1, 3).reshape(B, L, self.H * self.dh)
        return self.wo(o), None


# --------------------------------------------------------------------------- #
#  Block + backbone
# --------------------------------------------------------------------------- #
class KimiLinearBlock(nnx.Module):
    """One Kimi Linear block: pre-norm + residual token mixer, then pre-norm +
    residual channel mixer.

        h = x + mixer(RMSNorm(x))            # GDN-2 or MLA (both return (out, state))
        y = h + channel(RMSNorm(h))          # MoE, or dense SwiGLU on the first layer
    """

    def __init__(
        self, d_model: int, mixer: nnx.Module, channel: nnx.Module, *, rngs: nnx.Rngs
    ):
        self.mixer_norm = RMSNorm(d_model, rngs=rngs)
        self.mixer = mixer
        self.channel_norm = RMSNorm(d_model, rngs=rngs)
        self.channel = channel

    def __call__(self, x, state=None):
        mixed, new_state = self.mixer(self.mixer_norm(x), state)
        x = x + mixed
        x = x + self.channel(self.channel_norm(x))
        return x, new_state


class KimiLinearBackbone(nnx.Module):
    """Stack of KimiLinearBlocks interleaving GDN-2 and MLA at the Kimi Linear ratio.

    Pattern for hybrid_ratio=3:  G G G M | G G G M | ...   (MLA every 4th layer).
    Layer 0's channel mixer is a dense SwiGLU (no MoE), per the paper.

    Imports your GatedDeltaNet2 from gdn2_layer — make sure it uses the corrected
    GatedRMSNorm (see patch_note()).
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        *,
        # token-mixer dims
        num_heads: int = 16,
        head_k_dim: int = 128,
        head_v_dim: int = 128,
        num_v_heads: int | None = None,
        chunk_size: int = 64,
        # global-layer dims
        mla_q_lora: int = 512,
        mla_kv_lora: int = 256,
        # channel-mixer dims
        d_ff: int = 2048,
        n_routed: int = 8,
        n_shared: int = 1,
        top_k: int = 2,
        hybrid_ratio: int = 3,
        rngs: nnx.Rngs,
    ):
        from gated_deltanet_2.layer import GatedDeltaNet2  # your (patched) token mixer

        self.hybrid_ratio = hybrid_ratio
        period = hybrid_ratio + 1
        layers = []
        for i in range(n_layers):
            is_global = (i % period) == hybrid_ratio
            if is_global:
                mixer = MLAttentionNoPE(
                    d_model,
                    num_heads,
                    head_v_dim,
                    q_lora=mla_q_lora,
                    kv_lora=mla_kv_lora,
                    rngs=rngs,
                )
            else:
                mixer = GatedDeltaNet2(
                    d_model,
                    num_heads=num_heads,
                    head_k_dim=head_k_dim,
                    head_v_dim=head_v_dim,
                    num_v_heads=num_v_heads,
                    chunk_size=chunk_size,
                    rngs=rngs,
                )
            channel = (
                SwiGLU(d_model, d_ff, rngs=rngs)
                if i == 0
                else MoE(d_model, d_ff, n_routed, n_shared, top_k, rngs=rngs)
            )
            layers.append(KimiLinearBlock(d_model, mixer, channel, rngs=rngs))
        self.layers = nnx.data(layers)  # NNX 0.12+: wrap submodule lists

    def __call__(self, x):
        # Training forward (no decode cache). State threading for autoregressive
        # decoding lives in the inference loop, not here.
        for blk in self.layers:
            x, _ = blk(x)
        return x


# --------------------------------------------------------------------------- #
def patch_note():
    """How to wire the corrected GatedRMSNorm into your existing GatedDeltaNet2.

    In GatedDeltaNet2.__init__, REMOVE:
        self.gate_proj = nnx.Linear(d_model, v_proj_dim, use_bias=False, rngs=rngs)
        self.o_norm    = GatedRMSNorm(self.dv, rngs=rngs)
    and ADD:
        self.o_norm = GatedRMSNorm(
            head_dim=self.dv, d_model=d_model,
            inner_dim=self.Hv * self.dv, gate_rank=self.dv, rngs=rngs)

    In GatedDeltaNet2.__call__, REPLACE:
        gate = self.gate_proj(x).reshape(B, L, self.Hv, self.dv)
        O = self.o_norm(O, gate).reshape(B, L, self.Hv * self.dv)
    with:
        O = self.o_norm(O, x)            # low-rank sigmoid gate computed inside, from x

    OPTIONAL (parameter parity on the decay, matching Kimi Linear's low-rank α):
        replace  self.f_proj = nnx.Linear(d_model, k_proj_dim, use_bias=True, ...)
        with     self.f_proj = LowRankLinear(d_model, self.dk, k_proj_dim,
                                             use_bias=True, rngs=rngs)
    """
    return patch_note.__doc__
