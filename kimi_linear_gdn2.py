"""
Kimi Linear (GDN-2 variant) — the top-level decoder-only language model, in JAX /
Flax NNX. ANNOTATED against "Kimi Linear: An Expressive, Efficient Attention
Architecture."

WHAT KIMI LINEAR IS (paper, Sec. 3 / Fig. 2)
--------------------------------------------
A *hybrid* linear-attention transformer. Most layers use a cheap, O(L) linear-
attention token mixer (the paper's "Kimi Delta Attention", KDA); a minority use
ordinary softmax full attention (Multi-head Latent Attention, MLA). The two are
interleaved at a fixed **3:1 ratio** — three linear layers for every one full-
attention layer — which the paper finds recovers full-attention quality at a
fraction of the KV-cache and compute cost.

  • KDA layers carry positional information implicitly through their recurrence,
    so the full-attention layers need NO positional encoding. Hence the MLA layers
    here are NoPE (see multi_latent_attention/attention.py).
  • Every layer's channel mixer (FFN) is a DeepSeek-V3 / Moonlight-style MoE.

THIS FILE'S ONE DELIBERATE SUBSTITUTION
---------------------------------------
We replace KDA with **Gated DeltaNet-2** ("Decoupling Erase and Write in Linear
Attention", arXiv:2605.22791). Both are gated-delta-rule linear attentions with
fine-grained (channel-wise) gating; GDN-2's twist is a separate erase gate `b` and
write gate `w` instead of the single `beta` that KDA/GDN share. Everything else of
Kimi Linear — the 3:1 hybrid schedule, NoPE MLA, MoE FFN, pre-norm residual blocks
— is kept as in the paper. See gated_deltanet_2/layer.py for that token mixer.

BLOCK STRUCTURE (standard pre-norm transformer; Fig. 2)
-------------------------------------------------------
    x = x + TokenMixer(RMSNorm(x))     # TokenMixer = GDN-2 (linear) OR MLA (full)
    x = x + ChannelMixer(RMSNorm(x))   # ChannelMixer = MoE (or a dense SwiGLU MLP)

MODEL = Embed -> [DecoderLayer] * n_layers -> RMSNorm -> LM head.

TWO FORWARD MODES
-----------------
  • Training / full sequence:  model(input_ids)  — parallel, GDN-2 via its chunkwise
    core, MLA via a full causal-attention matrix.
  • Streaming / inference:     model.step(ids, caches) and model.generate(...)  —
    reuses per-layer state across calls so each new token is O(1) work for the GDN-2
    layers (fixed-size recurrent state) and O(context) for the few MLA layers (growing
    latent cache). See GatedDeltaNet2.step / GroupedQueryLatentAttention.step.
"""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

# Reuse the building blocks already implemented and verified in this repo.
from gated_deltanet_2.layer import GatedDeltaNet2, RMSNorm
from multi_latent_attention.attention import GroupedQueryLatentAttention
from multi_latent_attention.moe import GroupedGemmMoE


# --------------------------------------------------------------------------- #
#  Configuration
#
#  Defaults are deliberately TINY so the whole model trains on a laptop CPU. The
#  paper's 48B-A3B numbers are quoted in comments for reference; only the *ratios*
#  and structure matter for understanding — scale up by raising the dims/layers.
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class KimiLinearConfig:
    vocab_size: int = 256  # paper: 160k; tiny here (byte-level demo)
    d_model: int = 256  # model width  (paper 1.3B: 2048)
    n_layers: int = 8  # depth        (paper 1.3B: 27)

    # --- Hybrid schedule: which layers are FULL attention (MLA) vs linear (GDN-2) ---
    # full_attn_period = 4 places one MLA layer every 4th layer (indices 3, 7, ...),
    # i.e. a 3:1 linear:full ratio — exactly Kimi Linear's hybrid recipe (Sec. 3.2).
    full_attn_period: int = 4

    # --- GDN-2 token mixer (the KDA replacement) — see gated_deltanet_2/layer.py ---
    gdn_num_heads: int = 4  # H key/query heads   (paper 1.3B: 16)
    gdn_head_k_dim: int = 64  # d_k                 (paper: 128)
    gdn_head_v_dim: int = 64  # d_v                 (paper: 128)
    gdn_num_v_heads: int | None = None  # H_v for GQA value heads; None -> = num_heads
    gdn_chunk_size: int = 64  # chunkwise block size C (paper App.: 64).
    #   NOTE: the GDN-2 chunkwise core requires every fed sequence length to be a
    #   multiple of this C (it reshapes L into L/C chunks). Keep seq_len % C == 0.
    gdn_conv_size: int = 4  # short-conv kernel width
    gdn_expanded_erase: bool = False  # erase gate in [0,2] (neg-eigenvalue variant)

    # --- MLA full-attention layers (NoPE) — see multi_latent_attention/attention.py ---
    mla_num_q_heads: int = 8  # query heads
    mla_num_kv_heads: int = 2  # KV/latent heads (GQA); q_heads must be a multiple
    mla_head_dim: int = 64  # per-head latent (rank) width
    max_seq_len: int = 512  # builds the causal mask; cap on trainable length

    # --- Channel mixer (FFN) ---
    use_moe: bool = True  # True -> MoE (faithful); False -> dense SwiGLU MLP
    moe_d_ff: int = 512  # per-expert hidden width (paper: 1408 at 1.3B)
    moe_n_routed: int = 8  # number of routed experts E (paper: 256)
    moe_n_shared: int = 1  # always-on shared experts
    moe_top_k: int = 2  # experts activated per token (paper: 8)
    mlp_d_ff: int = 768  # hidden width of the dense MLP fallback

    rms_eps: float = 1e-5


# --------------------------------------------------------------------------- #
#  Dense SwiGLU MLP — the non-MoE channel mixer.
#
#  Kimi Linear itself is an MoE model, so `use_moe=True` is the faithful path. This
#  small dense alternative exists only to make tiny single-GPU/CPU runs trivial and
#  to show the channel mixer in its simplest form: SwiGLU = (SiLU(xW_g) * xW_u) W_d.
# --------------------------------------------------------------------------- #
class SwiGLUMLP(nnx.Module):
    def __init__(self, d_model: int, d_ff: int, *, rngs: nnx.Rngs):
        self.gate = nnx.Linear(d_model, d_ff, use_bias=False, rngs=rngs)
        self.up = nnx.Linear(d_model, d_ff, use_bias=False, rngs=rngs)
        self.down = nnx.Linear(d_ff, d_model, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down(jax.nn.silu(self.gate(x)) * self.up(x))


# --------------------------------------------------------------------------- #
#  One decoder block: pre-norm token mixer + pre-norm channel mixer, both residual.
#
#  The ONLY thing that varies across layers is the token mixer: GDN-2 (linear) on
#  most layers, MLA (full attention) on the 3:1 schedule. The channel mixer (MoE or
#  dense MLP) is the same kind on every layer — this matches Kimi Linear, where the
#  hybrid is in the *attention*, not the FFN.
# --------------------------------------------------------------------------- #
class DecoderLayer(nnx.Module):
    def __init__(self, cfg: KimiLinearConfig, layer_idx: int, *, rngs: nnx.Rngs):
        # 3:1 schedule: this layer is full-attention iff it is the last of its period.
        self.is_full_attn = (layer_idx + 1) % cfg.full_attn_period == 0

        # Pre-norm before the token mixer (Fig. 2). RMSNorm reused from the GDN-2 layer.
        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.rms_eps, rngs=rngs)

        if self.is_full_attn:
            # Full attention: NoPE Multi-head Latent Attention (absorbed/GQA form).
            self.token_mixer = GroupedQueryLatentAttention(
                embed_dim=cfg.d_model,
                num_q_heads=cfg.mla_num_q_heads,
                num_kv_heads=cfg.mla_num_kv_heads,
                head_dim=cfg.mla_head_dim,
                dropout_rate=0.0,
                seq_length=cfg.max_seq_len,
                rngs=rngs,
            )
        else:
            # Linear attention: Gated DeltaNet-2 (the KDA substitute).
            self.token_mixer = GatedDeltaNet2(
                d_model=cfg.d_model,
                num_heads=cfg.gdn_num_heads,
                head_k_dim=cfg.gdn_head_k_dim,
                head_v_dim=cfg.gdn_head_v_dim,
                num_v_heads=cfg.gdn_num_v_heads,
                chunk_size=cfg.gdn_chunk_size,
                conv_size=cfg.gdn_conv_size,
                expanded_erase=cfg.gdn_expanded_erase,
                rngs=rngs,
            )

        # Pre-norm before the channel mixer.
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.rms_eps, rngs=rngs)

        # Channel mixer: MoE (faithful) or dense SwiGLU (minimal).
        self.is_moe = cfg.use_moe
        if cfg.use_moe:
            self.channel_mixer = GroupedGemmMoE(
                d_model=cfg.d_model,
                d_ff=cfg.moe_d_ff,
                n_routed=cfg.moe_n_routed,
                n_shared=cfg.moe_n_shared,
                top_k=cfg.moe_top_k,
                rngs=rngs,
            )
        else:
            self.channel_mixer = SwiGLUMLP(cfg.d_model, cfg.mlp_d_ff, rngs=rngs)

    def __call__(self, x: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        """x: [B, L, d_model] -> (x, aux_or_None).

        `aux` carries the MoE load-balancing diagnostics the training loop
        needs (aux loss + per-expert token counts for the router-bias update).
        """
        # --- token mixing (residual, pre-norm) ---
        h = self.norm1(x)
        if isinstance(self.token_mixer, GroupedQueryLatentAttention):
            h = self.token_mixer(h)
        else:
            # GDN-2 also returns its end-of-sequence recurrent state; unused in the
            # full-sequence (training) path. Streaming decode reuses it — see .step().
            h, _gdn_state = self.token_mixer(h)
        x = x + h

        # --- channel mixing (residual, pre-norm) ---
        y = self.norm2(x)
        m, aux = self.channel_mixer(y)
        x = x + m
        return x, aux

    def init_cache(self, batch_size: int, max_len: int, dtype=jnp.float32):
        """Per-layer streaming cache: a GDN2Cache (linear layer) or MLACache (MLA)."""
        return self.token_mixer.init_cache(batch_size, max_len, dtype)

    def step(self, x: jax.Array, cache):
        """Streaming forward for one block. x: [B, L, d_model] -> (x, new_cache).
        Only the token mixer is stateful; the channel mixer (MoE/MLP) is position-wise,
        so it needs no cache."""
        h = self.norm1(x)
        h, new_cache = self.token_mixer.step(h, cache)  # GDN-2 and MLA both expose .step
        x = x + h
        y = self.norm2(x)
        m = self.channel_mixer(y)
        if isinstance(self.channel_mixer, GroupedGemmMoE):
            m = m[0]  # MoE returns (out, aux); drop the aux during inference
        x = x + m
        return x, new_cache


# --------------------------------------------------------------------------- #
#  The full model.
# --------------------------------------------------------------------------- #
class KimiLinear(nnx.Module):
    """Decoder-only Kimi Linear LM with a GDN-2 linear-attention backbone."""

    def __init__(self, cfg: KimiLinearConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        # Token embedding table.
        self.embed = nnx.Embed(cfg.vocab_size, cfg.d_model, rngs=rngs)
        # Stack of decoder blocks. NOTE: in Flax NNX a plain Python list of submodules
        # is not tracked as state — it must be wrapped in nnx.List(...).
        self.layers = nnx.List(
            [DecoderLayer(cfg, i, rngs=rngs) for i in range(cfg.n_layers)]
        )
        # Final pre-head norm + untied LM head (Moonlight/DeepSeek do not tie weights;
        # to tie, drop lm_head and use `x @ self.embed.embedding.value.T` instead).
        self.norm_f = RMSNorm(cfg.d_model, eps=cfg.rms_eps, rngs=rngs)
        self.lm_head = nnx.Linear(
            cfg.d_model, cfg.vocab_size, use_bias=False, rngs=rngs
        )

    def __call__(
        self, input_ids: jax.Array, return_aux: bool = False
    ) -> tuple[jax.Array, dict[str, ArrayLike]]:
        """input_ids: int[B, L] -> logits[B, L, vocab]  (or (logits, aux) if return_aux).

        aux = {"aux_loss": scalar summed over MoE layers,
               "group_sizes": list of per-expert token counts, one entry per layer}.
        """
        x = self.embed(input_ids)  # [B, L, d_model]

        aux_loss: ArrayLike = 0.0
        group_sizes: list[
            ArrayLike
        ] = []  # one [E] vector per MoE layer, in layer order
        for layer in self.layers:
            x, aux = layer(x)

            aux_loss = aux_loss + aux["aux_loss"]
            group_sizes.append(aux["group_sizes"])

        x = self.norm_f(x)
        logits = self.lm_head(x)  # [B, L, vocab]

        return logits, {"aux_loss": aux_loss, "group_sizes": jnp.stack(group_sizes)}

    # ----------------------------------------------------------------------- #
    #  Streaming / inference.  Each layer carries its own cache (GDN-2: fixed-size
    #  recurrent state + conv state; MLA: growing latent cache).  Reusing them makes
    #  generation O(1) per token for the linear layers instead of re-reading history.
    # ----------------------------------------------------------------------- #
    def init_cache(
        self, batch_size: int, max_len: int | None = None, dtype=jnp.float32
    ) -> list:
        """Streaming caches for every layer. `max_len` (default cfg.max_seq_len) sizes
        the MLA latent buffers; GDN-2 layers ignore it (their state is fixed-size)."""
        max_len = max_len or self.cfg.max_seq_len
        return [layer.init_cache(batch_size, max_len, dtype) for layer in self.layers]

    def step(self, input_ids: jax.Array, caches: list) -> tuple[jax.Array, list]:
        """One streaming step. input_ids: int[B, L] (L = prompt length on prefill, or
        1 per decoded token). Returns (logits[B, L, vocab], new_caches)."""
        x = self.embed(input_ids)
        new_caches = []
        for layer, cache in zip(self.layers, caches):
            x, new_cache = layer.step(x, cache)
            new_caches.append(new_cache)
        x = self.norm_f(x)
        return self.lm_head(x), new_caches

    def generate(
        self, prompt_ids: jax.Array, max_new_tokens: int, max_len: int | None = None
    ) -> jax.Array:
        """Greedy autoregressive decode that REUSES each layer's state across steps.
        prompt_ids: int[B, P]. Returns the continuation int[B, max_new_tokens].

        Prefill consumes the whole prompt in one step (filling every layer's cache);
        each decode step then feeds back ONE token and carries the caches forward — the
        GDN-2 layers via their fixed-size recurrent state, the MLA layers via the
        growing latent cache. (Wrap `step` in nnx.jit for a fast decode loop.)"""
        B, P = prompt_ids.shape
        max_len = max_len or (P + max_new_tokens)
        caches = self.init_cache(B, max_len)
        logits, caches = self.step(prompt_ids, caches)  # prefill the prompt
        next_tok = jnp.argmax(logits[:, -1:], axis=-1)  # [B, 1] greedy
        outs = [next_tok]
        for _ in range(max_new_tokens - 1):
            logits, caches = self.step(next_tok, caches)  # decode one token
            next_tok = jnp.argmax(logits[:, -1:], axis=-1)
            outs.append(next_tok)
        return jnp.concatenate(outs, axis=1)  # [B, max_new_tokens]


def count_params(model: nnx.Module) -> int:
    """Total number of trainable parameters (sum of nnx.Param leaf sizes)."""
    return int(sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))))
