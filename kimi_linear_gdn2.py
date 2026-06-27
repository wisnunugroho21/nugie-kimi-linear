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

Scope: training / full-sequence forward only. Incremental decoding (threading each
GDN-2 layer's recurrent state + an MLA KV-cache) is intentionally omitted to keep
the reference minimal; the hooks for it are noted inline.
"""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp

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
    vocab_size: int = 256          # paper: 160k; tiny here (byte-level demo)
    d_model: int = 256             # model width  (paper 1.3B: 2048)
    n_layers: int = 8              # depth        (paper 1.3B: 27)

    # --- Hybrid schedule: which layers are FULL attention (MLA) vs linear (GDN-2) ---
    # full_attn_period = 4 places one MLA layer every 4th layer (indices 3, 7, ...),
    # i.e. a 3:1 linear:full ratio — exactly Kimi Linear's hybrid recipe (Sec. 3.2).
    full_attn_period: int = 4

    # --- GDN-2 token mixer (the KDA replacement) — see gated_deltanet_2/layer.py ---
    gdn_num_heads: int = 4         # H key/query heads   (paper 1.3B: 16)
    gdn_head_k_dim: int = 64       # d_k                 (paper: 128)
    gdn_head_v_dim: int = 64       # d_v                 (paper: 128)
    gdn_num_v_heads: int | None = None  # H_v for GQA value heads; None -> = num_heads
    gdn_chunk_size: int = 64       # chunkwise block size C (paper App.: 64).
    #   NOTE: the GDN-2 chunkwise core requires every fed sequence length to be a
    #   multiple of this C (it reshapes L into L/C chunks). Keep seq_len % C == 0.
    gdn_conv_size: int = 4         # short-conv kernel width
    gdn_expanded_erase: bool = False    # erase gate in [0,2] (neg-eigenvalue variant)

    # --- MLA full-attention layers (NoPE) — see multi_latent_attention/attention.py ---
    mla_num_q_heads: int = 8       # query heads
    mla_num_kv_heads: int = 2      # KV/latent heads (GQA); q_heads must be a multiple
    mla_head_dim: int = 64         # per-head latent (rank) width
    max_seq_len: int = 512         # builds the causal mask; cap on trainable length

    # --- Channel mixer (FFN) ---
    use_moe: bool = True           # True -> MoE (faithful); False -> dense SwiGLU MLP
    moe_d_ff: int = 512            # per-expert hidden width (paper: 1408 at 1.3B)
    moe_n_routed: int = 8          # number of routed experts E (paper: 256)
    moe_n_shared: int = 1          # always-on shared experts
    moe_top_k: int = 2             # experts activated per token (paper: 8)
    mlp_d_ff: int = 768            # hidden width of the dense MLP fallback

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

    def __call__(self, x):
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

    def __call__(self, x, return_aux: bool = False):
        """x: [B, L, d_model] -> (x, aux_or_None).

        `aux` (only when return_aux and this is an MoE layer) carries the MoE
        load-balancing diagnostics the training loop needs (aux loss + per-expert
        token counts for the router-bias update)."""
        # --- token mixing (residual, pre-norm) ---
        h = self.norm1(x)
        if self.is_full_attn:
            h = self.token_mixer(h)
        else:
            # GDN-2 also returns its end-of-sequence recurrent state; unused in
            # training. For streaming decode you'd carry this state across calls.
            h, _gdn_state = self.token_mixer(h)
        x = x + h

        # --- channel mixing (residual, pre-norm) ---
        y = self.norm2(x)
        if self.is_moe and return_aux:
            m, aux = self.channel_mixer(y, return_aux=True)
        else:
            m, aux = self.channel_mixer(y), None
        x = x + m
        return x, aux


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
        self.lm_head = nnx.Linear(cfg.d_model, cfg.vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, input_ids: jax.Array, return_aux: bool = False):
        """input_ids: int[B, L] -> logits[B, L, vocab]  (or (logits, aux) if return_aux).

        aux = {"aux_loss": scalar summed over MoE layers,
               "group_sizes": list of per-expert token counts, one entry per layer}.
        """
        x = self.embed(input_ids)  # [B, L, d_model]

        aux_loss = 0.0
        group_sizes = []  # one [E] vector per MoE layer, in layer order
        for layer in self.layers:
            x, aux = layer(x, return_aux=return_aux)
            if aux is not None:
                aux_loss = aux_loss + aux["aux_loss"]
                group_sizes.append(aux["group_sizes"])

        x = self.norm_f(x)
        logits = self.lm_head(x)  # [B, L, vocab]

        if return_aux:
            return logits, {"aux_loss": aux_loss, "group_sizes": group_sizes}
        return logits


def count_params(model: nnx.Module) -> int:
    """Total number of trainable parameters (sum of nnx.Param leaf sizes)."""
    return int(
        sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    )
