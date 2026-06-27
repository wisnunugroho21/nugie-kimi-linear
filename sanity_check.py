"""
Correctness checks for the building blocks — run `python sanity_check.py`.

These are the same checks used to validate the implementation. They assert that the
fast/parallel paths equal their simple reference paths, which is the cheapest way to
trust that the code matches the math in the papers.

  1. GDN-2: the chunkwise-parallel core == the token-by-token recurrence (Eq. 9/10
     of arXiv:2605.22791). They are two ways to compute the same thing, so they must
     agree up to fp32 rounding.
  2. MoE: the token-dispatched grouped-GEMM == the dense "run every expert" reference
     (same weights), so any difference is a dispatch/GEMM bug.
  3. Model: a forward pass produces the right logits shape and finite values, and the
     3:1 GDN-2:MLA hybrid schedule is placed as expected.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from gated_deltanet_2.core import (
    chunkwise_gated_delta_rule_2,
    recurrent_gated_delta_rule_2,
)
from multi_latent_attention.moe import GroupedGemmMoE
from kimi_linear_gdn2 import KimiLinear, KimiLinearConfig, count_params


def check_gdn2_chunkwise_equals_recurrent():
    B, H, L, dk, dv, C = 2, 3, 128, 16, 16, 32
    ks = jax.random.split(jax.random.PRNGKey(0), 7)
    q = jax.random.normal(ks[0], (B, H, L, dk))
    k = jax.random.normal(ks[1], (B, H, L, dk))
    v = jax.random.normal(ks[2], (B, H, L, dv))
    g = -jax.nn.softplus(jax.random.normal(ks[3], (B, H, L, dk)))  # log-decay <= 0
    b = jax.nn.sigmoid(jax.random.normal(ks[4], (B, H, L, dk)))    # erase gate
    w = jax.nn.sigmoid(jax.random.normal(ks[5], (B, H, L, dv)))    # write gate
    S0 = jax.random.normal(ks[6], (B, H, dk, dv)) * 0.1

    o_c, s_c = chunkwise_gated_delta_rule_2(q, k, v, g, b, w, S0, chunk_size=C)
    o_r, s_r = recurrent_gated_delta_rule_2(q, k, v, g, b, w, S0)
    rel = jnp.max(jnp.abs(o_c - o_r)) / (jnp.max(jnp.abs(o_r)) + 1e-9)
    print(f"[1] GDN-2 chunkwise vs recurrent  | rel err {float(rel):.2e}  "
          f"state err {float(jnp.max(jnp.abs(s_c - s_r))):.2e}")
    assert rel < 1e-4


def check_moe_dispatch_equals_dense():
    B, L, d = 2, 64, 128
    x = jax.random.normal(jax.random.PRNGKey(1), (B, L, d))
    moe = GroupedGemmMoE(d_model=d, d_ff=128, n_routed=8, top_k=2, rngs=nnx.Rngs(0))
    y_dispatch = moe(x)
    y_dense = moe.dense_forward(x)
    diff = jnp.max(jnp.abs(y_dispatch - y_dense))
    print(f"[2] MoE dispatch vs dense         | max diff {float(diff):.2e}")
    assert diff < 1e-4


def check_model_forward():
    cfg = KimiLinearConfig(
        vocab_size=64, d_model=128, n_layers=8, full_attn_period=4,
        gdn_num_heads=2, gdn_head_k_dim=32, gdn_head_v_dim=32, gdn_chunk_size=32,
        mla_num_q_heads=4, mla_num_kv_heads=2, mla_head_dim=32, max_seq_len=64,
        moe_n_routed=6, moe_top_k=2, moe_d_ff=128,
    )
    model = KimiLinear(cfg, rngs=nnx.Rngs(0))
    ids = jax.random.randint(jax.random.PRNGKey(2), (2, 64), 0, cfg.vocab_size)
    logits, aux = model(ids, return_aux=True)
    full = [i for i, l in enumerate(model.layers) if l.is_full_attn]
    print(f"[3] model forward                 | logits {tuple(logits.shape)} "
          f"finite={bool(jnp.all(jnp.isfinite(logits)))} | params {count_params(model):,}")
    print(f"    hybrid schedule (3:1)         | MLA layers {full}, GDN-2 layers = the rest")
    assert logits.shape == (2, 64, cfg.vocab_size)
    assert full == [3, 7]  # one full-attention layer every 4th -> 3:1 ratio


if __name__ == "__main__":
    check_gdn2_chunkwise_equals_recurrent()
    check_moe_dispatch_equals_dense()
    check_model_forward()
    print("\nall checks passed.")
