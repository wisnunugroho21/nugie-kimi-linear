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
  4. Streaming: decoding token-by-token while REUSING each layer's state (GDN-2's
     fixed-size recurrent + conv state, MLA's latent cache) == the full-sequence
     forward. The predicted (argmax) tokens must match exactly, so generation is
     equivalent to scoring the whole sequence at once.
  5. Chunkwise prefill: a single step() over a ragged-length prompt (whole chunks
     via the parallel chunkwise core + a recurrent tail) == the full forward on
     those positions, and decode continues seamlessly from the prefilled caches.
"""

import sys

import jax
import jax.numpy as jnp
from flax import nnx

# The diagnostic prints below use a few non-ASCII math symbols (e.g. Δ). On Windows
# the console defaults to cp1252, which cannot encode them and raises
# UnicodeEncodeError mid-print. Force UTF-8 output so the script runs everywhere.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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


def check_gdn2_strong_decay_stability():
    """Extreme decay stress test. The chunkwise core forms pairwise decay ratios
    exp(G_r - G_s) in log space (every exponent <= 0), so even a cumulative
    log-decay of ~ -20 * 64 within one chunk — which overflowed the old
    exp(-G)-based form into inf/NaN — must stay finite, match the recurrent
    reference, AND have finite gradients (the 'trains fine, then NaN' case)."""
    B, H, L, dk, dv, C = 2, 2, 128, 16, 16, 64
    ks = jax.random.split(jax.random.PRNGKey(3), 7)
    q = jax.random.normal(ks[0], (B, H, L, dk))
    k = jax.random.normal(ks[1], (B, H, L, dk))
    v = jax.random.normal(ks[2], (B, H, L, dv))
    g = jnp.minimum(-20.0 + jax.random.normal(ks[3], (B, H, L, dk)), 0.0)  # brutal decay
    b = jax.nn.sigmoid(jax.random.normal(ks[4], (B, H, L, dk)))
    w = jax.nn.sigmoid(jax.random.normal(ks[5], (B, H, L, dv)))
    S0 = jax.random.normal(ks[6], (B, H, dk, dv)) * 0.1

    o_c, s_c = chunkwise_gated_delta_rule_2(q, k, v, g, b, w, S0, chunk_size=C)
    o_r, s_r = recurrent_gated_delta_rule_2(q, k, v, g, b, w, S0)
    finite = bool(jnp.all(jnp.isfinite(o_c)) and jnp.all(jnp.isfinite(s_c)))
    rel = jnp.max(jnp.abs(o_c - o_r)) / (jnp.max(jnp.abs(o_r)) + 1e-9)

    # Gradient path: a NaN anywhere in the forward poisons every gradient.
    def loss(q, k, v, g, b, w, S0):
        o, s = chunkwise_gated_delta_rule_2(q, k, v, g, b, w, S0, chunk_size=C)
        return jnp.sum(o**2) + jnp.sum(s**2)

    grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4, 5, 6))(q, k, v, g, b, w, S0)
    grads_finite = bool(
        all(jnp.all(jnp.isfinite(gr)) for gr in grads)
    )
    print(f"[1b] GDN-2 strong-decay stress    | finite={finite} rel err {float(rel):.2e} "
          f"grads finite={grads_finite}")
    assert finite and grads_finite and rel < 1e-4


def check_moe_dispatch_equals_dense():
    B, L, d = 2, 64, 128
    x = jax.random.normal(jax.random.PRNGKey(1), (B, L, d))
    moe = GroupedGemmMoE(d_model=d, d_ff=128, n_routed=8, top_k=2, rngs=nnx.Rngs(0))
    y_dispatch, _aux = moe(x)  # MoE now always returns (out, aux)
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
    logits, aux = model(ids)  # the model always returns (logits, aux)
    full = [i for i, l in enumerate(model.layers) if l.is_full_attn]
    print(f"[3] model forward                 | logits {tuple(logits.shape)} "
          f"finite={bool(jnp.all(jnp.isfinite(logits)))} | params {count_params(model):,}")
    print(f"    hybrid schedule (3:1)         | MLA layers {full}, GDN-2 layers = the rest")
    assert logits.shape == (2, 64, cfg.vocab_size)
    assert full == [3, 7]  # one full-attention layer every 4th -> 3:1 ratio


def check_streaming_equals_full():
    cfg = KimiLinearConfig(
        vocab_size=64, d_model=128, n_layers=8, full_attn_period=4,
        gdn_num_heads=2, gdn_head_k_dim=32, gdn_head_v_dim=32, gdn_chunk_size=8,
        mla_num_q_heads=4, mla_num_kv_heads=2, mla_head_dim=32, max_seq_len=64,
        moe_n_routed=6, moe_top_k=2, moe_d_ff=128,
    )
    model = KimiLinear(cfg, rngs=nnx.Rngs(0))
    B, L = 2, 32
    ids = jax.random.randint(jax.random.PRNGKey(2), (B, L), 0, cfg.vocab_size)

    full_logits, _ = model(ids)  # full-sequence forward (chunkwise GDN-2 + full MLA)

    # Decode token-by-token, carrying the per-layer caches forward.
    caches = model.init_cache(B, max_len=L)
    stream = []
    for t in range(L):
        lt, caches = model.step(ids[:, t : t + 1], caches)
        stream.append(lt)
    stream_logits = jnp.concatenate(stream, axis=1)

    diff = float(jnp.max(jnp.abs(full_logits - stream_logits)))
    agree = float(jnp.mean((jnp.argmax(full_logits, -1) == jnp.argmax(stream_logits, -1))))
    gen = model.generate(ids[:, :8], max_new_tokens=10)  # greedy decode reusing state
    print(f"[4] streaming vs full forward     | max|Δlogits| {diff:.2e} "
          f"argmax-agreement {agree:.3f} | generate -> {tuple(gen.shape)}")
    assert diff < 1e-3 and agree == 1.0 and gen.shape == (B, 10)

    # Chunkwise prefill: one step() call with a RAGGED prompt length (13 = one
    # 8-chunk through the parallel core + a 5-token recurrent tail) must equal the
    # full forward on those positions, and decoding must continue seamlessly from
    # the prefilled caches — i.e. the chunkwise/recurrent split is invisible.
    P = 13
    pre_logits, pre_caches = model.step(ids[:, :P], model.init_cache(B, max_len=L))
    pdiff = float(jnp.max(jnp.abs(full_logits[:, :P] - pre_logits)))
    nxt_logits, _ = model.step(ids[:, P : P + 1], pre_caches)
    ndiff = float(jnp.max(jnp.abs(full_logits[:, P] - nxt_logits[:, 0])))
    print(f"[5] chunkwise prefill (ragged 13) | max|Δlogits| prefill {pdiff:.2e} "
          f"next-token {ndiff:.2e}")
    assert pdiff < 1e-3 and ndiff < 1e-3


if __name__ == "__main__":
    check_gdn2_chunkwise_equals_recurrent()
    check_gdn2_strong_decay_stability()
    check_moe_dispatch_equals_dense()
    check_model_forward()
    check_streaming_equals_full()
    print("\nall checks passed.")
