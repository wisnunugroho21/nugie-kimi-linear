"""
Degenerate-case equivalence test: Gated DeltaNet-2  ->  Kimi Delta Attention.

GDN-2's recurrence (your gdn2_core):
    S_t = (I - k_t e_t^T) Diag(alpha_t) S_{t-1} + k_t z_t^T,
    e_t = b_t ⊙ k_t,   z_t = w_t ⊙ v_t.

KDA's recurrence (Kimi Linear Eq. 1):
    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T.

Tie the two channel-wise gates to a single scalar:  b_t = beta_t·1,  w_t = beta_t·1.
Then  e_t = beta_t k_t  and  z_t = beta_t v_t, so

    (I - k_t e_t^T) = (I - beta_t k_t k_t^T),   k_t z_t^T = beta_t k_t v_t^T,

i.e. GDN-2 becomes KDA EXACTLY. This file feeds identical (q,k,v,g) to both, with
GDN-2 given b=w=beta·1, and asserts the chunkwise core, the recurrent core, and an
independent KDA reference all agree to fp32 tolerance.

The equivalence is a property of the RECURRENCE (the mathematically meaningful locus),
so the check is at core level — no L2Norm / conv / projections, which are identical on
both sides anyway and would only add noise.

Run:  python test_gdn2_kda_equivalence.py
Assumes your core is importable as `gdn2_core`.
"""

import jax
import jax.numpy as jnp

from gated_deltanet_2.core import (
    chunkwise_gated_delta_rule_2,
    recurrent_gated_delta_rule_2,
)

F32 = jnp.float32


# --------------------------------------------------------------------------- #
#  Independent KDA reference (Kimi Linear Eq. 1), token-by-token.
#  Deliberately written from the (I - beta k k^T) Diag(alpha) form directly,
#  NOT reusing GDN-2 code, so it is a genuine cross-check.
# --------------------------------------------------------------------------- #
def _kda_single(q, k, v, g, beta, S0):
    """q,k: [L,dk]  v: [L,dv]  g: [L,dk]  beta: [L]  S0: [dk,dv]."""
    alpha = jnp.exp(g.astype(F32))  # channel-wise decay, same as GDN-2
    q, k, v = q.astype(F32), k.astype(F32), v.astype(F32)

    def step(S, inp):
        qt, kt, vt, at, bt = inp
        S_bar = at[:, None] * S  # Diag(alpha_t) S_{t-1}
        read = S_bar.T @ kt  # S_bar^T k_t            [dv]
        S_new = S_bar + bt * (kt[:, None] * (vt - read)[None, :])
        #        = (I - beta k k^T) S_bar + beta k v^T
        o = S_new.T @ qt  # o_t = S_t^T q_t        [dv]
        return S_new, o

    S_final, O = jax.lax.scan(step, S0.astype(F32), (q, k, v, alpha, beta))
    return O, S_final


def kda_reference(q, k, v, g, beta, S0):
    """Batched [B,H,L,*] KDA reference. beta: [B,H,L]."""
    over_h = jax.vmap(_kda_single, in_axes=(0, 0, 0, 0, 0, 0), out_axes=(0, 0))
    over_b = jax.vmap(over_h, in_axes=(0, 0, 0, 0, 0, 0), out_axes=(0, 0))
    return over_b(q, k, v, g, beta, S0)


# --------------------------------------------------------------------------- #
def run_equivalence_test(
    B=2, H=3, L=128, dk=64, dv=64, chunk_size=64, seed=0, atol=2e-4, rtol=2e-4
):
    keys = jax.random.split(jax.random.PRNGKey(seed), 5)
    q = jax.random.normal(keys[0], (B, H, L, dk), F32)
    k = jax.random.normal(keys[1], (B, H, L, dk), F32)
    v = jax.random.normal(keys[2], (B, H, L, dv), F32)
    # L2-normalize q, k per head — the delta rule is only eigenvalue-stable on the
    # unit sphere (this is exactly the L2Norm the layer applies before the kernel).
    # Same normalized tensors go to BOTH sides, so the equivalence is unaffected.
    l2 = lambda t: t / (jnp.linalg.norm(t, axis=-1, keepdims=True) + 1e-6)
    q, k = l2(q), l2(k)
    # Mild negative log-decay keeps cumulative gamma^{-1}=exp(-G) in fp32-safe range.
    g = -0.1 * jax.nn.softplus(jax.random.normal(keys[3], (B, H, L, dk), F32))
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (B, H, L), F32))  # beta in (0,1)
    S0 = jnp.zeros((B, H, dk, dv), F32)

    # Tie GDN-2's channel-wise gates to the scalar beta -> KDA.
    b = jnp.broadcast_to(beta[..., None], (B, H, L, dk))  # erase gate = beta·1
    w = jnp.broadcast_to(beta[..., None], (B, H, L, dv))  # write gate = beta·1

    O_chunk, S_chunk = chunkwise_gated_delta_rule_2(q, k, v, g, b, w, S0, chunk_size)
    O_rec, S_rec = recurrent_gated_delta_rule_2(q, k, v, g, b, w, S0)
    O_kda, S_kda = kda_reference(q, k, v, g, beta, S0)

    def report(name, a, ref):
        d = float(jnp.max(jnp.abs(a - ref)))
        ok = bool(jnp.allclose(a, ref, atol=atol, rtol=rtol))
        print(f"  {name:<34} max|Δ| = {d:.3e}   {'PASS' if ok else 'FAIL'}")
        return ok

    print("GDN-2 (b=w=β·1)  vs  KDA reference")
    ok = True
    ok &= report("chunkwise output  vs KDA", O_chunk, O_kda)
    ok &= report("chunkwise state   vs KDA", S_chunk, S_kda)
    ok &= report("recurrent output  vs KDA", O_rec, O_kda)
    ok &= report("recurrent state   vs KDA", S_rec, S_kda)
    # Internal consistency of the two GDN-2 paths (independent of KDA):
    ok &= report("chunkwise output  vs recurrent", O_chunk, O_rec)

    assert ok, (
        "GDN-2 did NOT reduce to KDA under b=w=β·1 — check gate placement in the core."
    )
    print("\nAll equivalences hold: GDN-2 collapses to KDA when erase=write=β·1.\n")
    return ok


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", False)
    run_equivalence_test()
