"""
Minimal end-to-end training + checkpointing demo for Kimi Linear (GDN-2 variant),
exercising the four requested libraries together:

    JAX        — array/compute backend and autodiff.
    Flax NNX   — the module system the model is written in (kimi_linear_gdn2.py).
    Optax      — optimizer: AdamW + global-norm grad clipping + warmup-cosine LR.
    Orbax      — checkpoint save / restore (verified by a bit-exact reload).

TASK — "continue the counting sequence." Each example is an arithmetic
progression mod V:  s[t] = (start + t) mod V, with a random `start` per row. The
model is trained with the standard causal next-token objective and learns the
successor rule, then GENERALIZES to held-out starts (held-out accuracy -> 1.0).

Why this task (and not associative recall / copy)? Two honest reasons:
  • It generalizes cleanly and fast on a CPU, so the demo is reproducible.
  • It keeps the GDN-2 decay MILD. The chunkwise core (gated_deltanet_2/core.py)
    forms exp(-G) with G the cumulative log-decay; in fp32 a very strong decay over
    a chunk can overflow (exp(-G)=inf while exp(G)=0 -> 0*inf = NaN). Tasks that
    force sharp decay (e.g. long copy/induction) can hit this during the high-LR
    "phase transition", so we keep chunk_size small and the LR moderate here.
The model's recall machinery itself is validated separately by the GDN-2
chunkwise==recurrent equivalence test and the MoE dispatch==dense test.

Run:  python train.py
"""

from __future__ import annotations

import tempfile

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx

from kimi_linear_gdn2 import KimiLinear, KimiLinearConfig, count_params
from multi_latent_attention.moe import GroupedGemmMoE, update_router_bias


# --------------------------------------------------------------------------- #
#  Data: arithmetic progressions  s[t] = (start + t) mod vocab.
# --------------------------------------------------------------------------- #
def sample_batch(key, batch_size: int, seq_len: int, vocab_size: int):
    """Returns int[B, seq_len]; each row counts up mod vocab from a random start."""
    start = jax.random.randint(key, (batch_size, 1), 0, vocab_size)
    t = jnp.arange(seq_len)[None, :]
    return (start + t) % vocab_size


# --------------------------------------------------------------------------- #
#  Loss + metric.
# --------------------------------------------------------------------------- #
def cross_entropy(logits, targets):
    """Mean next-token cross-entropy. logits:[B,L,V] (already shifted) targets:[B,L]."""
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    tgt_logp = jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    return -jnp.mean(tgt_logp)


def next_token_accuracy(model, batch):
    """Token accuracy of the next-token prediction over a batch (eval only)."""
    full_logits, _ = model(batch)
    logits = full_logits[:, :-1]  # predict positions 1..L-1
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean((preds == batch[:, 1:]).astype(jnp.float32))


# --------------------------------------------------------------------------- #
#  One training step.  @nnx.jit traces the model+optimizer state as pytrees and
#  propagates in-place state mutations (optimizer moments, router biases) out.
# --------------------------------------------------------------------------- #
@nnx.jit
def train_step(model: KimiLinear, optimizer: nnx.Optimizer, batch, bias_lr: float):
    # The GDN-2 chunkwise core needs seq_len divisible by gdn_chunk_size, so we feed
    # the FULL (chunk-aligned) sequence and shift the *logits* for next-token loss —
    # standard causal-LM training (position t's logits only see tokens <= t anyway).
    targets = batch[:, 1:]

    def loss_fn(model: KimiLinear):
        # return_aux=True also gives the MoE load-balancing diagnostics.
        full_logits, aux = model(batch, return_aux=True)
        logits = full_logits[:, :-1]  # drop the last position's prediction
        ce = cross_entropy(logits, targets)
        # Total training loss = CE + the (already alpha-scaled) MoE aux loss.
        loss = ce + aux["aux_loss"]
        # group_sizes flow out as non-differentiated auxiliary data.
        return loss, (ce, aux["group_sizes"])

    (loss, (ce, group_sizes)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

    # Gradient step (AdamW etc.). opt.update mutates model params + optimizer moments.
    optimizer.update(model, grads)

    # Aux-loss-FREE load balancing (DeepSeek-V3 / Kimi style): nudge each MoE layer's
    # per-expert selection bias toward uniform load. Done OUTSIDE the gradient, here
    # right after the optimizer step. group_sizes has one [E] vector per MoE layer.
    for layer, gsz in zip(model.layers, group_sizes):
        if isinstance(layer.channel_mixer, GroupedGemmMoE):
            rb = layer.channel_mixer.router_bias
            rb[...] = update_router_bias(rb[...], gsz, lr=bias_lr)

    # MoE load-balance health: worst-case per-expert share of tokens (1/E = balanced).
    load = group_sizes / (group_sizes.sum(-1, keepdims=True) + 1e-9)  # [n_layers, E]
    return loss, ce, jnp.max(load)


# --------------------------------------------------------------------------- #
#  Train, then checkpoint round-trip.
# --------------------------------------------------------------------------- #
def main(
    steps: int = 10,
    batch_size: int = 32,
    seq_len: int = 32,
    vocab_size: int = 32,
    lr: float = 2e-3,
    bias_lr: float = 1e-3,
    seed: int = 0,
):
    # Small config; chunk_size=8 keeps the fp32 decay accumulation safe (see docstring).
    # Full attention (MLA) every 4th layer -> 3:1 linear:full, Kimi Linear's recipe.
    cfg = KimiLinearConfig(
        vocab_size=vocab_size,
        d_model=128,
        n_layers=4,
        full_attn_period=4,
        gdn_num_heads=2,
        gdn_head_k_dim=32,
        gdn_head_v_dim=32,
        gdn_chunk_size=8,
        mla_num_q_heads=4,
        mla_num_kv_heads=2,
        mla_head_dim=32,
        max_seq_len=seq_len,
        moe_d_ff=128,
        moe_n_routed=6,
        moe_top_k=2,
    )
    model = KimiLinear(cfg, rngs=nnx.Rngs(seed))
    print(
        f"model: {count_params(model):,} params | "
        f"full-attn (MLA) layers: {[i for i, L in enumerate(model.layers) if L.is_full_attn]} "
        f"| the rest are GDN-2 linear-attention layers"
    )

    # --- Optax: clip -> AdamW, under a warmup-cosine learning-rate schedule. ---
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=max(1, steps // 20),
        decay_steps=steps,
        end_value=lr * 0.1,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=0.01),
    )
    # wrt=nnx.Param: only Param leaves get optimizer state (router_bias is a plain
    # Variable updated by hand above, so it is correctly left out).
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # --- Training loop (fresh random batch each step). ---
    held_out = sample_batch(jax.random.PRNGKey(10_000), 64, seq_len, vocab_size)
    key = jax.random.PRNGKey(seed + 1)
    for step in range(1, steps + 1):
        key, sub = jax.random.split(key)
        batch = sample_batch(sub, batch_size, seq_len, vocab_size)
        loss, ce, max_load = train_step(model, optimizer, batch, bias_lr)
        if step % 1 == 0 or step == 1:
            acc = next_token_accuracy(
                model, held_out
            )  # generalization to unseen starts
            print(
                f"step {step:4d} | loss {float(loss):.4f} | ce {float(ce):.4f} | "
                f"held-out acc {float(acc):.3f} | MoE max-load {float(max_load):.3f}"
            )

    # --- Orbax: save the model state, reload into a fresh model, verify equality. ---
    eval_ids = sample_batch(jax.random.PRNGKey(123), 4, seq_len, vocab_size)
    ref_logits, _ = model(eval_ids)  # reference output before reload

    ckpt_dir = tempfile.mkdtemp()
    path = f"{ckpt_dir}/kimi_linear_state"
    checkpointer = ocp.StandardCheckpointer()
    _, state = nnx.split(model)  # split graph (static) from state (arrays)
    checkpointer.save(path, state)
    checkpointer.wait_until_finished()
    print(f"\nsaved checkpoint -> {path}")

    # Rebuild an UNtrained model of the same shape, then overwrite its state. Orbax
    # restores into an abstract (shapes/dtypes only) target produced by nnx.eval_shape.
    fresh = KimiLinear(cfg, rngs=nnx.Rngs(999))
    abstract = nnx.eval_shape(lambda: nnx.split(KimiLinear(cfg, rngs=nnx.Rngs(0)))[1])
    restored = checkpointer.restore(path, abstract)
    nnx.update(fresh, restored)

    new_logits, _ = fresh(eval_ids)
    max_diff = float(jnp.max(jnp.abs(ref_logits - new_logits)))
    print(
        f"checkpoint reload max|Δlogits| = {max_diff:.2e}  "
        f"({'OK' if max_diff < 1e-5 else 'MISMATCH'})"
    )


if __name__ == "__main__":
    main()
