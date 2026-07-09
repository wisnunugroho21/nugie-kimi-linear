"""
Complete training loop for Kimi Linear (GDN-2 variant) on Project Gutenberg, with
per-step evaluation and an interactive chat REPL at the end.

DATA PIPELINE (the four data libraries):
  - HuggingFace `datasets`     : stream raw Project Gutenberg text from the Hub.
  - HuggingFace `transformers` : GPT-2 byte-level BPE tokenizer (vocab 50257).
  - HuggingFace `tokenizers`   : the fast Rust tokenizer that `transformers` wraps.
  - Google `grain`             : shuffle / repeat / batch the token windows for training.
Then JAX / Flax NNX / Optax train the model defined in kimi_linear_gdn2.py.

FLOW:
  1. Stream + tokenize Gutenberg into one long token array (cached to disk).
  2. Cut it into fixed-length windows; hold out a slice for evaluation.
  3. Grain feeds shuffled, repeating, batched windows to the training step.
  4. Train; AFTER EVERY STEP also run an eval step so you can watch train vs. held-out
     loss/accuracy track progress (and overfitting) live.
  5. Drop into an infinite chat loop: type a prompt, the model continues it via the
     streaming `generate` (reusing each layer's recurrent/KV state, token by token).

Run:  python train_chat_gutenberg.py        # train, then chat (Ctrl-C during training
                                            # jumps straight to chat; type /exit to quit)

NOTE: this is a tiny model trained briefly on a CPU, so expect loss to drop but the
chat output to be only vaguely English — the point is a correct, complete pipeline.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Prompts/banners below use non-ASCII punctuation (em dashes). On Windows the console
# defaults to cp1252; force UTF-8 so printing (and echoing user prompts) never crashes.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from kimi_linear_gdn2 import KimiLinear, KimiLinearConfig, count_params
from optimizer import make_optimizer

# Reuse the exact training step (AdamW + grad clip + MoE router-bias update) and the
# cross-entropy from train.py, so this script only adds data + eval + chat on top.
from train import cross_entropy, train_step


# --------------------------------------------------------------------------- #
#  1. Data: stream Project Gutenberg -> tokens -> fixed-length windows.
# --------------------------------------------------------------------------- #
def load_gutenberg_tokens(tokenizer, n_tokens: int, cache_path: str) -> np.ndarray:
    """Stream Project Gutenberg, tokenize until we have `n_tokens` tokens, cache to
    disk. Returns an int32 array of token ids. Documents are joined by the EOS token
    so the model sees explicit boundaries (standard LM-data practice)."""
    if os.path.exists(cache_path):
        return np.load(cache_path)

    from datasets import load_dataset

    print(f"streaming Project Gutenberg, tokenizing ~{n_tokens:,} tokens ...")
    stream = load_dataset("sedthh/gutenberg_english", split="train", streaming=True)
    eos = tokenizer.eos_token_id or 0
    ids: list[int] = []
    for example in stream:
        text = (example.get("TEXT") or "").strip()
        if not text:
            continue
        ids.extend(tokenizer.encode(text))
        ids.append(eos)  # document separator
        if len(ids) >= n_tokens:
            break
    arr = np.asarray(ids[:n_tokens], dtype=np.int32)
    np.save(cache_path, arr)
    print(f"tokenized {arr.size:,} tokens -> cached at {cache_path}")
    return arr


def make_windows(ids: np.ndarray, seq_len: int) -> np.ndarray:
    """Cut the flat token stream into non-overlapping [N, seq_len] windows. Each window
    is one training example; the next-token loss inside it shifts by one position."""
    n = ids.size // seq_len
    return ids[: n * seq_len].reshape(n, seq_len)


# --------------------------------------------------------------------------- #
#  2. Evaluation step (no gradients) — run after every training step.
# --------------------------------------------------------------------------- #
@nnx.jit
def eval_step(model: KimiLinear, batch):
    """Held-out next-token cross-entropy + token accuracy on a fixed eval batch."""
    full_logits, _ = model(batch)  # full-sequence (chunkwise) forward, no aux needed
    logits = full_logits[:, :-1]
    targets = batch[:, 1:]
    ce = cross_entropy(logits, targets)
    acc = jnp.mean((jnp.argmax(logits, -1) == targets).astype(jnp.float32))
    return ce, acc


# --------------------------------------------------------------------------- #
#  3. Sampling-based text generation for the chat loop.
#
#  Uses the model's STREAMING api directly (init_cache + step), so generation reuses
#  each layer's state across tokens instead of re-reading the whole prefix every step.
#  Greedy `model.generate` exists too, but a tiny model loops badly under argmax, so
#  here we add temperature + top-k sampling for more varied (if still rough) text.
# --------------------------------------------------------------------------- #
def sample_next(logits_last, key, temperature: float, top_k: int):
    """logits_last: [B, V] -> sampled token ids [B]."""
    logits = logits_last.astype(jnp.float32) / max(temperature, 1e-6)
    if top_k and top_k < logits.shape[-1]:
        kth = jax.lax.top_k(logits, top_k)[0][:, -1:]  # k-th largest logit per row
        logits = jnp.where(logits < kth, -jnp.inf, logits)  # keep only the top-k
    return jax.random.categorical(key, logits, axis=-1)


def generate_text(
    model: KimiLinear,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 60,
    temperature: float = 0.8,
    top_k: int = 40,
    seed: int = 0,
) -> str:
    eos = tokenizer.eos_token_id
    prompt_ids = tokenizer.encode(prompt) or [eos or 0]
    x = jnp.asarray(prompt_ids, dtype=jnp.int32)[None]  # [1, P]

    # Cache big enough for the prompt + everything we will generate.
    caches = model.init_cache(1, max_len=len(prompt_ids) + max_new_tokens)
    logits, caches = model.step(x, caches)  # prefill the prompt

    key = jax.random.PRNGKey(seed)
    out_ids: list[int] = []
    for _ in range(max_new_tokens):
        key, sub = jax.random.split(key)
        nxt = sample_next(logits[:, -1], sub, temperature, top_k)  # [1]
        tok = int(nxt[0])
        if tok == eos:
            break
        out_ids.append(tok)
        logits, caches = model.step(nxt[None], caches)  # feed the new token back in
    return tokenizer.decode(out_ids).strip()


# --------------------------------------------------------------------------- #
#  4. Put it all together: build data, train with live eval, then chat.
# --------------------------------------------------------------------------- #
def main(
    n_tokens: int = 1_000_000,  # corpus size to stream/tokenize (cached after first run)
    seq_len: int = 128,  # window length; MUST be a multiple of gdn_chunk_size
    batch_size: int = 8,
    eval_batch: int = 8,
    steps: int = 10,
    lr: float = 3e-3,
    bias_lr: float = 1e-3,
    seed: int = 0,
    interactive: bool = True,
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # We tokenize whole documents (often > the GPT-2 1024 context) only to build a
    # token stream, so silence the harmless "sequence longer than max length" warning.
    tokenizer.model_max_length = int(1e12)

    # --- data ---
    cache_path = f"gutenberg_gpt2_{n_tokens}.npy"
    ids = load_gutenberg_tokens(tokenizer, n_tokens, cache_path)
    windows = make_windows(ids, seq_len)
    n_eval_win = max(eval_batch, windows.shape[0] // 10)
    train_windows, eval_windows = windows[:-n_eval_win], windows[-n_eval_win:]
    eval_fixed = jnp.asarray(eval_windows[:eval_batch])  # one fixed held-out batch
    print(
        f"corpus: {ids.size:,} tokens | windows: {train_windows.shape[0]:,} train, "
        f"{eval_windows.shape[0]:,} eval | seq_len {seq_len}"
    )

    # --- Grain training loader: shuffle -> repeat forever -> batch ---
    train_iter = iter(
        grain.MapDataset.source(train_windows)
        .shuffle(seed=seed)
        .repeat()
        .batch(batch_size)
    )

    # --- model (vocab is fixed by the tokenizer) ---
    cfg = KimiLinearConfig(
        vocab_size=tokenizer.vocab_size,  # 50257
        d_model=256,
        n_layers=4,
        full_attn_period=4,  # layer 3 is MLA, layers 0-2 are GDN-2 (3:1)
        gdn_num_heads=4,
        gdn_head_k_dim=64,
        gdn_head_v_dim=64,
        gdn_chunk_size=32,
        mla_num_q_heads=4,
        mla_num_kv_heads=2,
        mla_head_dim=64,
        max_seq_len=seq_len,
        moe_d_ff=512,
        moe_n_routed=8,
        moe_top_k=2,
    )
    model = KimiLinear(cfg, rngs=nnx.Rngs(seed))
    print(f"model: {count_params(model):,} params | vocab {cfg.vocab_size}")

    # --- Optax: clip -> Muon/AdamW (Moonlight recipe) under a warmup-cosine
    # schedule; same LR scale as AdamW thanks to Muon's consistent-RMS scaling
    # (see optimizer.py for the matrix/non-matrix split).
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=max(1, steps // 20),
        decay_steps=steps,
        end_value=lr * 0.1,
    )
    optimizer = make_optimizer(model, schedule, weight_decay=0.01)

    # --- training loop with per-step evaluation ---
    print("\ntraining (Ctrl-C to stop early and jump to chat)\n" + "-" * 64)
    try:
        for step in range(1, steps + 1):
            batch = next(train_iter)  # numpy int32 [B, seq_len] from Grain
            tr_loss, tr_ce, max_load = train_step(model, optimizer, batch, bias_lr)
            # Evaluate EVERY step on the fixed held-out batch to track progress.
            ev_ce, ev_acc = eval_step(model, eval_fixed)
            print(
                f"step {step:4d} | train ce {float(tr_ce):6.3f} | "
                f"eval ce {float(ev_ce):6.3f} ppl {float(jnp.exp(ev_ce)):7.1f} "
                f"acc {float(ev_acc):.3f} | MoE max-load {float(max_load):.3f}"
            )
    except KeyboardInterrupt:
        print("\n[training interrupted — switching to chat]")

    # --- interactive chat REPL (or a single sample generation if non-interactive) ---
    if interactive:
        chat(model, tokenizer)
    else:
        print("\n[sample]", generate_text(model, tokenizer, "The ", max_new_tokens=20))
    return model, tokenizer


def chat(model: KimiLinear, tokenizer):
    """Infinite interactive loop: read a prompt, print the model's continuation."""
    print("\n" + "=" * 64)
    print("chat — type a prompt and the model continues it.")
    print("commands: /exit to quit, /temp <f> to set temperature.  (Ctrl-D quits)")
    print("=" * 64)
    temperature = 0.8
    turn = 0
    while True:
        try:
            prompt = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return
        if not prompt:
            continue
        if prompt == "/exit":
            print("bye.")
            return
        if prompt.startswith("/temp"):
            try:
                temperature = float(prompt.split()[1])
                print(f"[temperature = {temperature}]")
            except (IndexError, ValueError):
                print("[usage: /temp 0.8]")
            continue
        turn += 1
        text = generate_text(
            model, tokenizer, prompt, temperature=temperature, seed=turn
        )
        print(f"model> {text}")


if __name__ == "__main__":
    main()
