"""
Muon optimizer setup for Kimi Linear (Optax + Flax NNX).

Moonlight ("Muon is Scalable for LLM Training", arXiv:2502.16982) — the Kimi
lineage's training recipe, also used for Kimi K2 — optimizes all hidden WEIGHT
MATRICES with Muon and everything else with AdamW. Muon takes the momentum-
averaged gradient of a matrix and orthogonalizes it with a few Newton-Schulz
iterations (approximate steepest descent under the spectral norm), which lets
every direction of the update contribute instead of being dominated by a few
large singular values. Orthogonalization is only defined for matrices, hence
the split:

    Muon : parameters that ACT as matrices in a matmul — all Linear kernels,
           and the MoE's stacked expert tensors [E, d_in, d_out] seen as E
           independent matrices (batched via MuonDimensionNumbers).
    AdamW: everything else — the embedding and LM head (Moonlight keeps both
           on AdamW), and all non-matrix parameters: biases, RMSNorm gains,
           the GDN-2 decay parameters A_log / dt_bias, and the depthwise
           short-conv kernels (shape [width, 1, C] — not a matmul matrix).

Two Moonlight adjustments make Muon a drop-in for an AdamW setup:
  * weight decay inside the Muon update (their Sec. 2.2), and
  * consistent-RMS scaling: each update is scaled by 0.2·sqrt(max(fan_in,
    fan_out)) so its RMS matches AdamW's empirical ~0.2 regardless of shape —
    letting Muon reuse AdamW's learning rate and weight-decay values unchanged.

`optax.contrib.muon` implements all of the above (including the internal
Muon/AdamW split); this module's only real job is the classification rule
`_muon_spec` saying which parameter is a hidden weight matrix.

A pleasant side effect vs the previous plain `optax.adamw(weight_decay=...)`:
weight decay now touches ONLY the weight matrices — biases, norm gains, and
the decay/gate parameters are no longer being pulled toward zero.
"""

from __future__ import annotations

import jax
import optax
from flax import nnx
from optax.contrib import MuonDimensionNumbers

# 2D matrix in the x @ W convention: rows are reduced, columns are the output.
_MATRIX = MuonDimensionNumbers(reduction_axis=0, output_axis=1)
# Stacked MoE expert weights [E, d_in, d_out]: axes 1/2 form the matrix, the
# unlisted axis 0 is an implicit batch axis (optax vmaps Newton-Schulz over it).
_EXPERT_STACK = MuonDimensionNumbers(reduction_axis=1, output_axis=2)


def _path_names(path) -> set[str]:
    """The attribute/key names along one pytree path, e.g. {'layers', '0',
    'token_mixer', 'q_proj', 'kernel', 'value'}."""
    names = set()
    for k in path:
        for attr in ("key", "name", "idx"):
            if hasattr(k, attr):
                names.add(str(getattr(k, attr)))
                break
    return names


def _muon_spec(path, leaf) -> MuonDimensionNumbers | None:
    """Classify one parameter: a MuonDimensionNumbers spec => Muon updates it
    (with that matrix layout); None => it goes to the AdamW side."""
    names = _path_names(path)

    # Moonlight: embedding + LM head stay on AdamW. A_log is 2D [H, dk] but is a
    # per-channel decay parameter, not a matmul weight — AdamW as well.
    if names & {"embed", "lm_head", "A_log"}:
        return None

    if names & {"w_in", "w_out"} and leaf.ndim == 3:
        return _EXPERT_STACK  # MoE experts: E stacked matrices

    if leaf.ndim == 2:
        return _MATRIX  # every Linear kernel (projections, gates, router, ...)

    # Biases, RMSNorm gains, dt_bias (1D) and depthwise conv kernels (3D but
    # not a matmul matrix) -> AdamW.
    return None


def _spec_tree(params):
    return jax.tree_util.tree_map_with_path(_muon_spec, params)


def make_optimizer(
    model: nnx.Module,
    learning_rate,
    weight_decay: float = 0.01,
    clip_norm: float = 1.0,
    verbose: bool = True,
) -> nnx.Optimizer:
    """Global-norm clip -> Muon (matrices) / AdamW (the rest), NNX-wrapped.

    `learning_rate` may be a float or an Optax schedule; thanks to Muon's
    consistent-RMS scaling it is the SAME learning-rate scale you would give
    AdamW. wrt=nnx.Param: only Param leaves get optimizer state (the MoE
    router_bias is a plain Variable updated by hand in the training loop, so
    it is correctly left out).
    """
    tx = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.contrib.muon(
            learning_rate=learning_rate,
            weight_decay=weight_decay,  # decays ONLY the Muon-side matrices
            adam_weight_decay=0.0,  # embed/head/biases/norms/decays: no decay
            consistent_rms=0.2,  # Moonlight: match AdamW's update RMS
            muon_weight_dimension_numbers=_spec_tree,
        ),
    )

    if verbose:
        params = nnx.state(model, nnx.Param)
        leaves = jax.tree_util.tree_leaves_with_path(params)
        n_muon = sum(l.size for p, l in leaves if _muon_spec(p, l) is not None)
        n_adam = sum(l.size for p, l in leaves if _muon_spec(p, l) is None)
        print(
            f"optimizer: Muon on {n_muon:,} matrix params, "
            f"AdamW on {n_adam:,} others (embed/head/biases/norms/decays)"
        )

    return nnx.Optimizer(model, tx, wrt=nnx.Param)
