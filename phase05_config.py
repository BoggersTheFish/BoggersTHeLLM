"""
Phase 0.5 — instrumentation + stability toggles (no architecture change).

Weights (w1, w2, w3) combine raw tension components:
  T_total = w1 * T_energy + w2 * T_alignment + w3 * T_entropy
Defaults (1.0, TENSION_LAMBDA, TENSION_MU) match pre-Phase05 formulas in sandbox.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Phase05Config:
    log_metrics: bool = False
    """If True, collect per-batch diagnostics and tracing; False keeps only control-flow tension computations."""

    batch_metrics_csv: str | None = None
    """Path to append per-batch CSV rows (requires log_metrics)."""

    enforce_negative_definite_diffusion: bool = False
    """If True, SimpleAttractorDynamics uses D = -(A^T A) - eps I (strictly negative definite)."""

    adaptive_window_dt: bool = False
    """EMA scale on window positional dt from tension (smooth, clamped)."""

    adaptive_dt_spike_thresh: float = 0.22
    adaptive_dt_low_thresh: float = 0.06
    adaptive_dt_smooth: float = 0.12
    adaptive_dt_min_scale: float = 1e-3
    adaptive_dt_max_scale: float = 1.0
    adaptive_dt_spike_factor: float = 0.88
    adaptive_dt_low_factor: float = 1.04

    tension_weights: Tuple[float, float, float] | None = None
    """
    (w_energy, w_align, w_entropy). None = use model buffers tension_lambda / tension_mu
    with energy weight 1.0 (legacy T = E + λ·align + μ·H).
    """

    tension_lambda: float = 0.0
    """
    Weight λ on window tension in energy descent:
    ``E = mean(sum_i energy_heads[i](wave_i)) + λ * mean(compute_tension_window(S))``.
    Set to 0.0 to use only the learned energy head. Differentiable in ``S`` (same ``T`` as
    ``TorchAttractorLanguageModel.compute_tension_window``). Not the model buffer
    ``tension_lambda`` used for alignment weight in the tension decomposition when
    ``tension_weights`` is None.
    """

    anchor_lambda: float = 0.0
    """
    Weight on nearest-token-embedding distance in window energy:
    ``… + anchor_lambda * mean_b mean_w min_v ||S[b,w] - embed[v]||_2`` with ``embed``
    detached so gradients flow through ``S`` only. 0 disables (skips ``cdist``).
    """

    anchor_force_strength: float = 0.0
    """
    Before each inner energy-descent step, relax toward the nearest detached token embedding:
    ``S <- S - strength * (S - anchor[nearest])`` per (B, W) position. 0 disables.
    """

    anchor_search_topk: int = 64
    """
    Token-anchor distance search (``anchor_lambda``, ``anchor_force_strength``): candidates are
    the top-``k`` tokens by ``readout_window`` logits per batch row; distances are only computed
    against those embeddings (``O(k)`` per position vs full ``V``).
    """

    enable_anchor_freeze: bool = False
    """
    In ``run_window_dynamics``, after each outer step, mark wave slices whose L2 distance to the
    nearest top-``k`` anchor (same slice) is below :attr:`anchor_freeze_threshold`; on the next step,
    zero energy-descent gradients on those slices (coupling / GOAT / anchor pull still apply).
    """

    anchor_freeze_threshold: float = 0.03
    """Per-wave L2 distance below which a slice is treated as converged (try 0.01–0.05 for unit-scale states)."""

    anchor_freeze_max_age: int = 0
    """
    If > 0, a slice may stay gradient-frozen for at most this many consecutive stable outer steps
    (counting from the first step it is below the threshold); then it is forced unfrozen even if
    still close to an anchor. ``0`` means no cap (stay frozen while stable).
    """

    anchor_contrastive_weight: float = 0.0
    """
    Trajectory training: add ``weight * mean_b (‖s_b - e_{t_b}‖² - mean_k ‖s_b - e_{n_{b,k}}‖²)``
    on the last window row ``s_b`` vs target embedding ``e_{t_b}`` and random negative token rows
    (negatives detached from the graph). 0 disables.
    """

    anchor_contrastive_num_negatives: int = 8
    """Number of random negative token embeddings per batch row (excluding the target id)."""

    multi_negative: bool = False
    num_negatives: int = 4
    """Contrastive negatives: random permutations of teacher batch, averaged cosine."""

    trajectory_temperature: float = 1.0
    """Divides (margin) inside ReLU; 1.0 = unchanged."""

    stagnation_delta_thresh: float = 1e-3
    """Fraction of window substeps with mean row-wise ||ΔS|| below this counts as stagnation."""

    trajectory_intermediate_ce_weight: float = 0.0
    """
    In trajectory training, add ``weight * mean_k CE(readout_window(S_k), target)`` over outer
    dynamics steps ``S_k`` (same target each step). 0 disables.
    """

    trajectory_guidance_nudge_scale: float = 0.0
    """
    Each outer dynamics step, before coupling / energy descent:
    ``S <- S + β * (T.detach() - S)`` when :meth:`~sandbox.TorchAttractorLanguageModel.run_window_dynamics`
    receives ``target_states`` (same shape as ``S``). 0 disables nudging.
    """

    trajectory_guidance_mse_weight: float = 0.0
    """
    Trajectory training: add ``weight * MSE(S_pred, T.detach())`` when batch ``target_states``
    are provided (typically precomputed ``(B, W, D)`` per window). 0 disables.
    """

    enable_state_normalization: bool = True
    """
    If True, apply ``F.normalize(S, dim=-1)`` after each window dynamics step (and after
    directional break updates). Differentiable; keeps each (B, W, D) row on the unit sphere in D.
    """

    enable_adaptive_attractor_dt: bool = False
    """Per batch row: adapt inner energy-descent step size from energy before/after each update."""

    adaptive_attractor_energy_rel_thresh: float = 0.01
    """Relative drop ``(E_before - E_after) / (|E_before|+eps)`` to treat as significant decrease."""

    adaptive_attractor_dt_min: float = 1e-8
    """Floor on adapted dt (per row)."""

    adaptive_attractor_dt_grow: float = 1.1
    """Multiply dt when energy decreased significantly (capped at initial dt per row)."""

    adaptive_attractor_dt_shrink: float = 0.5
    """Multiply dt when energy increased."""

    energy_reg_weight: float = 0.001
    """
    Trajectory loss adds ``weight * mean(E^2)`` where ``E`` is per-batch-row window potential
    (same as :meth:`~TorchAttractorLanguageModel._window_energy_per_batch_row` on ``S_pred``).
    """

