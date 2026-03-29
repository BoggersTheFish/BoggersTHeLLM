"""
Phase 2 — directional breaks, residual head mixing, stable window coupling, head-level tension weights.
Defaults preserve prior random breaks / linear mixing when flags are off.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Phase2Config:
    # False → legacy Gaussian break (token/window).
    enable_directional_break: bool = True
    break_base_strength: float = 0.1
    break_min_scale: float = 0.1
    break_max_scale: float = 2.0
    """Clamp for (T_target - T) / T_target scaling."""
    break_t_target: float = 0.12
    """Reference tension for α scaling (window / token scale)."""
    enable_break_rejection: bool = False
    """Revert break if tension rises and row cosine alignment drops."""

    enable_residual_mixing: bool = True
    """drift_lin = state + sigmoid(gate) * mixed_heads (vs mixed only)."""
    mixing_gate_init: float = 0.1
    """Initial sigmoid(gate_raw)."""

    interaction_reg_weight: float = 0.0
    """||C - I||² added to trajectory loss when window interaction enabled."""
    interaction_decay_tau: Optional[float] = None
    """If set, C is multiplied by exp(-|i-j|/tau) before einsum."""

    enable_head_tension_weighting: bool = False
    """Per-row softmax(-T_head_slice) over head drift outputs (head-level only)."""

    store_break_memory: bool = False
    """Store last pre/post break states on model for future attractor reuse."""
