"""
Phase 1 — structured dynamics & routing (multi-head diffusion, window coupling).

Defaults preserve Phase 0.5 behaviour (num_heads=1, interaction off, diversity off).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Phase1Config:
    num_heads: int = 1
    """H=1 with shared mode + identity W_mix matches single diffusion."""

    head_dim_mode: str = "shared"
    """'shared': each head uses full dim with its own D_h (H×D×D). 'split': partition D into H blocks (D % H == 0)."""

    interaction_scale: float = 0.01
    """Scales learnable cross-position coupling delta (additive to state)."""

    enable_window_interaction: bool = False
    """If True, apply einsum('bid,ij->bjd', S, C) after local dynamics step."""

    head_diversity_weight: float = 0.0
    """Penalty on mean pairwise cosine similarity of head drift directions (0 = off)."""

    enable_per_head_tension: bool = False
    """When True and log_metrics, log per-head geometry tension (split layout on D); scalar control path unchanged."""
