"""
Dynamical stability smoke tests for attractor dynamics (no training).

Test 1: random state with zero signal — norm should stay bounded / not diverge.
Test 2: perturbation — after settling, small noise + dynamics should not blow up.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_TS = REPO / "vendor" / "ts-llm"
if str(_TS) not in sys.path:
    sys.path.insert(0, str(_TS))

import sandbox as sb
from dynamics_vectorized import VectorizedWindowDynamics  # type: ignore[import]


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def vec_model(device: torch.device):
    D, W = 64, 4
    m = sb.TorchAttractorLanguageModel(
        512,
        state_dim=D,
        train_window_size=W,
        max_window_steps=12,
        convergence_epsilon=0.0,
        num_waves=1,
    ).to(device)
    m.dynamics = VectorizedWindowDynamics(
        state_dim=D,
        window_size=W,
        num_heads=4,
        rank=16,
        max_steps=12,
        dt=0.05,
        coupling=0.01,
        diag_eigen_min=-0.55,
        diag_eigen_max=-0.25,
    ).to(device)
    m.eval()
    return m


def test_random_state_convergence_bounded_norm(vec_model, device):
    """No token signal: dynamics.step with signal=None; ||S|| should remain finite and bounded."""
    torch.manual_seed(0)
    B, W, D = 2, vec_model.train_window_size, vec_model.state_dim
    S = torch.randn(B, W, D, device=device)
    S = torch.nn.functional.normalize(S, dim=-1)
    norms = []
    with torch.no_grad():
        for _ in range(24):
            S = vec_model.dynamics.step(S, signal=None)
            n = torch.linalg.vector_norm(S, dim=-1).mean().item()
            norms.append(n)
    assert all(torch.isfinite(torch.tensor(norms))), "non-finite norms"
    assert max(norms) < 50.0, f"norm diverged: max={max(norms)}"
    assert min(norms) > 1e-4, f"norm collapsed: min={min(norms)}"


def test_perturbation_recovery(vec_model, device):
    """Stable-ish S*, add noise, run a few steps — should not explode."""
    torch.manual_seed(1)
    B, W, D = 1, vec_model.train_window_size, vec_model.state_dim
    S = torch.zeros(B, W, D, device=device)
    with torch.no_grad():
        for _ in range(16):
            S = vec_model.dynamics.step(S, signal=None)
    S_star = S.clone()
    noise = 0.05 * torch.randn_like(S_star)
    S = S_star + noise
    with torch.no_grad():
        for _ in range(20):
            S = vec_model.dynamics.step(S, signal=None)
    delta = torch.linalg.vector_norm(S - S_star).item()
    assert torch.isfinite(S).all()
    assert delta < 500.0, f"perturbed trajectory diverged (delta={delta})"
