"""
Smoke test: training-aligned generation path (generate → forward_training_window → readout_window).
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


def test_generate_produces_output():
    tok = sb._build_tokenizer(mode="fallback", vocab_cap=512)
    model = sb.TorchAttractorLanguageModel(
        tok.n_vocab,
        state_dim=64,
        train_window_size=4,
        max_window_steps=3,
        convergence_epsilon=0.0,
    )
    model.tokenizer = tok
    model.eval()

    calls: list[str] = []

    orig_ftw = sb.TorchAttractorLanguageModel.forward_training_window

    def wrapped_ftw(self, ctx, collect_dynamics_metrics=False):
        calls.append("forward_training_window")
        return orig_ftw(self, ctx, collect_dynamics_metrics=collect_dynamics_metrics)

    sb.TorchAttractorLanguageModel.forward_training_window = wrapped_ftw  # type: ignore[assignment]

    try:
        text = model.generate(
            "Hello",
            max_tokens=5,
            temperature=1.0,
            top_k=min(28, tok.n_vocab),
        )
    finally:
        sb.TorchAttractorLanguageModel.forward_training_window = orig_ftw  # type: ignore[assignment]

    assert isinstance(text, str)
    assert len(text) > 0
    assert len(calls) == 5, "expected one forward_training_window per generated token"
    assert all(c == "forward_training_window" for c in calls)


def test_forward_training_window_uses_readout_window_shape():
    tok = sb._build_tokenizer(mode="fallback", vocab_cap=512)
    model = sb.TorchAttractorLanguageModel(
        tok.n_vocab,
        state_dim=32,
        train_window_size=4,
        max_window_steps=2,
    )
    model.tokenizer = tok
    model.eval()
    wid = model.window_ids_from_sequence(tok.encode("Hello") or [0])
    with torch.no_grad():
        logits = model.forward_training_window(wid)
    assert logits.shape == (tok.n_vocab,)
