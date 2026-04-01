"""
Verify ``embed_windows_batch`` matches stacking ``embed_window`` per row.

Run:
  python3 tests/test_embed_windows_batch.py
Or:
  pytest tests/test_embed_windows_batch.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import sandbox as sb  # noqa: E402


def test_embed_windows_batch_matches_stacked_embed_window() -> None:
    torch.manual_seed(12345)
    vocab_size = 512
    W = 8
    D = 64
    B = 32

    model = sb.TorchAttractorLanguageModel(
        vocab_size,
        state_dim=D,
        train_window_size=W,
        max_window_steps=2,
    )
    model.eval()
    device = next(model.parameters()).device

    context_tensor = torch.randint(0, vocab_size, (B, W), device=device, dtype=torch.long)
    contexts = [context_tensor[i].tolist() for i in range(B)]

    stacked = torch.stack([model.embed_window(ctx) for ctx in contexts], dim=0)
    batched = model.embed_windows_batch(context_tensor)

    assert stacked.shape == (B, W, D)
    assert batched.shape == (B, W, D)

    diff = (stacked - batched).abs()
    max_abs = float(diff.max().item())
    print(f"embed_windows_batch vs stack(embed_window): max_abs_diff = {max_abs:.3e}")

    # Same math; CPU is typically 0. Batched LayerNorm may differ at ~1e-7 on some CUDA builds.
    assert torch.allclose(stacked, batched, atol=1e-6, rtol=1e-6), (
        f"max_abs_diff={max_abs} exceeds tolerance 1e-6"
    )


if __name__ == "__main__":
    test_embed_windows_batch_matches_stacked_embed_window()
    print("OK", flush=True)
