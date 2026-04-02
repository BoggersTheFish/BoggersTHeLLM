#!/usr/bin/env python3
"""
TS workflow smoke: run_window_dynamics (simple + vectorized) + training-parity generate().
Avoids deprecated state_cache.logits / generate_with_cache for decoding.
Run from repo root: python scripts/ts_workflow_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
import torch.nn.functional as F

import sandbox as sb
from dynamics_vectorized import VectorizedWindowDynamics

STATE_DIM = 512
WINDOW_SIZE = 4
MAX_WINDOW_STEPS = 8
VOCAB_SIZE = 512  # FULL_VOCAB is []; use explicit size for smoke test


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = sb._build_tokenizer(mode="fallback", vocab_cap=512)
    model = sb.TorchAttractorLanguageModel(
        VOCAB_SIZE,
        state_dim=STATE_DIM,
        train_window_size=WINDOW_SIZE,
        max_window_steps=MAX_WINDOW_STEPS,
        num_waves=1,
    ).to(device)
    model.tokenizer = tok
    model.eval()

    # 2. Training-parity logits (readout_window), not state_cache.logits()
    wid = model.window_ids_from_sequence(tok.encode("hello") or [0])
    logits = model.forward_training_window(wid)
    print(f"forward_training_window logits shape: {tuple(logits.shape)}")
    _ = model.generate("hello", max_tokens=2, temperature=1.0, top_k=8)
    print("model.generate() OK (readout_window path)")

    # 3a. Simple dynamics (default)
    ids = list(range(WINDOW_SIZE))
    emb = model.embedder(torch.tensor(ids, device=device, dtype=torch.long))
    S = F.normalize(emb, dim=-1).unsqueeze(0)
    S_out, _logs, _ = model.run_window_dynamics(
        S, context_ids=[ids], record_tension_log=False
    )
    print(f"Simple dynamics S_out shape: {tuple(S_out.shape)}")

    # 3b. Vectorized dynamics (swap module, same run_window_dynamics loop)
    vec_dyn = VectorizedWindowDynamics(
        state_dim=STATE_DIM,
        window_size=WINDOW_SIZE,
        num_heads=4,
        rank=64,
        max_steps=MAX_WINDOW_STEPS,
    ).to(device)
    model.dynamics = vec_dyn
    rand_ids = torch.randint(0, VOCAB_SIZE, (WINDOW_SIZE,), device=device)
    S2 = F.normalize(model.embedder(rand_ids), dim=-1).unsqueeze(0)
    ctx = rand_ids.cpu().tolist()
    S_vec, _, _ = model.run_window_dynamics(
        S2.clone(), context_ids=[ctx], record_tension_log=False
    )
    print(f"Vectorized dynamics S_out shape: {tuple(S_vec.shape)}")

    print("[Cursor] Model and cache initialized, TS workflow ready.")


if __name__ == "__main__":
    main()
