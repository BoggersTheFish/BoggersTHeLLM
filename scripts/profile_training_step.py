#!/usr/bin/env python3
"""
Profile one training step (trajectory contrastive + aux CE path) with torch.profiler.

Focus regions (see table in printed report):
  run_window_dynamics, MultiHeadDynamics.drift, embed path,
  trajectory_contrastive_loss, auxiliary CE.

Usage (from repo root):
  python scripts/profile_training_step.py
  python scripts/profile_training_step.py --device cuda --batch-size 32 --report reports/profile_summary.txt

Warmup runs reduce compile/profiler overhead; CUDA activity requires GPU.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_TS = REPO / "vendor" / "ts-llm"
if str(_TS) not in sys.path:
    sys.path.insert(0, str(_TS))

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, schedule

import sandbox as sb
from phase05_config import Phase05Config


def _build_model(
    *,
    device: torch.device,
    vocab_size: int,
    state_dim: int,
    window_size: int,
    max_steps: int,
    vectorized: bool,
) -> sb.TorchAttractorLanguageModel:
    p05 = Phase05Config(log_metrics=False)
    model = sb.TorchAttractorLanguageModel(
        vocab_size,
        state_dim=state_dim,
        train_window_size=window_size,
        max_window_steps=max_steps,
        convergence_epsilon=0.0,
        num_waves=1,
        phase05=p05,
    )
    model = model.to(device)
    if vectorized:
        from dynamics_vectorized import VectorizedWindowDynamics  # type: ignore[import]

        if state_dim % 4 != 0:
            raise SystemExit("state_dim must be divisible by 4 for default vectorized heads")
        model.dynamics = VectorizedWindowDynamics(
            state_dim=state_dim,
            window_size=window_size,
            num_heads=4,
            rank=min(64, state_dim),
            max_steps=max_steps,
            dt=0.09,
            coupling=0.01,
        ).to(device)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile one attractor LM training step.")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--state-dim", type=int, default=128)
    ap.add_argument("--window-size", type=int, default=8)
    ap.add_argument("--max-window-steps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--active", type=int, default=3, help="Profiled steps after warmup")
    ap.add_argument("--vectorized", action="store_true", default=True)
    ap.add_argument(
        "--simple-dynamics",
        action="store_true",
        help="Disable VectorizedWindowDynamics; use per-wave WaveDynamics (token evolve path).",
    )
    ap.add_argument("--report", type=Path, default=None, help="Write text summary here")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    use_vec = bool(args.vectorized) and not bool(args.simple_dynamics)
    vocab_size = 512
    model = _build_model(
        device=device,
        vocab_size=vocab_size,
        state_dim=args.state_dim,
        window_size=args.window_size,
        max_steps=args.max_window_steps,
        vectorized=use_vec,
    )
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    B = max(2, args.batch_size)
    W = args.window_size
    g = torch.Generator()
    g.manual_seed(0)
    ctx_t = torch.randint(0, vocab_size, (B, W), generator=g)
    tgt_t = torch.randint(0, vocab_size, (B,), generator=g)
    contexts = ctx_t.tolist()
    targets = tgt_t.tolist()
    targets_t = torch.tensor(targets, device=device, dtype=torch.long)

    def train_step() -> None:
        opt.zero_grad(set_to_none=True)
        loss_traj, logits = model.trajectory_contrastive_loss_and_logits(contexts, targets)
        aux = F.cross_entropy(logits, targets_t)
        loss = loss_traj + 0.1 * aux
        loss.backward()
        opt.step()

    # Warmup (no profiler)
    for _ in range(max(0, args.warmup)):
        train_step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    prof = profile(
        activities=activities,
        schedule=schedule(wait=0, warmup=0, active=args.active, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )
    with prof:
        for _ in range(args.active):
            train_step()
            prof.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    key_averages = prof.key_averages()
    # Sort by CUDA time when available, else CPU
    def _sort_key(k):
        if device.type == "cuda" and k.cuda_time_total > 0:
            return k.cuda_time_total
        return k.cpu_time_total

    sorted_events = sorted(key_averages, key=_sort_key, reverse=True)

    lines: list[str] = []
    lines.append("=== BoggersTheLLM — training step profile ===")
    lines.append(f"device={device}  vectorized={use_vec}  B={B}  W={W}  D={args.state_dim}  outer_steps={args.max_window_steps}")
    lines.append("")
    lines.append("Top kernel / op hotspots (self time):")
    lines.append(f"{'% CUDA':>8} {'% CPU':>8} {'CUDA us':>12} {'CPU us':>12}  name")
    total_cuda = sum(k.cuda_time_total for k in sorted_events) or 1
    total_cpu = sum(k.cpu_time_total for k in sorted_events) or 1
    for k in sorted_events[:40]:
        pct_cu = 100.0 * k.cuda_time_total / total_cuda
        pct_cp = 100.0 * k.cpu_time_total / total_cpu
        name = k.key
        lines.append(
            f"{pct_cu:8.2f} {pct_cp:8.2f} {k.cuda_time_total:12.0f} {k.cpu_time_total:12.0f}  {name}"
        )

    lines.append("")
    lines.append("--- Interpretation (typical bottlenecks) ---")
    lines.append(
        "1. run_window_dynamics dominates when outer_steps×inner drift is large; "
        "look for aten::mm, einsum, addmm tied to MultiHeadDynamics.drift."
    )
    lines.append(
        "2. Embedding/indexing shows up as embedding_dense_backward / index_select; "
        "batching already reduces Python overhead vs per-token loops."
    )
    lines.append(
        "3. trajectory_contrastive_loss appears as cosine_similarity, add, mul on (B,W,D) tensors."
    )
    lines.append(
        "4. cross_entropy + softmax on vocab readout scales with V; aux CE adds a second CE on logits."
    )
    lines.append(
        "5. If %CPU is high relative to CUDA, check host-side loops, .item() syncs, or small batch."
    )
    if device.type == "cpu":
        lines.append("Note: CUDA columns are zero on CPU — use --device cuda for GPU utilization view.")

    report = "\n".join(lines) + "\n"
    print(report, flush=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(f"Wrote {args.report}", flush=True)


if __name__ == "__main__":
    main()
