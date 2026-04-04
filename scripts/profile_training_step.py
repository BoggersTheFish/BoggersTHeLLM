#!/usr/bin/env python3
"""
Profile a single training step with torch.profiler (CUDA + CPU + memory).

Modes
-----
  sandbox  — sandbox.TorchAttractorLanguageModel.trajectory_contrastive_loss_and_logits
             (+ aux CE + backward + optimizer), matching research training.
  ts-llm   — vendor TorchAttractorLanguageModel.training_step (+ backward + optimizer),
             matching vendor/ts-llm run_attractor_llm train path.

Usage (repo root)
-----------------
  python scripts/profile_training_step.py
  python scripts/profile_training_step.py --mode sandbox --device cuda --batch-size 32 --top 25
  python scripts/profile_training_step.py --mode ts-llm --device cuda --seq-len 8 --state-dim 512
  python scripts/profile_training_step.py --mode sandbox --trace traces/step.json

Throughput
----------
  After profiling, runs ``--throughput-iters`` timed ``train_step()`` calls (wall clock, no profiler).
  Writes ``benchmarks/training_throughput.json`` and prints ``=== Throughput ===`` summary.

View Chrome trace
-----------------
  Open chrome://tracing in Chromium, Load → select traces/step.json
  Or: pip install tensorboard && tensorboard --logdir=prof_logs  (if using tensorboard export)

CUDA timeline (system)
----------------------
  nsys profile -o trace.nsys-rep --trace=cuda,nvtx python scripts/profile_training_step.py --mode sandbox --device cuda

Memory
------
  CUDA: wrap step with torch.cuda.memory_allocated() / max_memory_allocated()
  This script prints profiler memory column when profile_memory=True.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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

try:
    from torch.profiler import tensorboard_trace_handler
except ImportError:  # pragma: no cover
    tensorboard_trace_handler = None  # type: ignore[misc, assignment]


def _build_sandbox_model(
    *,
    device: torch.device,
    vocab_size: int,
    state_dim: int,
    window_size: int,
    max_steps: int,
    vectorized: bool,
):
    import sandbox as sb
    from phase05_config import Phase05Config

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


def _build_ts_llm_model(*, device: torch.device, state_dim: int, seq_len: int):
    from attractor_llm.torch_model import TorchAttractorLanguageModel

    model = TorchAttractorLanguageModel(
        state_dim=state_dim,
        tokenizer=None,
        dynamics_type="multihead",
        num_heads=4,
        rank=min(64, state_dim),
        num_attractor_steps=8,
        num_converge_steps=8,
    ).to(device)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile one BoggersTheLLM training step.")
    ap.add_argument("--mode", choices=["sandbox", "ts-llm"], default="sandbox")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--state-dim", type=int, default=128)
    ap.add_argument("--window-size", type=int, default=8)
    ap.add_argument("--max-window-steps", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=8, help="(ts-llm) sequence length for training_step")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--active", type=int, default=3, help="Profiled steps after warmup")
    ap.add_argument(
        "--throughput-iters",
        type=int,
        default=32,
        help="Timed train_step() iterations for throughput (after profiling).",
    )
    ap.add_argument("--top", type=int, default=25, help="Number of ops to print")
    ap.add_argument("--vectorized", action="store_true", default=True)
    ap.add_argument(
        "--simple-dynamics",
        action="store_true",
        help="(sandbox) Disable VectorizedWindowDynamics.",
    )
    ap.add_argument("--report", type=Path, default=None, help="Write text summary here")
    ap.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="Write Chrome JSON trace (load in chrome://tracing)",
    )
    ap.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=None,
        help="If set, also emit TensorBoard profiler traces under this directory",
    )
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    opt: torch.optim.Optimizer
    train_step: callable

    if args.mode == "sandbox":
        use_vec = bool(args.vectorized) and not bool(args.simple_dynamics)
        vocab_size = 512
        model = _build_sandbox_model(
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

        meta = (
            f"mode=sandbox  vectorized={use_vec}  B={B}  W={W}  D={args.state_dim}  "
            f"outer_steps={args.max_window_steps}"
        )
    else:
        model = _build_ts_llm_model(
            device=device,
            state_dim=args.state_dim,
            seq_len=args.seq_len,
        )
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        L = args.seq_len
        vs = model.embedder.vocab_size
        g = torch.Generator()
        g.manual_seed(0)
        x = torch.randint(0, vs, (L,), device=device, generator=g)
        y = torch.randint(0, vs, (L,), device=device, generator=g)

        def train_step() -> None:
            opt.zero_grad(set_to_none=True)
            loss = model.training_step(x, y)
            loss.backward()
            opt.step()

        meta = f"mode=ts-llm  L={L}  D={args.state_dim}  V={vs}  na={model.num_attractor_steps}  nc={model.num_converge_steps}"

    # Warmup (no profiler)
    for _ in range(max(0, args.warmup)):
        train_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    trace_dir = args.tensorboard_dir
    tb_handler = None
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        if tensorboard_trace_handler is None:
            print(
                "Warning: tensorboard_trace_handler unavailable; "
                "install torch with profiler extras or omit --tensorboard-dir.",
                file=sys.stderr,
            )
        else:
            tb_handler = tensorboard_trace_handler(str(trace_dir))

    prof = profile(
        activities=activities,
        schedule=schedule(wait=0, warmup=0, active=args.active, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=tb_handler,
    )
    with prof:
        for _ in range(args.active):
            train_step()
            prof.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Wall-clock throughput (no profiler overhead): elapsed_time / number_of_steps
    bench_iters = max(1, int(args.throughput_iters))
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_tp0 = time.perf_counter()
    for _ in range(bench_iters):
        train_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_tp = time.perf_counter() - t_tp0

    step_time_ms = (elapsed_tp / bench_iters) * 1000.0
    batches_per_second = bench_iters / elapsed_tp if elapsed_tp > 0 else float("nan")
    if args.mode == "sandbox":
        B = max(2, args.batch_size)
        W = args.window_size
        tokens_per_step = B * W
        state_dim = args.state_dim
        dyn_steps = args.max_window_steps
    else:
        B = 1
        W = args.seq_len
        tokens_per_step = W
        state_dim = args.state_dim
        dyn_steps = args.seq_len
    tokens_per_second = (tokens_per_step * bench_iters) / elapsed_tp if elapsed_tp > 0 else float("nan")

    tp_lines = [
        "",
        "=== Throughput ===",
        f"batch_size: {B}",
        f"window_size: {W}",
        f"state_dim: {state_dim}",
        f"steps: {dyn_steps}",
        "",
        f"step_time_ms: {step_time_ms:.4f}",
        f"batches/sec: {batches_per_second:.4f}",
        f"tokens/sec: {tokens_per_second:.4f}",
        "",
    ]
    print("\n".join(tp_lines), flush=True)

    throughput_payload = {
        "mode": args.mode,
        "device": str(device),
        "batch_size": B,
        "window_size": W,
        "state_dim": state_dim,
        "max_window_steps": int(args.max_window_steps) if args.mode == "sandbox" else None,
        "seq_len": int(args.seq_len) if args.mode == "ts-llm" else None,
        "throughput_benchmark_iters": bench_iters,
        "elapsed_seconds": elapsed_tp,
        "step_time_ms": step_time_ms,
        "batches_per_second": batches_per_second,
        "tokens_per_second": tokens_per_second,
        "tokens_per_step": tokens_per_step,
    }
    bench_path = REPO / "benchmarks" / "training_throughput.json"
    bench_path.parent.mkdir(parents=True, exist_ok=True)
    bench_path.write_text(json.dumps(throughput_payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {bench_path}", flush=True)

    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(args.trace))
        print(f"Wrote Chrome trace: {args.trace}", flush=True)

    key_averages = prof.key_averages()

    # Profiler events only include cuda_time_total when CUDA activity was recorded.
    time_attr = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"

    def _event_us(ev: object, attr: str) -> float:
        return float(getattr(ev, attr, 0) or 0)

    def _sort_key(k: object) -> float:
        return _event_us(k, time_attr)

    sorted_events = sorted(key_averages, key=_sort_key, reverse=True)

    total_cuda = sum(_event_us(k, "cuda_time_total") for k in sorted_events) or 1
    total_cpu = sum(_event_us(k, "cpu_time_total") for k in sorted_events) or 1

    lines: list[str] = []
    lines.append("=== BoggersTheLLM — training step profile ===")
    lines.append(meta)
    lines.append(f"device={device}  profiled_steps={args.active}")
    if device.type == "cuda":
        lines.append(
            f"cuda_memory_peak_mib={torch.cuda.max_memory_allocated() / (1024 * 1024):.2f}"
        )
    lines.append("")
    lines.append(f"Top {args.top} ops by {'CUDA' if device.type == 'cuda' else 'CPU'} self time:")
    lines.append(f"{'% CUDA':>8} {'% CPU':>8} {'CUDA us':>12} {'CPU us':>12}  name")
    for k in sorted_events[: args.top]:
        cu = _event_us(k, "cuda_time_total")
        cp = _event_us(k, "cpu_time_total")
        pct_cu = 100.0 * cu / total_cuda
        pct_cp = 100.0 * cp / total_cpu
        lines.append(f"{pct_cu:8.2f} {pct_cp:8.2f} {cu:12.0f} {cp:12.0f}  {k.key}")

    lines.append("")
    lines.append("--- Notes ---")
    lines.append(
        "Percentages are fractions of summed profiler rows (not wall time). "
        "Use --trace for kernel-level timeline; measured share is authoritative on your GPU."
    )
    if args.mode == "sandbox":
        lines.append(
            "Sandbox: expect run_window_dynamics, embedding, readout (Linear W*D→V), "
            "trajectory_contrastive_loss (cosine), cross_entropy."
        )
    else:
        lines.append(
            "ts-llm: expect cdist/mm for logits, MultiHeadDynamics/converge_fixed loops, "
            "embedding gather for signals, precompute attractors over V rows."
        )
    if device.type == "cpu":
        lines.append("CUDA columns are zero on CPU — use --device cuda for GPU view.")

    report = "\n".join(lines) + "\n"
    print(report, flush=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(f"Wrote {args.report}", flush=True)
    if trace_dir is not None:
        print(f"TensorBoard traces under: {trace_dir}  (tensorboard --logdir={trace_dir})", flush=True)


if __name__ == "__main__":
    main()
