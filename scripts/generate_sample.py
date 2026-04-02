#!/usr/bin/env python3
"""
Load a training checkpoint and generate text using the same path as training:

embed_window → run_window_dynamics → readout_window / effective_temperature
(via TorchAttractorLanguageModel.generate, not readout(fast/slow)).

Checkpoint loading uses sandbox.load_model_from_checkpoint (same as inference_server).

Example:
  python scripts/generate_sample.py --checkpoint checkpoints/fast_run/ckpt_step0041283.pt \\
    --tokenizer tiktoken --prompts "Once upon a time" "The scientist discovered"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_TS = REPO / "vendor" / "ts-llm"
if str(_TS) not in sys.path:
    sys.path.insert(0, str(_TS))

import sandbox as sb


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate text with attractor LM checkpoint.")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tokenizer", choices=("tiktoken", "fallback"), default="tiktoken")
    ap.add_argument("--vocab-cap", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=28)
    ap.add_argument(
        "--prompts",
        nargs="*",
        default=[
            "Once upon a time",
            "The scientist discovered",
            "In a small village",
        ],
    )
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = sb.load_model_from_checkpoint(
        args.checkpoint,
        tokenizer_mode=args.tokenizer,
        vocab_cap=args.vocab_cap,
        device=device,
    )

    for p in args.prompts:
        text = model.generate(
            p,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(f"PROMPT: {p!r}", flush=True)
        print(text, flush=True)
        print("---", flush=True)


if __name__ == "__main__":
    main()
