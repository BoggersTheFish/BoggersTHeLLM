#!/usr/bin/env python3
"""
Record mean window state at each outer attractor step for a few sentences (PCA trajectory).

Outputs:
  trajectory_pca2.npy — shape (num_sentences, num_steps, 2)
  trajectory_plot.png   — if matplotlib available

Example:
  python analysis/trajectory_visualization.py --sentences 4 --max-steps 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_TS = REPO / "vendor" / "ts-llm"
if str(_TS) not in sys.path:
    sys.path.insert(0, str(_TS))

import sandbox as sb
from dynamics_vectorized import VectorizedWindowDynamics  # type: ignore[import]


def _run_and_record_steps(
    model: sb.TorchAttractorLanguageModel,
    ids: list[int],
    device: torch.device,
) -> list[torch.Tensor]:
    """Return list of mean state vectors (D,) after each outer step (hack via re-run with increasing cap)."""
    W = model.train_window_size
    if len(ids) < W:
        ids = ids + [0] * (W - len(ids))
    ids = ids[-W:]
    traj: list[torch.Tensor] = []
    max_steps = model.max_window_steps
    with torch.no_grad():
        ids_t = torch.tensor(ids, device=device, dtype=torch.long)
        emb = model.embedder(ids_t)
        emb = model.norm(emb)
        S0 = F.normalize(emb, dim=-1).unsqueeze(0)
        saved_max = model.max_window_steps
        try:
            for k in range(1, max_steps + 1):
                model.max_window_steps = k
                S = S0.clone()
                S_out, _, _ = model.run_window_dynamics(
                    S, collect_metrics=False, record_tension_log=False, context_ids=[ids]
                )
                traj.append(S_out.mean(dim=(0, 1)).detach().cpu())
        finally:
            model.max_window_steps = saved_max
    return traj


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize attractor trajectories in PCA space.")
    ap.add_argument("--output-dir", type=Path, default=REPO / "analysis_out")
    ap.add_argument("--sentences", type=int, default=5)
    ap.add_argument("--state-dim", type=int, default=128)
    ap.add_argument("--window-size", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    try:
        from sklearn.decomposition import PCA  # type: ignore[import]
    except ImportError as e:
        raise SystemExit("pip install scikit-learn") from e

    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    V = 512
    model = sb.TorchAttractorLanguageModel(
        V,
        state_dim=args.state_dim,
        train_window_size=args.window_size,
        max_window_steps=args.max_steps,
        convergence_epsilon=0.0,
        num_waves=1,
    ).to(device)
    model.dynamics = VectorizedWindowDynamics(
        state_dim=args.state_dim,
        window_size=args.window_size,
        num_heads=4,
        rank=min(64, args.state_dim),
        max_steps=args.max_steps,
    ).to(device)
    model.eval()

    rng = torch.Generator()
    rng.manual_seed(args.seed)
    all_vecs: list[torch.Tensor] = []
    per_sentence: list[list[torch.Tensor]] = []
    for _ in range(args.sentences):
        sent_ids = torch.randint(0, V, (args.window_size,), generator=rng).tolist()
        steps = _run_and_record_steps(model, sent_ids, device)
        per_sentence.append(steps)
        all_vecs.extend(steps)

    X = torch.stack(all_vecs, dim=0).numpy()
    pca = PCA(n_components=2)
    pca.fit(X)

    curves = []
    for steps in per_sentence:
        Z = pca.transform(torch.stack(steps, dim=0).numpy())
        curves.append(Z)
    arr = np.stack(curves, axis=0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "trajectory_pca2.npy", arr)
    print(f"Wrote {args.output_dir / 'trajectory_pca2.npy'} shape={arr.shape}", flush=True)

    try:
        import matplotlib.pyplot as plt  # type: ignore[import]

        plt.figure(figsize=(6, 5))
        for i, Z in enumerate(curves):
            plt.plot(Z[:, 0], Z[:, 1], "-o", ms=3, alpha=0.7, label=f"s{i}")
        plt.title("Window mean state trajectory (PCA-2)")
        plt.legend(fontsize=7, loc="best")
        plt.tight_layout()
        plt.savefig(args.output_dir / "trajectory_plot.png", dpi=150)
        plt.close()
        print(f"Wrote {args.output_dir / 'trajectory_plot.png'}", flush=True)
    except ImportError:
        print("matplotlib not installed; skip plot", flush=True)


if __name__ == "__main__":
    main()
