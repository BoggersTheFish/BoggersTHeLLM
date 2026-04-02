#!/usr/bin/env python3
"""
Collect final window states from many random contexts and run PCA (2D) for manifold check.

Optional: UMAP if umap-learn is installed (--umap).

Example:
  python analysis/state_clustering.py --output-dir analysis_out --samples 256
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


def main() -> None:
    ap = argparse.ArgumentParser(description="PCA/UMAP on final attractor states.")
    ap.add_argument("--output-dir", type=Path, default=REPO / "analysis_out")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--state-dim", type=int, default=128)
    ap.add_argument("--window-size", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--umap", action="store_true", help="Also run UMAP (requires umap-learn)")
    args = ap.parse_args()

    try:
        from sklearn.decomposition import PCA  # type: ignore[import]
    except ImportError as e:
        raise SystemExit("Install scikit-learn: pip install scikit-learn") from e

    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    V = 1024
    W = args.window_size
    D = args.state_dim
    model = sb.TorchAttractorLanguageModel(
        V,
        state_dim=D,
        train_window_size=W,
        max_window_steps=args.max_steps,
        convergence_epsilon=0.0,
        num_waves=1,
    ).to(device)
    model.dynamics = VectorizedWindowDynamics(
        state_dim=D,
        window_size=W,
        num_heads=4,
        rank=min(64, D),
        max_steps=args.max_steps,
    ).to(device)
    model.eval()

    finals: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(args.samples):
            ids = torch.randint(0, V, (W,), device=device)
            emb = model.embedder(ids)
            emb = model.norm(emb)
            S = F.normalize(emb, dim=-1).unsqueeze(0)
            S_out, _, _ = model.run_window_dynamics(
                S, collect_metrics=False, record_tension_log=False
            )
            finals.append(S_out.reshape(-1).cpu())

    X = torch.stack(finals, dim=0).numpy()
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "state_finals_pca2.npy", Z)
    np.save(args.output_dir / "state_finals_raw.npy", X)
    print(f"PCA explained_variance_ratio_={pca.explained_variance_ratio_}", flush=True)
    print(f"Wrote {args.output_dir / 'state_finals_pca2.npy'}", flush=True)

    if args.umap:
        try:
            import umap  # type: ignore[import]

            reducer = umap.UMAP(n_components=2, random_state=args.seed)
            U = reducer.fit_transform(X)
            np.save(args.output_dir / "state_finals_umap2.npy", U)
            print(f"Wrote {args.output_dir / 'state_finals_umap2.npy'}", flush=True)
        except ImportError:
            print("umap-learn not installed; skip UMAP", flush=True)

    try:
        import matplotlib.pyplot as plt  # type: ignore[import]

        plt.figure(figsize=(6, 5))
        plt.scatter(Z[:, 0], Z[:, 1], s=8, alpha=0.6)
        plt.title("Final states (PCA-2)")
        plt.tight_layout()
        plt.savefig(args.output_dir / "state_clustering_pca.png", dpi=150)
        plt.close()
        print(f"Wrote {args.output_dir / 'state_clustering_pca.png'}", flush=True)
    except ImportError:
        print("matplotlib not installed; skip PNG", flush=True)


if __name__ == "__main__":
    main()
