#!/usr/bin/env python3
"""
Plot batch metrics CSV from sandbox.py (--phase05-batch-metrics-csv).

Includes Phase 0.5 core columns and, when present in the file, Phase 1–2 extras
(phase1_interaction_rms, phase2_break_*, etc.).

  python3 scripts/plot_phase05_metrics.py metrics/phase05_batches.csv --out plots/

Requires matplotlib.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _read_floats(path: Path) -> dict[str, list[float]]:
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols: dict[str, list[float]] = {}
        for row in r:
            for k, v in row.items():
                if k is None:
                    continue
                cols.setdefault(k, [])
                try:
                    cols[k].append(float(v) if v != "" else float("nan"))
                except ValueError:
                    cols[k].append(float("nan"))
    return cols


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Phase 0.5 batch metrics CSV.")
    ap.add_argument("csv_path", type=Path)
    ap.add_argument("--out", type=Path, default=Path("phase05_plots"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    d = _read_floats(args.csv_path)
    if not d:
        raise SystemExit("empty or missing CSV")

    step = d.get("global_step", list(range(len(next(iter(d.values()))))))

    def _plot(ykey: str, title: str, fname: str, ylabel: str | None = None):
        if ykey not in d:
            return
        y = d[ykey]
        if len(y) != len(step):
            return
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(step, y, lw=0.8, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("global_step")
        ax.set_ylabel(ylabel or ykey)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out / fname, dpi=150)
        plt.close(fig)

    _plot("outer_mean_T_total", "Window tension (mean over outer steps)", "tension_outer_mean.png")
    _plot("student_T_total", "Student window T_total (final state)", "tension_student_final.png")
    _plot("student_T_energy", "T_energy (final student state)", "tension_energy.png")
    _plot("student_T_align", "T_alignment (final student state)", "tension_align.png")
    _plot("student_T_entropy", "T_entropy (final student state)", "tension_entropy.png")
    _plot("cos_pos", "Mean cosine(pred, teacher)", "cosine_pos.png")
    _plot("cos_neg", "Mean cosine(pred, negatives)", "cosine_neg.png")
    _plot("margin", "Margin cos_pos - cos_neg", "margin.png")
    _plot("state_norm_mean", "Mean ||state|| (window dynamics)", "state_norm_mean.png")
    _plot("phase1_interaction_rms", "Phase 1: window C interaction RMS", "phase1_interaction_rms.png")
    _plot(
        "phase2_break_direction_norm_mean",
        "Phase 2: mean ||break direction|| (window breaks)",
        "phase2_break_direction_norm.png",
    )
    _plot(
        "phase2_break_applied_alpha_mean",
        "Phase 2: mean break step α",
        "phase2_break_alpha.png",
    )
    _plot(
        "phase2_break_delta_tension_mean",
        "Phase 2: mean ΔT across breaks",
        "phase2_break_delta_tension.png",
    )
    _plot(
        "phase2_break_delta_alignment_mean",
        "Phase 2: mean Δcos (alignment) across breaks",
        "phase2_break_delta_alignment.png",
    )
    _plot(
        "phase2_head_weight_entropy",
        "Phase 2: head weight entropy (if tension weighting)",
        "phase2_head_weight_entropy.png",
    )
    _plot(
        "phase2_interaction_reg_loss",
        "Phase 2: ‖C−I‖² (logged scalar when reg enabled)",
        "phase2_interaction_reg.png",
    )
    print(f"Wrote figures under {args.out.resolve()}")


if __name__ == "__main__":
    main()
