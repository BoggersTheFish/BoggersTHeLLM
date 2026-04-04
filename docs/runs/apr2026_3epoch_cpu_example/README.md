# Example: 3-epoch TinyStories CPU run (2026-04-04)

This folder documents a **complete, copy-pasted console run** of `sandbox.py` on **CPU**: TinyStories with the same corpus cache as the longer [meaningful_apr2026](../meaningful_apr2026/README.md) reference, but only **3 epochs** and **no** `--epoch-metrics-csv` / `--eval-results-json` (metrics still print at end of each epoch).

Use it to sanity-check **wall time**, **loss curves**, **Phase 0 baseline block**, **sample generations**, and **debug dynamics** lines before committing to a 10-epoch job.

---

## Command (as run)

```bash
python3 sandbox.py \
  --dataset-source tinystories \
  --hf-max-rows 50000 \
  --hf-max-chars 1500000 \
  --tokenizer tiktoken \
  --vocab-cap 8192 \
  --val-fraction 0.1 \
  --window-size 8 \
  --state-dim 128 \
  --num-waves 4 \
  --vectorized-num-heads 4 \
  --batch-size 64 \
  --max-epochs 3 \
  --lr 0.001 \
  --grad-clip 1.0 \
  --lr-decay-every 5 \
  --lr-gamma 0.8 \
  --checkpoint-dir checkpoints/meaningful_run \
  --save-every 500
```

**Git revision at run time (from printed baseline):** `d65dd64`.

**Corpus cache:** `data/cache/hf/tinystories_46047f2aaf7b6ad7399c.txt` (same key as the 10-epoch committed run).

---

## Dynamics depth note

The printed **Phase 0** block for this run shows **`num_dynamics_steps: 16`** (and tension vectors of length **16**). That matches the **previous default** `MAX_WINDOW_STEPS = 16` on that revision.

The repository default is now **`MAX_WINDOW_STEPS = 32`** (CLI: `--num-dynamics-steps` / `--max-window-steps`). To **reproduce this run’s depth** on current `main`, pass **`--num-dynamics-steps 16`**. To use the new default, omit the flag (expect somewhat slower steps and different metrics).

---

## Scaled data (from log)

| Quantity | Value |
|----------|--------|
| `total_tokens` | 355,425 |
| `train_tokens` | 319,874 |
| `val_tokens` | 35,543 |
| `train_windows` | 319,866 |
| `val_windows` | 35,535 |
| Batches per epoch | 4,998 |
| Device | CPU |
| Vocab | 8192 (tiktoken) |

---

## Epoch summary (from end-of-epoch lines)

| Epoch | Wall (s) | mean_loss | train_CE | val_CE | val_traj |
|-------|----------|-----------|----------|--------|----------|
| 1 | 1033.9 | 4.0222 | 5.1614 | 4.9507 | 0.003135 |
| 2 | 1030.3 | 3.7331 | 4.3312 | 4.8564 | 0.003094 |
| 3 | 1028.5 | 3.6845 | 4.2371 | 4.8071 | 0.003100 |

**Total training wall time:** ~3300 s (~55 min) for 3 epochs.

The **10-epoch** reference in [`docs/TRAINING_RUN_LOG.md`](../../TRAINING_RUN_LOG.md) lands near **`val_CE` ~4.81**; this 3-epoch stop is **not** fully converged but shows the same downward trend.

---

## Qualitative output (abbreviated)

- **Generations:** Story-like but unpolished TinyStories tone; order-sensitive `compare_prompts` (L2 distance in window space nonzero, cosine away from 1).
- **Debug dynamics:** `mean_cos(step)≈0.9997`, tension `T` decreasing along the 16-step curve, `energy_wave_*` logged per wave.

---

## Full console transcript

The **verbatim** terminal output (progress bars, batch prints, baseline block, prompts) is in:

- **[`EXAMPLE_RUN_OUTPUT.md`](EXAMPLE_RUN_OUTPUT.md)**

---

## Related

| Resource | Role |
|----------|------|
| [`../meaningful_apr2026/README.md`](../meaningful_apr2026/README.md) | Committed **CSV + eval JSON** for the **10-epoch** run |
| [`../../TRAINING_RUN_LOG.md`](../../TRAINING_RUN_LOG.md) | Chronological run log |
| [Root README](../../../README.md) | Install + **Option A1** full command with metrics files |

Weights are **local only** under `--checkpoint-dir` (not in git; see `.gitignore`).
