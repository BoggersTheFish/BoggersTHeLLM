# Meaningful CPU run artifacts (Apr 2026)

These files are a **frozen snapshot** from one completed `sandbox.py` training job: TinyStories with `--hf-max-chars 1500000`, 10 epochs, trajectory loss, CPU. They are useful as a **reference curve** and **eval baseline**; numbers will differ slightly if you re-run with the same flags (shuffle, hardware, PyTorch version).

| File | Role |
|------|------|
| [`metrics_meaningful.csv`](metrics_meaningful.csv) | One row per **epoch**: training objective, CE, trajectory contrast, LR, step count. |
| [`eval_meaningful.json`](eval_meaningful.json) | **Post-train** validation summary + subset of CLI config + checkpoint path used for eval. |

**Corpus cache key:** `tinystories_46047f2aaf7b6ad7399c.txt` under `data/cache/hf/` (content hash + caps; rebuilds if HF options change). **Weights** for this run live under `checkpoints/meaningful_run/` locally and are **not** in git (see `.gitignore`).

---

## `metrics_meaningful.csv` columns

| Column | Meaning |
|--------|--------|
| `epoch` | 1-based epoch index. |
| `loss_mode` | Training objective family (`trajectory` here). |
| `mean_loss` | Epoch mean of the **scalar minimized by backward** (trajectory + weighted aux terms), not raw CE alone. |
| `train_ce` | Mean cross-entropy on the **training** split (readout / token prediction), epoch aggregate. |
| `val_ce` | Mean cross-entropy on the **validation** split (same token-level val as training). |
| `train_traj_contrast` | Training-side trajectory / contrastive term (scale is **not** comparable to `val_traj_contrast`). |
| `val_traj_contrast` | Validation-side trajectory contrast metric (full-val mean; typically much smaller than train). |
| `mean_final_step_tension` | If logged: dynamics tension at last inner step (often empty depending on build flags). |
| `max_batch_loss` | Largest batch **objective** in the epoch (outlier batches). |
| `lr` | Optimizer learning rate after any epoch scheduling. |
| `global_step` | Total optimizer steps completed end of epoch (`batches_per_epoch × epoch`). |
| `tscore_evolves` | TS-Core evolution counter when that path is active (0 here). |
| `tscore_last_tension` | Last reported TS-Core tension when used (0 here). |

---

## `eval_meaningful.json` fields

| Path | Meaning |
|------|--------|
| `base.val_ce` | Final validation cross-entropy (same definition as epoch table `val_ce` for last epoch; may be computed in eval pass). |
| `base.val_ppl` | `exp(val_ce)` (perplexity). |
| `base.val_traj_contrast` | Validation trajectory contrast at eval time. |
| `base.val_windows` | Number of validation **windows** (not tokens). |
| `base.val_metrics_reliable` | Whether val metrics are considered valid (e.g. enough windows). |
| `checkpoint` | Relative path to the **checkpoint file** used when writing this eval (final step save). |
| `config.*` | Subset of run configuration (dataset, tokenizer, caps, loss mode, seed). Paths are **repo-relative** in the committed copy. |

To reproduce: see root [README.md](../../../README.md) → **First real training run** → **Option A1 (CPU-sized)**.
