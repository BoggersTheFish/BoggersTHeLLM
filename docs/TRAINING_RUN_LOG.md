# Training run log

Chronological notes for **substantive** `sandbox.py` runs (real corpora, multi-epoch). Integration smokes on `data/corpus.txt` alone are not logged here.

| Field | Meaning |
|--------|--------|
| **Date** | UTC or local as noted |
| **Hardware** | CPU / GPU, rough throughput |
| **Corpus** | HF source + cache keying (`--hf-max-rows`, `--hf-max-chars`) |
| **Metrics** | End-of-run `train_CE`, `val_CE` from epoch summary; `eval_results.json` if used |

---

## 2026-04-04 — TinyStories CPU, 3 epochs (full transcript example)

**Goal:** Short wall-clock run (~55 min total) on the same **1.5M-char** TinyStories cache as the 10-epoch reference, with **verbatim** console output committed for onboarding and debugging.

**Corpus / hardware:** Same TinyStories cache and batching as the **2026-04-02** 10-epoch entry below: `tinystories_46047f2aaf7b6ad7399c.txt`, **CPU**, ~17 min/epoch, **4998 batches/epoch** (batch 64).

**Dynamics depth:** Printed **`dynamics_steps=16`** and 16-step tension vectors — that matches **`MAX_WINDOW_STEPS = 16`** on git **`d65dd64`**. Current **`main`** defaults **`MAX_WINDOW_STEPS = 32`**; pass **`--num-dynamics-steps 16`** to match this run’s depth.

**Metrics (epoch 3):** `mean_loss` 3.6845, `train_CE` 4.2371, `val_CE` 4.8071, `val_traj_contrast` ~0.0031. **Total wall** ~3300 s for 3 epochs.

**Artifacts (committed):** [`docs/runs/apr2026_3epoch_cpu_example/README.md`](runs/apr2026_3epoch_cpu_example/README.md) — command, tables, and link to **[`EXAMPLE_RUN_OUTPUT.md`](runs/apr2026_3epoch_cpu_example/EXAMPLE_RUN_OUTPUT.md)** (full progress bars, checkpoints, Phase 0 baseline block, sample generations, debug attractor lines, `compare_prompts`).

**Weights (local only):** `checkpoints/meaningful_run/ckpt_step*.pt` — not in git.

---

## 2026-04-02 — TinyStories CPU slice (verified meaningful run)

**Goal:** Real LM signal on consumer CPU without multi-day epochs.

**Corpus:** `--dataset-source tinystories --hf-max-rows 50000 --hf-max-chars 1500000` → cached `data/cache/hf/tinystories_46047f2aaf7b6ad7399c.txt` (~355k BPE tokens after tiktoken; ~1.5M UTF-8 chars materialized).

**Hardware:** CPU, ~3.85–3.9 optimizer steps/s, **~21.5–22 min/epoch**, **~3.9 h** for 10 epochs (4998 batches/epoch, batch 64).

**Model / train:** `window_size=8`, `state_dim=128`, `num_waves=4`, `vectorized_num_heads=4`, `vocab_cap=8192`, `tiktoken`, trajectory + `token_aux_ce=0.2`, `readout_aux_alpha=0.15`, `lr=0.001`, `grad_clip=1.0`, LR decay every 5 epochs ×0.8 from epoch 5.

**Metrics (epoch 10):** `mean_loss` ~3.63, `train_CE` ~3.88, `val_CE` ~4.81 (`val_ppl` ≈ exp(4.81) ≈ 122). `val_traj_contrast` snapshot ~0.096 (full-val mean; not comparable to train trajectory loss scale).

**Qualitative:** Fixed-prompt generations show TinyStories-like story tone, imperfect grammar, and **order-sensitive** `compare_prompts` (nonzero L2, cosine not equal to 1).

**Artifacts (committed):** [`docs/runs/meaningful_apr2026/`](runs/meaningful_apr2026/README.md) — `metrics_meaningful.csv` (per-epoch table), `eval_meaningful.json` (final val + config subset). Field glossary: that folder’s **README**.

**Weights (local only):** `checkpoints/meaningful_run/ckpt_step*.pt` — not in git (`.gitignore`).

**Full command:** see README → [First real training run](../README.md#first-real-training-run-public-corpus--checkpoint--eval-json) → **Option A1 (CPU-sized)**.

---

## How to add an entry

1. Copy the section template above.
2. Keep metrics copy-paste friendly (no need to paste full generation blobs into git).
3. Link checkpoint dir and CSV names if reproducibility matters.
