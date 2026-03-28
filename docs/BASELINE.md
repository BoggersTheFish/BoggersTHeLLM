# Phase 0 — Baseline and success criteria

Phase 0 locks a **reproducible reference** before scaling data, windows, or model size. Use it to answer “did the change help?” without guessing.

## How to record a baseline run

1. From the repo root, with your venv active:

   ```bash
   python sandbox.py --baseline-out docs/BASELINE_LAST_RUN.txt
   ```

   Use the same command later (same defaults, or document any flags you changed).

2. The script prints a **Phase 0 baseline** block at the end (metrics + three fixed generations). `--baseline-out` saves that block to a file for git or notes.

3. Optionally copy the printed block (or `docs/BASELINE_LAST_RUN.txt`) into the **“Recorded baseline”** section below and commit.

**Fixed generation prompts** (defined in `sandbox.py` as `BASELINE_PROMPT_1` … `BASELINE_PROMPT_3`) are always the same so outputs are comparable across runs.

## Metrics to care about

| Metric | Meaning |
|--------|--------|
| **train_CE** (last epoch) | Smoothed cross-entropy on training windows, eval-style (no train noise). Compare across runs. |
| **val_CE** (last epoch) | Same, on held-out lines. Noisy if the val set is tiny; use **trend** after scale-up, not a single number. |
| **mean_loss** (last epoch) | Training objective `CE − ENTROPY_WEIGHT × entropy`; good for optimization, not 1:1 with CE. |

Architecture changes (tension-adaptive dynamics, symplectic readout) will change absolute CE values vs older commits—re-record baseline after major `sandbox.py` updates.

## v1 scale-up success check (agreed criteria)

Treat a change as **successful for v1** when **both** hold:

1. **Calibration:** **val_CE** is **lower than this baseline** (same seed/split settings), or at least not worse while **train_CE** improves — i.e. no clear collapse to memorized noise.
2. **Subjective quality:** On the **three fixed prompts**, text shows **less pointless repetition** (fewer tight loops like the same function word or phrase) than the baseline generations, without becoming random gibberish.

Optional: note wall time and epoch count if you change data size or model size.

---

## Recorded baseline

Fill this in after your official Phase 0 run (or paste from `docs/BASELINE_LAST_RUN.txt`).

| Field | Value |
|-------|--------|
| Date (UTC) | |
| Git commit | |
| Command | `python sandbox.py --baseline-out docs/BASELINE_LAST_RUN.txt` |
| Last epoch train_CE | |
| Last epoch val_CE | |
| Last epoch mean_loss | |
| Notes | |

**Generation snapshot:** attach `docs/BASELINE_LAST_RUN.txt` or paste the three **generation baseline** sections here.

---

## Example (replace with your numbers)

This row is illustrative only; it does not replace running the script yourself.

| Field | Example |
|-------|---------|
| Last epoch train_CE | ~8.x |
| Last epoch val_CE | ~8.x |
| Last epoch mean_loss | ~1.x |

After scaling, you want **train_CE / val_CE** to move in a sensible direction together, and the three prompt outputs to read **less repetitively** than this example snapshot.
