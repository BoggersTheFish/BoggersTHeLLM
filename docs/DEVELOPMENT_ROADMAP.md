# Development roadmap — BoggersTheLLM

This roadmap is **practical**: it aligns with the current **multi-wave** stack in `sandbox.py` and separates **measurement**, **performance**, and **model research**. It replaces the older “no testing until batch &lt; 1s” gate with explicit decision points—you should still profile before large spend.

---

## Pillar A — Truthful measurement

**Goal:** Know val CE, tension, and sample quality on a fixed protocol.

- Use **stream split + enough val windows** (`BASELINE.md`, `MIN_VAL_WINDOWS`).
- Log **epoch CSV** + optional **`--phase05-batch-metrics-csv`**.
- For decoding, **`model.generate`** only (training-parity readout path).
- **Checkpoint discipline:** save `use_readout_fusion`, `num_waves`, `use_readout_fusion`, vectorized metadata in `config` (already partially done).

**Exit:** You can compare two runs on the same corpus and say which improved val CE / samples without confounding tokenizer or readout mismatch.

---

## Pillar B — Throughput and cost

**Goal:** Make one optimizer step as cheap as possible on your GPU without changing math silently.

- Profile: `scripts/profile_training_step.py` (adjust `--simple-dynamics` / vectorized as needed).
- Targets are **environment-specific**; record GPU model, batch size, W, D, `max_window_steps` when claiming a number.
- Candidates: fewer `.item()` syncs in hot loops (ongoing), compiled inner dynamics (`dyn._step`), larger batches if memory allows.

**Exit:** Documented “steps/sec” or “batch time” for at least one reference config.

---

## Pillar C — Window attractor (current architecture)

**Goal:** Stable relaxation with clear semantics.

- **Energy:** per-wave `energy_heads`, sum for total; gradients respect wave boundaries.
- **Optional:** `anchor_freeze` to skip energy updates for converged wave slices; tune `anchor_freeze_threshold` with embedding scale.
- **Tension / Phase 2 breaks:** keep behavior documented when changing `W` or `num_waves`.

**Exit:** Ablations show effect of freeze / waves on val CE or training stability.

---

## Pillar D — Token path and multi-wave interaction

**Goal:** `evolve_token` path stays consistent with window training.

- **WaveDynamics** + **wave_interaction** when not using vectorized `mhd`.
- **Vectorized** path: `VectorizedWindowDynamics` operates on **wave_dim** with head count dividing `wave_dim`.

**Exit:** Smoke tests + one trained checkpoint that loads and generates on both `--dynamics simple` and `vectorized`.

---

## Pillar E — Data scale

**Order of scale (recommended):**

1. Synthetic / tiny — CI and smoke only.
2. **TinyStories** — first real trend checking.
3. **FineWeb-Edu sample** or large local text — scale-up.

**Exit:** Same metrics CSV schema so epochs are comparable across scales.

---

## What we are *not* committing to here

- Attention / transformer blocks inside this repo (out of scope for this roadmap).
- Guaranteed SOTA perplexity vs GPT-style LMs (different inductive bias).

---

## Rule of thumb

**Profile → change one knob → re-baseline.** Avoid stacking architecture flags without a control run.
