# Architecture Changes

## 2026-04 — Training observability and reproducibility

**Checkpoints:** Newer **`torch.save`** payloads include **`training_config`** (CLI hyperparameters). **`load_model_from_checkpoint`** reads it when present and warns on gaps.

**Training eval:** Each epoch, **`EVAL_PROMPTS`** in **`evaluation/prompts.py`** drives **`model.generate`** samples; outputs go to **`logs/eval_epoch_{N}.txt`**.

**Profiling:** **`scripts/profile_training_step.py`** appends wall-clock throughput and writes **`benchmarks/training_throughput.json`**.

**Decoding:** **`state_cache.generate_with_cache`** delegates to **`model.generate`**; legacy **`cache.logits()`** remains non-parity.

---

## 2026-04-04 — Attractor Relaxation Horizon

**Component:** Window attractor dynamics

**Change:**

```text
MAX_WINDOW_STEPS
16 → 32
```

**Location:** `sandbox.py`

**Reason:**

Training diagnostics showed the relaxation process stopping before fully settling into attractor minima.

Increasing the iteration budget allows deeper convergence within the window-state dynamics.

**Impact:**

- Deeper attractor basins
- Slightly longer training step
- Improved trajectory stability

**See also:** [CHANGELOG.md](../CHANGELOG.md), [README.md](../README.md) (*Attractor Dynamics Depth Update*), [BASELINE.md](BASELINE.md).
