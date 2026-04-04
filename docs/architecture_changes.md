# Architecture Changes

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
