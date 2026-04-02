# Generation failure analysis (attractor LM)

If `scripts/generate_sample.py` produces repetitive tokens, garbage, or divergence, work through these buckets before any architectural change. **Do not** replace the attractor core with transformers or attention.

## 1. Unstable attractor dynamics

**Signs:** exploding or NaN logits; state norms drifting in `--debug` / phase05 logs; high `convergence_failures` in `metrics_attractor_steps.csv`.

**Checks:**

- Run `pytest tests/test_attractor_stability.py`.
- Profile with `python scripts/profile_training_step.py --device cuda`.
- Temporarily lower `--vectorized-dt`, enable `--vectorized-strong-diffusion`, or tighten `--convergence-epsilon` (fewer steps only if still stable).

## 2. Weak or misaligned readout

**Signs:** flat CE during training; good trajectory loss but poor readout CE; tokenizer vocab mismatch vs checkpoint; custom code calling `readout_window` instead of `readout_window_logits` when `--readout-fusion` is on.

**Checks:**

- Ensure `--token-aux-ce > 0` or `--readout-aux-alpha > 0` during training.
- Match `--tokenizer` and `--vocab-cap` to the checkpoint (see `generate_sample.py` warning).
- Inspect `readout` vs `readout_window` / `readout_window_logits` gradients in a short overfit run on a tiny corpus.

## 3. Insufficient training signal

**Signs:** CE stuck high; val/train trajectory loss barely moves.

**Checks:**

- More data (`--hf-max-rows`), more epochs, learning rate.
- Trajectory batch size ≥ 2; contrastive negatives (`--phase05-num-negatives`).

## 4. Trajectory collapse

**Signs:** all final states similar (PCA variance ~0); margin in trajectory loss saturates; repetitive generation.

**Checks:**

- Run `python analysis/state_clustering.py` and `python analysis/trajectory_visualization.py`.
- Slightly increase `DRIFT_MIN` / entropy floor only if you understand the tradeoff (see `sandbox.py`).

## Minimal fixes (preferred order)

1. Stability: dt, strong diffusion diag, negdef diffusion for **simple** dynamics (`--phase05-enforce-negdef-diffusion`).
2. Training signal: aux CE weights, epochs, data.
3. Inference: temperature, top-k, repeat penalties via `TorchAttractorLanguageModel.generate` (avoid deprecated `generate_with_cache` / `cache.logits`).
4. Only if needed: small readout or coupling tweaks—**not** a new architecture.
