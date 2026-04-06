# BoggersTheLanguageModel — Complete Technical Audit

**Last updated:** 2026-04-06 (synced with `sandbox.py` CLI defaults: `MAX_WINDOW_STEPS`, trajectory training, `readout_window` / `token_aux_ce` / `readout_aux_alpha`).

**Scope:** `TorchAttractorLanguageModel` in [`sandbox.py`](../sandbox.py) (lines ~208–2773 for the core class; additional classes `WaveDynamics`, `SimpleAttractorDynamics` follow).  
**Note:** This is distinct from the smaller `TorchAttractorLanguageModel` in `vendor/ts-llm/attractor_llm/torch_model.py` (distance-to-attractor baseline).

The class body is **large** (~2500+ lines for the main class alone). This document **does not** duplicate every line; it inventories **all** submodules, tensors, and control flow with **file references**.

---

## 1. Full Model Architecture

### 1.1 Class location and extent

| Item | Location |
|------|----------|
| `TorchAttractorLanguageModel` | [`sandbox.py` L208–~2773](../sandbox.py) |
| `WaveDynamics` | [`sandbox.py` L2775+](../sandbox.py) |
| `SimpleAttractorDynamics` | [`sandbox.py` L2823+](../sandbox.py) |
| Positional coupling helpers | [`sandbox.py` L3179+](../sandbox.py) |

### 1.2 Constructor: submodules and I/O shapes

Assume **V** = `vocab_size`, **D** = `state_dim`, **W** = `train_window_size`, **H** = `num_waves`, **d_w** = `wave_dim = state_dim // num_waves` (must divide evenly).

| Submodule | Definition | Typical I/O |
|-----------|------------|-------------|
| `embedder` | `nn.Embedding(V, D)` | ids `(B, W)` or `(W,)` → `(B, W, D)` or `(W, D)` after norm |
| `norm` | `LayerNorm(D, elementwise_affine=False)` | `(…, D)` → `(…, D)` |
| `readout` | `Linear(D, V, bias=False)` | `(B, D)` → `(B, V)` — **legacy** single-vector decode |
| `readout_window` | `Linear(W * D, V, bias=False)` | `(B, W*D)` → `(B, V)` — **primary** window readout (flat path) |
| `readout_fusion` | optional `Linear(D, D, bias=True)` | `(B, W, D)` → `(B, W, D)` when enabled |
| Factored readout | optional `readout_query` `(D,)`, `readout_proj` `Linear(D, V)` | pool `(B, W, D)` → `(B, D)` → `(B, V)` |
| `wave_dynamics` | `ModuleList` of **H** × `WaveDynamics(d_w)` | per-wave `(1, d_w)` + signal → `(1, d_w)` inside `evolve_token` |
| `wave_interaction` | `Linear(H*d_w, H*d_w, bias=True)` | flattened wave stack `(H*d_w,)` → same; mixed back |
| `energy_heads` | H small MLPs: `Linear(d_w,d_w)→Tanh→Linear(d_w,1)` | `(B*W, d_w)` → `(B*W, 1)` per head |
| `phase1_window_C` | `(W, W)` learnable or zeros | Phase-1 global mix after inner step |
| `dynamics` | `None` initially; training may set `VectorizedWindowDynamics` or use `WaveDynamics` path | `evolve_token`: `step(fw, signal)` when `dynamics` has `mhd` or `step` |

**Buffers / learnable scalars (non-exhaustive):**  
`slow_decay`, `slow_lr`, `gamma`, `w_fast`, `w_slow`, `generation_temperature`, `signal_eps`, `wave_interaction_strength`, `window_tension_tol`, `window_tension_high`, `_phase2_C_dist_mask`, `tension_lambda`, `tension_mu`, `tension_tol`, `tension_break_thresh`, `tension_noise_gain`, `temperature_raw`, `position_*`, `interaction_*`, `agent_blend_weight`, etc. — see [`sandbox.py` L235–460](../sandbox.py).

---

## 2. Attractor State

### 2.1 Window tensor (primary training representation)

- **Shape:** **`S ∈ ℝ^{B×W×D}`** (or `(W, D)` unbatched).
- **Wave layout:** Last dimension **D** splits as **H contiguous blocks of length d_w**: reshape `(B, W, H, d_w)`.

### 2.2 Initialization (`embed_window` / `embed_windows_batch`)

1. `emb = norm(embedder(ids))`  
2. Row L2 normalize: `emb / ||emb||_2` per row  

See [`sandbox.py` L2211–2235](../sandbox.py).

### 2.3 Dual fast / slow path (`evolve_token`)

Used alongside **token-level** dynamics (not the main `run_window_dynamics` training loop):

- **`fast_state`, `slow_state`:** each **`(D,)`** — initialized to zeros in `_init_dual_state` ([`sandbox.py` L1225–1230](../sandbox.py)).
- **Slow update:** `slow = (1 - slow_decay) * slow + slow_lr * fast`, then clip norm ([`sandbox.py` L1356–1359](../sandbox.py)).
- **Combined:** `_symplectic_combined(fast, slow)` for readout / tension ([`sandbox.py` L1360–1361](../sandbox.py)).

---

## 3. Wave System

### 3.1 `WaveDynamics` ([`sandbox.py` L2775–2820](../sandbox.py))

- **Residual MLP:** `d_w → (2*d_w hidden, min 16) → GELU → d_w`.
- **`residual_scale`:** learnable; update `x + tanh(residual_scale) * MLP(x)`.
- **Signal:** optional **additive** `x = x + signal` (same shape).
- **Noise:** if `training` and `noise_scale > 0`, Gaussian noise scaled by `noise_scale * noise_scale_mul`.

### 3.2 Per-wave update (`_wave_dynamics_evolve`)

- Input **`fw`, `sigw`:** `(num_waves, wave_dim)`.
- Each channel i: `wave_dynamics[i](fw[i], sigw[i])` → concat → `(num_waves, wave_dim)` ([`sandbox.py` L1232–1245](../sandbox.py)).

### 3.3 Cross-wave mixing (`_apply_wave_cross_interaction`)

- `flat = fw.reshape(-1)`  → shape **`(H * d_w,)`**
- `inter = wave_interaction(flat)` → same shape
- Reshape to `(H, d_w)`; **`fw + alpha * inter`**, `alpha = wave_interaction_strength` ([`sandbox.py` L1247–1262](../sandbox.py)).

---

## 4. Token Embedding & Input

### 4.1 Window path (training / `generate`)

- **Token IDs** → **`embedder`** → **`norm`** → **L2 normalize rows** → **`(B, W, D)`**.

### 4.2 Positional coupling (not sinusoidal PE)

- Uses **learned** `position_gamma_raw` (softplus), `interaction_scale_raw`, `interaction_dt_raw`, `position_asym`.
- **`positional_coupling_delta`:** Laplacian-style **`sum_j w_ij (S_j - S_i)`** with `w_ij ∝ exp(-γ|i-j|)` ([`sandbox.py` L3179–3203](../sandbox.py)).

### 4.3 GOAT (optional)

- If `_goat_mgr` and `context_tensor` set: gather **bonus** per token, broadcast to **`(B, W, D)`** additive signal ([`sandbox.py` L1502–1515](../sandbox.py)).

### 4.4 `get_signal` (token path)

- Normalized embedding + **`gamma` * context vector** + optional GOAT bonus; renormalized ([`sandbox.py` L1188–1211](../sandbox.py)).

---

## 5. State Evolution

### 5.1 `_single_window_step` ([`sandbox.py` L1452–1534](../sandbox.py))

Order of operations:

1. **Positional coupling** on wave-flattened state (`_window_coupling_flat` → delta → `_from_window_coupling_flat`).
2. Optional **GOAT** additive signal.
3. **`_apply_token_anchor_force`** (optional pull toward readout top-k embeddings).
4. **`_window_energy_gradient_step`:** **`S ← S - dt ∇_S E`** (autograd on scalar batch energy).
5. Optional clamp (inference / adaptive dt paths).
6. **`_phase1_global_interaction_step`:** `S + scale * (C^T S)` style mix ([`sandbox.py` L1010–1034](../sandbox.py)).
7. **`_normalize_window_state_if_enabled`:** optional per-wave-sphere normalize.

### 5.2 Energy descent details

- **Scalar energy:** mean over batch of `_window_energy_per_batch_row(S)` ([`sandbox.py` L820–821](../sandbox.py)).
- **`∇_S E`** via `torch.autograd.grad(energy, S_leaf, create_graph=...)` ([`sandbox.py` L860–879](../sandbox.py)).
- **Per-wave L2 clamp** in `_clamp_window_waves_norm` ([`sandbox.py` L621–628](../sandbox.py)).

### 5.3 Truncated BPTT ([`sandbox.py` L1704–1710](../sandbox.py))

- Every **`bptt_chunk_size`** outer steps: **`S = S.detach()`** at chunk boundary.
- **`keep_history`:** true for inner steps except last in chunk so gradients can flow within chunk.

---

## 6. Trajectory / Inference Steps

| Parameter | Role | Default / CLI |
|-----------|------|----------------|
| `max_window_steps` | **Hard cap** on outer `run_window_dynamics` iterations | `MAX_WINDOW_STEPS` (32 in [`sandbox.py` L97–99](../sandbox.py)); CLI `--num-dynamics-steps` ([`sandbox.py` L4199](../sandbox.py)) |
| `convergence_epsilon` | Early exit if **mean batch energy** change \< ε | Model field; CLI `--convergence-epsilon` (typ. 0 = off) |
| `min_attractor_steps` | Minimum outer steps before ε exit | ≥ 2 ([`sandbox.py` L1656–1658](../sandbox.py)) |
| `convergence_steps` / `max_convergence_steps` | **Inner** loop in **`evolve_token`** (token path) | Defaults [`sandbox.py` L213–214, L88–92](../sandbox.py) |

**Early exit ([`sandbox.py` L1994–2000](../sandbox.py)):**  
If `conv_eps > 0` and `step + 1 >= min_steps` and `|E_now - E_prev| < conv_eps` → break, set `_last_convergence_triggered`.

---

## 7. Energy / Attractor Mechanics

### 7.1 Energy

- **Per-wave:** `energy_heads[i](wave_slice)` → scalar per (B, W) after reshape ([`sandbox.py` L594–608](../sandbox.py)).
- **`energy(S)`:** sum over waves → `(B, W, 1)` ([`sandbox.py` L630–642](../sandbox.py)).
- **`_window_energy_per_batch_row`:** adds optional **`λ * tension_window`**, **`anchor_lambda * anchor_distance`** ([`sandbox.py` L752–773](../sandbox.py)).

### 7.2 Update

**Gradient descent:** `S ← S - dt ∇ E` (see §5.1).

### 7.3 Stability

- **L2 clamps** on wave rows and window rows ([`sandbox.py` L775–787, L621–628](../sandbox.py)).
- **Training:** `clip_grad_norm_(model.parameters(), args.grad_clip)` ([`sandbox.py` L5441–5442](../sandbox.py)).
- **Tension-driven breaks:** low-tension jitter, high-tension escape ([`sandbox.py` L1889–1976](../sandbox.py)).

### 7.4 Collapse / oscillation heuristics

- **`diagnostics()`** and `_attractor_diag_flag` / training loop: **COLLAPSED**, **OSCILLATING**, **DRIFT**, **HEALTHY** ([`sandbox.py` L540–592, L5260–5292, L5458–5473](../sandbox.py)).
- **Geometry:** `_last_state_geom_cos_mean`, `_last_state_geom_dist_mean` ([`sandbox.py` L2445–2455](../sandbox.py)).

### 7.5 Multiple basins

- **Structural:** H separate **energy heads** + **H wave channels** + **mixing** — multiple degrees of freedom; **no** formal guarantee of disjoint basins in code.

---

## 8. Contrastive Trajectory Loss

### 8.1 `trajectory_contrastive_loss` ([`sandbox.py` L2140–2169](../sandbox.py))

1. Flatten `state_a`, `state_b` → `(B, W*D)`.
2. **`F.normalize(..., dim=-1)`** (global L2 on flattened vector).
3. **Positive:** `pos = sum(a * b, dim=-1)` (cosine).
4. **Negative:** `b[torch.randperm(B)]` **or** multi-negative random permutations.
5. **`margin = (0.2 - pos + neg) / tau`**, **`tau = trajectory_temperature`** (Phase05).
6. **Loss:** `F.relu(margin).mean() * tau`.

### 8.2 `trajectory_contrastive_loss_and_logits` ([`sandbox.py` L2263–2500](../sandbox.py))

- Runs **`run_window_dynamics`** with **`return_intermediate_states=True`**.
- Stacks states: **`S_pred_aligned = states[:-1]`**, **`S_tgt = states[1:].detach()`** — **consecutive outer-step** pairs on the **same** trajectory.
- Adds optional: intermediate CE, entropy floor, repulsion, Phase1/2 regs, energy reg, anchor contrast, guidance MSE.
- Returns **`(loss_traj, logits)`** with logits from **`_readout_window_logits(S_pred)`**.

---

## 9. Readout / Token Prediction

### 9.1 `_readout_window_logits` ([`sandbox.py` L505–538](../sandbox.py))

- Input **`S`:** `(W, D)`, `(B, W*D)`, or `(B, W, D)` normalized via `_window_batch_full_state`.
- Optional **`readout_fusion`:** `Linear(D,D)` on last dim.
- **Flat:** `flatten` → `readout_window` → `(B, V)`.
- **Factored:** softmax attention over W with `readout_query`, then `readout_proj`.

### 9.2 Temperature

- **`effective_temperature`:** `softplus(temperature_raw).clamp(min=1e-6)` ([`sandbox.py` L1171–1172](../sandbox.py)).

### 9.3 Joint training

- **`trajectory_contrastive_loss_and_logits`** combines trajectory loss + **readout logits**; CE added via **`token_aux_ce`**, **`readout_aux_alpha`**, intermediate CE when weights \> 0 ([`sandbox.py` L5388–5418](../sandbox.py)).

### 9.4 Teacher forcing

- **`forward_training_window`:** fixed context window from **teacher tokens** ([`sandbox.py` L2237–2256](../sandbox.py)).
- Trajectory loss uses **student** dynamics with **aligned consecutive** states (§8.2).

---

## 10. Training Loop

| Item | Value / location |
|------|------------------|
| Optimizer | `Adam`, `weight_decay=1e-5` ([`sandbox.py` L4786](../sandbox.py)) |
| LR | `--lr` default **0.001** ([`sandbox.py` L4268](../sandbox.py)); for CPU TinyStories, **`3e-4`**–**`1e-4`** with **`--token-aux-ce 0.5`** is often better when CE drifts up (see README **A1c**) |
| Grad clip | `--grad_clip` ([`sandbox.py` L5441](../sandbox.py)) |
| Batch | `--trajectory-batch-size` default **64** (`TRAJECTORY_BATCH_SIZE_DEFAULT`, L106) |
| Window | `--window-size` (e.g. 8 in example runs) |
| Unrolling | `max_window_steps` outer iterations; BPTT `bptt_chunk_size` |
| Metrics CSV | `--epoch-metrics-csv`; columns include CE, traj contrast, `mean_final_step_tension` ([`sandbox.py` L5777–5790](../sandbox.py)) |

---

## 11. Dataset

- **CLI:** `--corpus`, HF-style `--dataset-path`, `--tokenizer` `tiktoken` | `fallback`, `--vocab-cap` (default 32768), `--hf-max-chars`, etc. ([`sandbox.py` L4148–4310](../sandbox.py)).
- **Windows:** `build_sequence_dataset` — sliding (context, next_token), skips weak/repetitive spans ([`sandbox.py` L3267–3283](../sandbox.py)).
- **Example eval:** TinyStories, tiktoken GPT-2 BPE, vocab cap 8192 — see [`eval_cpu_meaningful.json`](../eval_cpu_meaningful.json).

---

## 12. Scaling

**Parameters (rough):**

- **Embedding:** `V * D`
- **readout_window:** `V * W * D` (dominant when W large)
- **readout_fusion:** `D^2`
- **wave_interaction:** `(H*d_w)^2 = D^2`
- **energy_heads:** O(H * d_w^2)
- **`max_window_steps`:** multiplies **compute** per forward, not parameter count

**Doubling `wave_dim`:** requires increasing **`state_dim`** (or halving H); increases per-wave MLP and interaction width.

**Doubling `num_waves`:** with fixed **`state_dim`**, **`wave_dim` halves**; more heads, narrower channels.

**Inference:** cost scales **~linearly** in **`max_window_steps`** per generated token (one full window relax per step).

---

## 13. Metrics / Geometry

| Attribute | Meaning |
|-----------|---------|
| `_last_traj_cos_pos` / `_last_traj_cos_neg` / `_last_traj_margin` | From `trajectory_contrastive_loss` ([`sandbox.py` L2165–2168](../sandbox.py)) |
| `_last_state_geom_*` | Pairwise cosine / L2 geometry of **final row** states in batch ([`sandbox.py` L2445–2455](../sandbox.py)) |
| `_last_window_tension_curve` | Mean **T_total** after each outer step ([`sandbox.py` L2110–2133](../sandbox.py)) |
| `mean_final_step_tension` / `mean_final_T` | Epoch mean of **last** tension in curve per batch ([`sandbox.py` L5421–5423, L5589–5590](../sandbox.py)) |

---

## 14. Generation ([`sandbox.py` L2681–2772](../sandbox.py))

1. `reset_readout_trajectory()`
2. Loop **max_tokens**: `window_ids_from_sequence(generated_ids)` → **`forward_training_window`** → **`_sample_next_token_id`**
3. **Sampling:** temperature, **top-k** (`GEN_TOP_K = 28`), repeat penalties on recent tokens
4. **State carry:** **No** persistent relaxed tensor — each step **re-embeds** the last **W** tokens; dynamics run fresh on that window.

---

## 15. Ablations / Special Cases

| Case | Behavior |
|------|----------|
| **`num_waves = 1`** | Single `wave_dim = D`; still valid; tests sometimes use this ([`tests/test_attractor_stability.py`](../tests/test_attractor_stability.py)). |
| **Single intermediate / pairwise traj** | If \< 2 outer states, trajectory core loss is **zero** ([`sandbox.py` L2351–2352](../sandbox.py)). |
| **Automated ablations** | **Not** a full benchmark suite in-repo; behavior is **code-defined**, systematic ablation reports are **not** committed. |

---

## 16. TS / Attractor Manifolds

**Logged (observational):**

- Pairwise **cosine** / **L2** on batch final states
- **Tension** curves and step cosines / delta norms
- **Phase05** batch CSV when `log_metrics` enabled

**Not implemented:** automatic **prompt → cluster** labeling, **semantic similarity** of prompts vs state distance, or **phase transition** detection beyond manual inspection of metrics.

---

## 17. Practical

- **Micro-benchmark:** [`benchmarks/training_throughput.json`](../benchmarks/training_throughput.json) — CPU, ~195 ms/step for listed config, **~2.6k tokens/s** (batch 64, window 8).
- **GPU memory / epoch time:** **not** fixed; depend on `D`, `W`, `V`, `max_window_steps`, batch size.
- **Bottleneck:** typically **outer-step count × energy backward** per batch; Python overhead if GOAT / heavy logging.

---

## 18. Evidence for “Meaningful Basins”

**Mechanisms that encourage structure:**

- Learned **energy heads** + **gradient descent** in state space
- **Trajectory contrastive** alignment of consecutive relaxations
- **Tension** / **break** / **repulsion** / **entropy floor** (when enabled)
- **Anchor** forces and contrastive losses (Phase05 weights)

**Missing for strong claims:**

- Controlled **memorization** vs **generalization** tests
- **Clustering** of final states vs **shuffled labels**
- **Quantitative** prompt–state correspondence across a benchmark
- **Ablations** archived as standard experiment harness

---

*Generated as a repository audit. Update this file when `sandbox.py` architecture or defaults change.*
