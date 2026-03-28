# woke-baby-llm

A **minimal, single-file** PyTorch experiment: a small-vocabulary “language model” built from **continuous dynamics** instead of attention or transformers. State follows a **trajectory**; meaning is **path-dependent** (order and context matter).

Repository: [github.com/BoggersTheFish/woke-baby-llm](https://github.com/BoggersTheFish/woke-baby-llm)

## What it does

- **Embeddings** feed a **context-conditioned signal** (base direction plus a γ-weighted context vector, then renormalized). Recent token signals can be **blended** (light “multi-agent” mix) via a learnable gate.
- **Fast + slow memory**: fast state is stepped by learned dynamics; slow memory uses **decay + learnable slow_lr** (`slow = (1 − slow_decay)·slow + slow_lr·fast`) with a **norm cap** on slow so it cannot dominate.
- **Tension-adaptive partial convergence**: each token runs up to **`max_convergence_steps`** inner dynamics steps. A scalar **tension** \(T \approx |\Delta E| + \lambda(1-\cos(\text{fast},\text{slow})) + \mu H(\text{logits})\) (energy drift, fast/slow misalignment, prediction entropy) drives **early exit** when \(T\) is below **`tension_tol`**, **extra noise** when tension was high, and a **break** perturbation when \(T\) exceeds **`tension_break_thresh`**. This replaces a fixed step count when the attractor is already stable.
- **Symplectic-style readout**: the combined state uses a **midpoint in fast** between the token’s start and end (`0.5·(fast_start + fast_end)`) with static slow for that sub-step, then `w_fast·mid + w_slow·slow`, before normalization and linear readout.
- **Dynamics** (stable): learned diffusion, **tanh**-bounded nonlinearity, **damping**, **β-scaled** input signal, **tension-scaled** Gaussian noise, then **unit normalization** of the fast state after each step.
- **Decoding**: primary path is a **linear readout** from the normalized combined state. A **`next_token_logits_distance`** helper keeps the distance-to-embedding baseline for experiments.
- **Training**: **sliding-window** sequences (default **6** tokens of context → predict next), corpus loaded from **`data/corpus.txt`** by default (one sentence per line; `#` line comments), duplicated and shuffled per epoch, **cross-entropy with label smoothing**, minus entropy bonus, **light bigram logit bias** on embeddings, **anti-repetition** logit shaping on the training context, and **entropy floor** (nats) when logits are too peaked. Optional **train/val split** reports validation CE each epoch (train CE printed for comparison).
- **Generation**: **temperature** sampling with **tension-adaptive** scaling when last tension exceeds the tolerance, **top-k** truncation, **repetition penalties** on recent token ids, optional **debug** attractor / diversity metrics, and **`compare_prompts()`** for trajectory distance between prompts.

There is **no** attention, no transformer blocks, and no external model—only **`sandbox.py`** plus **`requirements.txt`**.

## Limitations and what needs work

This repo is a **research sandbox**, not a production language model.

- **Data vs vocabulary**: The model is trained on **dozens of short sentences** against a **512-word** vocabulary. That is far too little signal for fluent text; the network tends toward **phrase-level templates** even with tension-aware dynamics and decoding.
- **Coherence**: Do not expect topic continuity, factual answers, or long-range syntax. Improvements need **more and more varied training text**, **longer windows**, **more epochs or capacity**, and/or **different objectives**—not only dynamics tweaks.
- **Throughput**: The bundled script runs **on CPU** by default; **pre-training is slow** at full epoch counts. Reduce `NUM_EPOCHS` or corpus size while iterating.

Contributions are welcome if you want to extend the experiment (data pipeline, evaluation, or architecture).

## Requirements

- Python 3.10+ recommended  
- [PyTorch](https://pytorch.org/) and NumPy (see `requirements.txt`)

On many Linux distributions the system Python is **PEP 668** “externally managed”; use a venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python sandbox.py
```

Default training text is **`data/corpus.txt`** (next to `sandbox.py`). Use your own file:

```bash
python sandbox.py --corpus path/to/sentences.txt
```

Useful flags: `--val-fraction 0.05` (validation cross-entropy after each epoch; `0` disables), `--seed 42`, `--epoch-copies 2` (repeat the sentence list per epoch before shuffling), `--baseline-out path` (Phase 0 snapshot; see `docs/BASELINE.md`). Run `python sandbox.py --help` for details.

On startup the script prints **corpus coverage**: how many lines are long enough after dropping out-of-vocabulary words. The **512-token vocabulary** is built from (1) a few legacy seed sentences, (2) **all unique words in `data/corpus.txt`**, then (3) filling to 512 from a large word blob. Custom `--corpus` files are not added to the vocab at runtime—use words already in the vocab or edit the default corpus / vocab construction.

This runs pre-training (epochs, sliding windows, progress lines), then sample generations, attractor debug, and prompt comparisons. Training is **CPU-heavy**; adjust epochs or corpus size if needed.

## Main knobs (in `sandbox.py`)

| Idea | Where |
|------|--------|
| Window length / epochs / entropy bonus | `WINDOW_SIZE`, `NUM_EPOCHS`, `ENTROPY_WEIGHT`, `ENTROPY_FLOOR`, `DRIFT_MIN` |
| Data CLI | `--corpus`, `--val-fraction`, `--seed`, `--epoch-copies`, `--baseline-out` |
| Training regularization | `LABEL_SMOOTHING`, `BIGRAM_TRAIN_WEIGHT`, `TRAIN_LOGIT_NOISE` |
| Generation sampling | `GEN_TOP_K`, `GEN_REPEAT_LOGIT_PENALTY`, `GEN_NO_REPEAT_LAST_EXTRA`, `GEN_TENSION_TEMP_SCALE`, `generation_temperature` (constructor arg) |
| Slow memory | `slow_decay`; learnable `slow_lr` |
| Tension / adaptive steps | `TENSION_LAMBDA`, `TENSION_MU`, `TENSION_TOL`, `MAX_CONVERGENCE_STEPS`, `TENSION_BREAK_THRESH`, `max_convergence_steps` (constructor) |
| Readout / context | `w_fast`, `w_slow`; learnable `gamma`, `agent_blend_weight` |
| Base inner steps per token | `convergence_steps` |
| Dynamics | `beta`, `noise_scale`, `lambda_decay`, `signal_scale` in `SimpleAttractorDynamics` |

## API sketch

- `TorchAttractorLanguageModel` — `get_signal`, `evolve_token`, `next_token_logits`, `next_token_logits_distance`, `generate`, `encode_prompt`, `reset_readout_trajectory`, `compute_tension` (used inside `evolve_token`)
- `compare_prompts(model, prompt_a, prompt_b)` — distance / cosine between final combined states
- `build_sequence_dataset(tokens, window_size)` — sliding (context, target) pairs

## License

[MIT](LICENSE)
