# woke-baby-llm

A **minimal, single-file** PyTorch experiment: a small-vocabulary “language model” built from **continuous dynamics** instead of attention or transformers. State follows a **trajectory**; meaning is **path-dependent** (order and context matter).

Repository: [github.com/BoggersTheFish/woke-baby-llm](https://github.com/BoggersTheFish/woke-baby-llm)

## What it does

- **Embeddings** feed a **context-conditioned signal** (base direction plus a γ-weighted context vector, then renormalized).
- **Fast + slow memory**: fast state is stepped by learned dynamics; slow memory uses **decay + learning rate** (`slow = (1 − slow_decay)·slow + slow_lr·fast`) with an optional **norm cap** on slow so it cannot dominate.
- **Partial convergence** per token (several dynamics steps per symbol), so the system does not fully relax on every token.
- **Dynamics** (stable): learned diffusion, **tanh**-bounded nonlinearity, **damping** on the state, **β-scaled** input signal, small Gaussian noise, then **unit normalization** of the fast state after each step.
- **Decoding**: primary path is a **linear readout** from a **normalized combined state** (`w_fast·fast + w_slow·slow`, defaults include reduced slow weight). A **`next_token_logits_distance`** helper keeps the older distance-to-embedding baseline for experiments.
- **Training**: **sliding-window** sequences (default **6** tokens of context → predict next), **~50** short themed sentences in the corpus (duplicated and shuffled per epoch), **cross-entropy with label smoothing**, minus entropy bonus, **light bigram logit bias** on embeddings, **anti-repetition** logit shaping on the training context, and **entropy floor** (nats) when logits are too peaked.
- **Generation**: **temperature** sampling, **top-k** truncation, **repetition penalties** on recent token ids (including an extra penalty on the immediate last token), optional **debug** attractor / diversity metrics, and **`compare_prompts()`** for trajectory distance between prompts.

There is **no** attention, no transformer blocks, and no external model—only **`sandbox.py`** plus **`requirements.txt`**.

## Limitations and what needs work

This repo is a **research sandbox**, not a production language model.

- **Data vs vocabulary**: The model is trained on **dozens of short sentences** against a **512-word** vocabulary. That is far too little signal for fluent text; the network tends toward **phrase-level templates** (“system / effect / mind / dog …”) even when decoding penalties remove **immediate** duplicate-token loops.
- **Coherence**: Do not expect topic continuity, factual answers, or long-range syntax. Improvements here need **more and more varied training text**, **longer windows**, **more epochs or capacity**, and/or **different objectives**—not only stronger sampling penalties.
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

This runs bundled pre-training (epochs, sliding windows, progress lines), then sample generations, optional attractor debug, and prompt comparisons. Training is **CPU-heavy**; adjust epochs or corpus size if needed.

## Main knobs (in `sandbox.py`)

| Idea | Where |
|------|--------|
| Window length / epochs / entropy bonus | `WINDOW_SIZE`, `NUM_EPOCHS`, `ENTROPY_WEIGHT`, `ENTROPY_FLOOR`, `DRIFT_MIN` |
| Training regularization | `LABEL_SMOOTHING`, `BIGRAM_TRAIN_WEIGHT`, `TRAIN_LOGIT_NOISE` |
| Generation sampling | `GEN_TOP_K`, `GEN_REPEAT_LOGIT_PENALTY`, `GEN_NO_REPEAT_LAST_EXTRA`, `generation_temperature` (constructor arg) |
| Slow memory | `slow_decay`, `slow_lr`; combined readout `w_fast`, `w_slow` |
| Steps per token | `convergence_steps`; `step_token()` for one step |
| Context in the signal | `gamma` (learnable) |
| Dynamics | `beta`, `noise_scale`, `lambda_decay`, `signal_scale` in `SimpleAttractorDynamics` |

## API sketch

- `TorchAttractorLanguageModel` — `get_signal`, `evolve_token`, `next_token_logits`, `next_token_logits_distance`, `generate`, `encode_prompt`, `reset_readout_trajectory`
- `compare_prompts(model, prompt_a, prompt_b)` — distance / cosine between final combined states
- `build_sequence_dataset(tokens, window_size)` — sliding (context, target) pairs

## License

[MIT](LICENSE)
