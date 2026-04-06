# BoggersTheLLM documentation index

Start with the [README](../README.md) in the repo root for install, training, and CLI.

| Document | Purpose |
|----------|---------|
| [BOGGERS_THE_LANGUAGE_MODEL_AUDIT.md](BOGGERS_THE_LANGUAGE_MODEL_AUDIT.md) | **Full technical audit:** `TorchAttractorLanguageModel` submodules, shapes, dynamics, losses, training loop, metrics |
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | **Current state:** what is implemented, gaps, recommended next steps |
| [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md) | Phased engineering and research priorities (updated for multi-wave stack) |
| [API_DISCOVERY.md](API_DISCOVERY.md) | Vendored TS-OS repos + `sandbox.py` / model entrypoints |
| [BASELINE.md](BASELINE.md) | How to record Phase 0 baselines and which metrics to watch |
| [architecture_changes.md](architecture_changes.md) | Dated architecture / default changes (e.g. relaxation horizon 2026-04) |
| [FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md) | Symptoms and checks when training looks flat or wrong |
| [MILESTONE_TRAINING.md](MILESTONE_TRAINING.md) | Milestone-oriented training notes |
| [TRAINING_RUN_LOG.md](TRAINING_RUN_LOG.md) | Log of verified multi-epoch runs (metrics, corpus caps, wall time) |
| [runs/meaningful_apr2026/README.md](runs/meaningful_apr2026/README.md) | **Committed** epoch CSV + eval JSON from the Apr 2026 **10-epoch** CPU TinyStories run (field meanings) |
| [runs/apr2026_3epoch_cpu_example/README.md](runs/apr2026_3epoch_cpu_example/README.md) | **Example:** full **3-epoch** CPU transcript + metrics (`EXAMPLE_RUN_OUTPUT.md`) |
| [`../evaluation/prompts.py`](../evaluation/prompts.py) | **`EVAL_PROMPTS`** — fixed strings for end-of-epoch generation during `sandbox.py` training (`logs/eval_epoch_*.txt`) |
| [`../benchmarks/training_throughput.json`](../benchmarks/training_throughput.json) | **Output** of `scripts/profile_training_step.py` (throughput JSON; overwritten each run) |
| [`../scripts/profile_training_step.py`](../scripts/profile_training_step.py) | Profiler + wall-clock throughput (`--throughput-iters`, writes JSON above) |

Submodule docs live under `vendor/*` (GOAT-TS, TS-Core, ts-llm) and are not duplicated here.
