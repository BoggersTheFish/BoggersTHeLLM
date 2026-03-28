# Phase 0 API Discovery

Verified entrypoints from `vendor/GOAT-TS` and `vendor/TS-Core` as of the pinned submodule SHAs.

## GOAT-TS

Python-only. Import path root: `vendor/GOAT-TS/src/`.

### Node / Edge / Wave / MemoryState models
`src/graph/models.py`
- `Node(node_id, label, node_type, activation, state: MemoryState, ...)` — dataclass (slots)
- `Edge(src_id, dst_id, relation, weight)` — dataclass
- `Wave(wave_id, label, source, tension, ...)` — cognitive episode record
- `MemoryState` — StrEnum: `ACTIVE`, `DORMANT`, `DEEP`

### Wave propagation
`src/graph/wave_propagation.py`
- `run_wave_propagation(nodes, edges, input_text|seed_ids, *, max_hops, decay, threshold, use_torch) -> (nodes, WavePropagationResult)` — full pipeline; GPU via PyTorch when available
- `propagate_wave(node_ids, edges, nodes, seed_ids, ...) -> WavePropagationResult`
- `WavePropagationResult(activations: dict[str,float], iterations, converged, seed_ids)`

### Tension (GOAT-TS constraint-graph sense)
`src/reasoning/tension.py`
- `compute_tension(positions: dict[str, np.ndarray], expected_distances: dict[tuple,float]) -> TensionResult`
- `TensionResult(score: float, high_tension_pairs: list[...])`

### Memory state transitions (maps to Phase 6 ACTIVE → DORMANT → DEEP)
`src/memory_manager.py`
- `memory_tick(nodes, low_activation_ticks, *, decay_rate, active_threshold, dormant_threshold, ticks_to_deep) -> (nodes, ticks)`
- `apply_decay_and_transitions(nodes, decay_rate, ...)` — one tick: decay + state transition
- `promote_to_deep_after_ticks(nodes, low_activation_ticks, ...)` — DORMANT → DEEP after N ticks
- Thresholds: `ACTIVE_THRESHOLD=0.5`, `DORMANT_THRESHOLD=0.1`, `TICKS_TO_DEEP=3`

### Graph engine
`src/graph/graph_engine.py` — higher-level graph CRUD + persistence (optional; not required for Phase 0 smoke)

---

## TS-Core

Rust crate (`vendor/TS-Core/src/rust/`) with **optional** PyO3 bindings (`ts_core_kernel`).
Falls back gracefully to pure-Python equivalent when Rust extension is not built.

**Integration boundary: pure-Python import** — no Rust build required.
Import root: `vendor/TS-Core/src/python/`.

### TSCore (primary class)
`src/python/core.py`  —  `from src.python.core import TSCore`

```python
ts = TSCore(damping=0.35, data_dir=Path("~/.tscore"), on_propagate=callback)
```

Key methods:
- `ts.add_node(node_id, activation, stability)` — register a node in the TS graph
- `ts.add_edge(fr, to, weight)` — add constraint edge
- `ts.propagate_wave(*, quiet) -> (tension: float, icarus_line: str)` — one tick; Rust if built, else Python
- `ts.run_until_stable(max_ticks=64, eps=1e-5) -> int` — iterate until tension delta < eps
- `ts.measure_tension() -> float` — std-dev of activations (scalar)
- `ts.factory_evolve()` — append a new stability node (self-improvement tick)
- Wave 12 OS path: `ts.kernel_wave12 = True` activates 9-phase strongest-node scheduler

### Wave 12 (11-step → 9-phase scheduler)
`_python_wave12_propagate_blob` inside TSCore: 9 phases including strongest-node lock, 3×propagation, icarus wings seal, self-validation, pages-island persist.
**"11-step WaveCycleRunner"** referenced in the plan = `run_until_stable(max_ticks=11)` plus one `factory_evolve()` per stable run.

### Language substrate node registration pattern
```python
# Phase 5 shim pattern — register attractor model as a native TS-Core node
ts.add_node("llm_substrate", activation=0.5, stability=0.5)
ts.add_edge("ts_native", "llm_substrate", weight=1.0)
# Push tension scalar from sandbox into TS graph each step:
ts.graph["nodes"]["llm_substrate"]["activation"] = float(sandbox_tension_val)
tension, _ = ts.propagate_wave(quiet=True)
```

### Rust bindings (optional, Phase 2+)
`src/rust/bindings.rs`, `kernel.rs`, `lib.rs` expose:
- `rust_propagate_wave(graph_json, damping)` via `ts_core_kernel` PyO3 module
- `rust_wave12_propagate(graph_json, damping)` — Wave 12 scheduler

Build: `cd vendor/TS-Core && maturin develop` (requires Rust toolchain + maturin).
Not required for Phases 0–1.

---

## ts-llm

Python package: `vendor/ts-llm/attractor_llm/`

- `tokenizer.py` — `AttractorTokenizer` wrapping tiktoken or pure-Python BPE
- `hierarchy.py` — `HierarchicalAttractorLLM` with explicit fast/slow timescale split
- `torch_core.py` — low-rank diffusion operator, vectorized step functions
- `torch_model.py` — full `TorchAttractorModel` with window dynamics
- `datasets.py` — dataset helpers for real token streams
- `training.py` — training loop helpers

Reuse pattern: **import from submodule** (`sys.path.insert(0, "vendor/ts-llm")`).

---

## Integration map (Phases → APIs)

| Phase | Key API |
|-------|---------|
| 0 smoke | `TSCore.run_until_stable`, `TSCore.measure_tension` |
| 1 vocab | `attractor_llm.tokenizer.AttractorTokenizer` |
| 2 vectorize | `attractor_llm.torch_core` low-rank diffusion |
| 3 cache | internal `fast_state`/`slow_memory` tensors from `sandbox.py` |
| 4 data | `attractor_llm.datasets` helpers |
| 5 shim | `TSCore.add_node`, `TSCore.propagate_wave`, `TSCore.factory_evolve` |
| 6 memory | `memory_manager.memory_tick`, `MemoryState` thresholds |
| 7 serve | `TSCore` as sidecar, `sandbox.py` inference loop |
| 8 eval | `TSCore.run_until_stable(max_ticks=11)` |
