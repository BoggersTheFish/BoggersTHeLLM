"""
Wave C — Rolling state cache (legacy inference helper).

.. deprecated::
    For **next-token generation**, use :meth:`sandbox.TorchAttractorLanguageModel.generate`
    instead of :func:`generate_with_cache`. Training aligns logits with
    ``readout_window(flatten(S)) / effective_temperature()`` via
    :meth:`~sandbox.TorchAttractorLanguageModel.forward_training_window`;
    :meth:`AttractorStateCache.logits` uses ``readout(combined)`` **without**
    ``readout_window`` or ``effective_temperature``, which skews sampling vs training.

``step()`` still runs the same ``run_window_dynamics`` embedding geometry as training
and remains useful for experiments. ``generate_with_cache`` is a deprecated shim that
calls :meth:`sandbox.TorchAttractorLanguageModel.generate`; avoid ``logits()`` for decoding.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    import sandbox as sb  # type: ignore[import]

_LOGITS_MISMATCH_WARNED = False


@dataclass
class AttractorStateCache:
    """
    Mutable inference-state container tied to one TorchAttractorLanguageModel instance.

    .. deprecated::
        Prefer :meth:`sandbox.TorchAttractorLanguageModel.generate` for text generation
        (training-parity readout). This cache's :meth:`logits` path is not equivalent
        to the training readout.

    Attributes
    ----------
    fast_state : Tensor (D,) — last post-dynamics row of the rolling window (unit-norm).
    slow_memory : Tensor (D,) — slow decaying memory.
    phrase_table : list of (token_ids, state_snapshot) pairs — rolling window history.
    token_history : flat list[int] of all stepped token ids (including warmup).
    """
    model: "sb.TorchAttractorLanguageModel"
    fast_state: torch.Tensor = field(init=False)
    slow_memory: torch.Tensor = field(init=False)
    phrase_table: list = field(default_factory=list)
    token_history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        device = self.model.embedder.weight.device
        dtype = self.model.embedder.weight.dtype
        D = self.model.state_dim
        self.fast_state = torch.zeros(D, device=device, dtype=dtype)
        self.slow_memory = torch.zeros(D, device=device, dtype=dtype)

    def reset(self) -> None:
        """Clear all state (new conversation / generation)."""
        device = self.model.embedder.weight.device
        dtype = self.model.embedder.weight.dtype
        D = self.model.state_dim
        self.fast_state = torch.zeros(D, device=device, dtype=dtype)
        self.slow_memory = torch.zeros(D, device=device, dtype=dtype)
        self.phrase_table.clear()
        self.token_history.clear()

    def step(self, token_id: int) -> torch.Tensor:
        """
        Evolve state with one new token via the same window pipeline as training
        (``run_window_dynamics`` on ``(1, W, D)``), then EMA slow memory.

        Returns the current ``fast_state`` ``(D,)`` after the update.
        """
        model = self.model
        device = self.fast_state.device
        dtype = self.fast_state.dtype

        seq = self.token_history + [token_id]
        ids = model.window_ids_from_sequence(seq)

        with torch.no_grad():
            ids_t = torch.tensor(ids, device=device, dtype=torch.long)
            emb = model.embedder(ids_t)
            emb = model.norm(emb)
            S = F.normalize(emb, dim=-1).unsqueeze(0)
            S_out, _, _ = model.run_window_dynamics(
                S,
                collect_metrics=False,
                record_tension_log=False,
                context_ids=[ids],
            )
            if S_out.dim() == 2:
                S_out = S_out.unsqueeze(0)
            new_fast = S_out[0, -1, :].clone()
            new_fast = F.normalize(new_fast, dim=-1)

        slow_lr = float(model.slow_lr.detach())
        slow_dec = float(model.slow_decay.detach())
        new_slow = (1.0 - slow_dec) * self.slow_memory + slow_lr * new_fast
        slow_n = torch.linalg.vector_norm(new_slow)
        max_slow = 3.0
        if slow_n > max_slow:
            new_slow = new_slow * (max_slow / slow_n)

        self.fast_state = new_fast
        self.slow_memory = new_slow
        self.token_history.append(token_id)

        W = model.train_window_size
        self.phrase_table.append((token_id, new_fast.detach().clone()))
        if len(self.phrase_table) > W:
            self.phrase_table.pop(0)

        return self.fast_state

    def logits(self) -> torch.Tensor:
        """
        .. deprecated::
            Uses ``readout(D)`` only — **not** ``readout_window(W·D) / effective_temperature``
            (training path). For parity with training, call
            ``forward_training_window(window_ids_from_sequence(...))`` or ``model.generate``.
        """
        global _LOGITS_MISMATCH_WARNED
        if not _LOGITS_MISMATCH_WARNED:
            warnings.warn(
                "AttractorStateCache.logits() uses readout(fast/slow), not readout_window; "
                "logits differ from training. Use TorchAttractorLanguageModel.generate() "
                "for aligned decoding.",
                FutureWarning,
                stacklevel=2,
            )
            _LOGITS_MISMATCH_WARNED = True
        model = self.model
        w_fast = float(model.w_fast)
        w_slow = float(model.w_slow)
        combined = w_fast * self.fast_state + w_slow * self.slow_memory
        n = torch.linalg.vector_norm(combined)
        if n > 1e-8:
            combined = combined / n

        with torch.no_grad():
            logits = model.readout(combined)
        return logits

    def warmup(self, prompt_ids: list[int]) -> None:
        """
        Seed the cache by stepping through prompt token ids one at a time.
        Prefer :meth:`TorchAttractorLanguageModel.generate` for decoding; this is for cache experiments only.
        """
        for tid in prompt_ids:
            self.step(tid)


# --------------------------------------------------------------------------
# generate_with_cache — rolling window inference
# --------------------------------------------------------------------------

def generate_with_cache(
    model: "sb.TorchAttractorLanguageModel",
    cache: AttractorStateCache,
    prompt: str,
    max_tokens: int = 40,
    temperature: float = 1.0,
    top_k: int = 28,
    repeat_penalty: float = 1.35,
    no_repeat_last_extra: float = 5.0,
    reset: bool = True,
) -> str:
    """
    .. deprecated::
        Legacy wrapper — now delegates to :meth:`TorchAttractorLanguageModel.generate`.
        The ``cache`` / ``reset`` arguments are ignored; kept for API compatibility only.

    Parameters
    ----------
    model : TorchAttractorLanguageModel
    cache : ignored (compatibility only)
    prompt : seed text
    max_tokens : tokens to generate
    temperature : sampling temperature
    top_k : top-k truncation
    repeat_penalty : passed to ``model.generate``
    no_repeat_last_extra : passed to ``model.generate``
    reset : ignored (compatibility only)
    """
    warnings.warn(
        "generate_with_cache is deprecated; it now calls model.generate() only. "
        "Pass prompt and sampling kwargs directly to TorchAttractorLanguageModel.generate.",
        FutureWarning,
        stacklevel=2,
    )
    if reset:
        cache.reset()
    return model.generate(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        no_repeat_last_extra=no_repeat_last_extra,
    )


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sandbox as sb  # type: ignore[import]

    print("[wave-c] state_cache self-test ...", flush=True)

    tok = sb._build_tokenizer(mode="fallback", vocab_cap=512)
    model = sb.TorchAttractorLanguageModel(tok.n_vocab, state_dim=512, train_window_size=4)
    model.tokenizer = tok
    model.eval()
    cache = AttractorStateCache(model)

    # Test 1: step produces finite states
    for word in ["the cat sat on"]:
        for tid in tok.encode(word)[:4]:
            cache.step(tid)
    assert torch.isfinite(cache.fast_state).all(), "fast_state is not finite"
    assert torch.isfinite(cache.slow_memory).all(), "slow_memory is not finite"
    print(f"  test 1 PASS — fast_state norm={cache.fast_state.norm():.4f}  slow_norm={cache.slow_memory.norm():.4f}", flush=True)

    # Test 2: training-parity readout (readout_window), not cache.logits()
    wid = model.window_ids_from_sequence(cache.token_history)
    plogits = model.forward_training_window(wid)
    assert torch.isfinite(plogits).all(), "forward_training_window logits not finite"
    print(f"  test 2 PASS — readout_window logits shape={plogits.shape}", flush=True)

    # Test 3: aligned generation path
    text = model.generate("the quick brown fox", max_tokens=10, temperature=1.0, top_k=28)
    assert len(text.split()) > 0, "model.generate returned empty string"
    print(f"  test 3 PASS — generated: {text!r}", flush=True)

    # Test 4: phrase_table rolling window
    assert len(cache.phrase_table) <= model.train_window_size, "phrase_table exceeds window_size"
    print(f"  test 4 PASS — phrase_table len={len(cache.phrase_table)} <= window_size={model.train_window_size}", flush=True)

    print("\n[wave-c] ALL TESTS PASSED", flush=True)
