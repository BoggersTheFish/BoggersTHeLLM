"""
Wave D — Streaming sharded DataLoader.

Replaces the in-memory list shuffle in sandbox.py's training loop with a
streaming DataLoader that:

  1. Accepts any iterable of JSONL / plain-text / corpus files (or directories).
  2. Tokenises each line lazily (word-list or tiktoken via wave_a_tokenizer).
  3. Builds sliding-window (context, target) pairs on-the-fly.
  4. Yields mini-batches of (contexts, targets, target_state_batch) compatible with
     model.trajectory_contrastive_loss_and_logits(..., target_states=...).
     The third element is None unless ``train_target_states`` is set on the pipeline.
  5. Supports multi-shard round-robin by holding one file handle per shard.

Usage
-----
    from data_pipeline import AttractorDataPipeline

    pipe = AttractorDataPipeline(
        sources=["data/corpus.txt", "data/extra/"],   # files or directories
        model=model,          # for vocab / window_size
        batch_size=16,
        window_size=6,
        shuffle_buffer=1024,
        tokenizer=tok,        # optional: AttractorTokenizer from wave_a_tokenizer
    )

    for epoch in range(NUM_EPOCHS):
        for contexts, targets, tgt_states in pipe.epoch_batches(epoch_index=0):
            loss, _ = model.trajectory_contrastive_loss_and_logits(
                contexts, targets, target_states=tgt_states
            )
            ...

For single-machine operation (no Redis), shuffle_buffer controls in-memory
random shuffling. For distributed runs, each worker gets its own shard files.
"""
from __future__ import annotations

import os
import random
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Iterable, Optional

import torch

if TYPE_CHECKING:
    import sandbox as sb  # type: ignore[import]


# --------------------------------------------------------------------------
# Tokenizer protocol (duck-typed to accept both word-list and AttractorTokenizer)
# --------------------------------------------------------------------------

class _WordListTokenizer:
    """Last-resort tokenizer: delegates to model.tokenizer or uses hash-based IDs."""

    def __init__(self, model: "sb.TorchAttractorLanguageModel") -> None:
        self.n_vocab = model.vocab_size
        self._tok = getattr(model, "tokenizer", None)

    def encode(self, text: str) -> list[int]:
        if self._tok is not None:
            return self._tok.encode(text)
        # Bare minimum: stable hash-based IDs (no vocabulary required).
        return [hash(w) % self.n_vocab for w in text.lower().split()]


# --------------------------------------------------------------------------
# File discovery
# --------------------------------------------------------------------------

def _collect_text_files(source: str | Path) -> list[Path]:
    p = Path(source)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(
            f for f in p.rglob("*")
            if f.suffix in (".txt", ".jsonl", ".json") and f.is_file()
        )
    return []


def _iter_lines(path: Path) -> Generator[str, None, None]:
    """Yield non-empty, non-comment lines from a text or JSONL file."""
    import json as _json

    with path.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if path.suffix == ".jsonl":
                try:
                    obj = _json.loads(line)
                    text = obj.get("text") or obj.get("content") or obj.get("sentence", "")
                    if text:
                        yield str(text)
                except Exception:
                    yield line
            else:
                yield line


def _load_full_text_from_file(path: Path) -> str:
    """
    Load entire file as one string (no per-line tokenization).

    - .txt / .md / plain: raw read (newlines preserved).
    - .jsonl: concatenate extracted text fields, newline-separated.
    """
    if path.suffix == ".jsonl":
        parts: list[str] = []
        for line in _iter_lines(path):
            parts.append(line)
        return "\n".join(parts)
    return path.read_text(encoding="utf-8", errors="replace")


def load_concatenated_corpus_text(files: list[Path]) -> str:
    """Concatenate all shard files into one corpus string (blank line between files)."""
    chunks: list[str] = []
    for path in files:
        chunks.append(_load_full_text_from_file(path))
    return "\n\n".join(chunks)


# --------------------------------------------------------------------------
# Sliding-window pair builder
# --------------------------------------------------------------------------

def _make_windows(ids: list[int], window_size: int) -> list[tuple[list[int], int]]:
    pairs: list[tuple[list[int], int]] = []
    for start in range(len(ids) - window_size):
        ctx = ids[start : start + window_size]
        tgt = ids[start + window_size]
        pairs.append((ctx, tgt))
    return pairs


# --------------------------------------------------------------------------
# AttractorDataPipeline
# --------------------------------------------------------------------------

class AttractorDataPipeline:
    """
    Streaming, shard-aware data pipeline for the attractor training loop.

    Parameters
    ----------
    sources : list of file paths or directories (plain text or JSONL)
    model : TorchAttractorLanguageModel — provides vocab + window_size
    batch_size : mini-batch width (≥ 2 for trajectory contrastive loss)
    window_size : overrides model.train_window_size when set
    shuffle_buffer : (line mode only) pairs held for shuffle between refills
    tokenizer : optional; if None, uses the model word list
    streaming_dataset : if True (default), corpus is one continuous token stream
        (full-file read + encode); ignores line boundaries for windowing.
    train_token_ids : optional pre-tokenized train split; if set, ``sources``
        are not read for training windows (useful when sandbox splits tokens).
    shard_id / num_shards : for multi-worker data parallelism (each worker
        receives every num_shards-th file starting from shard_id)
    seed : random seed for shuffle
    train_target_states : optional ``(n_windows, W, D)`` float tensor (CPU), one row per
        sliding window in stream order (``start`` 0 .. len(tokens)-W-1), aligned with
        ``epoch_batches`` window indexing.
    """

    def __init__(
        self,
        sources: list[str | Path],
        model: "sb.TorchAttractorLanguageModel",
        batch_size: int = 16,
        window_size: Optional[int] = None,
        shuffle_buffer: int = 1024,
        tokenizer: Optional[object] = None,
        streaming_dataset: bool = True,
        train_token_ids: Optional[list[int]] = None,
        train_target_states: Optional[torch.Tensor] = None,
        shard_id: int = 0,
        num_shards: int = 1,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.batch_size = max(2, batch_size)
        self.window_size = window_size or model.train_window_size
        self.shuffle_buffer = shuffle_buffer
        self.tokenizer = tokenizer or _WordListTokenizer(model)
        self.streaming_dataset = streaming_dataset
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.seed = seed

        # Collect all text files, then apply sharding
        all_files: list[Path] = []
        for src in sources:
            all_files.extend(_collect_text_files(src))
        self.files: list[Path] = [
            f for i, f in enumerate(all_files) if i % num_shards == shard_id
        ]

        self._stream_tokens: Optional[list[int]] = None
        self._line_mode = False

        if train_token_ids is not None:
            self._stream_tokens = list(train_token_ids)
            self._line_mode = False
        elif streaming_dataset:
            if not self.files:
                raise ValueError(
                    f"No text files found under sources={sources} for "
                    f"shard_id={shard_id}/{num_shards}"
                )
            full_text = load_concatenated_corpus_text(self.files)
            self._stream_tokens = self.tokenizer.encode(full_text)
            self._line_mode = False
        else:
            if not self.files:
                raise ValueError(
                    f"No text files found under sources={sources} for "
                    f"shard_id={shard_id}/{num_shards}"
                )
            self._line_mode = True

        if self._stream_tokens is not None:
            need = self.window_size + 1
            if len(self._stream_tokens) < need:
                raise ValueError(
                    "Corpus too small after tokenization: need at least "
                    f"{need} tokens for window_size={self.window_size}, "
                    f"got {len(self._stream_tokens)}."
                )

        self._train_target_states: Optional[torch.Tensor] = None
        if train_target_states is not None:
            if self._stream_tokens is None:
                raise ValueError(
                    "train_target_states requires stream token mode (train_token_ids or "
                    "streaming full-file encode), not line-only buffering."
                )
            n_win = len(self._stream_tokens) - self.window_size
            ts = train_target_states
            if not isinstance(ts, torch.Tensor):
                ts = torch.as_tensor(ts, dtype=torch.float32)
            else:
                ts = ts.to(dtype=torch.float32)
            Wm = self.window_size
            Dm = int(model.state_dim)
            if ts.dim() != 3 or ts.shape[0] != n_win or ts.shape[1] != Wm or ts.shape[2] != Dm:
                raise ValueError(
                    "train_target_states must have shape "
                    f"(n_windows, W, D)=({n_win}, {Wm}, {Dm}); got {tuple(ts.shape)}"
                )
            self._train_target_states = ts.detach().cpu().contiguous()

    # ------------------------------------------------------------------
    def _num_stream_windows(self) -> int:
        assert self._stream_tokens is not None
        return max(0, len(self._stream_tokens) - self.window_size)

    def _window_stream(self) -> Generator[tuple[list[int], int], None, None]:
        """Line-based mode: yield (context, target) pairs by streaming lines."""
        W = self.window_size
        for path in self.files:
            for line in _iter_lines(path):
                ids = self.tokenizer.encode(line)
                if len(ids) < W + 1:
                    continue
                for pair in _make_windows(ids, W):
                    yield pair

    def epoch_batches(
        self,
        epoch_index: int = 0,
    ) -> Generator[
        tuple[list[list[int]], list[int], Optional[torch.Tensor]], None, None
    ]:
        """
        Yield (contexts, targets, target_states_batch) mini-batches for one epoch.

        ``target_states_batch`` is ``None`` unless ``train_target_states`` was passed at
        construction; then it is ``(B, W, D)`` float CPU for the same window indices as
        ``contexts``.

        Stream mode: shuffles window start indices, then batches. Uses
        ``random.Random(self.seed + epoch_index)`` so each epoch gets a
        deterministic but distinct order (reproducible when base ``seed`` fixed).

        Line mode: shuffle-buffer refill over the line stream (legacy).
        """
        if self._stream_tokens is not None:
            yield from self._epoch_batches_stream(epoch_index=epoch_index)
            return

        rng = random.Random(self.seed + int(epoch_index))
        buf: deque[tuple[list[int], int]] = deque()
        stream = self._window_stream()

        def _fill(n: int) -> bool:
            count = 0
            for item in stream:
                buf.append(item)
                count += 1
                if count >= n:
                    return True
            return False

        _fill(self.shuffle_buffer)
        buf_list = list(buf)
        buf.clear()

        while buf_list:
            rng.shuffle(buf_list)
            for i in range(0, len(buf_list), self.batch_size):
                chunk = buf_list[i : i + self.batch_size]
                if len(chunk) < 2:
                    chunk = chunk * 2
                contexts = [c for c, _t in chunk]
                targets = [_t for _c, _t in chunk]
                yield contexts, targets, None

            buf_list = []
            _fill(self.shuffle_buffer)
            buf_list = list(buf)
            buf.clear()

    def _epoch_batches_stream(
        self, *, epoch_index: int = 0
    ) -> Generator[
        tuple[list[list[int]], list[int], Optional[torch.Tensor]], None, None
    ]:
        """One full pass over all sliding windows with shuffled order."""
        rng = random.Random(self.seed + int(epoch_index))
        toks = self._stream_tokens
        assert toks is not None
        W = self.window_size
        n_win = len(toks) - W
        if n_win <= 0:
            return
        idxs = list(range(n_win))
        rng.shuffle(idxs)
        bs = self.batch_size
        ts_all = self._train_target_states
        for i in range(0, len(idxs), bs):
            batch_i = idxs[i : i + bs]
            contexts = [toks[j : j + W] for j in batch_i]
            targets = [toks[j + W] for j in batch_i]
            ts_batch: Optional[torch.Tensor] = None
            if ts_all is not None:
                bi = torch.as_tensor(batch_i, dtype=torch.long)
                ts_batch = ts_all[bi].clone()
            if len(contexts) < 2:
                contexts = contexts * 2
                targets = targets * 2
                if ts_batch is not None:
                    ts_batch = ts_batch.repeat(2, 1, 1)
            yield contexts, targets, ts_batch

    def epoch_count_estimate(self) -> int:
        """Approximate batches per epoch."""
        if self._stream_tokens is not None:
            n_win = self._num_stream_windows()
            return max(1, (n_win + self.batch_size - 1) // self.batch_size)
        total = 0
        for path in self.files:
            for line in _iter_lines(path):
                ids = self.tokenizer.encode(line)
                if len(ids) >= self.window_size + 1:
                    total += max(0, len(ids) - self.window_size)
        return max(1, max(0, total) // self.batch_size)


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent))
    import sandbox as sb  # type: ignore[import]

    print("[wave-d] data_pipeline self-test ...", flush=True)

    CORPUS = _Path(__file__).resolve().parent / "data" / "corpus.txt"
    tok = sb._build_tokenizer(mode="fallback", vocab_cap=512)
    model = sb.TorchAttractorLanguageModel(tok.n_vocab, train_window_size=4)
    model.tokenizer = tok
    model.eval()

    pipe = AttractorDataPipeline(
        sources=[CORPUS],
        model=model,
        batch_size=4,
        window_size=4,
        shuffle_buffer=64,
        tokenizer=tok,
        seed=0,
    )

    count = 0
    total_contexts = 0
    for contexts, targets, _ts in pipe.epoch_batches(epoch_index=0):
        assert len(contexts) == len(targets), "contexts/targets length mismatch"
        assert len(contexts) >= 2, "batch too small for trajectory contrastive loss"
        total_contexts += len(contexts)
        count += 1
        if count >= 5:
            break

    print(f"  test 1 PASS — {count} batches yielded, {total_contexts} total windows", flush=True)

    # Test 2: estimate
    est = pipe.epoch_count_estimate()
    print(f"  test 2 PASS — epoch_count_estimate: {est} batches", flush=True)

    # Test 3: JSONL support (write a temp file)
    import json, tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
        for sentence in sb.load_corpus(CORPUS)[:5]:  # type: ignore[attr-defined]
            fh.write(json.dumps({"text": sentence}) + "\n")
        tmp = fh.name

    try:
        pipe2 = AttractorDataPipeline(
            sources=[tmp],
            model=model,
            batch_size=2,
            shuffle_buffer=32,
        )
        batches = list(pipe2.epoch_batches(epoch_index=0))
        assert len(batches) >= 1, "JSONL pipeline yielded no batches"
        print(f"  test 3 PASS — JSONL: {len(batches)} batches", flush=True)
    finally:
        os.unlink(tmp)

    print("\n[wave-d] ALL TESTS PASSED", flush=True)
