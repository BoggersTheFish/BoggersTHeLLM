"""
Load public Hugging Face datasets into a plain UTF-8 text file for sandbox training.

Each dataset row is written as one line (internal newlines collapsed to spaces) so
`eval_harness.py --corpus` line-based splits stay well-defined.

Requires: pip install datasets
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# --- dataset ids (Hugging Face Hub) ---
TINYSTORIES_ID = "roneneldan/TinyStories"
FINEWEB_EDU_ID = "HuggingFaceFW/fineweb-edu"
FINEWEB_EDU_CONFIG = "sample-10BT"


def _norm_line(text: str) -> str:
    return " ".join(text.split())


def _cache_file(cache_dir: Path, source: str, max_rows: int, max_chars: int) -> Path:
    key = f"{source}|rows={max_rows}|chars={max_chars}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{source.replace('-', '_')}_{h}.txt"


def ensure_hf_corpus_file(
    source: str,
    *,
    cache_dir: Path,
    max_rows: int,
    max_chars: int,
    refresh: bool = False,
) -> Path:
    """
    Return path to a text file containing concatenated examples.

    Parameters
    ----------
    source
        ``tinystories`` or ``fineweb-edu``.
    cache_dir
        Directory for cached ``.txt`` (reused across runs with same limits).
    max_rows
        Maximum rows to read from the dataset (after sharding / streaming).
    max_chars
        If > 0, stop after this many UTF-8 characters (approximate corpus cap).
    refresh
        If True, ignore cache and rebuild.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "Hugging Face datasets are not installed. Run: pip install datasets"
        ) from exc

    out = _cache_file(cache_dir, source, max_rows, max_chars)
    if out.is_file() and not refresh:
        print(f"[hf] Using cached corpus: {out}", flush=True)
        return out

    print(f"[hf] Building corpus → {out} (this may download data once)...", flush=True)

    if source == "tinystories":
        split = f"train[:{max_rows}]"
        ds = load_dataset(TINYSTORIES_ID, split=split)
        rows = []
        for ex in ds:
            t = ex.get("text") or ex.get("story") or ""
            if isinstance(t, str) and t.strip():
                rows.append(_norm_line(t))
    elif source == "fineweb-edu":
        # Streaming: sample-10BT is a public subset; read only the first max_rows rows.
        try:
            ds = load_dataset(
                FINEWEB_EDU_ID,
                name=FINEWEB_EDU_CONFIG,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not open {FINEWEB_EDU_ID} (config {FINEWEB_EDU_CONFIG!r}). "
                "Upgrade datasets: pip install -U datasets. Original error: "
                f"{exc}"
            ) from exc
        rows = []
        for i, ex in enumerate(ds):
            if i >= max_rows:
                break
            t = ex.get("text") or ""
            if isinstance(t, str) and t.strip():
                rows.append(_norm_line(t))
    else:
        raise ValueError(f"unknown HF dataset source: {source!r}")

    text_parts: list[str] = []
    total_chars = 0
    for line in rows:
        if max_chars > 0 and total_chars >= max_chars:
            break
        if max_chars > 0:
            remain = max_chars - total_chars
            if len(line) > remain:
                line = line[:remain]
        text_parts.append(line)
        total_chars += len(line) + 1  # newline

    blob = "\n".join(text_parts)
    if not blob.strip():
        raise RuntimeError(
            f"[hf] No text extracted for source={source!r}. "
            "Check network access and dataset availability."
        )

    out.write_text(blob + "\n", encoding="utf-8")
    print(
        f"[hf] Wrote {len(text_parts)} lines, ~{len(blob)} chars → {out}",
        flush=True,
    )
    return out


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Download HF corpus to a .txt file.")
    p.add_argument(
        "source",
        choices=("tinystories", "fineweb-edu"),
        help="Dataset to materialize",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "cache" / "hf",
    )
    p.add_argument("--max-rows", type=int, default=50_000)
    p.add_argument("--max-chars", type=int, default=0, help="0 = no cap")
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()
    path = ensure_hf_corpus_file(
        args.source,
        cache_dir=args.cache_dir,
        max_rows=args.max_rows,
        max_chars=args.max_chars,
        refresh=args.refresh,
    )
    print(path)
    sys.exit(0)


if __name__ == "__main__":
    main()
