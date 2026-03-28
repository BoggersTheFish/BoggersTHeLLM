#!/usr/bin/env python3
"""
Deterministic synthetic natural-language corpus for local experiments.

No network access. Token count uses GPT-2 byte-pair encoding (tiktoken) so
length is stable and comparable to common subword pipelines.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import tiktoken

# Short public-domain-flavor fragments (facts, nature, civic life) + connective variety.
_ADJECTIVES = (
    "ancient", "quiet", "busy", "open", "narrow", "broad", "gentle", "wild",
    "clear", "clouded", "bright", "dim", "warm", "cool", "dry", "wet",
    "still", "swift", "slow", "tall", "low", "deep", "shallow", "fair",
    "sturdy", "fragile", "plain", "rich", "humble", "proud", "calm", "eager",
)

_NOUNS = (
    "river", "harbor", "bridge", "market", "school", "library", "garden",
    "orchard", "meadow", "forest", "village", "city", "harbor", "lighthouse",
    "clock", "station", "road", "path", "gate", "tower", "hall", "kitchen",
    "workshop", "stable", "field", "hill", "valley", "shore", "island",
    "harvest", "winter", "spring", "summer", "autumn", "morning", "evening",
    "neighbor", "traveler", "teacher", "farmer", "mason", "baker", "sailor",
    "child", "parent", "counsel", "record", "ledger", "map", "letter", "song",
)

_VERBS = (
    "rests", "waits", "turns", "opens", "closes", "flows", "rises", "falls",
    "gleams", "fades", "echoes", "lingers", "passes", "returns", "gathers",
    "scatters", "measures", "marks", "keeps", "holds", "lends", "reads",
    "writes", "sings", "listens", "learns", "teaches", "builds", "mends",
)

_PLACE_PREP = ("near", "beyond", "beside", "under", "along", "toward", "past", "around")
_PLACES = (
    "the old wall", "the harbor lights", "the market square", "the river bend",
    "the orchard row", "the schoolyard", "the stone bridge", "the open field",
    "the forest edge", "the village green", "the city gate", "the quiet lane",
    "the hill road", "the shore path", "the morning mist", "the evening tide",
)

_STRUCT_OPENERS = (
    "In practice, ", "For the record, ", "By custom, ", "In that season, ",
    "On fair days, ", "When work slowed, ", "After the rain, ", "Before dusk, ",
    "Near the harbor, ", "Along the river, ", "At the crossroads, ",
)

_STRUCT_MIDDLES = (
    "people traded news more than goods; ",
    "children learned rhymes before rules; ",
    "the council kept minutes in plain ink; ",
    "travelers shared water and a careful map; ",
    "the bell marked hours that felt the same; ",
    "the ledger grew honest line by line; ",
    "the choir practiced until the tune held; ",
    "the masons squared each stone to the last; ",
)

_STRUCT_CLOSERS = (
    "no one hurried the ending.",
    "the work was enough for the day.",
    "memory outlasted the weather.",
    "kindness cost little and lasted.",
    "the story stayed shorter than the road.",
)


def _sentence_pool(rng: random.Random, size: int = 420) -> list[str]:
    """Build a medium-sized pool of varied declarative sentences."""
    out: list[str] = []
    for _ in range(size):
        if rng.random() < 0.55:
            out.append(
                f"The {rng.choice(_ADJECTIVES)} {rng.choice(_NOUNS)} "
                f"{rng.choice(_VERBS)} {rng.choice(_PLACE_PREP)} {rng.choice(_PLACES)}."
            )
        else:
            out.append(
                f"{rng.choice(_STRUCT_OPENERS)}"
                f"{rng.choice(_STRUCT_MIDDLES)}"
                f"{rng.choice(_STRUCT_CLOSERS)}"
            )
    # Deterministic light paraphrases
    extra: list[str] = []
    for s in out[:120]:
        if s.startswith("The "):
            extra.append(s.replace("The ", "Yet the ", 1))
        else:
            extra.append(s + " Still, the day remained fair.")
    out.extend(extra)
    return out


def _one_paragraph(rng: random.Random, pool: list[str]) -> str:
    n = rng.randint(4, 9)
    picks = [rng.choice(pool) for _ in range(n)]
    # Vary connectors
    glue = rng.choice(("\n", " ", "\n"))
    if glue == "\n":
        return "\n".join(picks)
    joined = picks[0]
    for p in picks[1:]:
        conj = rng.choice((" Moreover, ", " Meanwhile, ", " Still, ", " Then "))
        joined += conj.strip() + " " + p[0].lower() + p[1:]
    return joined


def generate_corpus(path: Path | str, target_tokens: int = 20000, seed: int = 42) -> None:
    """
    Append paragraphs until ``tiktoken`` GPT-2 encoding length is at least
    ``target_tokens``, then write UTF-8 text to ``path``.
    """
    path = Path(path)
    rng = random.Random(seed)
    enc = tiktoken.get_encoding("gpt2")
    pool = _sentence_pool(rng)
    parts: list[str] = []
    text = ""
    while len(enc.encode(text)) < target_tokens:
        parts.append(_one_paragraph(rng, pool))
        text = "\n\n".join(parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a deterministic synthetic text corpus.")
    ap.add_argument("--out", type=Path, required=True, help="Output .txt path")
    ap.add_argument("--tokens", type=int, default=20000, help="Target token count (GPT-2 BPE)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    generate_corpus(args.out, target_tokens=args.tokens, seed=args.seed)
    enc = tiktoken.get_encoding("gpt2")
    got = len(enc.encode(Path(args.out).read_text(encoding="utf-8")))
    print(f"Wrote {args.out}  ({got} GPT-2 tokens)", flush=True)


if __name__ == "__main__":
    main()
