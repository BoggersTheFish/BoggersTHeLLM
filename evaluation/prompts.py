"""
Fixed prompts for end-of-epoch generation during `sandbox.py` training.

Each epoch the loop runs ``model.generate(prompt, max_tokens=120)`` for every
string below and appends results to ``logs/eval_epoch_{epoch}.txt`` (1-based epoch).
"""

EVAL_PROMPTS = [
    "Once upon a time there was a curious robot who",
    "The scientist discovered that the strange signal",
    "In a small town near the mountains, a boy found",
    "The quick brown fox jumps over the lazy dog because",
    "A detective entered the dark room and noticed",
    "Why do people dream about impossible places?",
    "If gravity suddenly stopped working, the first thing that would happen is",
    "Two friends were arguing about whether machines could think when",
]
