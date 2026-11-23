"""Prompt-pool helpers shared by the generators.

  parse_numbered(text, n)   split a "1. ... 2. ..." model reply into items
  _classify_prompt(p)       screen a generated prompt ("hard" = drop: refers to files/links/prior turns, etc.)
  _trigrams / _jaccard      word-trigram near-duplicate filter
"""
from __future__ import annotations

import re


_NUMBERED = re.compile(r"^\s*\d+[\.\)]\s*(.+?)\s*$")


def parse_numbered(text: str, n: int) -> list[str]:
    out = []
    for line in text.splitlines():
        m = _NUMBERED.match(line)
        if m:
            out.append(m.group(1))
    return out[:n]


_HARD_BOGUS_PATTERNS = [
    ("attachment_ref", re.compile(r"\b(attached|attachment|the file|the document|the link|uploaded|provided document|the doc)\b", re.IGNORECASE)),
    ("image_ref",      re.compile(r"\b(the image|the picture|the photo|the screenshot|the chart|the graph|the diagram|the video|in the image|in the picture)\b", re.IGNORECASE)),
    ("backref",        re.compile(r"\b(as i mentioned|as i said|as discussed|previously discussed|in your last|in your previous|as we discussed|the previous response)\b", re.IGNORECASE)),
    ("pronoun_orphan", re.compile(r"^(That|This|It|These|Those|They|He|She)\s+(is|are|was|were|will|would|seems|seem|looks|look)\b")),
    ("too_short",      re.compile(r"^.{0,25}$")),
]


_SOFT_FLAG_PATTERNS = [
    ("no_terminal_punct", re.compile(r"[^.!?\"')\]]\s*$")),
]


def _classify_prompt(prompt: str) -> tuple[str, str]:
    """Return ('hard', reason) | ('soft', reason) | ('ok', '')."""
    for name, pat in _HARD_BOGUS_PATTERNS:
        if pat.search(prompt):
            return ("hard", name)
    for name, pat in _SOFT_FLAG_PATTERNS:
        if pat.search(prompt):
            return ("soft", name)
    return ("ok", "")


def _trigrams(s: str) -> set[str]:
    """Lowercased character-trigram set, non-alphanumerics collapsed to single
    spaces. Captures slot-fill clones like "I am struggling ... low-light forest"
    vs "I am struggling ... low-light dusk" without being confused by case or
    incidental punctuation."""
    norm = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    return {norm[i:i+3] for i in range(max(0, len(norm) - 2))}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0
