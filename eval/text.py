from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> tuple[str, ...]:
    """Return deterministic, punctuation-insensitive ASCII tokens."""

    return tuple(_TOKEN_RE.findall(text.casefold()))


def normalize_text(text: str) -> str:
    """Normalize text for robust anchor and required-fact matching."""

    return " ".join(tokens(text))
