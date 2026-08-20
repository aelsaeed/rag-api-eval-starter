import re
from collections import Counter

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "which",
    "why",
    "with",
}


def _stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        stem = token[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Return normalized lexical tokens shared by indexing and querying."""

    return [
        _stem(token) for token in TOKEN_PATTERN.findall(text.casefold()) if token not in STOP_WORDS
    ]


def lexical_overlap_score(document_tokens: list[str], query_tokens: list[str]) -> float:
    """Score how much of a query is represented in a document.

    Duplicate query terms count once so repeated words cannot inflate the result.
    """

    query_terms = set(query_tokens)
    if not document_tokens or not query_terms:
        return 0.0
    document_terms = Counter(document_tokens)
    return sum(term in document_terms for term in query_terms) / len(query_terms)
