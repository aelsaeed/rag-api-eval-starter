from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eval.models import EvalCase, GoldContext
from eval.text import normalize_text

_CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_CASE_FIELDS = {
    "id",
    "question",
    "answerable",
    "gold_contexts",
    "required_fact_groups",
    "tags",
}
_CONTEXT_FIELDS = {"source", "anchor", "relevance"}


class DatasetValidationError(ValueError):
    """Raised when the golden dataset does not satisfy its schema."""


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _string_list(value: Any, field: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise DatasetValidationError(f"{field} must be a list of strings")
    result = tuple(_required_string(item, field) for item in value)
    if not allow_empty and not result:
        raise DatasetValidationError(f"{field} must not be empty")
    if len(set(result)) != len(result):
        raise DatasetValidationError(f"{field} must not contain duplicates")
    return result


def _parse_gold_context(value: Any, case_id: str, index: int) -> GoldContext:
    field = f"case {case_id!r} gold_contexts[{index}]"
    if not isinstance(value, dict):
        raise DatasetValidationError(f"{field} must be an object")
    unknown = set(value) - _CONTEXT_FIELDS
    if unknown:
        raise DatasetValidationError(f"{field} has unknown fields: {sorted(unknown)}")

    source = _required_string(value.get("source"), f"{field}.source")
    anchor = _required_string(value.get("anchor"), f"{field}.anchor")
    if not normalize_text(anchor):
        raise DatasetValidationError(f"{field}.anchor must contain searchable text")
    relevance = value.get("relevance", 1)
    if isinstance(relevance, bool) or not isinstance(relevance, int) or not 1 <= relevance <= 3:
        raise DatasetValidationError(f"{field}.relevance must be an integer from 1 to 3")
    return GoldContext(source=source, anchor=anchor, relevance=relevance)


def parse_case(value: Any) -> EvalCase:
    if not isinstance(value, dict):
        raise DatasetValidationError("each JSONL record must be an object")
    unknown = set(value) - _CASE_FIELDS
    if unknown:
        raise DatasetValidationError(f"record has unknown fields: {sorted(unknown)}")

    case_id = _required_string(value.get("id"), "id")
    if not _CASE_ID_RE.fullmatch(case_id):
        raise DatasetValidationError(f"case id {case_id!r} must match {_CASE_ID_RE.pattern}")
    question = _required_string(value.get("question"), f"case {case_id!r}.question")
    answerable = value.get("answerable")
    if not isinstance(answerable, bool):
        raise DatasetValidationError(f"case {case_id!r}.answerable must be a boolean")

    raw_contexts = value.get("gold_contexts")
    if not isinstance(raw_contexts, list):
        raise DatasetValidationError(f"case {case_id!r}.gold_contexts must be a list")
    contexts = tuple(
        _parse_gold_context(context, case_id, index) for index, context in enumerate(raw_contexts)
    )
    context_keys = [(context.source, normalize_text(context.anchor)) for context in contexts]
    if len(set(context_keys)) != len(context_keys):
        raise DatasetValidationError(f"case {case_id!r} has duplicate gold contexts")

    raw_groups = value.get("required_fact_groups")
    if not isinstance(raw_groups, list):
        raise DatasetValidationError(
            f"case {case_id!r}.required_fact_groups must be a list of string lists"
        )
    fact_groups = tuple(
        _string_list(group, f"case {case_id!r}.required_fact_groups[{index}]", allow_empty=False)
        for index, group in enumerate(raw_groups)
    )
    normalized_groups = [tuple(normalize_text(alias) for alias in group) for group in fact_groups]
    if any(not alias for group in normalized_groups for alias in group):
        raise DatasetValidationError(
            f"case {case_id!r}.required_fact_groups must contain searchable text"
        )
    if any(len(set(group)) != len(group) for group in normalized_groups):
        raise DatasetValidationError(
            f"case {case_id!r}.required_fact_groups contains duplicate aliases"
        )
    if len(set(normalized_groups)) != len(normalized_groups):
        raise DatasetValidationError(f"case {case_id!r} has duplicate required fact groups")

    tags = _string_list(value.get("tags", []), f"case {case_id!r}.tags")

    if answerable and (not contexts or not fact_groups):
        raise DatasetValidationError(
            f"answerable case {case_id!r} requires gold contexts and required fact groups"
        )
    if not answerable and (contexts or fact_groups):
        raise DatasetValidationError(
            f"unanswerable case {case_id!r} must not define contexts or facts"
        )

    return EvalCase(
        id=case_id,
        question=question,
        answerable=answerable,
        gold_contexts=contexts,
        required_fact_groups=fact_groups,
        tags=tags,
    )


def load_cases(path: str | Path) -> tuple[EvalCase, ...]:
    dataset_path = Path(path)
    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                case = parse_case(raw)
            except (json.JSONDecodeError, DatasetValidationError) as exc:
                raise DatasetValidationError(f"{dataset_path}:{line_number}: {exc}") from exc
            if case.id in seen_ids:
                raise DatasetValidationError(
                    f"{dataset_path}:{line_number}: duplicate case id {case.id!r}"
                )
            seen_ids.add(case.id)
            cases.append(case)
    if not cases:
        raise DatasetValidationError(f"{dataset_path}: evaluation dataset is empty")
    return tuple(cases)


def dataset_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_gold_contexts(cases: Sequence[EvalCase], corpus: Mapping[str, str]) -> None:
    errors: list[str] = []
    normalized_corpus = {source: normalize_text(text) for source, text in corpus.items()}
    for case in cases:
        for context in case.gold_contexts:
            document = normalized_corpus.get(context.source)
            if document is None:
                errors.append(f"{case.id}: unknown source {context.source!r}")
                continue
            if normalize_text(context.anchor) not in document:
                errors.append(
                    f"{case.id}: anchor not found in source {context.source!r}: {context.anchor!r}"
                )
    if errors:
        raise DatasetValidationError("invalid gold contexts: " + "; ".join(errors))
