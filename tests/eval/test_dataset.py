import json
from pathlib import Path

import pytest

from eval.adapters import load_corpus
from eval.dataset import (
    DatasetValidationError,
    load_cases,
    parse_case,
    validate_gold_contexts,
)
from eval.generate import generate

ROOT = Path(__file__).resolve().parents[2]


def _record() -> dict:
    return {
        "id": "case-001",
        "question": "What is supported?",
        "answerable": True,
        "gold_contexts": [{"source": "doc.md", "anchor": "Markdown is supported.", "relevance": 2}],
        "required_fact_groups": [["markdown"], ["supported", "available"]],
        "tags": ["capability"],
    }


def test_curated_dataset_has_unique_ids_and_valid_corpus_anchors() -> None:
    cases = load_cases(ROOT / "data/eval.jsonl")

    assert len(cases) == 12
    assert len({case.id for case in cases}) == len(cases)
    assert sum(case.answerable for case in cases) == 10
    assert sum(not case.answerable for case in cases) == 2
    validate_gold_contexts(cases, load_corpus(ROOT / "data/sample_docs"))


def test_load_cases_rejects_duplicate_ids_with_line_number(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.jsonl"
    line = json.dumps(_record())
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match=r":2: duplicate case id 'case-001'"):
        load_cases(path)


@pytest.mark.parametrize("anchor", ["", "!!!"])
def test_parse_case_rejects_empty_or_unsearchable_anchor(anchor: str) -> None:
    record = _record()
    record["gold_contexts"][0]["anchor"] = anchor

    with pytest.raises(DatasetValidationError, match="anchor"):
        parse_case(record)


def test_unanswerable_case_cannot_define_gold_facts() -> None:
    record = _record()
    record["answerable"] = False

    with pytest.raises(DatasetValidationError, match="must not define contexts or facts"):
        parse_case(record)


def test_gold_anchor_must_exist_in_declared_source() -> None:
    case = parse_case(_record())

    with pytest.raises(DatasetValidationError, match="anchor not found"):
        validate_gold_contexts((case,), {"doc.md": "Only text uploads are supported."})


def test_generator_writes_review_candidates_without_overwriting_gold_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidates.jsonl"

    generate(str(path), str(ROOT / "data/sample_docs"), total=3)
    cases = load_cases(path)

    assert len(cases) == 3
    assert all("review-required" in case.tags for case in cases)
    validate_gold_contexts(cases, load_corpus(ROOT / "data/sample_docs"))
