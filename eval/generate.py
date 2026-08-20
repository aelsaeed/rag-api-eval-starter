import json
import re
from hashlib import sha256
from pathlib import Path


def _sentences(text: str) -> list[str]:
    body = " ".join(
        line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")
    )
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", body)
        if len(sentence.strip()) > 20
    ]


def generate(dataset_path: str, docs_path: str, total: int = 30) -> None:
    docs_dir = Path(docs_path)
    candidates: list[tuple[str, str]] = []
    documents = sorted((*docs_dir.glob("*.md"), *docs_dir.glob("*.txt")))
    for document in documents:
        candidates.extend(
            (document.name, sentence)
            for sentence in _sentences(document.read_text(encoding="utf-8"))
        )

    if not candidates:
        raise ValueError("No documents found for dataset generation")
    if total <= 0:
        raise ValueError("total must be positive")

    records = []
    for source, sentence in candidates[:total]:
        subject = sentence.split(" ")[0:5]
        digest = sha256(f"{source}\0{sentence}".encode()).hexdigest()[:10]
        question = f"What does the documentation say about {' '.join(subject)}?"
        records.append(
            {
                "id": f"candidate-{digest}",
                "question": question,
                "answerable": True,
                "gold_contexts": [{"source": source, "anchor": sentence, "relevance": 1}],
                "required_fact_groups": [[sentence]],
                "tags": ["synthetic", "review-required"],
            }
        )

    output_path = Path(dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    # Synthetic records are candidates for human review, never the curated gold set.
    generate("data/eval.candidates.jsonl", "data/sample_docs")
