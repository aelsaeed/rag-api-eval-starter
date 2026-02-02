import json
from pathlib import Path


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 20]


def generate(dataset_path: str, docs_path: str, total: int = 30) -> None:
    docs_dir = Path(docs_path)
    sentences = []
    for doc in docs_dir.glob("*.md"):
        sentences.extend(_sentences(doc.read_text(encoding="utf-8")))
    for doc in docs_dir.glob("*.txt"):
        sentences.extend(_sentences(doc.read_text(encoding="utf-8")))

    if not sentences:
        raise ValueError("No documents found for dataset generation")

    records = []
    for idx in range(total):
        sentence = sentences[idx % len(sentences)]
        subject = sentence.split(" ")[0:5]
        question = f"What does the documentation say about {' '.join(subject)}?"
        records.append(
            {
                "question": question,
                "answer": sentence,
                "contexts": [sentence],
                "ground_truths": [sentence],
            }
        )

    output_path = Path(dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    generate("data/eval.jsonl", "data/sample_docs")
