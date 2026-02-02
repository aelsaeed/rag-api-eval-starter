import argparse
import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness


def load_dataset(path: str) -> Dataset:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def render_report(result: dict) -> str:
    lines = ["# RAGAS Evaluation Report", "", "## Metrics"]
    for key, value in result.items():
        lines.append(f"- **{key}**: {value:.4f}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    result_dict = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    report = render_report(result_dict)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
