import json
from pathlib import Path

from eval.compare import (
    render_ablation_json,
    render_ablation_markdown,
    run_ablation,
)
from eval.metrics import enforce_thresholds
from eval.models import Thresholds

ROOT = Path(__file__).resolve().parents[2]


def test_real_strategy_ablation_is_deterministic_and_hybrid_passes_gates() -> None:
    reports = run_ablation(ROOT / "data/eval.jsonl", ROOT / "data/sample_docs")

    enforce_thresholds(reports["hybrid"], Thresholds())
    assert reports["hybrid"].summary.mrr_at_5 > reports["dense"].summary.mrr_at_5
    assert reports["lexical"].summary.mrr_at_5 >= reports["hybrid"].summary.mrr_at_5

    markdown = render_ablation_markdown(reports)
    payload = json.loads(render_ablation_json(reports))
    assert "| hybrid | 0.900 | 1.000 | 1.000 | 0.950 |" in markdown
    assert set(payload["strategies"]) == {"dense", "lexical", "hybrid"}
