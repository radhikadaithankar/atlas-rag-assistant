from __future__ import annotations

from pathlib import Path

import pytest

from atlas.config import load_settings
from atlas.evaluation.answer_metrics import (
    FAITHFULNESS_NOT_GROUND_TRUTH,
    PendingScoreError,
    assign_metric,
    citation_form_ok,
    citation_precision,
    estimate_gemini_cost,
    linear_percentile,
)
from atlas.evaluation.dataset import load_questions, unique_quotes
from atlas.evaluation.report import render_report
from atlas.evaluation.retrieval_metrics import (
    containment,
    mean,
    recall_at_k,
    reciprocal_rank,
)
from atlas.evaluation.runner import RunFlags, execute, judge_supported, preflight
from atlas.generation.base import GenerationResult


def test_recall_and_mrr_are_hand_checked() -> None:
    retrieved = ["no", "the red buoy marks the channel", "later"]
    quotes = ["red buoy"]
    assert recall_at_k(retrieved, quotes, k=5) == 1.0
    assert reciprocal_rank(retrieved, quotes) == 0.5
    assert recall_at_k(["nothing"], ["red buoy"], k=5) == 0.0


def test_containment_counts_quotes_that_fit() -> None:
    score, found, total = containment(
        ["short quote here"], ["short quote", "this quote is not present"]
    )
    assert found == 1
    assert total == 2
    assert score == 0.5


def test_mean_and_percentile() -> None:
    assert mean([1.0, 3.0]) == 2.0
    assert linear_percentile([0.0, 10.0, 20.0], 0.5) == 10.0


def test_citation_precision_abstain_is_zero() -> None:
    assert (
        citation_precision(
            abstained=True, citation_quotes=["quote"], gold_quotes=["quote"]
        )
        == 0.0
    )
    assert (
        citation_precision(
            abstained=False, citation_quotes=["quote"], gold_quotes=["quote"]
        )
        == 1.0
    )


def test_citation_form() -> None:
    assert citation_form_ok(
        abstained=False,
        citations=[("a", "red buoy")],
        retrieved={"a": "the red buoy marks the channel"},
    )
    assert citation_form_ok(abstained=True, citations=[], retrieved={})
    assert not citation_form_ok(
        abstained=False,
        citations=[("missing", "red buoy")],
        retrieved={},
    )


def test_pending_row_refuses_a_number() -> None:
    row: dict[str, object] = {"status": "pending", "value": None}
    with pytest.raises(PendingScoreError):
        assign_metric(row, 1.0)


def test_gemini_cost_formula_matches_the_published_product() -> None:
    assert estimate_gemini_cost(46827, 5345) == pytest.approx(0.0274106)


def test_judge_can_mark_a_fluent_unsupported_answer() -> None:
    class Agree:
        def generate(self, _system: str, _user: str) -> GenerationResult:
            return GenerationResult('{"supported": true}', None, None, 1.0)

    assert judge_supported(
        "The clinic flew to Mars.", ["The clinic folds linen."], Agree()
    )
    assert FAITHFULNESS_NOT_GROUND_TRUTH


def test_provider_flags_do_not_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    target = tmp_path / "results.json"
    code = execute(
        ["--vector", "--hybrid", "--rerank", "--answers", "--results", str(target)]
    )
    assert code == 2
    assert not target.exists()
    message = preflight(RunFlags(vector=True), load_settings())
    assert message is not None
    assert "No score was written" in message


def test_dataset_has_fifty_three_questions() -> None:
    questions = load_questions(Path("data/evaluation/v1/questions.json"))
    assert len(questions) == 53
    assert sum(1 for question in questions if question.has_gold) == 45
    assert sum(1 for question in questions if not question.has_gold) == 8
    assert len(unique_quotes(questions)) == 39


def test_published_report_copies_measured_scores_and_labels_faithfulness() -> None:
    import json

    results = json.loads(
        Path("data/evaluation/v1/results.json").read_text(encoding="utf-8")
    )
    report = Path("docs/evaluation_report.md").read_text(encoding="utf-8")
    rendered = render_report(results)
    assert rendered == report
    assert "not ground truth" in report
    assert "0.9743589743589743" in report
    assert "0.0274106" in report
    assert "0.9111111111111111" in report
    assert "940.5741750015295" in report
    assert "The judge can agree with a fluent unsupported answer." in report
