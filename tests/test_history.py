"""Test the history module — SQLite trend tracking.

Uses a temporary database, no real Ollama connection needed.
"""

import os
import tempfile
import pytest
from baby_llm_nanny.history import (
    init_db, save_run_to_db, list_runs, format_trend_report,
    save_retry_results, get_retry_stats,
)
from baby_llm_nanny.report import build_results
from baby_llm_nanny.evaluator import EvalResult
from baby_llm_nanny.runner import ModelResponse
from baby_llm_nanny.prompts.prompts import TestPrompt, PROMPTS


@pytest.fixture
def tmp_db():
    """Provide a temporary SQLite DB path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    os.unlink(path)  # Remove the file so init_db creates fresh
    yield path
    if os.path.exists(path):
        os.unlink(path)


def _make_mock_report(model="test-model"):
    """Create a mock RunReport for testing."""
    prompts = PROMPTS[:5]
    responses = [
        ModelResponse(model=model, prompt=p.prompt, response="mock response",
                      response_time_sec=1.0, eval_count=10, prompt_eval_count=5)
        for p in prompts
    ]
    evaluations = [
        EvalResult(passed=True, score=1.0, detail="Test passed", extracted="mock"),
        EvalResult(passed=True, score=1.0, detail="Test passed", extracted="mock"),
        EvalResult(passed=False, score=0.0, detail="Test failed", extracted="mock"),
        EvalResult(passed=True, score=0.5, detail="Partial", extracted="mock"),
        EvalResult(passed=True, score=1.0, detail="Test passed", extracted="mock"),
    ]
    return build_results(model, prompts, responses, evaluations)


class TestInitDb:
    def test_init_db_creates_file(self, tmp_db):
        path = init_db(tmp_db)
        assert os.path.exists(path)

    def test_init_db_idempotent(self, tmp_db):
        init_db(tmp_db)
        path = init_db(tmp_db)  # Should not error
        assert os.path.exists(path)


class TestSaveRun:
    def test_save_and_list_run(self, tmp_db):
        report = _make_mock_report("test-model")
        run_id = save_run_to_db(report, tmp_db)
        assert run_id > 0

        runs = list_runs(tmp_db)
        assert len(runs) == 1
        assert runs[0]["model"] == "test-model"
        assert runs[0]["total_prompts"] == 5

    def test_save_multiple_runs(self, tmp_db):
        report1 = _make_mock_report("model-a")
        report2 = _make_mock_report("model-b")
        save_run_to_db(report1, tmp_db)
        save_run_to_db(report2, tmp_db)

        runs = list_runs(tmp_db)
        assert len(runs) == 2

    def test_list_runs_by_model(self, tmp_db):
        report1 = _make_mock_report("model-a")
        report2 = _make_mock_report("model-b")
        save_run_to_db(report1, tmp_db)
        save_run_to_db(report2, tmp_db)

        runs = list_runs(tmp_db, model="model-a")
        assert len(runs) == 1
        assert runs[0]["model"] == "model-a"


class TestRetryResults:
    def test_save_and_get_retry_stats(self, tmp_db):
        report = _make_mock_report("test-model")
        run_id = save_run_to_db(report, tmp_db)

        retry_results = [
            {"attempt": 1, "passed": True, "score": 1.0, "response": "answer1"},
            {"attempt": 2, "passed": False, "score": 0.0, "response": "answer2"},
            {"attempt": 3, "passed": True, "score": 1.0, "response": "answer3"},
        ]
        save_retry_results(run_id, "test-prompt", retry_results, tmp_db)

        stats = get_retry_stats(tmp_db, run_id)
        assert len(stats) == 1
        assert stats[0]["prompt_id"] == "test-prompt"
        assert stats[0]["attempts"] == 3
        assert stats[0]["passes"] == 2


class TestTrendReport:
    def test_empty_trend_report(self, tmp_db):
        text = format_trend_report(tmp_db)
        assert "No historical runs" in text

    def test_trend_report_with_data(self, tmp_db):
        report = _make_mock_report("model-a")
        save_run_to_db(report, tmp_db)

        text = format_trend_report(tmp_db)
        assert "Historical Trends" in text
        assert "model-a" in text