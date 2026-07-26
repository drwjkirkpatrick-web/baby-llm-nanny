"""Test the report module — report building and formatting.

Uses mock data, no Ollama connection needed.
"""

import json
import os
import tempfile
import pytest
from baby_llm_nanny.report import (
    build_results, format_terminal_report, save_json_report,
    RunReport, PromptResult, CategorySummary,
)
from baby_llm_nanny.evaluator import EvalResult
from baby_llm_nanny.runner import ModelResponse
from baby_llm_nanny.prompts.prompts import TestPrompt, PROMPTS


def _make_mock_data():
    """Create mock prompts, responses, and evaluations for testing."""
    prompts = PROMPTS[:5]
    responses = [
        ModelResponse(model="test-model", prompt=p.prompt, response="mock response",
                      response_time_sec=1.0)
        for p in prompts
    ]
    evaluations = [
        EvalResult(passed=True, score=1.0, detail="Test passed", extracted="mock"),
        EvalResult(passed=True, score=1.0, detail="Test passed", extracted="mock"),
        EvalResult(passed=False, score=0.0, detail="Test failed", extracted="mock"),
        EvalResult(passed=True, score=0.5, detail="Partial", extracted="mock"),
        EvalResult(passed=True, score=1.0, detail="Test passed", extracted="mock"),
    ]
    return prompts, responses, evaluations


class TestBuildResults:
    def test_build_report(self):
        prompts, responses, evaluations = _make_mock_data()
        report = build_results(
            model="test-model", prompts=prompts, responses=responses,
            evaluations=evaluations, system_prompt_name="none",
        )
        assert report.model == "test-model"
        assert report.total_prompts == 5
        assert report.total_passed == 4  # 4 have score 1.0 or 0.5 → passed=True

    def test_report_properties(self):
        prompts, responses, evaluations = _make_mock_data()
        report = build_results("test-model", prompts, responses, evaluations)
        assert report.overall_score >= 0.0
        assert 0.0 <= report.overall_pass_rate <= 1.0
        assert report.total_time_sec > 0

    def test_category_summaries_built(self):
        prompts, responses, evaluations = _make_mock_data()
        report = build_results("test-model", prompts, responses, evaluations)
        assert len(report.category_summaries) > 0
        for cat, cs in report.category_summaries.items():
            assert cs.total > 0
            assert cs.passed + cs.partial + cs.failed == cs.total


class TestFormatTerminalReport:
    def test_report_has_header(self):
        prompts, responses, evaluations = _make_mock_data()
        report = build_results("test-model", prompts, responses, evaluations)
        text = format_terminal_report(report)
        assert "baby-llm-nanny" in text
        assert "test-model" in text

    def test_report_has_overall_section(self):
        prompts, responses, evaluations = _make_mock_data()
        report = build_results("test-model", prompts, responses, evaluations)
        text = format_terminal_report(report)
        assert "OVERALL" in text

    def test_report_has_category_section(self):
        prompts, responses, evaluations = _make_mock_data()
        report = build_results("test-model", prompts, responses, evaluations)
        text = format_terminal_report(report)
        assert "BY CATEGORY" in text

    def test_verbose_shows_details(self):
        prompts, responses, evaluations = _make_mock_data()
        report = build_results("test-model", prompts, responses, evaluations)
        text = format_terminal_report(report, verbose=True)
        assert "Prompt:" in text
        assert "Response:" in text

    def test_non_verbose_minimal(self):
        prompts, responses, evaluations = _make_mock_data()
        report = build_results("test-model", prompts, responses, evaluations)
        text = format_terminal_report(report, verbose=False)
        assert "--verbose" in text  # hint at bottom


class TestSaveJsonReport:
    def test_save_json(self):
        prompts, responses, evaluations = _make_mock_data()
        report = build_results("test-model", prompts, responses, evaluations)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_json_report(report, path)
            with open(path) as f:
                data = json.load(f)
            assert data["model"] == "test-model"
            assert "results" in data
            assert "category_summaries" in data
            assert len(data["results"]) == 5
        finally:
            os.unlink(path)

    def test_json_has_overall_scores(self):
        prompts, responses, evaluations = _make_mock_data()
        report = build_results("test-model", prompts, responses, evaluations)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_json_report(report, path)
            with open(path) as f:
                data = json.load(f)
            assert "overall_score" in data
            assert "overall_pass_rate" in data
        finally:
            os.unlink(path)

    def test_save_json_creates_dirs(self):
        prompts, responses, evaluations = _make_mock_data()
        report = build_results("test-model", prompts, responses, evaluations)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "report.json")
            save_json_report(report, path)
            assert os.path.exists(path)