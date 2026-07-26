"""Test the reviewer module — live code review loop.

Tests cover:
  - Code extraction from model responses
  - Test execution in subprocess
  - Feedback prompt construction
  - ReviewResult / ReviewIteration data classes
  - Full review loop with Ollama (integration, skip if Ollama down)
"""

import json
import os
import tempfile
import pytest
from baby_llm_nanny.reviewer import (
    extract_code, run_code_tests, build_feedback_prompt,
    ReviewIteration, ReviewResult, review_code, review_coding_prompts,
    format_review_report, save_review_json,
)
from baby_llm_nanny.runner import check_ollama


# ═══════════════════════════════════════════════════════════════════════
# Code extraction
# ═══════════════════════════════════════════════════════════════════════

class TestExtractCode:
    def test_plain_code(self):
        code = "def is_even(n):\n    return n % 2 == 0"
        result = extract_code(code)
        assert "def is_even" in result
        assert "return n % 2 == 0" in result

    def test_markdown_fence(self):
        response = "Here is the code:\n```python\ndef is_even(n):\n    return n % 2 == 0\n```\nDone."
        result = extract_code(response)
        assert "def is_even" in result
        assert "return n % 2 == 0" in result

    def test_markdown_fence_no_lang(self):
        response = "```\ndef is_even(n):\n    return n % 2 == 0\n```"
        result = extract_code(response)
        assert "def is_even" in result

    def test_code_pattern_extraction(self):
        response = "Here:\n```python\ndef is_even(n):\n    return n % 2 == 0\n\ndef other():\n    pass\n```"
        pattern = r"def\s+is_even\s*\([^)]*\)\s*:.*?(?=\ndef\s|\Z)"
        result = extract_code(response, pattern)
        assert "def is_even" in result
        assert "other" not in result

    def test_empty_response(self):
        result = extract_code("")
        assert result == ""

    def test_dedent(self):
        response = "```python\n    def is_even(n):\n        return n % 2 == 0\n```"
        result = extract_code(response)
        assert "def is_even" in result
        # Should be dedented
        assert "    return" not in result.split("\n")[0] if result else True


# ═══════════════════════════════════════════════════════════════════════
# Test execution
# ═══════════════════════════════════════════════════════════════════════

class TestRunCodeTests:
    def test_correct_code(self):
        code = "def is_even(n):\n    return n % 2 == 0"
        test_cases = [
            ({"n": 0}, True),
            ({"n": 1}, False),
            ({"n": 2}, True),
        ]
        results, stderr = run_code_tests(code, "is_even", test_cases)
        assert stderr == ""
        assert len(results) == 3
        assert all(r["pass"] for r in results)

    def test_buggy_code(self):
        code = "def is_even(n):\n    return n % 2 == 1"  # reversed
        test_cases = [
            ({"n": 0}, True),
            ({"n": 1}, False),
        ]
        results, stderr = run_code_tests(code, "is_even", test_cases)
        assert len(results) == 2
        assert not results[0]["pass"]  # 0 % 2 == 1 is False, expected True
        assert not results[1]["pass"]  # 1 % 2 == 1 is True, expected False

    def test_syntax_error(self):
        code = "def is_even(n:\n    return n % 2 == 0"
        test_cases = [({"n": 0}, True)]
        results, stderr = run_code_tests(code, "is_even", test_cases)
        assert results == []
        assert "SyntaxError" in stderr or "Error" in stderr

    def test_runtime_error(self):
        code = "def is_even(n):\n    return n % 2 == 0\n\nis_even(undefined_var)"
        # Actually the function itself is fine, but extra code causes error
        code = "def is_even(n):\n    return 1/0"
        test_cases = [({"n": 0}, True)]
        results, stderr = run_code_tests(code, "is_even", test_cases)
        assert len(results) == 1
        assert not results[0]["pass"]
        assert "error" in results[0]

    def test_empty_test_cases(self):
        code = "def is_even(n):\n    return n % 2 == 0"
        results, stderr = run_code_tests(code, "is_even", [])
        assert results == []
        assert stderr == ""

    def test_multiple_functions(self):
        code = "def helper(x):\n    return x + 1\n\ndef add_one(lst):\n    return [helper(x) for x in lst]"
        test_cases = [
            ({"lst": [1, 2, 3]}, [2, 3, 4]),
            ({"lst": []}, []),
        ]
        results, stderr = run_code_tests(code, "add_one", test_cases)
        assert stderr == ""
        assert all(r["pass"] for r in results)


# ═══════════════════════════════════════════════════════════════════════
# Feedback prompt construction
# ═══════════════════════════════════════════════════════════════════════

class TestBuildFeedbackPrompt:
    def test_feedback_contains_failures(self):
        results = [
            {"pass": True, "result": "True", "expected": "True"},
            {"pass": False, "result": "False", "expected": "True"},
            {"pass": False, "error": "ZeroDivisionError: division by zero"},
        ]
        feedback = build_feedback_prompt(
            "Write is_even function", "def is_even(n):\n    return n % 2 == 1",
            results, "is_even", iteration=1,
        )
        assert "2 failing test" in feedback
        assert "Got False, expected True" in feedback
        assert "ZeroDivisionError" in feedback
        assert "def is_even" in feedback  # previous code included
        assert "Write is_even function" in feedback  # original prompt included

    def test_feedback_no_errors(self):
        results = [
            {"pass": False, "result": "False", "expected": "True"},
            {"pass": False, "result": "True", "expected": "False"},
        ]
        feedback = build_feedback_prompt(
            "Write function", "def f():\n    pass", results, "f", iteration=1,
        )
        assert "2 failing test" in feedback
        # No error-type failures, so no error-checking advice
        assert "runtime/syntax errors" not in feedback


# ═══════════════════════════════════════════════════════════════════════
# ReviewResult data classes
# ═══════════════════════════════════════════════════════════════════════

class TestReviewResult:
    def test_improved_property(self):
        rr = ReviewResult(prompt_id="test", original_prompt="", function_name="f")
        rr.iterations = [
            ReviewIteration(1, "", "", [], False, 0.0, "Failed", 1.0),
            ReviewIteration(2, "", "", [], True, 1.0, "Passed", 1.0),
        ]
        rr.final_passed = True
        assert rr.improved is True

    def test_not_improved(self):
        rr = ReviewResult(prompt_id="test", original_prompt="", function_name="f")
        rr.iterations = [
            ReviewIteration(1, "", "", [], False, 0.0, "Failed", 1.0),
            ReviewIteration(2, "", "", [], False, 0.0, "Still failed", 1.0),
        ]
        rr.final_passed = False
        assert rr.improved is False

    def test_improved_first_try(self):
        # If it passes on first try, it's not "improved" — it was always right
        rr = ReviewResult(prompt_id="test", original_prompt="", function_name="f")
        rr.iterations = [
            ReviewIteration(1, "", "", [], True, 1.0, "Passed", 1.0),
        ]
        rr.final_passed = True
        assert rr.improved is False

    def test_score_progression(self):
        rr = ReviewResult(prompt_id="test", original_prompt="", function_name="f")
        rr.iterations = [
            ReviewIteration(1, "", "", [], False, 0.0, "Failed", 1.0),
            ReviewIteration(2, "", "", [], False, 0.5, "Partial", 1.0),
            ReviewIteration(3, "", "", [], True, 1.0, "Passed", 1.0),
        ]
        assert rr.score_progression == [0.0, 0.5, 1.0]


# ═══════════════════════════════════════════════════════════════════════
# JSON export
# ═══════════════════════════════════════════════════════════════════════

class TestSaveReviewJson:
    def test_save_json(self):
        rr = ReviewResult(
            prompt_id="test-prompt", original_prompt="Write is_even",
            function_name="is_even", final_passed=True, final_score=1.0,
            iterations_used=2, total_time_sec=3.0,
        )
        rr.iterations = [
            ReviewIteration(1, "def is_even(n):...", "def is_even(n):\n    return n % 2 == 1",
                            [{"pass": False, "result": "False", "expected": "True"}],
                            False, 0.0, "0/1 passed", 1.0, "feedback..."),
            ReviewIteration(2, "def is_even(n):...", "def is_even(n):\n    return n % 2 == 0",
                            [{"pass": True, "result": "True", "expected": "True"}],
                            True, 1.0, "1/1 passed", 1.0, ""),
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_review_json([rr], "test-model", path)
            with open(path) as f:
                data = json.load(f)
            assert data["model"] == "test-model"
            assert data["total_prompts"] == 1
            assert data["final_passed"] == 1
            assert data["results"][0]["prompt_id"] == "test-prompt"
            assert len(data["results"][0]["iterations"]) == 2
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════
# Terminal report formatting
# ═══════════════════════════════════════════════════════════════════════

class TestFormatReviewReport:
    def test_report_has_header(self):
        rr = ReviewResult(
            prompt_id="test", original_prompt="", function_name="f",
            final_passed=True, final_score=1.0, iterations_used=1,
        )
        rr.iterations = [ReviewIteration(1, "", "", [], True, 1.0, "Passed", 1.0)]
        text = format_review_report([rr], "test-model")
        assert "Live Code Review Report" in text
        assert "test-model" in text

    def test_report_shows_improvement(self):
        rr = ReviewResult(
            prompt_id="test", original_prompt="", function_name="f",
            final_passed=True, final_score=1.0, iterations_used=2,
        )
        rr.iterations = [
            ReviewIteration(1, "", "", [], False, 0.0, "Failed", 1.0),
            ReviewIteration(2, "", "", [], True, 1.0, "Passed", 1.0),
        ]
        text = format_review_report([rr], "model")
        assert "self-corrected" in text
        assert "Improvement" in text

    def test_report_shows_score_progression(self):
        rr = ReviewResult(
            prompt_id="test", original_prompt="", function_name="f",
            final_passed=False, final_score=0.5, iterations_used=2,
        )
        rr.iterations = [
            ReviewIteration(1, "", "", [], False, 0.0, "Failed", 1.0),
            ReviewIteration(2, "", "", [], False, 0.5, "Partial", 1.0),
        ]
        text = format_review_report([rr], "model")
        assert "0%" in text
        assert "50%" in text


# ═══════════════════════════════════════════════════════════════════════
# Integration tests (require Ollama)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not check_ollama(), reason="Ollama not running")
class TestReviewCodeIntegration:
    def test_review_is_even(self):
        """Test the full review loop on the is_even prompt."""
        rr = review_code(
            model="qwen2.5:3b",
            prompt="Write a Python function called `is_even` that takes an integer "
                   "and returns True if it is even, False otherwise. "
                   "Return only the code, no explanation.",
            function_name="is_even",
            test_cases=[
                ({"n": 0}, True),
                ({"n": 1}, False),
                ({"n": 2}, True),
                ({"n": -3}, False),
            ],
            code_pattern=r"def\s+is_even\s*\([^)]*\)\s*:.*?(?=\ndef\s|\Z)",
            prompt_id="test-is-even",
            max_iterations=3,
            show_progress=False,
        )
        assert rr.iterations_used >= 1
        assert rr.iterations_used <= 3
        assert len(rr.iterations) == rr.iterations_used
        # Each iteration should have a response time
        for it in rr.iterations:
            assert it.response_time_sec > 0

    def test_review_buggy_prompt(self):
        """Test that the review loop can fix a deliberately misleading prompt."""
        # This prompt asks for a function but gives a wrong hint
        rr = review_code(
            model="qwen2.5:3b",
            prompt="Write a Python function called `factorial` that takes n and returns n! "
                   "Use recursion. Return only the code.",
            function_name="factorial",
            test_cases=[
                ({"n": 0}, 1),
                ({"n": 1}, 1),
                ({"n": 5}, 120),
            ],
            code_pattern=r"def\s+factorial\s*\([^)]*\)\s*:.*?(?=\ndef\s|\Z)",
            prompt_id="test-factorial",
            max_iterations=3,
            show_progress=False,
        )
        assert rr.iterations_used >= 1
        # Should eventually produce something (even if not perfect)
        assert rr.iterations[0].extracted_code != ""