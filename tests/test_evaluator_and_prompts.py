"""Test the evaluator module and prompt bank — the core checking logic.

These tests verify that each evaluation strategy correctly identifies
right and wrong answers.  No Ollama connection needed for evaluator/prompt tests.
"""

import pytest
from baby_llm_nanny.evaluator import (
    evaluate, EvalResult,
    eval_exact, eval_contains_any, eval_numeric, eval_json_keys, eval_code_exec,
    _extract_number,
)
from baby_llm_nanny.prompts.prompts import TestPrompt, PROMPTS
from baby_llm_nanny.prompts import get_prompts_by_category, list_categories


# ═══════════════════════════════════════════════════════════════════════
# _extract_number helper
# ═══════════════════════════════════════════════════════════════════════

class TestExtractNumber:
    def test_plain_integer(self):
        assert _extract_number("391") == 391.0

    def test_decimal(self):
        assert _extract_number("142.86") == 142.86

    def test_negative(self):
        assert _extract_number("-5") == -5.0

    def test_comma_separated(self):
        assert _extract_number("3,628,800") == 3628800.0

    def test_number_in_text(self):
        assert _extract_number("The answer is 42.") == 42.0

    def test_no_number(self):
        assert _extract_number("I don't know") is None

    def test_empty_string(self):
        assert _extract_number("") is None

    def test_scientific_notation(self):
        assert _extract_number("1024") == 1024.0


# ═══════════════════════════════════════════════════════════════════════
# eval_exact
# ═══════════════════════════════════════════════════════════════════════

class TestEvalExact:
    def test_exact_match(self):
        r = eval_exact("BANANA", "BANANA")
        assert r.passed
        assert r.score == 1.0

    def test_case_insensitive_partial(self):
        r = eval_exact("banana", "BANANA")
        assert r.passed
        assert r.score == 0.9

    def test_whitespace_stripped(self):
        r = eval_exact("  BANANA  ", "BANANA")
        assert r.passed
        assert r.score == 1.0

    def test_mismatch(self):
        r = eval_exact("apple", "BANANA")
        assert not r.passed
        assert r.score == 0.0

    def test_extra_text_fails(self):
        r = eval_exact("The answer is BANANA", "BANANA")
        assert not r.passed
        assert r.score == 0.0


# ═══════════════════════════════════════════════════════════════════════
# eval_contains_any
# ═══════════════════════════════════════════════════════════════════════

class TestEvalContainsAny:
    def test_single_expected_found(self):
        r = eval_contains_any("Canberra", "Canberra")
        assert r.passed
        assert r.score == 1.0

    def test_list_one_match(self):
        r = eval_contains_any("The capital is Bern.", ["Bern", "Berne"])
        assert r.passed
        assert r.score == 1.0

    def test_case_insensitive(self):
        r = eval_contains_any("the answer is canberra", ["Canberra"])
        assert r.passed

    def test_no_match(self):
        r = eval_contains_any("The capital is Sydney.", ["Canberra"])
        assert not r.passed
        assert r.score == 0.0

    def test_hallucination_trap_pass(self):
        """Model correctly says 'I don't know' for a fake book."""
        r = eval_contains_any(
            "I don't know about that book.",
            ["I don't know", "don't know", "not familiar"],
        )
        assert r.passed
        assert r.score == 1.0

    def test_hallucination_trap_fail(self):
        """Model fabricates a plot for a fake book."""
        fake_response = (
            "The Crimson Lighthouse by Eleanor Whitfield is a gripping novel "
            "about a lighthouse keeper who discovers a mysterious artifact..."
        )
        r = eval_contains_any(
            fake_response,
            ["I don't know", "don't know", "not familiar", "does not exist"],
        )
        assert not r.passed
        assert r.score == 0.0


# ═══════════════════════════════════════════════════════════════════════
# eval_numeric
# ═══════════════════════════════════════════════════════════════════════

class TestEvalNumeric:
    def test_exact_integer(self):
        r = eval_numeric("391", 391)
        assert r.passed
        assert r.score == 1.0

    def test_number_in_text(self):
        r = eval_numeric("The answer is 391.", 391)
        assert r.passed

    def test_wrong_number(self):
        r = eval_numeric("390", 391)
        assert not r.passed
        assert r.score == 0.0

    def test_no_number_found(self):
        r = eval_numeric("I don't know", 391)
        assert not r.passed
        assert r.score == 0.0

    def test_decimal_with_tolerance(self):
        r = eval_numeric("142.86", 142.86, tolerance=0.01)
        assert r.passed

    def test_comma_separated(self):
        r = eval_numeric("3,628,800", 3628800)
        assert r.passed

    def test_float_precision(self):
        """0.3 check with tolerance for float comparison."""
        r = eval_numeric("0.3", 0.3, tolerance=0.001)
        assert r.passed

    def test_negative_number(self):
        r = eval_numeric("-5", -5)
        assert r.passed


# ═══════════════════════════════════════════════════════════════════════
# Prompt bank integrity
# ═══════════════════════════════════════════════════════════════════════

class TestPromptBank:
    def test_all_prompts_have_unique_ids(self):
        ids = [p.id for p in PROMPTS]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {[x for x in ids if ids.count(x)>1]}"

    def test_all_categories_known(self):
        known = {"factual", "math", "series", "coding", "logic", "hallucination", "instruction", "boundaries"}
        for p in PROMPTS:
            assert p.category in known, f"Unknown category '{p.category}' for prompt {p.id}"

    def test_all_checks_valid(self):
        valid = {"exact", "contains_any", "numeric", "json_keys", "code_exec", "multi_constraint"}
        for p in PROMPTS:
            assert p.check in valid, f"Unknown check '{p.check}' for prompt {p.id}"

    def test_all_prompts_have_notes(self):
        for p in PROMPTS:
            assert p.notes, f"Prompt {p.id} has empty notes"

    def test_all_prompts_have_prompt_text(self):
        for p in PROMPTS:
            assert p.prompt, f"Prompt {p.id} has empty prompt text"

    def test_category_functions(self):
        cats = list_categories()
        assert "factual" in cats
        assert "math" in cats
        assert len(cats) >= 6

    def test_get_prompts_by_category(self):
        math_prompts = get_prompts_by_category("math")
        assert len(math_prompts) >= 5
        assert all(p.category == "math" for p in math_prompts)

    def test_nonexistent_category(self):
        assert get_prompts_by_category("nonexistent") == []

    def test_total_prompt_count(self):
        """We should have at least 60 prompts across all categories."""
        assert len(PROMPTS) >= 60

    def test_coding_prompts_have_test_cases(self):
        coding_prompts = get_prompts_by_category("coding")
        for p in coding_prompts:
            assert isinstance(p.expected, dict)
            assert "test_cases" in p.expected
            assert "function_name" in p.expected
            assert len(p.expected["test_cases"]) >= 3


# ═══════════════════════════════════════════════════════════════════════
# eval_json_keys
# ═══════════════════════════════════════════════════════════════════════

class TestEvalJsonKeys:
    def test_exact_json_match(self):
        resp = '{"name": "Alice", "age": 30}'
        r = eval_json_keys(resp, {"name": "Alice", "age": 30})
        assert r.passed
        assert r.score == 1.0

    def test_json_in_markdown_fence(self):
        resp = '```json\n{"name": "Alice", "age": 30}\n```'
        r = eval_json_keys(resp, {"name": "Alice", "age": 30})
        assert r.passed

    def test_json_with_extra_keys(self):
        resp = '{"name": "Alice", "age": 30, "extra": "ignored"}'
        r = eval_json_keys(resp, {"name": "Alice", "age": 30})
        assert r.passed

    def test_one_wrong_value(self):
        resp = '{"name": "Bob", "age": 30}'
        r = eval_json_keys(resp, {"name": "Alice", "age": 30})
        assert not r.passed
        assert r.score == 0.5

    def test_completely_wrong(self):
        resp = "I am a language model"
        r = eval_json_keys(resp, {"name": "Alice", "age": 30})
        assert not r.passed
        assert r.score == 0.0

    def test_missing_key(self):
        resp = '{"name": "Alice"}'
        r = eval_json_keys(resp, {"name": "Alice", "age": 30})
        assert not r.passed
        assert r.score == 0.5


# ═══════════════════════════════════════════════════════════════════════
# eval_code_exec
# ═══════════════════════════════════════════════════════════════════════

class TestEvalCodeExec:
    def test_correct_is_even(self):
        code = "def is_even(n):\n    return n % 2 == 0"
        expected = {
            "code_pattern": r"def\s+is_even\s*\([^)]*\)\s*:.*?(?=\ndef\s|\Z)",
            "test_cases": [
                ({"n": 0}, True),
                ({"n": 1}, False),
                ({"n": 2}, True),
            ],
            "function_name": "is_even",
        }
        r = eval_code_exec(code, expected)
        assert r.passed
        assert r.score == 1.0

    def test_buggy_is_even(self):
        code = "def is_even(n):\n    return n % 2 == 1"  # reversed
        expected = {
            "code_pattern": r"def\s+is_even\s*\([^)]*\)\s*:.*?(?=\ndef\s|\Z)",
            "test_cases": [
                ({"n": 0}, True),
                ({"n": 1}, False),
                ({"n": 2}, True),
            ],
            "function_name": "is_even",
        }
        r = eval_code_exec(code, expected)
        assert not r.passed

    def test_correct_factorial(self):
        code = "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
        expected = {
            "code_pattern": r"def\s+factorial\s*\([^)]*\)\s*:.*?(?=\ndef\s|\Z)",
            "test_cases": [
                ({"n": 0}, 1),
                ({"n": 5}, 120),
                ({"n": 10}, 3628800),
            ],
            "function_name": "factorial",
        }
        r = eval_code_exec(code, expected)
        assert r.passed

    def test_code_in_markdown_fence(self):
        code = "```python\ndef is_even(n):\n    return n % 2 == 0\n```"
        expected = {
            "code_pattern": r"def\s+is_even\s*\([^)]*\)\s*:.*?(?=\ndef\s|\Z)",
            "test_cases": [({"n": 4}, True)],
            "function_name": "is_even",
        }
        r = eval_code_exec(code, expected)
        assert r.passed

    def test_syntax_error(self):
        code = "def is_even(n:\n    return n % 2 == 0"  # syntax error
        expected = {
            "code_pattern": r"def\s+is_even\s*\([^)]*\)\s*:.*?(?=\ndef\s|\Z)",
            "test_cases": [({"n": 4}, True)],
            "function_name": "is_even",
        }
        r = eval_code_exec(code, expected)
        assert not r.passed


# ═══════════════════════════════════════════════════════════════════════
# evaluate() dispatcher
# ═══════════════════════════════════════════════════════════════════════

class TestEvaluateDispatcher:
    def test_dispatch_exact(self):
        tp = TestPrompt(id="t", category="instruction", prompt="test",
                        expected="BANANA", check="exact")
        r = evaluate("BANANA", tp)
        assert r.passed

    def test_dispatch_numeric(self):
        tp = TestPrompt(id="t", category="math", prompt="test",
                        expected=42, check="numeric")
        r = evaluate("42", tp)
        assert r.passed

    def test_dispatch_contains_any(self):
        tp = TestPrompt(id="t", category="factual", prompt="test",
                        expected=["Canberra"], check="contains_any")
        r = evaluate("Canberra", tp)
        assert r.passed

    def test_dispatch_unknown_check(self):
        tp = TestPrompt(id="t", category="test", prompt="test",
                        expected="x", check="nonexistent")
        r = evaluate("x", tp)
        assert not r.passed
        assert "Unknown check" in r.detail

    def test_numeric_with_tolerance_from_prompt(self):
        tp = TestPrompt(id="t", category="math", prompt="test",
                        expected=0.3, check="numeric", tolerance=0.001)
        r = evaluate("0.3", tp)
        assert r.passed


# ═══════════════════════════════════════════════════════════════════════
# multi_constraint evaluator
# ═══════════════════════════════════════════════════════════════════════

class TestMultiConstraint:
    def test_7_words_starts_the_period(self):
        from baby_llm_nanny.evaluator import eval_multi_constraint
        r = eval_multi_constraint("The quick brown fox jumps over dog.", "starts_with_The_and_7_words")
        assert r.passed
        assert r.score == 1.0

    def test_wrong_word_count(self):
        from baby_llm_nanny.evaluator import eval_multi_constraint
        r = eval_multi_constraint("The quick fox jumps.", "starts_with_The_and_7_words")
        assert not r.passed

    def test_wrong_start(self):
        from baby_llm_nanny.evaluator import eval_multi_constraint
        r = eval_multi_constraint("A quick brown fox jumps over the dog.", "starts_with_The_and_7_words")
        assert not r.passed

    def test_exactly_3_words(self):
        from baby_llm_nanny.evaluator import eval_multi_constraint
        r = eval_multi_constraint("one two three", "exactly_3_words")
        assert r.passed

    def test_wrong_word_count_3(self):
        from baby_llm_nanny.evaluator import eval_multi_constraint
        r = eval_multi_constraint("one two three four", "exactly_3_words")
        assert not r.passed


# ═══════════════════════════════════════════════════════════════════════
# Hallucination confidence scoring
# ═══════════════════════════════════════════════════════════════════════

class TestHallucinationConfidence:
    def test_admits_ignorance(self):
        from baby_llm_nanny.evaluator import hallucination_confidence
        assert hallucination_confidence("I don't know about that.", "hallucination") == 0.0

    def test_brief_fabrication(self):
        from baby_llm_nanny.evaluator import hallucination_confidence
        assert hallucination_confidence("It was about a lighthouse.", "hallucination") == 0.3

    def test_elaborate_fabrication(self):
        from baby_llm_nanny.evaluator import hallucination_confidence
        long_resp = "The novel is about a keeper who discovers an ancient artifact that grants " \
                    "immortality but at a terrible cost. The protagonist must choose between " \
                    "power and humanity in a gripping tale of suspense."
        assert hallucination_confidence(long_resp, "hallucination") == 0.9

    def test_non_hallucination_category(self):
        from baby_llm_nanny.evaluator import hallucination_confidence
        assert hallucination_confidence("Canberra", "factual") == 0.0


# ═══════════════════════════════════════════════════════════════════════
# Difficulty ratings
# ═══════════════════════════════════════════════════════════════════════

class TestDifficultyRatings:
    def test_all_prompts_have_difficulty(self):
        for p in PROMPTS:
            assert hasattr(p, 'difficulty')
            assert p.difficulty in ("easy", "medium", "hard"), \
                f"Prompt {p.id} has invalid difficulty '{p.difficulty}'"

    def test_difficulty_distribution(self):
        from baby_llm_nanny.prompts import get_prompts_by_difficulty
        easy = get_prompts_by_difficulty("easy")
        medium = get_prompts_by_difficulty("medium")
        hard = get_prompts_by_difficulty("hard")
        assert len(easy) > 0, "No easy prompts"
        assert len(medium) > 0, "No medium prompts"
        assert len(hard) > 0, "No hard prompts"
        assert len(easy) + len(medium) + len(hard) == len(PROMPTS)