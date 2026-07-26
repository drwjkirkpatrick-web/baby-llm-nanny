"""Evaluator — check model responses against known-correct answers.

Each evaluation strategy is a function that takes the model's response text
and the expected answer, and returns an EvalResult with:
  - passed:  bool — did the response fully pass?
  - score:   float in [0.0, 1.0] — 1.0 = perfect, 0.0 = wrong, 0.5 = partial
  - detail:  str — human-readable explanation of the verdict
  - extracted: str — what we extracted from the response for checking
"""

from __future__ import annotations

import re
import json as json_mod
from dataclasses import dataclass
from typing import Any

# Sentinel for code_exec evaluation — actual code execution happens in the
# caller (which has the model response text and the expected dict).  This keeps
# the evaluator module importable without subprocess side-effects at import time.


@dataclass
class EvalResult:
    """Result of evaluating a single response."""
    passed: bool
    score: float
    detail: str
    extracted: str | None = None


# ─────────────────────────────────────────────────────────────────────
# Helper: extract numbers from text
# ─────────────────────────────────────────────────────────────────────

def _extract_number(text: str) -> float | None:
    """Try to find a number in the response text.

    Handles:
      - Plain integers: "391"
      - Decimals: "142.86"
      - Negative: "-5"
      - Comma-separated: "3,628,800"
      - Scientific: "1e5"
      - Fractions in text: "142.857..." → 142.857
    Returns None if no number found.
    """
    if not text:
        return None

    text = text.strip()

    # Try direct float parse first (for exact "391" or "142.86")
    cleaned = text.replace(",", "").replace(" ", "")
    try:
        return float(cleaned)
    except ValueError:
        pass

    # Search for first number pattern in the text
    # Match integers, decimals, negatives, and comma-separated numbers
    patterns = [
        r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?",  # 3,628,800 or 1,234.56
        r"-?\d+\.?\d*",                      # 391 or 142.86 or -5
        r"-?\d+e-?\d+",                      # 1e5
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            val = m.group().replace(",", "")
            try:
                return float(val)
            except ValueError:
                continue

    return None


# ─────────────────────────────────────────────────────────────────────
# Evaluation strategies
# ─────────────────────────────────────────────────────────────────────

def eval_exact(response: str, expected: str) -> EvalResult:
    """Response must exactly match the expected string (case-sensitive, stripped)."""
    resp = response.strip()
    exp = expected.strip() if isinstance(expected, str) else str(expected).strip()
    if resp == exp:
        return EvalResult(passed=True, score=1.0, detail="Exact match", extracted=resp)
    # Check case-insensitive as partial
    if resp.lower() == exp.lower():
        return EvalResult(passed=True, score=0.9, detail="Match (case-insensitive)", extracted=resp)
    return EvalResult(
        passed=False, score=0.0,
        detail=f"Expected '{exp}', got '{resp}'", extracted=resp
    )


def eval_contains_any(response: str, expected: list[str] | str) -> EvalResult:
    """Response must contain at least one of the expected substrings (case-insensitive).

    For hallucination traps, the expected list includes phrases like "I don't know",
    "don't know", "cannot", etc. — if any of these appear, the model passed by
    correctly admitting ignorance.
    """
    resp_lower = response.lower().strip()
    if isinstance(expected, str):
        expected = [expected]

    for exp in expected:
        if exp.lower() in resp_lower:
            return EvalResult(
                passed=True, score=1.0,
                detail=f"Found expected phrase: '{exp}'",
                extracted=exp,
            )
    # For hallucination traps, if the model gives a long confident answer,
    # it's a hallucination — score 0
    return EvalResult(
        passed=False, score=0.0,
        detail=f"Response does not contain any expected answer. "
               f"Expected one of: {expected}. Got: '{response[:200]}'",
        extracted=response[:200],
    )


def eval_numeric(response: str, expected: float | int, tolerance: float = 0.0) -> EvalResult:
    """Response must contain a number matching the expected value within tolerance.

    Tolerance is absolute (|response - expected| <= tolerance).
    """
    extracted_num = _extract_number(response)
    if extracted_num is None:
        return EvalResult(
            passed=False, score=0.0,
            detail=f"No number found in response: '{response[:200]}'",
            extracted=None,
        )

    expected_num = float(expected)
    diff = abs(extracted_num - expected_num)

    if diff <= tolerance or diff < 0.001:  # exact match (with float slack)
        return EvalResult(
            passed=True, score=1.0,
            detail=f"Number match: {extracted_num} == {expected_num}",
            extracted=str(extracted_num),
        )
    elif tolerance > 0 and diff <= tolerance * 2:
        # Close but slightly outside tolerance — partial credit
        return EvalResult(
            passed=False, score=0.5,
            detail=f"Close: {extracted_num} vs expected {expected_num} "
                   f"(diff {diff:.4f}, tolerance {tolerance})",
            extracted=str(extracted_num),
        )
    else:
        return EvalResult(
            passed=False, score=0.0,
            detail=f"Wrong number: got {extracted_num}, expected {expected_num}",
            extracted=str(extracted_num),
        )


def eval_json_keys(response: str, expected: dict) -> EvalResult:
    """Response must be valid JSON with the expected key-value pairs.

    Small models often wrap JSON in markdown fences or add extra prose.
    We try to extract JSON from the response and compare keys.
    """
    # Try to extract JSON from the response
    # Models often add ```json ... ``` or prose around it
    json_text = response.strip()

    # Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
    if fence_match:
        json_text = fence_match.group(1).strip()

    # Try to find a bare { ... } block
    if not json_text.startswith("{"):
        brace_match = re.search(r"\{.*\}", response, re.DOTALL)
        if brace_match:
            json_text = brace_match.group(0)

    try:
        parsed = json_mod.loads(json_text)
    except json_mod.JSONDecodeError:
        return EvalResult(
            passed=False, score=0.0,
            detail=f"Response is not valid JSON: '{response[:200]}'",
            extracted=json_text[:200],
        )

    if not isinstance(parsed, dict):
        return EvalResult(
            passed=False, score=0.0,
            detail=f"JSON is not an object: {type(parsed).__name__}",
            extracted=json_text[:200],
        )

    # Check each expected key
    correct = 0
    total = len(expected)
    wrong_keys = []
    for key, val in expected.items():
        if key in parsed and parsed[key] == val:
            correct += 1
        else:
            got = parsed.get(key, "<missing>")
            wrong_keys.append(f"{key}: expected {val}, got {got}")

    score = correct / total
    passed = score == 1.0
    detail = f"JSON keys: {correct}/{total} correct"
    if wrong_keys:
        detail += f". Wrong: {'; '.join(wrong_keys)}"

    return EvalResult(passed=passed, score=score, detail=detail, extracted=json_text[:200])


# ─────────────────────────────────────────────────────────────────────
# Code execution evaluation
# ─────────────────────────────────────────────────────────────────────

def eval_code_exec(response: str, expected: dict) -> EvalResult:
    """Extract Python code from the response, execute it, and run test cases.

    Expected dict format:
        {
            "code_pattern": regex to extract the function definition,
            "test_cases": [({"arg": val}, expected_result), ...],
            "function_name": "function_name"
        }
    """
    import subprocess
    import textwrap

    # Extract code from response
    # Models often wrap code in markdown fences
    code_text = response.strip()

    # Try markdown fences first
    fence_match = re.search(r"```(?:python)?\s*(.*?)\s*```", response, re.DOTALL)
    if fence_match:
        code_text = fence_match.group(1).strip()

    # Try regex pattern to extract the function
    pattern = expected.get("code_pattern", "")
    func_name = expected.get("function_name", "")
    test_cases = expected.get("test_cases", [])

    # If no function found via pattern, try to use the full code block
    func_code = code_text
    if pattern:
        m = re.search(pattern, code_text, re.DOTALL)
        if m:
            func_code = m.group(0)
        # If pattern doesn't match but we have a code fence, use that

    # Dedent the code (models often indent inside fences)
    func_code = textwrap.dedent(func_code)

    # Build the test script
    test_script = func_code + "\n\n"
    test_script += f"# --- Auto-generated tests ---\n"
    test_script += f"import json\n"
    test_script += f"results = []\n"
    for i, (args, exp) in enumerate(test_cases):
        args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
        test_script += f"try:\n"
        test_script += f"    _result = {func_name}({args_str})\n"
        test_script += f"    _expected = {repr(exp)}\n"
        test_script += f"    _pass = _result == _expected\n"
        test_script += f"    results.append({{'pass': _pass, 'result': repr(_result), 'expected': repr(_expected)}})\n"
        test_script += f"except Exception as _e:\n"
        test_script += f"    results.append({{'pass': False, 'error': str(_e)}})\n"
    test_script += f"print(json.dumps(results))\n"

    # Execute in a subprocess for isolation
    try:
        proc = subprocess.run(
            ["python3", "-c", test_script],
            capture_output=True, text=True, timeout=10,
            cwd="/tmp",
        )
    except subprocess.TimeoutExpired:
        return EvalResult(
            passed=False, score=0.0,
            detail="Code execution timed out (10s)",
            extracted=func_code[:500],
        )
    except Exception as e:
        return EvalResult(
            passed=False, score=0.0,
            detail=f"Failed to execute code: {e}",
            extracted=func_code[:500],
        )

    if proc.returncode != 0:
        # Syntax error or runtime error before tests
        stderr = proc.stderr[:500] if proc.stderr else "(no stderr)"
        return EvalResult(
            passed=False, score=0.0,
            detail=f"Code error: {stderr}",
            extracted=func_code[:500],
        )

    # Parse test results
    try:
        results = json_mod.loads(proc.stdout.strip())
    except json_mod.JSONDecodeError:
        return EvalResult(
            passed=False, score=0.0,
            detail=f"Could not parse test output: '{proc.stdout[:200]}'",
            extracted=func_code[:500],
        )

    passed_count = sum(1 for r in results if r.get("pass"))
    total = len(results)
    score = passed_count / total if total else 0.0
    failed = [r for r in results if not r.get("pass")]

    if score == 1.0:
        detail = f"All {total} test cases passed"
    else:
        fail_details = []
        for r in failed:
            if "error" in r:
                fail_details.append(f"Error: {r['error']}")
            else:
                fail_details.append(f"Got {r.get('result')}, expected {r.get('expected')}")
        detail = f"{passed_count}/{total} passed. Failed: {'; '.join(fail_details)}"

    return EvalResult(
        passed=score == 1.0,
        score=score,
        detail=detail,
        extracted=func_code[:500],
    )


# ─────────────────────────────────────────────────────────────────────
# Multi-constraint evaluation
# ─────────────────────────────────────────────────────────────────────

def eval_multi_constraint(response: str, expected: str) -> EvalResult:
    """Evaluate responses against custom multi-constraint rules.

    Expected is a special string key indicating which constraints to check:
      - "starts_with_The_and_7_words": must start with 'The', have 7 words, end with '.'
      - "exactly_3_words": must have exactly 3 words
    """
    resp = response.strip()
    constraints_passed = 0
    constraints_total = 0
    details = []

    if expected == "starts_with_The_and_7_words":
        # Constraint 1: starts with 'The'
        constraints_total += 1
        if resp.startswith("The "):
            constraints_passed += 1
        else:
            details.append(f"Does not start with 'The' (starts with '{resp[:20]}')")

        # Constraint 2: exactly 7 words
        constraints_total += 1
        words = resp.split()
        if len(words) == 7:
            constraints_passed += 1
        else:
            details.append(f"Has {len(words)} words, expected 7")

        # Constraint 3: ends with period
        constraints_total += 1
        if resp.endswith("."):
            constraints_passed += 1
        else:
            details.append("Does not end with period")

    elif expected == "exactly_3_words":
        constraints_total += 1
        words = resp.split()
        if len(words) == 3:
            constraints_passed += 1
        else:
            details.append(f"Has {len(words)} words, expected 3")
    else:
        return EvalResult(passed=False, score=0.0,
                          detail=f"Unknown multi-constraint key: {expected}")

    score = constraints_passed / constraints_total if constraints_total else 0.0
    passed = score == 1.0
    detail = f"{constraints_passed}/{constraints_total} constraints met"
    if details:
        detail += ". " + "; ".join(details)

    return EvalResult(passed=passed, score=score, detail=detail, extracted=resp)


# ─────────────────────────────────────────────────────────────────────
# Hallucination confidence scoring
# ─────────────────────────────────────────────────────────────────────

def hallucination_confidence(response: str, category: str) -> float:
    """Estimate how confidently a model hallucinates.

    Returns a float 0.0–1.0 where higher = more confident fabrication.

    Heuristic: for hallucination-category prompts, if the model gives a long,
    confident answer instead of admitting ignorance, longer = more confident
    fabrication. For non-hallucination categories, returns 0.0.

    This is a proxy metric — response length correlates with confident
    fabrication in small models. A model that says "I don't know" is brief;
    a model that fabricates tends to elaborate.
    """
    if category != "hallucination":
        return 0.0

    resp = response.strip()
    if not resp:
        return 0.0

    # Check if the model admitted ignorance (good behavior)
    ignorance_phrases = [
        "i don't know", "don't know", "not familiar", "no information",
        "cannot", "not aware", "don't have", "doesn't exist",
        "not real", "fictional", "no record", "no such",
    ]
    resp_lower = resp.lower()
    for phrase in ignorance_phrases:
        if phrase in resp_lower:
            return 0.0  # Model correctly admitted ignorance → no fabrication

    # Model didn't admit ignorance → it's fabricating
    # Use response length as confidence proxy:
    #   < 50 chars  → 0.3 (brief fabrication)
    #   50-150 chars → 0.6 (moderate fabrication)
    #   > 150 chars → 0.9 (elaborate fabrication)
    length = len(resp)
    if length < 50:
        return 0.3
    elif length < 150:
        return 0.6
    else:
        return 0.9


# ─────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────

# Map check names to evaluator functions
EVALUATORS = {
    "exact": lambda resp, exp, **kw: eval_exact(resp, exp),
    "contains_any": lambda resp, exp, **kw: eval_contains_any(resp, exp),
    "numeric": lambda resp, exp, **kw: eval_numeric(resp, exp, kw.get("tolerance", 0.0)),
    "json_keys": lambda resp, exp, **kw: eval_json_keys(resp, exp),
    "code_exec": lambda resp, exp, **kw: eval_code_exec(resp, exp),
    "multi_constraint": lambda resp, exp, **kw: eval_multi_constraint(resp, exp),
}


def evaluate(response: str, tp) -> EvalResult:
    """Evaluate a model response against a TestPrompt.

    Args:
        response: The model's response text.
        tp:       TestPrompt dataclass with .check, .expected, .tolerance.

    Returns:
        EvalResult with passed/score/detail.
    """
    check = tp.check
    if check not in EVALUATORS:
        return EvalResult(
            passed=False, score=0.0,
            detail=f"Unknown check type: '{check}'",
        )
    evaluator_fn = EVALUATORS[check]
    return evaluator_fn(response, tp.expected, tolerance=tp.tolerance)