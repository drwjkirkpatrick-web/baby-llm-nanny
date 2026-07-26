"""Live code review module — iterative generate→test→feedback→regenerate loop.

Designed for hackathon environments where students use only a local Qwen 2.5
model.  The reviewer:

1. Sends a coding prompt to the local LLM.
2. Extracts the generated code.
3. Runs it against test cases in an isolated subprocess.
4. If tests fail, constructs specific feedback (which tests failed, what errors
   occurred) and re-queries the model with the feedback appended.
5. Repeats until all tests pass or max_iterations is reached.

This gets the absolute best out of small local models by catching their
mistakes and giving them a chance to self-correct — something a human reviewer
would do in a live hackathon setting.
"""

from __future__ import annotations

import re
import json
import subprocess
import textwrap
import time
from dataclasses import dataclass, field
from typing import Optional

from .runner import query_model, ModelResponse, check_ollama


# ─────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ReviewIteration:
    """A single iteration of the review loop."""
    iteration: int
    response: str
    extracted_code: str
    test_results: list[dict]
    passed: bool
    score: float
    detail: str
    response_time_sec: float
    feedback_given: str = ""  # The feedback prompt sent for the next iteration


@dataclass
class ReviewResult:
    """Complete result of a live code review session."""
    prompt_id: str
    original_prompt: str
    function_name: str
    iterations: list[ReviewIteration] = field(default_factory=list)
    final_passed: bool = False
    final_score: float = 0.0
    total_time_sec: float = 0.0
    iterations_used: int = 0

    @property
    def improved(self) -> bool:
        """True if the model went from failing to passing across iterations.

        Requires at least 2 iterations — if it passes on the first try,
        it was never broken, so it didn't 'improve'.
        """
        if len(self.iterations) < 2:
            return False
        first_passed = self.iterations[0].passed
        return (not first_passed) and self.final_passed

    @property
    def score_progression(self) -> list[float]:
        """Score at each iteration — shows whether the model is improving."""
        return [it.score for it in self.iterations]


# ─────────────────────────────────────────────────────────────────────
# Code extraction
# ─────────────────────────────────────────────────────────────────────

def extract_code(response: str, code_pattern: str = "") -> str:
    """Extract Python code from a model response.

    Tries markdown fences first, then regex pattern, then full response.
    """
    code_text = response.strip()

    # Try markdown fences
    fence_match = re.search(r"```(?:python)?\s*(.*?)\s*```", response, re.DOTALL)
    if fence_match:
        code_text = fence_match.group(1).strip()

    # Try regex pattern to extract the function
    if code_pattern:
        m = re.search(code_pattern, code_text, re.DOTALL)
        if m:
            code_text = m.group(0)

    # Dedent (models often indent inside fences)
    code_text = textwrap.dedent(code_text)
    return code_text


# ─────────────────────────────────────────────────────────────────────
# Test execution
# ─────────────────────────────────────────────────────────────────────

def run_code_tests(code: str, function_name: str,
                   test_cases: list[tuple]) -> tuple[list[dict], str]:
    """Execute code against test cases in an isolated subprocess.

    Returns (results_list, stderr_text).
    Each result dict has:
      - "pass": bool
      - "result": repr of actual result (or None on error)
      - "expected": repr of expected result
      - "error": str (only if an exception occurred)
    """
    test_script = code + "\n\n"
    test_script += "# --- Auto-generated tests ---\n"
    test_script += "import json\n"
    test_script += "results = []\n"
    for args, exp in test_cases:
        args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
        test_script += "try:\n"
        test_script += f"    _result = {function_name}({args_str})\n"
        test_script += f"    _expected = {repr(exp)}\n"
        test_script += f"    _pass = _result == _expected\n"
        test_script += "    results.append({'pass': _pass, 'result': repr(_result), 'expected': repr(_expected)})\n"
        test_script += "except Exception as _e:\n"
        test_script += "    results.append({'pass': False, 'error': str(_e)})\n"
    test_script += "print(json.dumps(results))\n"

    try:
        proc = subprocess.run(
            ["python3", "-c", test_script],
            capture_output=True, text=True, timeout=10,
            cwd="/tmp",
        )
    except subprocess.TimeoutExpired:
        return [], "Code execution timed out (10s)"
    except Exception as e:
        return [], f"Failed to execute code: {e}"

    if proc.returncode != 0:
        stderr = proc.stderr[:1000] if proc.stderr else "(no stderr)"
        return [], stderr

    try:
        results = json.loads(proc.stdout.strip())
    except json.JSONDecodeError:
        return [], f"Could not parse test output: '{proc.stdout[:200]}'"

    return results, ""


# ─────────────────────────────────────────────────────────────────────
# Feedback construction
# ─────────────────────────────────────────────────────────────────────

def build_feedback_prompt(original_prompt: str, code: str, results: list[dict],
                           function_name: str, iteration: int) -> str:
    """Construct a feedback prompt that tells the model what went wrong.

    The feedback is specific and actionable:
    - Which test cases failed
    - What the model returned vs what was expected
    - Any runtime/syntax errors
    - A clear instruction to fix the issues
    """
    failures = [r for r in results if not r.get("pass")]
    errors = [r for r in results if "error" in r]

    feedback = f"""
Your previous attempt had {len(failures)} failing test case(s) out of {len(results)} total.

Here is your previous code:
```python
{code}
```

Here are the specific failures that need to be fixed:
"""
    for i, r in enumerate(failures):
        if "error" in r:
            feedback += f"\n  Test {i+1}: ERROR — {r['error']}\n"
        else:
            feedback += f"\n  Test {i+1}: Got {r.get('result')}, expected {r.get('expected')}\n"

    if errors:
        feedback += "\nThere were runtime/syntax errors. Check your code for:\n"
        feedback += "- Correct function name and parameters\n"
        feedback += "- Proper indentation and syntax\n"
        feedback += "- No infinite loops or recursion without base case\n"

    feedback += f"""
Please fix the issues and provide the corrected `{function_name}` function.
Return only the corrected Python code, no explanation.

Original task:
{original_prompt}
"""
    return feedback


# ─────────────────────────────────────────────────────────────────────
# The review loop
# ─────────────────────────────────────────────────────────────────────

def review_code(
    model: str,
    prompt: str,
    function_name: str,
    test_cases: list[tuple],
    code_pattern: str = "",
    prompt_id: str = "custom",
    host: str = "localhost",
    port: int = 11434,
    timeout: int = 120,
    temperature: float = 0.0,
    seed: int = 42,
    max_iterations: int = 3,
    show_progress: bool = True,
) -> ReviewResult:
    """Run the live code review loop for a single coding prompt.

    Args:
        model:          Ollama model name (e.g. "qwen2.5:3b").
        prompt:         The coding prompt to send to the model.
        function_name:  Name of the function the model should generate.
        test_cases:     List of (args_dict, expected_result) tuples.
        code_pattern:   Regex to extract the function from the response.
        prompt_id:      Identifier for this prompt (for reporting).
        host/port:      Ollama server address.
        timeout:        Per-query timeout in seconds.
        temperature:    Generation temperature.
        seed:           Random seed.
        max_iterations: Maximum number of generate→test→fix iterations.
        show_progress:  Print progress to stdout.

    Returns:
        ReviewResult with all iterations and final outcome.
    """
    result = ReviewResult(
        prompt_id=prompt_id,
        original_prompt=prompt,
        function_name=function_name,
    )

    current_prompt = prompt
    total_start = time.time()

    for iteration in range(1, max_iterations + 1):
        if show_progress:
            print(f"    🔄 Iteration {iteration}/{max_iterations} for {prompt_id}...", end="", flush=True)

        # Query the model
        resp = query_model(
            model, current_prompt,
            host=host, port=port, timeout=timeout,
            temperature=temperature, seed=seed + iteration - 1,  # vary seed per iteration
        )

        if resp.error:
            if show_progress:
                print(f" ❌ Error: {resp.error}")
            result.iterations.append(ReviewIteration(
                iteration=iteration,
                response=resp.response,
                extracted_code="",
                test_results=[],
                passed=False, score=0.0,
                detail=f"Model error: {resp.error}",
                response_time_sec=resp.response_time_sec,
            ))
            break

        # Extract code
        code = extract_code(resp.response, code_pattern)

        # Run tests
        test_results, stderr = run_code_tests(code, function_name, test_cases)

        if test_results:
            passed_count = sum(1 for r in test_results if r.get("pass"))
            total = len(test_results)
            score = passed_count / total if total else 0.0
            passed = score == 1.0

            # Build detail string
            if passed:
                detail = f"All {total} tests passed"
            else:
                fail_details = []
                for r in test_results:
                    if not r.get("pass"):
                        if "error" in r:
                            fail_details.append(f"Error: {r['error']}")
                        else:
                            fail_details.append(f"Got {r.get('result')}, expected {r.get('expected')}")
                detail = f"{passed_count}/{total} passed. Failed: {'; '.join(fail_details)}"
        else:
            # stderr error (syntax error, timeout, etc.)
            passed_count = 0
            score = 0.0
            passed = False
            detail = f"Execution error: {stderr[:300]}"

        # Build feedback for next iteration if needed
        feedback = ""
        if not passed and iteration < max_iterations:
            feedback = build_feedback_prompt(
                prompt, code, test_results, function_name, iteration
            )

        result.iterations.append(ReviewIteration(
            iteration=iteration,
            response=resp.response,
            extracted_code=code,
            test_results=test_results,
            passed=passed,
            score=score,
            detail=detail,
            response_time_sec=resp.response_time_sec,
            feedback_given=feedback,
        ))

        if show_progress:
            if passed:
                print(f" ✅ {score:.0%} ({resp.response_time_sec:.1f}s)")
            else:
                print(f" ❌ {score:.0%} ({resp.response_time_sec:.1f}s) — requeuing...")

        # If passed, we're done
        if passed:
            break

        # Otherwise, use the feedback as the next prompt
        current_prompt = feedback

    result.total_time_sec = time.time() - total_start
    result.iterations_used = len(result.iterations)
    if result.iterations:
        result.final_passed = result.iterations[-1].passed
        result.final_score = result.iterations[-1].score

    return result


# ─────────────────────────────────────────────────────────────────────
# Batch review — review all coding prompts
# ─────────────────────────────────────────────────────────────────────

def review_coding_prompts(
    model: str,
    prompts: list,
    host: str = "localhost",
    port: int = 11434,
    timeout: int = 120,
    temperature: float = 0.0,
    seed: int = 42,
    max_iterations: int = 3,
    show_progress: bool = True,
) -> list[ReviewResult]:
    """Run the live code review loop for all coding-category prompts.

    Args:
        prompts: List of TestPrompt objects (should be coding category).
        Other args: same as review_code.

    Returns:
        List of ReviewResult objects, one per prompt.
    """
    results = []
    for tp in prompts:
        if tp.check != "code_exec":
            if show_progress:
                print(f"    ⏭️  Skipping {tp.id} (not code_exec check)")
            continue

        expected = tp.expected
        rr = review_code(
            model=model,
            prompt=tp.prompt,
            function_name=expected["function_name"],
            test_cases=expected["test_cases"],
            code_pattern=expected.get("code_pattern", ""),
            prompt_id=tp.id,
            host=host, port=port, timeout=timeout,
            temperature=temperature, seed=seed,
            max_iterations=max_iterations,
            show_progress=show_progress,
        )
        results.append(rr)
    return results


# ─────────────────────────────────────────────────────────────────────
# Terminal formatting
# ─────────────────────────────────────────────────────────────────────

def format_review_report(results: list[ReviewResult], model: str) -> str:
    """Format a terminal report for live code review results."""
    from .report import GREEN, RED, YELLOW, BOLD, RESET, _c

    lines = []
    lines.append("")
    lines.append("═" * 64)
    lines.append(f"  🔬 {_c('Live Code Review Report', BOLD)} — {model}")
    lines.append("═" * 64)

    total = len(results)
    passed = sum(1 for r in results if r.final_passed)
    improved = sum(1 for r in results if r.improved)

    lines.append(f"  Prompts reviewed: {total}")
    lines.append(f"  Final pass rate:  {_c(f'{passed}/{total}', GREEN if passed == total else YELLOW)}")
    lines.append(f"  Self-corrected:   {_c(str(improved), GREEN if improved > 0 else '')}"
                f" (went from failing → passing)")
    lines.append("")

    # Per-prompt details
    lines.append("─" * 64)
    lines.append(f"  {_c('PER-PROMPT REVIEW DETAILS', BOLD)}")
    lines.append("─" * 64)

    for rr in results:
        icon = _c("✅", GREEN) if rr.final_passed else _c("❌", RED)
        improved_tag = f" {_c('(self-corrected!)', GREEN)}" if rr.improved else ""
        lines.append(f"  {icon} {rr.prompt_id} [{rr.iterations_used} iterations]{improved_tag}")
        lines.append(f"     Function: {rr.function_name}")
        lines.append(f"     Score progression: {' → '.join(f'{s:.0%}' for s in rr.score_progression)}")
        lines.append(f"     Time: {rr.total_time_sec:.1f}s")

        for it in rr.iterations:
            it_icon = _c("✅", GREEN) if it.passed else _c("❌", RED)
            lines.append(f"     Iteration {it.iteration}: {it_icon} {it.score:.0%} — {it.detail[:120]}")

        lines.append("")

    # Summary statistics
    lines.append("─" * 64)
    lines.append(f"  {_c('SUMMARY', BOLD)}")
    lines.append("─" * 64)
    total_iterations = sum(r.iterations_used for r in results)
    total_time = sum(r.total_time_sec for r in results)
    lines.append(f"  Total iterations: {total_iterations}")
    lines.append(f"  Total time:       {total_time:.1f}s")
    lines.append(f"  Avg iterations:   {total_iterations / total:.1f}" if total else "")
    if total > 0:
        first_pass = sum(1 for r in results if r.iterations and r.iterations[0].passed)
        lines.append(f"  First-try pass:   {first_pass}/{total} ({first_pass/total:.0%})")
        lines.append(f"  After review:    {passed}/{total} ({passed/total:.0%})")
        improvement = passed - first_pass
        if improvement > 0:
            lines.append(f"  {_c(f'Improvement: +{improvement} prompts fixed by review loop', GREEN)}")
    lines.append("")
    lines.append("═" * 64)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# JSON export for review results
# ─────────────────────────────────────────────────────────────────────

def save_review_json(results: list[ReviewResult], model: str, filepath: str) -> str:
    """Save review results as JSON."""
    import os
    from datetime import datetime
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    data = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "total_prompts": len(results),
        "final_passed": sum(1 for r in results if r.final_passed),
        "self_corrected": sum(1 for r in results if r.improved),
        "results": [
            {
                "prompt_id": r.prompt_id,
                "function_name": r.function_name,
                "iterations_used": r.iterations_used,
                "final_passed": r.final_passed,
                "final_score": r.final_score,
                "improved": r.improved,
                "score_progression": r.score_progression,
                "total_time_sec": r.total_time_sec,
                "iterations": [
                    {
                        "iteration": it.iteration,
                        "passed": it.passed,
                        "score": it.score,
                        "detail": it.detail,
                        "response_time_sec": it.response_time_sec,
                        "extracted_code": it.extracted_code[:1000],
                        "test_results": it.test_results,
                    }
                    for it in r.iterations
                ],
            }
            for r in results
        ],
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return filepath