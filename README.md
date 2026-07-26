# 🍼 baby-llm-nanny

**Hallucination and quality screening for small local LLMs on headless Jetson Orin Nano.**

When you're running a 3B-parameter model on an edge device, you need to know exactly where it fails. Baby-llm-nanny sends a curated bank of test prompts to your local Ollama models and automatically evaluates the responses against known-correct answers — catching hallucinations, bad math, logic errors, and instruction-following failures.

## 🔬 Live Code Review (v0.3.0)

The **live code review module** implements an iterative generate→test→feedback→regenerate loop — exactly what a human reviewer would do in a hackathon:

1. Sends a coding prompt to the local LLM
2. Extracts the generated code
3. Runs it against test cases in an isolated subprocess
4. If tests fail, constructs specific feedback ("Test 2: got False, expected True") and re-queries the model
5. Repeats until all tests pass or `--max-iterations` is reached

This gets the absolute best out of small local models by catching their mistakes and giving them a chance to self-correct.

```bash
# Review all built-in coding prompts (9 prompts, up to 3 iterations each)
baby-llm-nanny qwen2.5:3b --review

# Review with 5 iterations max
baby-llm-nanny qwen2.5:3b --review --max-iterations 5

# Review a custom prompt with your own test cases
baby-llm-nanny qwen2.5:3b \
  --review-prompt "Write a Python function called square that takes n and returns n*n. Return only the code." \
  --review-tests-file my_tests.py \
  --max-iterations 3

# Save review results as JSON
baby-llm-nanny qwen2.5:3b --review --review-json reviews/qwen_review.json
```

### Custom test files

Create a `.py` file with `test_cases`, `function_name`, and optionally `code_pattern`:

```python
# my_tests.py
function_name = "square"
code_pattern = r"def\s+square\s*\([^)]*\)\s*:.*?(?=\ndef\s|\Z)"
test_cases = [
    ({"n": 2}, 4),
    ({"n": 3}, 9),
    ({"n": 0}, 0),
    ({"n": -4}, 16),
]
```

### Sample review output

```
🔬 Live Code Review Report — qwen2.5:3b
  Prompts reviewed: 9
  Final pass rate:  5/9
  Self-corrected:   0 (went from failing → passing)

  ✅ coding-is-even [1 iterations]
     Score progression: 100%
     Iteration 1: ✅ 100% — All 5 tests passed

  ❌ coding-fizzbuzz [3 iterations]
     Score progression: 0% → 0% → 0%
     Iteration 1: ❌ 0% — Error: can't multiply sequence by non-int of type 'str'
     Iteration 2: ❌ 0% — Error: can't multiply sequence by non-int of type 'str'
     Iteration 3: ❌ 0% — Error: can't multiply sequence by non-int of type 'str'

  SUMMARY
  First-try pass:   5/9 (56%)
  After review:    5/9 (56%)
```

### What the review loop reveals

Testing Qwen 2.5 3B reveals that:
- **Simple functions** (is_even, factorial, reverse_string, palindrome, count_vowels) pass on first try
- **FizzBuzz** — the model tries to multiply strings (`"Fizz" * (i % 3 == 0)`) instead of using if/elif
- **Parameter naming** — the model generates `max_of_list(arr)` instead of `max_of_list(lst)`, ignoring the parameter name in the prompt
- **Complex algorithms** (binary search, merge sorted) — the 3B model can't generate correct implementations even with feedback
- The feedback loop correctly identifies and reports each error, but the 3B model often repeats the same mistake — valuable information for hackathon mentors

## What it tests

| Category | What it probes | # Prompts |
|---|---|---|
| **factual** | Verifiable real-world facts (capitals, symbols, geography) | 8 |
| **math** | Arithmetic with exact numeric answers (multiplication, division, percentages, compound interest, set theory) | 12 |
| **series** | Next-in-sequence (powers, Fibonacci, primes, squares, Tribonacci, geometric) | 8 |
| **coding** | Code generation verified by actual execution (is_even, factorial, FizzBuzz, binary search, merge sorted, palindrome) | 9 |
| **logic** | Deductive reasoning (syllogisms, modus tollens, Knights & Knaves, Monty Hall) | 9 |
| **hallucination** | Fake books, battles, scientists, laws, mountains, medications, papers, quotes | 8 |
| **instruction** | Exact format compliance (single word, JSON, multi-constraint, reverse order) | 9 |
| **boundaries** | Self-knowledge and calibration (future predictions, training cutoff, subjective vs objective) | 4 |
| **Total** | | **67** |

Each prompt is tagged with **difficulty**: easy, medium, or hard.

## Quick start

```bash
# Install
cd baby-llm-nanny
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
uv pip install pytest

# Make sure Ollama is running
ollama serve &

# Run full evaluation against qwen2.5:3b
baby-llm-nanny qwen2.5:3b

# Live code review (generate→test→fix loop)
baby-llm-nanny qwen2.5:3b --review --max-iterations 3

# Run specific categories
baby-llm-nanny qwen2.5:3b --categories math,hallucination

# Filter by difficulty
baby-llm-nanny qwen2.5:3b --difficulty hard

# Compare multiple models side-by-side
baby-llm-nanny --compare qwen2.5:3b,gemma2:2b,llama3.2:3b

# Save reports in multiple formats
baby-llm-nanny qwen2.5:3b --json reports/qwen.json --csv reports/qwen.csv --html reports/qwen.html

# Run with system prompt strategy
baby-llm-nanny qwen2.5:3b --system-prompt careful

# Retry each prompt 3 times for consistency analysis
baby-llm-nanny qwen2.5:3b --retries 3

# Save to SQLite history for trend tracking
baby-llm-nanny qwen2.5:3b --save-history

# View historical trends
baby-llm-nanny --history

# List available models or prompts
baby-llm-nanny --list-models
baby-llm-nanny --list-prompts
```

## v0.3.0 — Live Code Review

New module: `baby_llm_nanny/reviewer.py` — iterative code review loop.

- **`--review`**: Run all coding prompts through the generate→test→fix loop
- **`--review-prompt TEXT`**: Review a custom coding prompt (bypasses prompt bank)
- **`--review-tests-file F`**: Load test cases from a `.py` file
- **`--max-iterations N`**: Max fix attempts (default: 3)
- **`--review-json PATH`**: Save review results as JSON
- Reports show score progression per iteration, self-correction detection, and first-try vs after-review pass rates

## v0.2.0 — 10 improvements

1. **Multi-model comparison** (`--compare m1,m2,m3`): Side-by-side table.
2. **Token efficiency metrics**: Tokens/sec from Ollama API.
3. **Difficulty ratings**: Easy/medium/hard on all prompts.
4. **CSV export** (`--csv path`): Spreadsheet-friendly output.
5. **20 new prompts**: Compound interest, Tribonacci, Knights & Knaves, Monty Hall, binary search, etc.
6. **Color-coded terminal output**: ANSI green/yellow/red.
7. **Historical trend tracking** (`--save-history`, `--history`): SQLite DB.
8. **Retry/consistency analysis** (`--retries N`): Per-prompt variance.
9. **Hallucination confidence scoring**: Response length as fabrication proxy.
10. **HTML report export** (`--html path`): Self-contained styled file.

## How it works

1. **Prompt bank** (`prompts/prompts.py`): 67 test prompts across 8 categories, each with a known-correct answer, evaluation strategy, and difficulty rating.

2. **Runner** (`runner.py`): Sends prompts to Ollama via HTTP API. Uses `temperature=0.0` and fixed seed for reproducibility. Zero external dependencies.

3. **Evaluator** (`evaluator.py`): Six evaluation strategies:
   - `exact` — string must match exactly
   - `contains_any` — response must contain at least one acceptable substring
   - `numeric` — extracts and compares numbers with optional tolerance
   - `json_keys` — parses JSON (even from markdown fences) and checks key-value pairs
   - `code_exec` — extracts Python code, executes it in a subprocess, runs test cases
   - `multi_constraint` — checks multiple constraints simultaneously

4. **Reviewer** (`reviewer.py`): Live code review loop — generate→test→feedback→regenerate. Constructs specific actionable feedback from test failures and re-queries the model.

5. **Report** (`report.py`): Terminal output with color, JSON/CSV/HTML export, multi-model comparison table.

6. **History** (`history.py`): SQLite database for storing runs over time and detecting regressions.

## System prompts

| Strategy | Description |
|---|---|
| `none` | No system prompt (baseline) |
| `careful` | "Think step by step. If you're not sure, say 'I don't know'." |
| `expert` | Detailed reasoning and calibration instructions |
| `concise` | "Follow output format instructions exactly." |

## Project structure

```
baby-llm-nanny/
├── baby_llm_nanny/
│   ├── __init__.py
│   ├── cli.py            # CLI entry point with all flags
│   ├── runner.py         # Ollama HTTP client + token efficiency
│   ├── evaluator.py      # 6 evaluation strategies + hallucination scoring
│   ├── reviewer.py       # Live code review loop (generate→test→fix)
│   ├── report.py         # Terminal + JSON + CSV + HTML + comparison
│   ├── history.py        # SQLite trend tracking + retry analysis
│   └── prompts/
│       ├── __init__.py
│       └── prompts.py    # 67 test prompts with difficulty ratings
├── tests/
│   ├── test_evaluator_and_prompts.py   # 80+ unit tests
│   ├── test_runner.py                  # 9 integration tests (needs Ollama)
│   ├── test_report.py                  # 20+ report/export tests
│   ├── test_history.py                 # 10+ SQLite tests
│   └── test_reviewer.py               # 24 review loop tests
├── pyproject.toml
└── README.md
```

## Running tests

```bash
# Unit tests (no Ollama needed)
.venv/bin/python -m pytest tests/test_evaluator_and_prompts.py tests/test_report.py tests/test_history.py tests/test_reviewer.py -q

# Integration tests (requires Ollama running)
.venv/bin/python -m pytest tests/test_runner.py -q

# All tests
.venv/bin/python -m pytest tests/ -q
```

## License

MIT

## Author

Walker Kirkpatrick — built for getting the best out of small LLMs on NVIDIA Jetson Orin Nano.