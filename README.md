# 🍼 baby-llm-nanny

**Hallucination and quality screening for small local LLMs on headless Jetson Orin Nano.**

When you're running a 3B-parameter model on an edge device, you need to know exactly where it fails. Baby-llm-nanny sends a curated bank of test prompts to your local Ollama models and automatically evaluates the responses against known-correct answers — catching hallucinations, bad math, logic errors, and instruction-following failures.

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

## v0.2.0 — 10 new improvements

1. **Multi-model comparison** (`--compare m1,m2,m3`): Side-by-side table comparing pass rates, scores, token speed, and category breakdowns across models.

2. **Token efficiency metrics**: Reports tokens/second from the Ollama API, plus total tokens consumed. Visible in terminal, JSON, CSV, and HTML reports.

3. **Difficulty ratings**: Every prompt is tagged easy, medium, or hard. Reports include a difficulty breakdown showing where the model struggles most.

4. **CSV export** (`--csv path`): Spreadsheet-friendly output with one row per prompt, including all metrics.

5. **20 new prompts**: Harder math (compound interest, set theory), Tribonacci series, Knights & Knaves logic, Monty Hall, binary search, merge sorted lists, adversarial hallucination traps (fake medication, fake paper in real journal, modified famous quote), multi-constraint instruction following.

6. **Color-coded terminal output**: ANSI green/yellow/red for scores and status indicators. Disable with `--no-color`.

7. **Historical trend tracking** (`--save-history`, `--history`): SQLite database stores each run. View trends over time and detect regressions vs the previous run.

8. **Retry/consistency analysis** (`--retries N`): Runs each prompt N times with different seeds. Reports when a model gives inconsistent answers across retries.

9. **Hallucination confidence scoring**: For hallucination-category prompts, estimates how confidently the model fabricates (0.0 = admitted ignorance, 0.3 = brief fabrication, 0.9 = elaborate fabrication). Based on response length as a proxy.

10. **HTML report export** (`--html path`): Self-contained HTML file with styled tables, color-coded scores, and full per-prompt details.

## How it works

1. **Prompt bank** (`prompts/prompts.py`): 67 test prompts across 8 categories, each with a known-correct answer, evaluation strategy, and difficulty rating.

2. **Runner** (`runner.py`): Sends prompts to Ollama via HTTP API. Uses `temperature=0.0` and fixed seed for reproducibility. Zero external dependencies.

3. **Evaluator** (`evaluator.py`): Six evaluation strategies:
   - `exact` — string must match exactly
   - `contains_any` — response must contain at least one acceptable substring
   - `numeric` — extracts and compares numbers with optional tolerance
   - `json_keys` — parses JSON (even from markdown fences) and checks key-value pairs
   - `code_exec` — extracts Python code, executes it in a subprocess, runs test cases
   - `multi_constraint` — checks multiple constraints simultaneously (word count, starting word, etc.)

4. **Report** (`report.py`): Terminal output with color, JSON/CSV/HTML export, multi-model comparison table.

5. **History** (`history.py`): SQLite database for storing runs over time and detecting regressions.

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
│   ├── cli.py            # CLI entry point with all new flags
│   ├── runner.py         # Ollama HTTP client + token efficiency
│   ├── evaluator.py      # 6 evaluation strategies + hallucination scoring
│   ├── report.py         # Terminal + JSON + CSV + HTML + comparison
│   ├── history.py        # SQLite trend tracking + retry analysis
│   └── prompts/
│       ├── __init__.py
│       └── prompts.py    # 67 test prompts with difficulty ratings
├── tests/
│   ├── test_evaluator_and_prompts.py   # 80+ unit tests
│   ├── test_runner.py                  # 9 integration tests (needs Ollama)
│   ├── test_report.py                  # 20+ report/export tests
│   └── test_history.py                  # 10+ SQLite tests
├── pyproject.toml
└── README.md
```

## Running tests

```bash
# Unit tests (no Ollama needed)
.venv/bin/python -m pytest tests/test_evaluator_and_prompts.py tests/test_report.py tests/test_history.py -q

# Integration tests (requires Ollama running)
.venv/bin/python -m pytest tests/test_runner.py -q

# All tests
.venv/bin/python -m pytest tests/ -q
```

## License

MIT

## Author

Walker Kirkpatrick — built for getting the best out of small LLMs on NVIDIA Jetson Orin Nano.