# 🍼 baby-llm-nanny

**Hallucination and quality screening for small local LLMs on headless Jetson Orin Nano.**

When you're running a 3B-parameter model on an edge device, you need to know exactly where it fails. Baby-llm-nanny sends a curated bank of test prompts to your local Ollama models and automatically evaluates the responses against known-correct answers — catching hallucinations, bad math, logic errors, and instruction-following failures.

## What it tests

| Category | What it probes | # Prompts |
|---|---|---|
| **factual** | Verifiable real-world facts (capitals, symbols, geography) | 8 |
| **math** | Arithmetic with exact numeric answers (multiplication, division, percentages, word problems) | 8 |
| **series** | Next-in-sequence (powers, Fibonacci, primes, squares, arithmetic, geometric) | 6 |
| **coding** | Code generation verified by actual execution (is_even, factorial, reverse_string, FizzBuzz, max_of_list) | 5 |
| **logic** | Deductive reasoning with deterministic answers (syllogisms, modus tollens, spatial reasoning) | 6 |
| **hallucination** | Fake books, battles, scientists, laws, and mountains the model should refuse to answer | 5 |
| **instruction** | Exact format compliance (single word, single number, JSON, no-explanation) | 6 |
| **boundaries** | Self-knowledge and calibration (future predictions, training cutoff, subjective vs objective) | 4 |
| **Total** | | **48** |

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

# Save JSON report
baby-llm-nanny qwen2.5:3b --json reports/qwen2.5-3b.json

# Use the "careful" system prompt to see if it improves scores
baby-llm-nanny qwen2.5:3b --system-prompt careful

# Compare models
baby-llm-nanny qwen2.5:3b --json reports/qwen.json
baby-llm-nanny gemma2:2b --json reports/gemma.json
baby-llm-nanny llama3.2:3b --json reports/llama.json
baby-llm-nanny phi3:3.8b --json reports/phi3.json

# List available models
baby-llm-nanny --list-models

# List all test prompts
baby-llm-nanny --list-prompts
```

## How it works

1. **Prompt bank** (`baby_llm_nanny/prompts/prompts.py`): 48 test prompts across 8 categories, each with a known-correct answer and an evaluation strategy.

2. **Runner** (`baby_llm_nanny/runner.py`): Sends prompts to Ollama via the HTTP API (`localhost:11434`). Uses `temperature=0.0` and a fixed seed for reproducibility. Zero external dependencies — uses only Python stdlib `urllib`.

3. **Evaluator** (`baby_llm_nanny/evaluator.py`): Checks responses against expected answers using five strategies:
   - `exact` — string must match exactly
   - `contains_any` — response must contain at least one acceptable substring
   - `numeric` — extracts and compares numbers with optional tolerance
   - `json_keys` — parses JSON (even from markdown fences) and checks key-value pairs
   - `code_exec` — extracts Python code, executes it in a subprocess, and runs test cases

4. **Report** (`baby_llm_nanny/report.py`): Generates a terminal table with per-category breakdowns and per-prompt details. Exports JSON for comparison across runs.

## System prompts

Baby-llm-nanny supports four system prompt strategies to test whether prompting improves small model accuracy:

| Strategy | Description |
|---|---|
| `none` | No system prompt (baseline) |
| `careful` | "Think step by step. If you're not sure, say 'I don't know'." |
| `expert` | Detailed reasoning and calibration instructions |
| `concise` | "Follow output format instructions exactly." |

## Sample output

```
🍼 baby-llm-nanny v0.1.0
   Model:          qwen2.5:3b
   Prompts:        48
   System prompt:  none
   Temperature:    0.0
   Seed:           42

════════════════════════════════════════════════════════════════
  🍼 baby-llm-nanny — Report for qwen2.5:3b
════════════════════════════════════════════════════════════════

  OVERALL
  Prompts:       48
  Passed:        32 / 48 (66.7%)
  Avg score:     0.681

  BY CATEGORY
  Category         Pass     Partial  Fail     AvgScore   AvgTime
  ──────────────────────────────────────────────────────
  coding           4        0        1        0.800      2.3
  factual          5        0        3        0.625      1.2
  hallucination    3        0        2        0.600      1.5
  instruction      4        1        1        0.767      1.1
  logic             5        0        1        0.833      1.4
  math             6        0        2        0.750      1.5
  series           5        0        1        0.833      1.3
  boundaries       2        0        2        0.500      1.2
```

## Project structure

```
baby-llm-nanny/
├── baby_llm_nanny/
│   ├── __init__.py
│   ├── cli.py            # CLI entry point
│   ├── runner.py         # Ollama HTTP API client
│   ├── evaluator.py      # Response checking (5 strategies)
│   ├── report.py         # Terminal + JSON report generation
│   └── prompts/
│       ├── __init__.py
│       └── prompts.py    # 48 test prompts across 8 categories
├── tests/
│   ├── test_evaluator_and_prompts.py   # 64 unit tests
│   ├── test_runner.py                  # 9 integration tests (needs Ollama)
│   └── test_report.py                  # 14 report tests
├── pyproject.toml
└── README.md
```

## Design decisions

- **Zero external dependencies**: Uses only Python stdlib (`urllib`, `json`, `re`, `subprocess`). No `requests`, no `aiohttp`, no `pydantic` — because on a headless Jetson, fewer dependencies means fewer things break.
- **Deterministic by default**: `temperature=0.0`, `seed=42` — so you get reproducible results across runs and can compare models fairly.
- **Code execution isolation**: Generated code runs in a subprocess with a 10-second timeout, not in the main process.
- **Hallucination traps**: The prompt bank includes fake books, fake battles, fake scientists, fake laws, and fake mountains. A model that fabricates answers to these fails; a model that says "I don't know" passes.
- **Real verification**: Coding prompts are verified by actually executing the generated code against test cases — not by pattern matching or LLM-as-judge.

## Running tests

```bash
# Unit tests (no Ollama needed)
.venv/bin/python -m pytest tests/test_evaluator_and_prompts.py tests/test_report.py -q

# Integration tests (requires Ollama running)
.venv/bin/python -m pytest tests/test_runner.py -q

# All tests
.venv/bin/python -m pytest tests/ -q
```

## License

MIT

## Author

Walker Kirkpatrick — built for getting the best out of small LLMs on NVIDIA Jetson Orin Nano.