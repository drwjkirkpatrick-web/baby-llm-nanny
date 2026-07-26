"""Test prompt bank for baby-llm-nanny.

Each prompt has:
- id: unique identifier
- category: which evaluation category
- prompt: the text sent to the model
- expected: the canonical correct answer (string or list of acceptable strings)
- check: evaluation strategy name (see evaluator.py)
- notes: what this test is designed to catch

Categories:
  factual        - verifiable real-world facts
  math           - arithmetic with exact numeric answers
  series         - number series / next-in-sequence
  coding         - code generation, verified by execution
  logic          - deductive reasoning with deterministic answers
  hallucination  - traps where the model should admit ignorance, not fabricate
  instruction    - exact format compliance
  boundaries     - self-knowledge / calibration
"""

from .prompts import PROMPTS, get_prompts_by_category, get_prompts_by_difficulty, list_categories

__all__ = ["PROMPTS", "get_prompts_by_category", "get_prompts_by_difficulty", "list_categories"]