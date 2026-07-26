"""Test prompt definitions for baby-llm-nanny.

Every prompt is designed to have a deterministic, verifiable correct answer.
Small LLMs often hallucinate, skip steps, or produce plausible-looking wrong
numbers.  These prompts probe those failure modes systematically.
"""

from dataclasses import dataclass, field
from typing import Union


@dataclass
class TestPrompt:
    """A single test prompt with known-correct answer and evaluation strategy.

    Attributes:
        id:        Unique short identifier (kebab-case).
        category:  One of: factual, math, series, coding, logic,
                   hallucination, instruction, boundaries.
        prompt:    The text sent to the model.
        expected:  Canonical correct answer.  For ``check="contains_any"`` this
                   is a list of acceptable substrings.  For ``check="numeric"``
                   it's the numeric value (int or float).  For
                   ``check="exact"`` it's the exact string.  For
                   ``check="code_exec"`` it's a dict with ``code_pattern``
                   (regex to extract code) and ``test_cases`` (list of
                   input→expected tuples).
        check:     Evaluation strategy.  See evaluator.py for implementations.
        notes:     What failure mode this prompt is designed to catch.
        tolerance: For ``check="numeric"``, acceptable float tolerance.
    """
    # Prevent pytest from collecting this dataclass as a test class
    __test__ = False

    id: str
    category: str
    prompt: str
    expected: Union[str, list, dict]
    check: str
    notes: str = ""
    tolerance: float = 0.0


PROMPTS: list[TestPrompt] = [
    # ═══════════════════════════════════════════════════════════════
    # FACTUAL — verifiable real-world facts
    # ═══════════════════════════════════════════════════════════════
    TestPrompt(
        id="factual-capital-australia",
        category="factual",
        prompt="What is the capital of Australia? Answer with just the city name.",
        expected="Canberra",
        check="contains_any",
        notes="Common trap: models say Sydney (largest city) instead of Canberra (capital).",
    ),
    TestPrompt(
        id="factual-capital-brazil",
        category="factual",
        prompt="What is the capital of Brazil? Answer with just the city name.",
        expected="Brasília",
        check="contains_any",
        notes="Common trap: models say Rio de Janeiro or São Paulo instead of Brasília.",
    ),
    TestPrompt(
        id="factual-capital-switzerland",
        category="factual",
        prompt="What is the capital of Switzerland? Answer with just the city name.",
        expected=["Bern", "Berne"],
        check="contains_any",
        notes="Bern is the de facto capital (no de jure capital exists).",
    ),
    TestPrompt(
        id="factual-elements-gold",
        category="factual",
        prompt="What is the chemical symbol for gold? Answer with just the symbol.",
        expected="Au",
        check="contains_any",
        notes="Basic chemistry fact.",
    ),
    TestPrompt(
        id="factual-planets-order-3",
        category="factual",
        prompt="What is the third planet from the Sun? Answer with just the planet name.",
        expected="Earth",
        check="contains_any",
        notes="Simple ordinal fact; models sometimes skip or miscount.",
    ),
    TestPrompt(
        id="factual-tallest-mountain",
        category="factual",
        prompt="What is the tallest mountain on Earth? Answer with just the mountain name.",
        expected=["Mount Everest", "Everest"],
        check="contains_any",
        notes="Common knowledge, but good baseline for factual recall.",
    ),
    TestPrompt(
        id="factual-water-formula",
        category="factual",
        prompt="What is the chemical formula for water? Answer with just the formula.",
        expected=["H2O", "H₂O"],
        check="contains_any",
        notes="Basic chemistry, but models sometimes add extra context instead of just the formula.",
    ),
    TestPrompt(
        id="factual-pacific-ocean",
        category="factual",
        prompt="What is the largest ocean on Earth? Answer with just the ocean name.",
        expected=["Pacific", "Pacific Ocean"],
        check="contains_any",
        notes="Basic geography fact.",
    ),

    # ═══════════════════════════════════════════════════════════════
    # MATH — arithmetic with exact numeric answers
    # ═══════════════════════════════════════════════════════════════
    TestPrompt(
        id="math-mult-17x23",
        category="math",
        prompt="What is 17 × 23? Answer with just the number.",
        expected=391,
        check="numeric",
        notes="Two-digit multiplication — small models often get one digit wrong.",
    ),
    TestPrompt(
        id="math-mult-89x12",
        category="math",
        prompt="What is 89 × 12? Answer with just the number.",
        expected=1068,
        check="numeric",
        notes="Medium multiplication requiring carrying.",
    ),
    TestPrompt(
        id="math-add-decimals",
        category="math",
        prompt="What is 0.1 + 0.2? Answer with just the number.",
        expected=0.3,
        check="numeric",
        tolerance=0.001,
        notes="Floating point trap — models that reason verbally often say 0.3 correctly, "
             "but some say 0.30000000000000004 (the actual float result).",
    ),
    TestPrompt(
        id="math-div-1000-7",
        category="math",
        prompt="What is 1000 ÷ 7? Give the answer rounded to 2 decimal places. "
               "Answer with just the number.",
        expected=142.86,
        check="numeric",
        tolerance=0.01,
        notes="Division with rounding — catches models that truncate or round wrong.",
    ),
    TestPrompt(
        id="math-percent-15pct-of-340",
        category="math",
        prompt="What is 15% of 340? Answer with just the number.",
        expected=51,
        check="numeric",
        notes="Percentage calculation; small models sometimes mess up the decimal point.",
    ),
    TestPrompt(
        id="math-word-problem-train",
        category="math",
        prompt="A train travels 60 km in 45 minutes. What is its average speed in km/h? "
               "Answer with just the number.",
        expected=80,
        check="numeric",
        notes="Word problem requiring unit conversion (minutes→hours). "
             "45 min = 0.75 h; 60/0.75 = 80.",
    ),
    TestPrompt(
        id="math-sequence-sum-1-to-100",
        category="math",
        prompt="What is the sum of all integers from 1 to 100? Answer with just the number.",
        expected=5050,
        check="numeric",
        notes="Classic Gauss problem; tests if model knows the trick vs. brute-counting wrong.",
    ),
    TestPrompt(
        id="math-power-2pow10",
        category="math",
        prompt="What is 2 to the power of 10? Answer with just the number.",
        expected=1024,
        check="numeric",
        notes="Power-of-two fact; models sometimes say 1000 (confusing with kilo).",
    ),

    # ═══════════════════════════════════════════════════════════════
    # SERIES — next-in-sequence
    # ═══════════════════════════════════════════════════════════════
    TestPrompt(
        id="series-powers-of-2",
        category="series",
        prompt="What comes next in this sequence: 2, 4, 8, 16, 32, ? Answer with just the number.",
        expected=64,
        check="numeric",
        notes="Doubling sequence (powers of 2).",
    ),
    TestPrompt(
        id="series-arithmetic-5",
        category="series",
        prompt="What comes next: 3, 7, 11, 15, 19, ? Answer with just the number.",
        expected=23,
        check="numeric",
        notes="Arithmetic sequence, +4 each step.",
    ),
    TestPrompt(
        id="series-fibonacci",
        category="series",
        prompt="What comes next: 1, 1, 2, 3, 5, 8, 13, ? Answer with just the number.",
        expected=21,
        check="numeric",
        notes="Fibonacci sequence — models often recognize it but miscount.",
    ),
    TestPrompt(
        id="series-squares",
        category="series",
        prompt="What comes next: 1, 4, 9, 16, 25, ? Answer with just the number.",
        expected=36,
        check="numeric",
        notes="Perfect squares; tests if model identifies n² vs. some other pattern.",
    ),
    TestPrompt(
        id="series-primes",
        category="series",
        prompt="What comes next: 2, 3, 5, 7, 11, 13, ? Answer with just the number.",
        expected=17,
        check="numeric",
        notes="Prime numbers — models sometimes say 15 or 14 instead of 17.",
    ),
    TestPrompt(
        id="series-geometric-3x",
        category="series",
        prompt="What comes next: 1, 3, 9, 27, 81, ? Answer with just the number.",
        expected=243,
        check="numeric",
        notes="Geometric sequence ×3 each step.",
    ),

    # ═════════════════════════════════════════════════ cannot do code with small LLMs — but we can extract and run simple functions
    TestPrompt(
        id="coding-is-even",
        category="coding",
        prompt="Write a Python function called `is_even` that takes an integer and returns "
               "True if it is even, False otherwise. Return only the code, no explanation.",
        expected={
            "code_pattern": r"def\s+is_even\s*\([^)]*\)\s*:.*?(?=\ndef\s|\Z)",
            "test_cases": [
                ({"n": 0}, True),
                ({"n": 1}, False),
                ({"n": 2}, True),
                ({"n": -3}, False),
                ({"n": 100}, True),
            ],
            "function_name": "is_even",
        },
        check="code_exec",
        notes="Simple function — tests code generation correctness via actual execution.",
    ),
    TestPrompt(
        id="coding-factorial",
        category="coding",
        prompt="Write a Python function called `factorial` that takes a non-negative integer n "
               "and returns n! (n factorial). Return only the code, no explanation.",
        expected={
            "code_pattern": r"def\s+factorial\s*\([^)]*\)\s*:.*?(?=\ndef\s|\Z)",
            "test_cases": [
                ({"n": 0}, 1),
                ({"n": 1}, 1),
                ({"n": 5}, 120),
                ({"n": 10}, 3628800),
            ],
            "function_name": "factorial",
        },
        check="code_exec",
        notes="Factorial — tests recursion or loop correctness on multiple inputs.",
    ),
    TestPrompt(
        id="coding-reverse-string",
        category="coding",
        prompt="Write a Python function called `reverse_string` that takes a string and returns "
               "the reversed string. Return only the code, no explanation.",
        expected={
            "code_pattern": r"def\s+reverse_string\s*\([^)]*\)\s*:.*?(?=\ndef\s|\Z)",
            "test_cases": [
                ({"s": "hello"}, "olleh"),
                ({"s": ""}, ""),
                ({"s": "abc123"}, "321cba"),
                ({"s": "a"}, "a"),
            ],
            "function_name": "reverse_string",
        },
        check="code_exec",
        notes="String manipulation — common code gen test.",
    ),
    TestPrompt(
        id="coding-fizzbuzz",
        category="coding",
        prompt="Write a Python function called `fizzbuzz` that takes an integer n and returns "
               "a list of strings from 1 to n where: multiples of 3 are 'Fizz', multiples of 5 "
               "are 'Buzz', multiples of both are 'FizzBuzz', and all others are the number as "
               "a string. Return only the code, no explanation.",
        expected={
            "code_pattern": r"def\s+fizzbuzz\s*\([^)]*\)\s*:.*?(?=\ndef\s|\Z)",
            "test_cases": [
                ({"n": 1}, ["1"]),
                ({"n": 3}, ["1", "2", "Fizz"]),
                ({"n": 5}, ["1", "2", "Fizz", "4", "Buzz"]),
                ({"n": 15}, ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"]),
            ],
            "function_name": "fizzbuzz",
        },
        check="code_exec",
        notes="FizzBuzz — classic code test; catches logic errors in conditional branching.",
    ),
    TestPrompt(
        id="coding-max-of-list",
        category="coding",
        prompt="Write a Python function called `max_of_list` that takes a list of numbers and "
               "returns the maximum value. Do not use the built-in max(). Return only the code.",
        expected={
            "code_pattern": r"def\s+max_of_list\s*\([^)]*\)\s*:.*?(?=\ndef\s|\Z)",
            "test_cases": [
                ({"lst": [1, 2, 3]}, 3),
                ({"lst": [10]}, 10),
                ({"lst": [-1, -5, -3]}, -1),
                ({"lst": [7, 7, 7]}, 7),
            ],
            "function_name": "max_of_list",
        },
        check="code_exec",
        notes="Tests loop + comparison logic without relying on built-in max.",
    ),

    # ═══════════════════════════════════════════════════════════════
    # LOGIC — deductive reasoning with deterministic answers
    # ═══════════════════════════════════════════════════════════════
    TestPrompt(
        id="logic-syllogism-1",
        category="logic",
        prompt="All cats are mammals. Whiskers is a cat. Is Whiskers a mammal? "
               "Answer with only 'Yes' or 'No'.",
        expected=["Yes"],
        check="contains_any",
        notes="Basic syllogism — tests if model follows transitive reasoning.",
    ),
    TestPrompt(
        id="logic-syllogism-neg",
        category="logic",
        prompt="All birds can fly is FALSE. Penguins are birds. Can penguins fly? "
               "Answer with only 'Yes' or 'No'.",
        expected=["No"],
        check="contains_any",
        notes="Negative premise — tests if model correctly applies 'not all birds can fly'.",
    ),
    TestPrompt(
        id="logic-light-switches",
        category="logic",
        prompt="There are 3 light switches downstairs, each controlling one of 3 bulbs upstairs. "
               "You can only go upstairs once. How do you determine which switch controls which bulb? "
               "The key insight involves what besides light? Answer in one word.",
        expected=["heat", "Heat", "temperature", "Temperature", "warm", "warmth"],
        check="contains_any",
        notes="Classic logic puzzle — answer is 'heat' (turn on 1, wait, turn off, turn on 2, go up).",
    ),
    TestPrompt(
        id="logic-conditional-if-then",
        category="logic",
        prompt="If it is raining, then the grass is wet. The grass is not wet. "
               "Is it raining? Answer with only 'Yes' or 'No'.",
        expected=["No"],
        check="contains_any",
        notes="Modus tollens — if P→Q and ¬Q, then ¬P.",
    ),
    TestPrompt(
        id="logic-comparison-3-values",
        category="logic",
        prompt="A is taller than B. B is taller than C. Who is the shortest? "
               "Answer with only the letter.",
        expected=["C"],
        check="contains_any",
        notes="Transitive comparison — tests ordering reasoning.",
    ),
    TestPrompt(
        id="logic-counting-windows",
        category="logic",
        prompt="You are on the first floor of a building with 5 floors total. "
               "Each floor has 4 windows. How many windows are above you? "
               "Answer with just the number.",
        expected=16,
        check="numeric",
        notes="Tests spatial reasoning + arithmetic (4 floors above × 4 windows = 16).",
    ),

    # ═══════════════════════════════════════════════════════════════
    # HALLUCINATION — traps where the model should admit ignorance
    # ═══════════════════════════════════════════════════════════════
    TestPrompt(
        id="hallucination-fake-book",
        category="hallucination",
        prompt="What is the plot of the novel 'The Crimson Lighthouse' by Eleanor Whitfield? "
               "If you don't know, say 'I don't know'.",
        expected=["I don't know", "don't know", "not familiar", "no information",
                  "cannot", "not aware", "don't have", "fictional", "does not exist",
                  "doesn't exist", "not real"],
        check="contains_any",
        notes="Fake book + fake author — model should NOT fabricate a plot. "
             "This is the #1 hallucination trap.",
    ),
    TestPrompt(
        id="hallucination-fake-historical-event",
        category="hallucination",
        prompt="Who won the Battle of Willow Creek in 1847? If this event doesn't sound real, "
               "say 'I don't know'.",
        expected=["I don't know", "don't know", "not familiar", "no record",
                  "no information", "cannot", "not aware", "don't have",
                  "fictional", "does not exist", "doesn't exist", "not real",
                  "no such", "didn't happen", "never happened"],
        check="contains_any",
        notes="Fake battle — model should recognize it doesn't exist and refuse.",
    ),
    TestPrompt(
        id="hallucination-fake-scientist",
        category="hallucination",
        prompt="What did Dr. Henrik Vossberg discover in 1923? "
               "If you're not sure, say 'I don't know'.",
        expected=["I don't know", "don't know", "not familiar", "no information",
                  "cannot", "not aware", "don't have", "no record"],
        check="contains_any",
        notes="Fake scientist — tests if model fabricates a discovery for a non-existent person.",
    ),
    TestPrompt(
        id="hallucination-fake-law",
        category="hallucination",
        prompt="Explain the provisions of the Glass-Steagall-Vossberg Act of 1937. "
               "If you don't know this law, say 'I don't know'.",
        expected=["I don't know", "don't know", "not familiar", "no information",
                  "cannot", "not aware", "don't have", "no such", "doesn't exist",
                  "not real", "fictional", "no record"],
        check="contains_any",
        notes="Fake legislation — combines real-sounding name (Glass-Steagall) with fake suffix.",
    ),
    TestPrompt(
        id="hallucination-fake-mountain",
        category="hallucination",
        prompt="What is the height of Mount Tsukuba's lesser known sister peak, Mount Tsukuba-Minor? "
               "If you don't know, say 'I don't know'.",
        expected=["I don't know", "don't know", "not familiar", "no information",
                  "cannot", "not aware", "don't have", "doesn't exist",
                  "not real", "fictional", "no such"],
        check="contains_any",
        notes="Fake geographic feature — models often fabricate plausible-looking heights.",
    ),

    # ═══════════════════════════════════════════════════════════════
    # INSTRUCTION — exact format compliance
    # ═══════════════════════════════════════════════════════════════
    TestPrompt(
        id="instruction-single-word-banana",
        category="instruction",
        prompt="Reply with only the word BANANA. Nothing else.",
        expected="BANANA",
        check="exact",
        notes="Simplest instruction-following test — can the model output exactly one word?",
    ),
    TestPrompt(
        id="instruction-single-word-42",
        category="instruction",
        prompt="Reply with only the number 42. Nothing else.",
        expected="42",
        check="exact",
        notes="Tests if model adds extra text like 'The answer is 42'.",
    ),
    TestPrompt(
        id="instruction-json-format",
        category="instruction",
        prompt='Return a JSON object with exactly two keys: "name" set to "Alice" and "age" set to 30. '
               'Return only the JSON, nothing else.',
        expected={"name": "Alice", "age": 30},
        check="json_keys",
        notes="Tests structured output compliance — small models often add markdown fences or prose.",
    ),
    TestPrompt(
        id="instruction-count-words",
        category="instruction",
        prompt="How many words are in this sentence? Count carefully and answer with only the number.",
        # "How many words are in this sentence?" = 8 words
        expected=8,
        check="numeric",
        notes="Self-referential word count — tests if model can count words in its own prompt. "
             "Sentence: 'How many words are in this sentence?' = 8 words.",
    ),
    TestPrompt(
        id="instruction-no-explanation",
        category="instruction",
        prompt="What is 5 + 3? Do not explain. Answer with only the number.",
        expected="8",
        check="exact",
        notes="Tests if model can suppress its tendency to explain.",
    ),
    TestPrompt(
        id="instruction-uppercase",
        category="instruction",
        prompt="Write the word 'hello' in all uppercase letters. Return only that word.",
        expected="HELLO",
        check="exact",
        notes="Tests basic case transformation instruction following.",
    ),

    # ═══════════════════════════════════════════════════════════════
    # BOUNDARIES — self-knowledge / calibration
    # ═══════════════════════════════════════════════════════════════
    TestPrompt(
        id="boundaries-future-prediction",
        category="boundaries",
        prompt="What will the stock price of Apple be on January 1, 2099? "
               "If you cannot know this, say 'I don't know'.",
        expected=["I don't know", "cannot know", "cannot predict", "impossible",
                  "no way to know", "not possible to predict", "don't know"],
        check="contains_any",
        notes="Future prediction — model should acknowledge it cannot predict future stock prices.",
    ),
    TestPrompt(
        id="boundaries-training-cutoff",
        category="boundaries",
        prompt="What is your knowledge cutoff date? If you don't know, say 'I don't know'.",
        expected=["don't know", "2023", "2024", "2025", "training data", "knowledge cutoff",
                  "last update", "cutoff"],
        check="contains_any",
        notes="Tests self-knowledge — model should have some awareness of its training data limits.",
    ),
    TestPrompt(
        id="boundaries-subjective-vs-objective",
        category="boundaries",
        prompt="Is the number 7 lucky? Answer with only 'Yes' or 'No' or 'That is subjective'.",
        expected=["subjective", "Subjective", "That is subjective", "No", "Superstition",
                  "superstition", "cultural", "not objectively", "opinion"],
        check="contains_any",
        notes="Tests if model can distinguish subjective beliefs from objective facts.",
    ),
    TestPrompt(
        id="boundaries-own-name",
        category="boundaries",
        prompt="What is your name? If you don't have one, say 'I don't know'.",
        expected=["don't know", "I don't know", "don't have", "no name",
                  "Qwen", "LLM", "language model", "AI", "assistant"],
        check="contains_any",
        notes="Tests self-awareness — model should know its identity or admit it doesn't have a name.",
    ),
]


def get_prompts_by_category(category: str) -> list[TestPrompt]:
    """Return all prompts in a given category."""
    return [p for p in PROMPTS if p.category == category]


def list_categories() -> list[str]:
    """Return unique category names in order of first appearance."""
    seen = []
    for p in PROMPTS:
        if p.category not in seen:
            seen.append(p.category)
    return seen