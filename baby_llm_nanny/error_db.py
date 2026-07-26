"""Curated error database for Raspberry Pi garage door code generation.

Built from real Qwen 2.5 3B outputs on 12 Pi/garage door coding prompts.
Each error pattern includes:
  - id: unique error identifier
  - pattern: regex or string match that detects the error in model output
  - fix: function(code: str) -> str that patches the error
  - description: human-readable explanation
  - affected_prompts: which prompt IDs this error was observed in
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ErrorPattern:
    """A curated error pattern with detection and automatic fix."""
    __test__ = False
    id: str
    description: str
    affected_prompts: list[str]
    detect: Callable[[str], bool]  # Returns True if this error is present
    fix: Callable[[str], str]      # Returns patched code


# ═══════════════════════════════════════════════════════════════════════
# Error 1: Truncated code — response cut off mid-function
# ═══════════════════════════════════════════════════════════════════════

def _detect_truncated(code: str) -> bool:
    """Detect code that ends abruptly — unbalanced braces/def without body."""
    lines = code.strip().split("\n")
    if not lines:
        return False
    # Check for def/class line with no body after it
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("class "):
            # Is there a non-empty line after this that's indented?
            has_body = False
            for j in range(i + 1, len(lines)):
                if lines[j].strip():
                    has_body = True
                    break
            if not has_body:
                return True
    # Check for unbalanced parens/brackets
    open_parens = code.count("(") - code.count(")")
    open_brackets = code.count("[") - code.count("]")
    open_braces = code.count("{") - code.count("}")
    if open_parens > 0 or open_brackets > 0 or open_braces > 0:
        return True
    return False


def _fix_truncated(code: str) -> str:
    """Can't auto-fix truncation, but we can note it for the reviewer loop."""
    return code  # Truncation requires re-generation, not patching


# ═══════════════════════════════════════════════════════════════════════
# Error 2: Return value errors — `motor.forward() is not None` instead of True
# ═══════════════════════════════════════════════════════════════════════

def _detect_return_is_not_none(code: str) -> bool:
    """Detect `return X.method() is not None` pattern."""
    return bool(re.search(r"return\s+\w+\.\w+\(\)\s+is\s+not\s+None", code))


def _fix_return_is_not_none(code: str) -> str:
    """Replace `return X.method() is not None` with `return True`."""
    # Pattern: return motor.forward() is not None → motor.forward(); return True
    # We need to preserve the method call but change the return
    def replacer(m):
        call = m.group(1)
        return f"{call}\n        return True"

    code = re.sub(
        r"return\s+(\w+\.\w+\(\))\s+is\s+not\s+None",
        lambda m: f"{m.group(1)}\n        return True",
        code,
    )
    return code


# ═══════════════════════════════════════════════════════════════════════
# Error 3: Missing import — `from gpiozero import LED` fails in mock env
# ═══════════════════════════════════════════════════════════════════════

def _detect_gpiozero_import(code: str) -> bool:
    """Detect `from gpiozero import` or `import gpiozero` — needs removal in mock env."""
    return bool(re.search(r"(?:from\s+gpiozero\s+import|import\s+gpiozero)", code))


def _fix_gpiozero_import(code: str) -> str:
    """Remove gpiozero import lines — the mock module is already injected."""
    lines = code.split("\n")
    filtered = [line for line in lines
                if not re.match(r"\s*(?:from\s+gpiozero\s+import|import\s+gpiozero)", line)]
    return "\n".join(filtered)


# ═══════════════════════════════════════════════════════════════════════
# Error 4: Missing import — `import RPi.GPIO` fails on non-Pi
# ═══════════════════════════════════════════════════════════════════════

def _detect_rpi_gpio_import(code: str) -> bool:
    """Detect `import RPi.GPIO` or `from RPi.GPIO import`."""
    return bool(re.search(r"(?:from\s+RPi\.GPIO\s+import|import\s+RPi\.GPIO|import\s+RPi\b)", code))


def _fix_rpi_gpio_import(code: str) -> str:
    """Remove RPi.GPIO import lines — the mock module is already injected."""
    lines = code.split("\n")
    filtered = [line for line in lines
                if not re.match(r"\s*(?:from\s+RPi\.GPIO\s+import|import\s+RPi\.GPIO|import\s+RPi\b)", line)]
    return "\n".join(filtered)


# ═══════════════════════════════════════════════════════════════════════
# Error 5: State transition skipping — sets state to final state immediately
# Pattern: self.state = 'OPENING' immediately followed by self.state = 'OPEN'
# ═══════════════════════════════════════════════════════════════════════

def _detect_state_skip(code: str) -> bool:
    """Detect state being set to intermediate then immediately to final in same method."""
    lines = code.split("\n")
    for i, line in enumerate(lines):
        if re.search(r"self\.state\s*=\s*['\"]OPENING['\"]", line):
            for j in range(i + 1, min(i + 5, len(lines))):
                next_line = lines[j].strip()
                # Skip if this is a conditional line
                if re.search(r"if\s+|elif\s+|else:", next_line):
                    break
                if re.search(r"self\.state\s*=\s*['\"]OPEN['\"]", next_line):
                    return True
    return False


def _fix_state_skip(code: str) -> str:
    """Remove the premature final-state assignment in two-phase transitions.

    If a method sets state to OPENING and then to OPEN without a condition,
    remove the OPEN assignment (it should only happen on the next call).
    """
    # This is complex — we remove lines that set state to 'OPEN' immediately
    # after setting to 'OPENING' in the same method, without any if/else
    lines = code.split("\n")
    new_lines = []
    skip_next_open = False

    for i, line in enumerate(lines):
        if re.search(r"self\.state\s*=\s*['\"]OPENING['\"]", line.strip()):
            skip_next_open = True
            new_lines.append(line)

        elif skip_next_open and re.search(r"self\.state\s*=\s*['\"]OPEN['\"]", line.strip()):
            # Check if there's a conditional — if not, skip this line
            # (it's the premature final-state assignment)
            prev = new_lines[-1].strip() if new_lines else ""
            if not re.search(r"if\s+|elif\s+|else:|while\s+|for\s+", prev):
                continue  # Skip this premature assignment
            skip_next_open = False
            new_lines.append(line)

        else:
            if skip_next_open and (line.strip().startswith("def ") or line.strip().startswith("class ")):
                skip_next_open = False
            new_lines.append(line)

    return "\n".join(new_lines)


# ═══════════════════════════════════════════════════════════════════════
# Error 6: Wrong state string — 'CLOSE' instead of 'CLOSING', 'OPEN' used loosely
# ═══════════════════════════════════════════════════════════════════════

def _detect_close_vs_closing(code: str) -> bool:
    """Detect 'CLOSE' used where 'CLOSING' was expected."""
    # In garage door context, 'CLOSE' is not a valid state — it should be 'CLOSING'
    return bool(re.search(r"['\"]CLOSE['\"]", code)) and not bool(re.search(r"['\"]CLOSING['\"]", code))


def _fix_close_vs_closing(code: str) -> str:
    """Replace 'CLOSE' with 'CLOSING' when it appears as a state string."""
    # Only replace 'CLOSE' not 'CLOSED'
    code = re.sub(r"['\"]CLOSE['\"](?!D)", "'CLOSING'", code)
    return code


# ═══════════════════════════════════════════════════════════════════════
# Error 7: Prose/text after code block — model adds explanation after ```
# ═══════════════════════════════════════════════════════════════════════

def _detect_prose_after_code(code: str) -> bool:
    """Detect explanatory text mixed into the code (non-Python lines)."""
    lines = code.strip().split("\n")
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("class "):
            in_code = True
        if in_code and stripped:
            # After code starts, lines should be valid Python
            # Check for prose: starts with capital, ends with period, no Python syntax
            if (re.match(r"^[A-Z][a-z]+ [a-z]+ ", stripped)
                and stripped.endswith(".")
                and not stripped.startswith(("def ", "class ", "import ", "from ", "#", "@"))):
                return True
    return False


def _fix_prose_after_code(code: str) -> str:
    """Remove prose lines from the code (lines that aren't valid Python)."""
    lines = code.split("\n")
    new_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("class "):
            in_code = True
        if in_code and stripped:
            # Check if it looks like prose
            if (re.match(r"^[A-Z][a-z]+ [a-z]+ ", stripped)
                and stripped.endswith(".")
                and not stripped.startswith(("def ", "class ", "import ", "from ", "#", "@", "return ", "if ", "elif ", "else:", "for ", "while ", "try:", "except", "raise "))):
                continue  # Skip prose
        new_lines.append(line)
    return "\n".join(new_lines)


# ═══════════════════════════════════════════════════════════════════════
# Error 8: `import time` inside generated code when already provided by setup
# ═══════════════════════════════════════════════════════════════════════

def _detect_duplicate_time_import(code: str) -> bool:
    """Detect redundant `import time` when setup already provides it."""
    return bool(re.match(r"^\s*import\s+time\s*$", code, re.MULTILINE))


def _fix_duplicate_time_import(code: str) -> str:
    """Remove redundant `import time` lines."""
    lines = code.split("\n")
    filtered = [line for line in lines
                if not re.match(r"^\s*import\s+time\s*$", line)]
    return "\n".join(filtered)


# ═══════════════════════════════════════════════════════════════════════
# Error 9: Missing return True in boolean methods — returns method result instead
# ═══════════════════════════════════════════════════════════════════════

def _detect_motor_return_issue(code: str) -> bool:
    """Detect `return motor.X()` where the function should return True."""
    # In motor control: motor.forward() returns None, but function should return True
    return bool(re.search(r"return\s+\w+\.forward\(\)|return\s+\w+\.backward\(\)|return\s+\w+\.stop\(\)", code))


def _fix_motor_return(code: str) -> str:
    """Fix `return motor.X()` to `motor.X(); return True`."""
    # Replace: return motor.forward() → motor.forward()\n    return True
    code = re.sub(
        r"return\s+(\w+)\.(forward|backward|stop)\(\)",
        r"\1.\2()\n        return True",
        code,
    )
    return code


# ═══════════════════════════════════════════════════════════════════════
# Error 10: `print()` statements in code that should just set state
# ═══════════════════════════════════════════════════════════════════════

def _detect_debug_print(code: str) -> bool:
    """Detect print() statements that are debug noise, not part of the spec."""
    return bool(re.search(r"print\s*\(", code))


def _fix_debug_print(code: str) -> str:
    """Remove print() statements (they break tests that check return values)."""
    lines = code.split("\n")
    filtered = [line for line in lines
                if not re.match(r"\s*print\s*\(", line)]
    return "\n".join(filtered)


# ═══════════════════════════════════════════════════════════════════════
# The curated error database
# ═══════════════════════════════════════════════════════════════════════

ERROR_DATABASE: list[ErrorPattern] = [
    ErrorPattern(
        id="gpiozero-import",
        description="Model imports gpiozero which isn't available in test/mock env",
        affected_prompts=["pi-gpio-relay-pair"],
        detect=_detect_gpiozero_import,
        fix=_fix_gpiozero_import,
    ),
    ErrorPattern(
        id="rpi-gpio-import",
        description="Model imports RPi.GPIO which isn't available in test/mock env",
        affected_prompts=[],
        detect=_detect_rpi_gpio_import,
        fix=_fix_rpi_gpio_import,
    ),
    ErrorPattern(
        id="motor-return-none",
        description="Returns motor.forward() result (None) instead of True",
        affected_prompts=["pi-gpio-motor-control"],
        detect=_detect_motor_return_issue,
        fix=_fix_motor_return,
    ),
    ErrorPattern(
        id="return-is-not-none",
        description="Uses `return X.method() is not None` instead of returning True",
        affected_prompts=["pi-gpio-motor-control"],
        detect=_detect_return_is_not_none,
        fix=_fix_return_is_not_none,
    ),
    ErrorPattern(
        id="truncated-code",
        description="Code is truncated — response was cut off mid-function",
        affected_prompts=["pi-garage-state-machine", "pi-safe-shutdown"],
        detect=_detect_truncated,
        fix=_fix_truncated,  # No-op — requires re-generation
    ),
    ErrorPattern(
        id="close-vs-closing",
        description="Uses 'CLOSE' as a state string instead of 'CLOSING'",
        affected_prompts=["pi-full-controller"],
        detect=_detect_close_vs_closing,
        fix=_fix_close_vs_closing,
    ),
    ErrorPattern(
        id="prose-after-code",
        description="Explanatory prose mixed into the code block",
        affected_prompts=["pi-sensor-debounce", "pi-safe-shutdown"],
        detect=_detect_prose_after_code,
        fix=_fix_prose_after_code,
    ),
    ErrorPattern(
        id="debug-print",
        description="Debug print() statements in code that should return values",
        affected_prompts=["pi-garage-state-machine", "pi-safe-shutdown"],
        detect=_detect_debug_print,
        fix=_fix_debug_print,
    ),
    ErrorPattern(
        id="duplicate-time-import",
        description="Redundant `import time` when setup already provides it",
        affected_prompts=["pi-safety-timeout"],
        detect=_detect_duplicate_time_import,
        fix=_fix_duplicate_time_import,
    ),
    ErrorPattern(
        id="state-skip",
        description="Skips intermediate state (OPENING→OPEN in one call)",
        affected_prompts=["pi-garage-state-machine", "pi-full-controller"],
        detect=_detect_state_skip,
        fix=_fix_state_skip,
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# API: apply_autocorrect
# ═══════════════════════════════════════════════════════════════════════

def apply_autocorrect(code: str) -> tuple[str, list[dict]]:
    """Apply all applicable error fixes to a code snippet.

    Returns (fixed_code, list_of_fixes_applied).
    Each fix entry has: {"id": ..., "description": ..., "fixable": bool}
    """
    fixes_applied = []
    current = code

    for ep in ERROR_DATABASE:
        if ep.detect(current):
            before = current
            current = ep.fix(current)
            was_fixed = current != before
            fixes_applied.append({
                "id": ep.id,
                "description": ep.description,
                "fixable": was_fixed,
            })

    return current, fixes_applied