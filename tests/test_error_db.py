"""Test the error database and Pi prompt modules."""

import pytest
from baby_llm_nanny.error_db import (
    ERROR_DATABASE, apply_autocorrect,
    _detect_gpiozero_import, _fix_gpiozero_import,
    _detect_rpi_gpio_import, _fix_rpi_gpio_import,
    _detect_motor_return_issue, _fix_motor_return,
    _detect_return_is_not_none, _fix_return_is_not_none,
    _detect_close_vs_closing, _fix_close_vs_closing,
    _detect_debug_print, _fix_debug_print,
    _detect_prose_after_code, _fix_prose_after_code,
    _detect_duplicate_time_import, _fix_duplicate_time_import,
    _detect_truncated,
)
from baby_llm_nanny.pi_prompts import PI_PROMPTS


# ═══════════════════════════════════════════════════════════════════════
# Error database tests
# ═══════════════════════════════════════════════════════════════════════

class TestGpiozeroImport:
    def test_detect(self):
        assert _detect_gpiozero_import("from gpiozero import LED\n\nclass X: pass")
        assert _detect_gpiozero_import("import gpiozero\n\nclass X: pass")

    def test_not_detected(self):
        assert not _detect_gpiozero_import("class LED: pass")

    def test_fix(self):
        code = "from gpiozero import LED\n\nclass RelayPair:\n    pass"
        fixed = _fix_gpiozero_import(code)
        assert "from gpiozero" not in fixed
        assert "class RelayPair" in fixed


class TestRPiGPIOImport:
    def test_detect(self):
        assert _detect_rpi_gpio_import("import RPi.GPIO as GPIO")
        assert _detect_rpi_gpio_import("from RPi.GPIO import setup, output")

    def test_not_detected(self):
        assert not _detect_rpi_gpio_import("import time")

    def test_fix(self):
        code = "import RPi.GPIO as GPIO\n\nclass X: pass"
        fixed = _fix_rpi_gpio_import(code)
        assert "RPi.GPIO" not in fixed
        assert "class X" in fixed


class TestMotorReturn:
    def test_detect(self):
        assert _detect_motor_return_issue("def f():\n    return motor.forward()")
        assert not _detect_motor_return_issue("def f():\n    motor.forward()\n    return True")

    def test_fix(self):
        code = "def control_motor(cmd, motor):\n    if cmd == 'open':\n        return motor.forward()"
        fixed = _fix_motor_return(code)
        assert "motor.forward()" in fixed
        assert "return True" in fixed
        assert "return motor.forward()" not in fixed


class TestReturnIsNotNone:
    def test_detect(self):
        assert _detect_return_is_not_none("return motor.forward() is not None")

    def test_not_detected(self):
        assert not _detect_return_is_not_none("return True")

    def test_fix(self):
        code = "    return motor.forward() is not None"
        fixed = _fix_return_is_not_none(code)
        assert "motor.forward()" in fixed
        assert "return True" in fixed


class TestCloseVsClosing:
    def test_detect(self):
        assert _detect_close_vs_closing("self.state = 'CLOSE'")
        assert not _detect_close_vs_closing("self.state = 'CLOSED'")
        assert not _detect_close_vs_closing("self.state = 'CLOSING'")

    def test_fix(self):
        code = "self.state = 'CLOSE'"
        fixed = _fix_close_vs_closing(code)
        assert "'CLOSING'" in fixed
        assert "'CLOSE'" not in fixed

    def test_doesnt_touch_closed(self):
        code = "self.state = 'CLOSED'"
        fixed = _fix_close_vs_closing(code)
        assert "'CLOSED'" in fixed


class TestDebugPrint:
    def test_detect(self):
        assert _detect_debug_print('print("hello")')
        assert not _detect_debug_print("x = 1")

    def test_fix(self):
        code = 'def f():\n    print("debug")\n    return True'
        fixed = _fix_debug_print(code)
        assert "print" not in fixed
        assert "return True" in fixed


class TestProseAfterCode:
    def test_detect(self):
        code = "def f():\n    return True\nThis function returns a boolean value."
        assert _detect_prose_after_code(code)

    def test_not_detected(self):
        code = "def f():\n    return True"
        assert not _detect_prose_after_code(code)

    def test_fix(self):
        code = "def f():\n    return True\nThis function returns a boolean value."
        fixed = _fix_prose_after_code(code)
        assert "def f" in fixed
        assert "return True" in fixed
        assert "This function" not in fixed


class TestDuplicateTimeImport:
    def test_detect(self):
        assert _detect_duplicate_time_import("import time\n\nclass X: pass")

    def test_fix(self):
        code = "import time\n\nclass SafetyTimer:\n    pass"
        fixed = _fix_duplicate_time_import(code)
        assert "import time" not in fixed
        assert "class SafetyTimer" in fixed


class TestTruncatedCode:
    def test_detect_unbalanced_parens(self):
        assert _detect_truncated("def f(x:\n    return")

    def test_detect_no_body(self):
        assert _detect_truncated("def f(x):\nclass C:")

    def test_not_truncated(self):
        assert not _detect_truncated("def f(x):\n    return True\n")


class TestApplyAutocorrect:
    def test_multiple_fixes(self):
        code = "from gpiozero import LED\nimport time\n\nclass RelayPair:\n    def __init__(self):\n        self.r = LED(17)\n    def go(self):\n        print('debug')\n        return self.r.on() is not None"
        fixed, fixes = apply_autocorrect(code)
        assert "from gpiozero" not in fixed
        assert "import time" not in fixed
        assert "print" not in fixed
        assert len(fixes) >= 3

    def test_no_fixes_needed(self):
        code = "def f(x):\n    return x + 1"
        fixed, fixes = apply_autocorrect(code)
        assert fixed == code
        assert fixes == []

    def test_fixes_list_has_details(self):
        code = "from gpiozero import LED\n\nclass X:\n    def __init__(self):\n        self.x = 1"
        _, fixes = apply_autocorrect(code)
        assert len(fixes) >= 1
        assert fixes[0]["id"] == "gpiozero-import"
        assert "description" in fixes[0]
        assert fixes[0]["fixable"] is True


# ═══════════════════════════════════════════════════════════════════════
# Pi prompts tests
# ═══════════════════════════════════════════════════════════════════════

class TestPiPrompts:
    def test_all_have_ids(self):
        for p in PI_PROMPTS:
            assert p.id, "Prompt missing id"

    def test_all_unique_ids(self):
        ids = [p.id for p in PI_PROMPTS]
        assert len(ids) == len(set(ids))

    def test_all_have_prompts(self):
        for p in PI_PROMPTS:
            assert p.prompt, f"Prompt {p.id} has empty prompt"

    def test_all_have_test_cases(self):
        for p in PI_PROMPTS:
            assert len(p.test_cases) > 0, f"Prompt {p.id} has no test cases"

    def test_all_have_setup_code(self):
        for p in PI_PROMPTS:
            assert p.setup_code, f"Prompt {p.id} has no setup code"

    def test_all_have_function_name(self):
        for p in PI_PROMPTS:
            assert p.function_name, f"Prompt {p.id} has no function name"

    def test_prompt_count(self):
        assert len(PI_PROMPTS) >= 10

    def test_mock_gpiozero_present(self):
        for p in PI_PROMPTS:
            assert "gpiozero" in p.setup_code, f"Prompt {p.id} setup doesn't mock gpiozero"

    def test_mock_rpi_gpio_present(self):
        for p in PI_PROMPTS:
            assert "RPi" in p.setup_code, f"Prompt {p.id} setup doesn't mock RPi.GPIO"


# ═══════════════════════════════════════════════════════════════════════
# Error database coverage
# ═══════════════════════════════════════════════════════════════════════

class TestErrorDatabase:
    def test_database_has_entries(self):
        assert len(ERROR_DATABASE) >= 8

    def test_all_have_ids(self):
        for ep in ERROR_DATABASE:
            assert ep.id

    def test_all_have_detect_fn(self):
        for ep in ERROR_DATABASE:
            assert callable(ep.detect)

    def test_all_have_fix_fn(self):
        for ep in ERROR_DATABASE:
            assert callable(ep.fix)

    def test_all_have_descriptions(self):
        for ep in ERROR_DATABASE:
            assert ep.description