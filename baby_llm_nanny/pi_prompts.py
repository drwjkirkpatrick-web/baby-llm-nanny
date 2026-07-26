"""Raspberry Pi garage door simulation prompts for baby-llm-nanny.

These prompts test the kinds of code hackathon students would write for a
garage door simulation on Raspberry Pi with both real GPIO and simulated modes.

Categories covered:
  - State machine logic (door states: CLOSED, OPENING, OPEN, CLOSING, STOPPED)
  - GPIO pin control (motor relays, limit switches, button input)
  - Sensor reading and debouncing
  - Safety timeout (auto-stop motor)
  - Obstacle detection (reverse on contact)
  - Timed auto-close
  - gpiozero API patterns
  - KeyboardInterrupt / cleanup handling

Each prompt has test cases that run in a mocked environment (no real GPIO needed).
The test harness mocks `gpiozero` and `RPi.GPIO` so code can be verified headless.
"""

from dataclasses import dataclass
from typing import Union


@dataclass
class PiPrompt:
    __test__ = False
    id: str
    prompt: str
    function_name: str
    test_cases: list
    code_pattern: str
    setup_code: str  # Mock GPIO/gpiozero setup injected before the model's code
    notes: str


# ─────────────────────────────────────────────────────────────────────
# Mock setup code — injected before the model's code in test subprocess
# This lets us run Pi code headless without real GPIO hardware
# ─────────────────────────────────────────────────────────────────────

MOCK_GPIOZERO = '''
# --- Mock gpiozero for headless testing ---
import sys
import time as _time

class _MockButton:
    """Mock gpiozero.Button for headless testing."""
    def __init__(self, pin, pull_up=True, bounce_time=None, hold_time=1, hold_repeat=False):
        self.pin = pin
        self.pull_up = pull_up
        self.bounce_time = bounce_time
        self.is_pressed = False
        self._when_pressed = None
        self._when_released = None
    @property
    def when_pressed(self):
        return self._when_pressed
    @when_pressed.setter
    def when_pressed(self, fn):
        self._when_pressed = fn
    @property
    def when_released(self):
        return self._when_released
    @when_released.setter
    def when_released(self, fn):
        self._when_released = fn
    def press(self):
        self.is_pressed = True
        if self._when_pressed:
            self._when_pressed()
    def release(self):
        self.is_pressed = False
        if self._when_released:
            self._when_released()

class _MockLED:
    """Mock gpiozero.LED for headless testing."""
    def __init__(self, pin):
        self.pin = pin
        self.is_lit = False
    def on(self):
        self.is_lit = True
    def off(self):
        self.is_lit = False
    def toggle(self):
        self.is_lit = not self.is_lit

class _MockMotor:
    """Mock gpiozero.Motor for headless testing."""
    def __init__(self, forward_pin, backward_pin):
        self.forward_pin = forward_pin
        self.backward_pin = backward_pin
        self.direction = None  # 'forward', 'backward', or None
    def forward(self):
        self.direction = 'forward'
    def backward(self):
        self.direction = 'backward'
    def stop(self):
        self.direction = None

class _MockPWMLED:
    """Mock gpiozero.PWMLED for headless testing."""
    def __init__(self, pin):
        self.pin = pin
        self.value = 0.0
    def on(self):
        self.value = 1.0
    def off(self):
        self.value = 0.0

class _MockServo:
    """Mock gpiozero.Servo for headless testing."""
    def __init__(self, pin):
        self.pin = pin
        self.value = 0.0
    def min(self):
        self.value = -1.0
    def mid(self):
        self.value = 0.0
    def max(self):
        self.value = 1.0

# Build mock module
class _MockGpiozero:
    Button = _MockButton
    LED = _MockLED
    Motor = _MockMotor
    PWMLED = _MockPWMLED
    Servo = _MockServo

sys.modules['gpiozero'] = _MockGpiozero

# --- Mock RPi.GPIO for headless testing ---
class _MockRPiGPIO:
    BCM = 11
    BOARD = 10
    IN = 1
    OUT = 0
    HIGH = 1
    LOW = 0
    PUD_UP = 22
    PUD_DOWN = 21
    RISING = 31
    FALLING = 32
    BOTH = 33

    _pin_states = {}
    _callbacks = {}

    @classmethod
    def setmode(cls, mode):
        cls._mode = mode
    @classmethod
    def setup(cls, pin, direction, pull_up_down=None, initial=None):
        cls._pin_states[pin] = direction
    @classmethod
    def output(cls, pin, value):
        cls._pin_states[pin] = value
    @classmethod
    def input(cls, pin):
        return cls._pin_states.get(pin, 0)
    @classmethod
    def add_event_detect(cls, pin, edge, callback=None, bouncetime=None):
        cls._callbacks[pin] = callback
    @classmethod
    def remove_event_detect(cls, pin):
        cls._callbacks.pop(pin, None)
    @classmethod
    def cleanup(cls, pin=None):
        if pin:
            cls._pin_states.pop(pin, None)
        else:
            cls._pin_states.clear()
            cls._callbacks.clear()
    @classmethod
    def setwarnings(cls, flag):
        pass

sys.modules['RPi'] = type(sys)('RPi')
sys.modules['RPi.GPIO'] = _MockRPiGPIO
'''

# Common setup used by most prompts — just the mock
GPIO_SETUP = MOCK_GPIOZERO

# ─────────────────────────────────────────────────────────────────────
# The prompts
# ─────────────────────────────────────────────────────────────────────

PI_PROMPTS: list[PiPrompt] = [

    # ═══════════════════════════════════════════════════════════════
    # STATE MACHINE — door state transitions
    # ═══════════════════════════════════════════════════════════════

    PiPrompt(
        id="pi-garage-state-machine",
        prompt=(
            "Write a Python class called `GarageDoor` that simulates a garage door "
            "with 5 states: CLOSED, OPENING, OPEN, CLOSING, STOPPED. "
            "The class should have:\n"
            "- A `state` attribute initialized to 'CLOSED'\n"
            "- A method `open_door()` that transitions CLOSED→OPENING→OPEN "
            "(sets state to 'OPENING' immediately, then 'OPEN' when called again)\n"
            "- A method `close_door()` that transitions OPEN→CLOSING→CLOSED "
            "(sets state to 'CLOSING' immediately, then 'CLOSED' when called again)\n"
            "- A method `stop()` that sets state to 'STOPPED'\n"
            "Return only the class definition, no explanation."
        ),
        function_name="GarageDoor",
        code_pattern=r"class\s+GarageDoor\s*:.*?(?=\nclass\s|\Z)",
        setup_code=GPIO_SETUP,
        test_cases=[
            ({"action": "open", "calls": 1}, "OPENING"),
            ({"action": "open", "calls": 2}, "OPEN"),
            ({"action": "close", "calls": 1}, "CLOSING"),
            ({"action": "close", "calls": 2}, "CLOSED"),
            ({"action": "stop", "calls": 1}, "STOPPED"),
        ],
        notes="Core state machine — tests if model can implement a 2-phase transition.",
    ),

    PiPrompt(
        id="pi-garage-state-guard",
        prompt=(
            "Write a Python class called `SafeGarageDoor` with a `state` attribute "
            "initialized to 'CLOSED'. "
            "Implement `open_door()` that only works if state is 'CLOSED' or 'STOPPED' "
            "(returns False otherwise), and `close_door()` that only works if state "
            "is 'OPEN' or 'STOPPED' (returns False otherwise). "
            "Each method returns True if the action was allowed, False if not. "
            "When allowed, open_door sets state to 'OPENING' and close_door sets "
            "state to 'CLOSING'. Return only the class definition."
        ),
        function_name="SafeGarageDoor",
        code_pattern=r"class\s+SafeGarageDoor\s*:.*?(?=\nclass\s|\Z)",
        setup_code=GPIO_SETUP,
        test_cases=[
            ({"from_state": "CLOSED", "action": "open"}, (True, "OPENING")),
            ({"from_state": "OPENING", "action": "close"}, (False, "OPENING")),
            ({"from_state": "OPEN", "action": "close"}, (True, "CLOSING")),
            ({"from_state": "CLOSING", "action": "open"}, (False, "CLOSING")),
            ({"from_state": "STOPPED", "action": "open"}, (True, "OPENING")),
            ({"from_state": "STOPPED", "action": "close"}, (True, "CLOSING")),
        ],
        notes="State guard — tests if model checks current state before allowing transitions.",
    ),

    # ═══════════════════════════════════════════════════════════════
    # GPIO CONTROL — motor relay control
    # ═══════════════════════════════════════════════════════════════

    PiPrompt(
        id="pi-gpio-motor-control",
        prompt=(
            "Write a Python function called `control_motor` that takes a string "
            "command ('open', 'close', 'stop') and a mock motor object that has "
            "methods `forward()`, `backward()`, and `stop()`. "
            "For 'open', call motor.forward(). "
            "For 'close', call motor.backward(). "
            "For 'stop', call motor.stop(). "
            "Return True for valid commands, False for unknown commands. "
            "Return only the function, no explanation."
        ),
        function_name="control_motor",
        code_pattern=r"def\s+control_motor\s*\([^)]*\)\s*:.*?(?=\ndef\s|\Z)",
        setup_code=GPIO_SETUP,
        test_cases=[
            ({"command": "open"}, (True, "forward")),
            ({"command": "close"}, (True, "backward")),
            ({"command": "stop"}, (True, None)),
            ({"command": "invalid"}, (False, None)),
        ],
        notes="Motor relay control — tests if model can map commands to motor methods.",
    ),

    PiPrompt(
        id="pi-gpio-relay-pair",
        prompt=(
            "Write a Python class called `RelayPair` that controls two relays "
            "for a garage door motor: relay_open (pin 17) and relay_close (pin 27). "
            "Use the mock `gpiozero.LED` class for each relay. "
            "Methods:\n"
            "- `activate_open()`: turns relay_open on, relay_close off\n"
            "- `activate_close()`: turns relay_close on, relay_open off\n"
            "- `deactivate()`: turns both relays off\n"
            "Return only the class definition."
        ),
        function_name="RelayPair",
        code_pattern=r"class\s+RelayPair\s*:.*?(?=\nclass\s|\Z)",
        setup_code=GPIO_SETUP,
        test_cases=[
            ({"action": "activate_open"}, (True, False)),
            ({"action": "activate_close"}, (False, True)),
            ({"action": "deactivate"}, (False, False)),
        ],
        notes="Two-relay control — tests GPIO output and mutual exclusion (never both on).",
    ),

    # ═══════════════════════════════════════════════════════════════
    # SENSOR — limit switches and debouncing
    # ═══════════════════════════════════════════════════════════════

    PiPrompt(
        id="pi-sensor-limit-switch",
        prompt=(
            "Write a Python function called `check_limit_switches` that takes two "
            "booleans: `top_switch` (True if door fully open) and `bottom_switch` "
            "(True if door fully closed). "
            "Return a string:\n"
            "- 'fully_open' if top_switch is True\n"
            "- 'fully_closed' if bottom_switch is True\n"
            "- 'in_transition' if both are False\n"
            "- 'error' if both are True (should never happen)\n"
            "Return only the function, no explanation."
        ),
        function_name="check_limit_switches",
        code_pattern=r"def\s+check_limit_switches\s*\([^)]*\)\s*:.*?(?=\ndef\s|\Z)",
        setup_code=GPIO_SETUP,
        test_cases=[
            ({"top_switch": True, "bottom_switch": False}, "fully_open"),
            ({"top_switch": False, "bottom_switch": True}, "fully_closed"),
            ({"top_switch": False, "bottom_switch": False}, "in_transition"),
            ({"top_switch": True, "bottom_switch": True}, "error"),
        ],
        notes="Limit switch reading — tests priority logic and error state detection.",
    ),

    PiPrompt(
        id="pi-sensor-debounce",
        prompt=(
            "Write a Python function called `debounce_read` that takes a function "
            "`read_fn` (which returns True or False) and an integer `reads` (default 3). "
            "Call `read_fn` `reads` times. If all reads agree (all True or all False), "
            "return that value. If reads disagree, return None (unstable). "
            "Return only the function, no explanation."
        ),
        function_name="debounce_read",
        code_pattern=r"def\s+debounce_read\s*\([^)]*\)\s*:.*?(?=\ndef\s|\Z)",
        setup_code=GPIO_SETUP,
        test_cases=[
            ({"read_fn": "lambda: True", "reads": 3}, True),
            ({"read_fn": "lambda: False", "reads": 3}, False),
            ({"read_fn": "lambda: None", "reads": 1}, None),  # None → unstable
        ],
        notes="Button debouncing — tests if model can implement repeated-read consensus.",
    ),

    # ═══════════════════════════════════════════════════════════════
    # SAFETY — timeout and obstacle detection
    # ═══════════════════════════════════════════════════════════════

    PiPrompt(
        id="pi-safety-timeout",
        prompt=(
            "Write a Python class called `SafetyTimer` that prevents a garage door "
            "motor from running too long. "
            "Constructor takes `max_seconds` (default 30). "
            "Methods:\n"
            "- `start()`: records the current time using `time.time()`\n"
            "- `is_expired()`: returns True if `max_seconds` have elapsed since "
            "`start()` was called, False otherwise\n"
            "- `reset()`: resets the start time to now\n"
            "Return only the class definition."
        ),
        function_name="SafetyTimer",
        code_pattern=r"class\s+SafetyTimer\s*:.*?(?=\nclass\s|\Z)",
        setup_code=GPIO_SETUP + "\nimport time\n",
        test_cases=[
            ({"max_seconds": 0, "wait": 0}, True),   # 0s timeout → immediately expired
            ({"max_seconds": 100, "wait": 0}, False),  # 100s timeout, no wait → not expired
        ],
        notes="Safety timeout — tests if model can track elapsed time and compare against a max.",
    ),

    PiPrompt(
        id="pi-safety-obstacle-detect",
        prompt=(
            "Write a Python function called `handle_obstacle` that takes:\n"
            "- `door_state` (string: 'CLOSING', 'OPENING', 'OPEN', 'CLOSED', 'STOPPED')\n"
            "- `obstacle_detected` (bool)\n"
            "If obstacle_detected is True and door_state is 'CLOSING', return 'REVERSE' "
            "(open the door back up).\n"
            "If obstacle_detected is True and door_state is 'OPENING', return 'STOP'.\n"
            "If obstacle_detected is False, return the current door_state unchanged.\n"
            "Return only the function, no explanation."
        ),
        function_name="handle_obstacle",
        code_pattern=r"def\s+handle_obstacle\s*\([^)]*\)\s*:.*?(?=\ndef\s|\Z)",
        setup_code=GPIO_SETUP,
        test_cases=[
            ({"door_state": "CLOSING", "obstacle_detected": True}, "REVERSE"),
            ({"door_state": "OPENING", "obstacle_detected": True}, "STOP"),
            ({"door_state": "CLOSING", "obstacle_detected": False}, "CLOSING"),
            ({"door_state": "OPEN", "obstacle_detected": True}, "OPEN"),
        ],
        notes="Obstacle detection — tests safety reversal logic (close→reverse, open→stop).",
    ),

    # ═══════════════════════════════════════════════════════════════
    # BUTTON HANDLING — debounced press, toggle logic
    # ═══════════════════════════════════════════════════════════════

    PiPrompt(
        id="pi-button-toggle",
        prompt=(
            "Write a Python class called `ButtonToggle` that tracks garage door state "
            "based on button presses. "
            "Constructor sets `is_open` to False (door starts closed). "
            "Method `on_press()` toggles `is_open` (False→True→False→...). "
            "Method `get_state()` returns 'OPEN' if is_open is True, 'CLOSED' otherwise. "
            "Return only the class definition."
        ),
        function_name="ButtonToggle",
        code_pattern=r"class\s+ButtonToggle\s*:.*?(?=\nclass\s|\Z)",
        setup_code=GPIO_SETUP,
        test_cases=[
            ({"presses": 0}, "CLOSED"),
            ({"presses": 1}, "OPEN"),
            ({"presses": 2}, "CLOSED"),
            ({"presses": 3}, "OPEN"),
        ],
        notes="Button toggle — tests simple state toggle on button press.",
    ),

    # ═══════════════════════════════════════════════════════════════
    # AUTO-CLOSE — timed auto-close after N seconds
    # ═══════════════════════════════════════════════════════════════

    PiPrompt(
        id="pi-auto-close-check",
        prompt=(
            "Write a Python function called `should_auto_close` that takes:\n"
            "- `door_state` (string)\n"
            "- `seconds_open` (float): how long the door has been open\n"
            "- `auto_close_after` (float): threshold in seconds (default 60)\n"
            "Return True if door_state is 'OPEN' AND seconds_open >= auto_close_after. "
            "Return False otherwise. "
            "Return only the function, no explanation."
        ),
        function_name="should_auto_close",
        code_pattern=r"def\s+should_auto_close\s*\([^)]*\)\s*:.*?(?=\ndef\s|\Z)",
        setup_code=GPIO_SETUP,
        test_cases=[
            ({"door_state": "OPEN", "seconds_open": 90, "auto_close_after": 60}, True),
            ({"door_state": "OPEN", "seconds_open": 30, "auto_close_after": 60}, False),
            ({"door_state": "CLOSED", "seconds_open": 90, "auto_close_after": 60}, False),
            ({"door_state": "OPENING", "seconds_open": 90, "auto_close_after": 60}, False),
        ],
        notes="Auto-close logic — tests conditional on both state and elapsed time.",
    ),

    # ═══════════════════════════════════════════════════════════════
    # CLEANUP — safe shutdown
    # ═══════════════════════════════════════════════════════════════

    PiPrompt(
        id="pi-safe-shutdown",
        prompt=(
            "Write a Python function called `safe_shutdown` that takes a motor object "
            "(with `.stop()` method) and a list of GPIO pin objects (each with `.off()` "
            "method). The function should:\n"
            "1. Call motor.stop()\n"
            "2. Call .off() on each pin in the list\n"
            "3. Return True if all cleanup succeeded\n"
            "If any call raises an exception, catch it and return False. "
            "Return only the function, no explanation."
        ),
        function_name="safe_shutdown",
        code_pattern=r"def\s+safe_shutdown\s*\([^)]*\)\s*:.*?(?=\ndef\s|\Z)",
        setup_code=GPIO_SETUP,
        test_cases=[
            ({"motor_good": True, "pins_good": True}, True),
            ({"motor_good": False, "pins_good": True}, False),
        ],
        notes="Safe shutdown — tests exception handling during GPIO cleanup.",
    ),

    # ═══════════════════════════════════════════════════════════════
    # INTEGRATION — full door controller
    # ═══════════════════════════════════════════════════════════════

    PiPrompt(
        id="pi-full-controller",
        prompt=(
            "Write a Python class called `GarageController` that combines state machine "
            "and motor control. "
            "Constructor takes a `motor` object (with .forward(), .backward(), .stop()).\n"
            "State attribute starts at 'CLOSED'. "
            "Methods:\n"
            "- `open()`: if state is 'CLOSED', set state to 'OPENING', call motor.forward(), "
            "return True. Otherwise return False.\n"
            "- `close()`: if state is 'OPEN', set state to 'CLOSING', call motor.backward(), "
            "return True. Otherwise return False.\n"
            "- `door_fully_open()`: if state is 'OPENING', set state to 'OPEN', call "
            "motor.stop(), return True. Otherwise return False.\n"
            "- `door_fully_closed()`: if state is 'CLOSING', set state to 'CLOSED', call "
            "motor.stop(), return True. Otherwise return False.\n"
            "Return only the class definition."
        ),
        function_name="GarageController",
        code_pattern=r"class\s+GarageController\s*:.*?(?=\nclass\s|\Z)",
        setup_code=GPIO_SETUP,
        test_cases=[
            ({"method": "open", "initial_state": "CLOSED"}, (True, "OPENING", "forward")),
            ({"method": "open", "initial_state": "OPEN"}, (False, "OPEN", None)),
            ({"method": "close", "initial_state": "OPEN"}, (True, "CLOSING", "backward")),
            ({"method": "door_fully_open", "initial_state": "OPENING"}, (True, "OPEN", None)),
            ({"method": "door_fully_closed", "initial_state": "CLOSING"}, (True, "CLOSED", None)),
        ],
        notes="Full controller — integration test combining state machine + motor control.",
    ),
]