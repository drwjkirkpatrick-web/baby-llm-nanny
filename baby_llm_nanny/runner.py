"""Ollama runner — query local models via the Ollama HTTP API.

Designed for headless Jetson Orin Nano with no GUI.  Uses only stdlib
``urllib`` so there are zero external dependencies to install.
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 11434
DEFAULT_TIMEOUT = 120  # seconds — small models can be slow on first load


@dataclass
class ModelResponse:
    """Response from a single model query."""
    model: str
    prompt: str
    response: str
    response_time_sec: float
    eval_count: Optional[int] = None  # tokens generated
    prompt_eval_count: Optional[int] = None  # tokens in prompt
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _api_url(host: str, port: int, endpoint: str) -> str:
    return f"http://{host}:{port}/api/{endpoint}"


def check_ollama(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                timeout: int = 5) -> bool:
    """Return True if Ollama server is reachable and responding."""
    try:
        url = _api_url(host, port, "tags")
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def list_models(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                timeout: int = 5) -> list[str]:
    """Return list of installed model names.  Empty list if server down."""
    if not check_ollama(host, port, timeout):
        return []
    url = _api_url(host, port, "tags")
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def query_model(
    model: str,
    prompt: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: int = DEFAULT_TIMEOUT,
    temperature: float = 0.0,
    seed: int = 42,
    num_ctx: int = 4096,
) -> ModelResponse:
    """Send a prompt to an Ollama model and return the response.

    Uses temperature=0.0 and a fixed seed for reproducibility — we want
    deterministic outputs to compare across runs.

    Args:
        model:      Model name (e.g. "qwen2.5:3b").
        prompt:      The prompt text.
        host:        Ollama server hostname.
        port:        Ollama server port.
        timeout:     Request timeout in seconds.
        temperature: 0.0 for deterministic output.
        seed:        Fixed seed for reproducibility.
        num_ctx:     Context window size in tokens.

    Returns:
        ModelResponse with the model's reply and timing info.
    """
    url = _api_url(host, port, "generate")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_ctx": num_ctx,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            elapsed = time.time() - start
            data = json.loads(raw)
            return ModelResponse(
                model=model,
                prompt=prompt,
                response=data.get("response", "").strip(),
                response_time_sec=elapsed,
                eval_count=data.get("eval_count"),
                prompt_eval_count=data.get("prompt_eval_count"),
            )
    except urllib.error.URLError as e:
        elapsed = time.time() - start
        return ModelResponse(
            model=model, prompt=prompt, response="",
            response_time_sec=elapsed,
            error=f"Connection error: {e.reason}",
        )
    except Exception as e:
        elapsed = time.time() - start
        return ModelResponse(
            model=model, prompt=prompt, response="",
            response_time_sec=elapsed,
            error=f"Query error: {e}",
        )


def run_prompt_set(
    model: str,
    prompts: list,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: int = DEFAULT_TIMEOUT,
    temperature: float = 0.0,
    seed: int = 42,
    num_ctx: int = 4096,
    show_progress: bool = True,
) -> list[ModelResponse]:
    """Run a list of TestPrompt objects against a model.

    Returns a list of ModelResponse objects, one per prompt.
    """
    results = []
    total = len(prompts)
    for i, tp in enumerate(prompts):
        if show_progress:
            cat = tp.category if hasattr(tp, "category") else "?"
            pid = tp.id if hasattr(tp, "id") else f"#{i}"
            print(f"  [{i+1}/{total}] {cat}/{pid} ...", end="", flush=True)
        resp = query_model(
            model, tp.prompt, host, port, timeout, temperature, seed, num_ctx
        )
        results.append(resp)
        if show_progress:
            status = "✓" if resp.ok else "✗ ERROR"
            print(f" {status} ({resp.response_time_sec:.1f}s)")
    return results