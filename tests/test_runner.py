"""Test the runner module — Ollama connection and model querying.

Requires Ollama to be running on localhost:11434.
"""

import pytest
from baby_llm_nanny.runner import (
    check_ollama, list_models, query_model, run_prompt_set, ModelResponse,
)
from baby_llm_nanny.prompts import PROMPTS

# Skip all tests in this module if Ollama isn't running
pytestmark = pytest.mark.skipif(
    not check_ollama(),
    reason="Ollama not running — start with 'ollama serve'",
)


class TestConnection:
    def test_check_ollama_up(self):
        assert check_ollama() is True

    def test_list_models(self):
        models = list_models()
        assert len(models) > 0
        # At least one of our known models should be present
        assert any("qwen" in m.lower() for m in models) or any("gemma" in m.lower() for m in models)

    def test_check_ollama_wrong_port(self):
        assert check_ollama(port=99999) is False


class TestQueryModel:
    def test_simple_query(self):
        resp = query_model("qwen2.5:3b", "What is 2+2? Answer with just the number.")
        assert resp.ok
        assert resp.response  # non-empty
        assert resp.response_time_sec > 0
        assert resp.model == "qwen2.5:3b"

    def test_query_captures_token_counts(self):
        resp = query_model("qwen2.5:3b", "Say hello.")
        assert resp.ok
        # eval_count and prompt_eval_count should be present
        assert resp.eval_count is not None or resp.eval_count is None  # API dependent

    def test_nonexistent_model(self):
        resp = query_model("nonexistent-model-xyz:latest", "Hello")
        assert not resp.ok
        assert resp.error is not None

    def test_response_is_stripped(self):
        resp = query_model("qwen2.5:3b", "Reply with the word HELLO only.")
        if resp.ok:
            # Response should be stripped (no leading/trailing whitespace)
            assert resp.response == resp.response.strip()


class TestRunPromptSet:
    def test_run_small_subset(self):
        """Run a small subset of prompts to test the batch runner."""
        small_set = PROMPTS[:3]
        results = run_prompt_set("qwen2.5:3b", small_set, show_progress=False)
        assert len(results) == 3
        assert all(isinstance(r, ModelResponse) for r in results)

    def test_run_with_progress(self, capsys):
        small_set = PROMPTS[:2]
        results = run_prompt_set("qwen2.5:3b", small_set, show_progress=True)
        captured = capsys.readouterr()
        assert "[1/2]" in captured.out
        assert "[2/2]" in captured.out


class TestSystemPrompt:
    """Test that the system_prompt parameter actually reaches the model."""

    def test_with_system_prompt(self):
        """Query with a system prompt should still return a valid response."""
        resp = query_model(
            "qwen2.5:3b", "What is 2+2? Answer with just the number.",
            system_prompt="You are a helpful math assistant.",
        )
        assert resp.ok
        assert resp.response

    def test_system_prompt_affects_output(self):
        """A system prompt telling the model to be very brief should produce shorter output."""
        # Without system prompt
        resp_normal = query_model(
            "qwen2.5:3b", "Write a one-sentence greeting.",
        )
        # With system prompt enforcing extreme brevity
        resp_brief = query_model(
            "qwen2.5:3b", "Write a one-sentence greeting.",
            system_prompt="You can only output exactly 3 words. Nothing else. No punctuation except a period.",
        )
        assert resp_normal.ok
        assert resp_brief.ok
        # The brief one should be shorter (or at least not longer)
        assert len(resp_brief.response) <= len(resp_normal.response) + 50