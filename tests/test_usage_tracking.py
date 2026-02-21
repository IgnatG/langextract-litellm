"""Tests for provider-level token usage tracking.

Verifies that ``last_usage``, ``total_usage``, and ``reset_usage()``
behave correctly across sync and async inference calls.
"""

from __future__ import annotations

from unittest import mock

import pytest

from langextract_litellm import LiteLLMLanguageModel, UsageStats


# ── Helpers ──────────────────────────────────────────────────────


def _mock_response(
    content: str = "ok",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    total_tokens: int = 30,
):
    """Build a mock litellm response with token usage."""
    choice = mock.MagicMock()
    choice.message.content = content
    usage = mock.MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = total_tokens
    resp = mock.MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _mock_response_no_usage(content: str = "ok"):
    """Build a mock litellm response without usage info."""
    choice = mock.MagicMock()
    choice.message.content = content
    resp = mock.MagicMock()
    resp.choices = [choice]
    resp.usage = None
    return resp


# ── UsageStats unit tests ────────────────────────────────────────


class TestUsageStats:
    """UsageStats dataclass behaviour."""

    def test_defaults_are_zero(self) -> None:
        """New instance should have all counters at zero."""
        stats = UsageStats()
        assert stats.prompt_tokens == 0
        assert stats.completion_tokens == 0
        assert stats.total_tokens == 0

    def test_iadd_accumulates(self) -> None:
        """In-place addition should sum all fields."""
        a = UsageStats(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        b = UsageStats(prompt_tokens=3, completion_tokens=7, total_tokens=10)
        a += b
        assert a.prompt_tokens == 8
        assert a.completion_tokens == 17
        assert a.total_tokens == 25

    def test_iadd_returns_self(self) -> None:
        """``+=`` should return the same instance."""
        a = UsageStats()
        b = UsageStats(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        result = a.__iadd__(b)
        assert result is a


# ── Sync inference usage tracking ────────────────────────────────


class TestSyncUsageTracking:
    """Usage tracking via ``infer()``."""

    def test_last_usage_populated(self) -> None:
        """``last_usage`` should reflect the most recent call."""
        provider = LiteLLMLanguageModel(model_id="litellm/gpt-4o")

        with mock.patch("litellm.completion") as m:
            m.return_value = _mock_response(
                prompt_tokens=15,
                completion_tokens=25,
                total_tokens=40,
            )
            list(provider.infer(["hello"]))

        assert provider.last_usage.prompt_tokens == 15
        assert provider.last_usage.completion_tokens == 25
        assert provider.last_usage.total_tokens == 40

    def test_last_usage_updates_per_prompt(self) -> None:
        """``last_usage`` should reflect the *last* prompt processed."""
        provider = LiteLLMLanguageModel(model_id="litellm/gpt-4o")

        resp1 = _mock_response(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        resp2 = _mock_response(prompt_tokens=20, completion_tokens=8, total_tokens=28)

        with mock.patch("litellm.completion") as m:
            m.side_effect = [resp1, resp2]
            list(provider.infer(["a", "b"]))

        # last_usage should be from the second prompt
        assert provider.last_usage.prompt_tokens == 20
        assert provider.last_usage.total_tokens == 28

    def test_total_usage_accumulates(self) -> None:
        """``total_usage`` should sum across all prompts."""
        provider = LiteLLMLanguageModel(model_id="litellm/gpt-4o")

        resp1 = _mock_response(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        resp2 = _mock_response(prompt_tokens=20, completion_tokens=8, total_tokens=28)

        with mock.patch("litellm.completion") as m:
            m.side_effect = [resp1, resp2]
            list(provider.infer(["a", "b"]))

        assert provider.total_usage.prompt_tokens == 30
        assert provider.total_usage.completion_tokens == 13
        assert provider.total_usage.total_tokens == 43

    def test_total_usage_across_calls(self) -> None:
        """``total_usage`` should accumulate across multiple infer calls."""
        provider = LiteLLMLanguageModel(model_id="litellm/gpt-4o")

        with mock.patch("litellm.completion") as m:
            m.return_value = _mock_response(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            )
            list(provider.infer(["a"]))
            list(provider.infer(["b"]))

        assert provider.total_usage.prompt_tokens == 20
        assert provider.total_usage.total_tokens == 30

    def test_no_usage_in_response(self) -> None:
        """When response has no usage, counters should stay at zero."""
        provider = LiteLLMLanguageModel(model_id="litellm/gpt-4o")

        with mock.patch("litellm.completion") as m:
            m.return_value = _mock_response_no_usage()
            list(provider.infer(["hello"]))

        assert provider.last_usage.total_tokens == 0
        assert provider.total_usage.total_tokens == 0

    def test_error_does_not_corrupt_usage(self) -> None:
        """Errors should not add to usage counters."""
        provider = LiteLLMLanguageModel(model_id="litellm/gpt-4o")

        with mock.patch("litellm.completion") as m:
            m.return_value = _mock_response(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            )
            list(provider.infer(["a"]))

        with mock.patch("litellm.completion") as m:
            m.side_effect = Exception("boom")
            list(provider.infer(["b"]))

        # total should still only reflect the first successful call
        assert provider.total_usage.prompt_tokens == 10
        assert provider.total_usage.total_tokens == 15


# ── Async inference usage tracking ───────────────────────────────


class TestAsyncUsageTracking:
    """Usage tracking via ``async_infer()``."""

    @pytest.mark.asyncio
    async def test_last_usage_populated(self) -> None:
        """``last_usage`` should reflect the most recent async call."""
        provider = LiteLLMLanguageModel(model_id="litellm/gpt-4o")

        with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as m:
            m.return_value = _mock_response(
                prompt_tokens=12,
                completion_tokens=18,
                total_tokens=30,
            )
            await provider.async_infer(["hello"])

        assert provider.last_usage.prompt_tokens == 12
        assert provider.last_usage.completion_tokens == 18

    @pytest.mark.asyncio
    async def test_total_usage_accumulates_async(self) -> None:
        """``total_usage`` should sum across concurrent async prompts."""
        provider = LiteLLMLanguageModel(model_id="litellm/gpt-4o")

        resp = _mock_response(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as m:
            m.return_value = resp
            await provider.async_infer(["a", "b", "c"])

        # 3 prompts × 10 prompt tokens each = 30
        assert provider.total_usage.prompt_tokens == 30
        assert provider.total_usage.total_tokens == 45


# ── reset_usage ──────────────────────────────────────────────────


class TestResetUsage:
    """``reset_usage()`` clears both counters."""

    def test_reset_clears_both(self) -> None:
        """After reset, both last and total should be zero."""
        provider = LiteLLMLanguageModel(model_id="litellm/gpt-4o")

        with mock.patch("litellm.completion") as m:
            m.return_value = _mock_response()
            list(provider.infer(["hello"]))

        assert provider.total_usage.total_tokens > 0

        provider.reset_usage()

        assert provider.last_usage.total_tokens == 0
        assert provider.total_usage.total_tokens == 0

    def test_usage_resumes_after_reset(self) -> None:
        """After reset, new calls should start accumulating fresh."""
        provider = LiteLLMLanguageModel(model_id="litellm/gpt-4o")

        with mock.patch("litellm.completion") as m:
            m.return_value = _mock_response(
                prompt_tokens=100, completion_tokens=50, total_tokens=150
            )
            list(provider.infer(["a"]))

        provider.reset_usage()

        with mock.patch("litellm.completion") as m:
            m.return_value = _mock_response(
                prompt_tokens=5, completion_tokens=3, total_tokens=8
            )
            list(provider.infer(["b"]))

        assert provider.total_usage.prompt_tokens == 5
        assert provider.total_usage.total_tokens == 8


# ── Snapshot isolation ───────────────────────────────────────────


class TestUsageSnapshotIsolation:
    """Returned UsageStats are copies, not live references."""

    def test_last_usage_is_snapshot(self) -> None:
        """Mutating the returned object should not affect the provider."""
        provider = LiteLLMLanguageModel(model_id="litellm/gpt-4o")

        with mock.patch("litellm.completion") as m:
            m.return_value = _mock_response(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            )
            list(provider.infer(["a"]))

        snapshot = provider.last_usage
        snapshot.prompt_tokens = 9999

        assert provider.last_usage.prompt_tokens == 10
