"""Tests for the LiteLLM provider async_infer implementation.

Verifies that:
- ``async_infer`` calls ``litellm.acompletion`` (not ``completion``).
- Concurrency is bounded by a shared ``asyncio.Semaphore``.
- Error handling matches the sync path.
- Internal keys (``max_workers``) are **not** forwarded to LiteLLM.
"""

from __future__ import annotations

import asyncio
from unittest import mock

import pytest

from langextract_litellm.provider import LiteLLMLanguageModel


def _make_provider(**kwargs):
    return LiteLLMLanguageModel(model_id="litellm/gpt-4o", **kwargs)


def _mock_response(content: str):
    """Build a minimal mock that looks like a litellm response."""
    choice = mock.MagicMock()
    choice.message.content = content
    resp = mock.MagicMock()
    resp.choices = [choice]
    return resp


class TestAsyncInfer:
    """Tests for LiteLLMLanguageModel.async_infer."""

    @pytest.mark.asyncio
    async def test_calls_acompletion(self):
        provider = _make_provider()
        mock_resp = _mock_response("Hello world")

        with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as m:
            m.return_value = mock_resp
            results = await provider.async_infer(["prompt1"])

        assert len(results) == 1
        assert results[0][0].score == 1.0
        assert results[0][0].output == "Hello world"
        m.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_batch_concurrency(self):
        """All prompts in a batch should be dispatched concurrently."""
        provider = _make_provider(max_workers=5)
        mock_resp = _mock_response("ok")

        with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as m:
            m.return_value = mock_resp
            results = await provider.async_infer(["p1", "p2", "p3", "p4", "p5"])

        assert len(results) == 5
        assert m.await_count == 5

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Errors produce score=0.0 outputs, not exceptions."""
        provider = _make_provider()

        with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as m:
            m.side_effect = RuntimeError("API fail")
            results = await provider.async_infer(["prompt1"])

        assert len(results) == 1
        assert results[0][0].score == 0.0
        assert results[0][0].output == "LLM inference failed"

    @pytest.mark.asyncio
    async def test_empty_response(self):
        provider = _make_provider()
        choice = mock.MagicMock()
        choice.message.content = ""
        resp = mock.MagicMock()
        resp.choices = [choice]

        with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as m:
            m.return_value = resp
            results = await provider.async_infer(["prompt1"])

        assert results[0][0].score == 0.0
        assert results[0][0].output == ""

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Verify semaphore bounds concurrent requests."""
        provider = _make_provider(max_workers=2)
        active = 0
        max_active = 0

        async def _slow_completion(**kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.05)
            active -= 1
            return _mock_response("ok")

        with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as m:
            m.side_effect = _slow_completion
            results = await provider.async_infer(["p1", "p2", "p3", "p4"])

        assert len(results) == 4
        assert max_active <= 2  # semaphore limit

    @pytest.mark.asyncio
    async def test_semaphore_reused_across_calls(self):
        """The same semaphore instance must be shared across calls."""
        provider = _make_provider(max_workers=3)
        sem1 = provider._get_semaphore()
        sem2 = provider._get_semaphore()
        assert sem1 is sem2

    @pytest.mark.asyncio
    async def test_max_workers_not_forwarded_to_litellm(self):
        """``max_workers`` must NOT appear in the kwargs sent to litellm."""
        provider = _make_provider(max_workers=4, temperature=0.7)
        mock_resp = _mock_response("ok")

        with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as m:
            m.return_value = mock_resp
            await provider.async_infer(["p1"])

        call_kwargs = m.call_args.kwargs
        assert "max_workers" not in call_kwargs
        assert call_kwargs["temperature"] == 0.7

    def test_max_workers_not_forwarded_to_litellm_sync(self):
        """``max_workers`` must NOT appear in sync ``completion()`` kwargs."""
        provider = _make_provider(max_workers=4, temperature=0.5)
        mock_resp = _mock_response("ok")

        with mock.patch("litellm.completion") as m:
            m.return_value = mock_resp
            # infer is a generator — consume the first result
            _ = list(provider.infer(["p1"]))

        call_kwargs = m.call_args.kwargs
        assert "max_workers" not in call_kwargs
        assert call_kwargs["temperature"] == 0.5

    def test_max_workers_stored_on_instance(self):
        """``max_workers`` should be stored on the instance, not in kwargs."""
        provider = _make_provider(max_workers=7)
        assert provider._max_workers == 7
        assert "max_workers" not in provider.provider_kwargs
