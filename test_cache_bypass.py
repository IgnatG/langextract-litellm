"""Tests for the multi-pass LiteLLM cache bypass.

Verifies that:
- Pass 0 (first pass) does NOT inject ``cache`` into litellm kwargs.
- Passes >= 1 inject ``cache={"no-cache": True}`` to bypass the
  LiteLLM response cache.
- ``pass_num`` is never forwarded to ``litellm.completion()`` /
  ``litellm.acompletion()``.
- Cache bypass works correctly for both sync ``infer`` and async
  ``async_infer`` paths.
"""

from __future__ import annotations

from unittest import mock

import pytest

from langextract_litellm.provider import LiteLLMLanguageModel


def _make_provider(**kwargs):
    """Create a provider instance with sensible defaults."""
    return LiteLLMLanguageModel(model_id="litellm/gpt-4o", **kwargs)


def _mock_response(content: str = "ok"):
    """Build a minimal mock that looks like a litellm response."""
    choice = mock.MagicMock()
    choice.message.content = content
    resp = mock.MagicMock()
    resp.choices = [choice]
    return resp


# ── Sync infer tests ─────────────────────────────────────────────


class TestSyncCacheBypass:
    """Cache bypass behaviour for ``infer()``."""

    def test_pass_0_no_cache_bypass(self):
        """First pass (pass_num=0) must NOT send cache bypass."""
        provider = _make_provider()
        mock_resp = _mock_response()

        with mock.patch("litellm.completion") as m:
            m.return_value = mock_resp
            list(provider.infer(["prompt"], pass_num=0))

        call_kwargs = m.call_args.kwargs
        assert "cache" not in call_kwargs

    def test_pass_1_cache_bypass(self):
        """Second pass (pass_num=1) must include cache bypass."""
        provider = _make_provider()
        mock_resp = _mock_response()

        with mock.patch("litellm.completion") as m:
            m.return_value = mock_resp
            list(provider.infer(["prompt"], pass_num=1))

        call_kwargs = m.call_args.kwargs
        assert call_kwargs["cache"] == {"no-cache": True}

    def test_pass_2_cache_bypass(self):
        """Third pass (pass_num=2) must also include cache bypass."""
        provider = _make_provider()
        mock_resp = _mock_response()

        with mock.patch("litellm.completion") as m:
            m.return_value = mock_resp
            list(provider.infer(["prompt"], pass_num=2))

        call_kwargs = m.call_args.kwargs
        assert call_kwargs["cache"] == {"no-cache": True}

    def test_no_pass_num_defaults_to_no_bypass(self):
        """When pass_num is omitted, default to pass 0 (no bypass)."""
        provider = _make_provider()
        mock_resp = _mock_response()

        with mock.patch("litellm.completion") as m:
            m.return_value = mock_resp
            list(provider.infer(["prompt"]))

        call_kwargs = m.call_args.kwargs
        assert "cache" not in call_kwargs

    def test_pass_num_not_forwarded_to_litellm(self):
        """``pass_num`` must never appear in litellm kwargs."""
        provider = _make_provider()
        mock_resp = _mock_response()

        with mock.patch("litellm.completion") as m:
            m.return_value = mock_resp
            list(provider.infer(["prompt"], pass_num=1))

        call_kwargs = m.call_args.kwargs
        assert "pass_num" not in call_kwargs

    def test_cache_bypass_preserves_instance_kwargs(self):
        """Cache bypass must coexist with provider instance kwargs."""
        provider = _make_provider(temperature=0.7, top_p=0.9)
        mock_resp = _mock_response()

        with mock.patch("litellm.completion") as m:
            m.return_value = mock_resp
            list(provider.infer(["prompt"], pass_num=1))

        call_kwargs = m.call_args.kwargs
        assert call_kwargs["cache"] == {"no-cache": True}
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.9


# ── Async infer tests ────────────────────────────────────────────


class TestAsyncCacheBypass:
    """Cache bypass behaviour for ``async_infer()``."""

    @pytest.mark.asyncio
    async def test_pass_0_no_cache_bypass(self):
        """First pass (pass_num=0) must NOT send cache bypass."""
        provider = _make_provider()
        mock_resp = _mock_response()

        with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as m:
            m.return_value = mock_resp
            await provider.async_infer(["prompt"], pass_num=0)

        call_kwargs = m.call_args.kwargs
        assert "cache" not in call_kwargs

    @pytest.mark.asyncio
    async def test_pass_1_cache_bypass(self):
        """Second pass (pass_num=1) must include cache bypass."""
        provider = _make_provider()
        mock_resp = _mock_response()

        with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as m:
            m.return_value = mock_resp
            await provider.async_infer(["prompt"], pass_num=1)

        call_kwargs = m.call_args.kwargs
        assert call_kwargs["cache"] == {"no-cache": True}

    @pytest.mark.asyncio
    async def test_pass_2_cache_bypass(self):
        """Third pass (pass_num=2) must also include cache bypass."""
        provider = _make_provider()
        mock_resp = _mock_response()

        with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as m:
            m.return_value = mock_resp
            await provider.async_infer(["prompt"], pass_num=2)

        call_kwargs = m.call_args.kwargs
        assert call_kwargs["cache"] == {"no-cache": True}

    @pytest.mark.asyncio
    async def test_no_pass_num_defaults_to_no_bypass(self):
        """When pass_num is omitted, default to pass 0 (no bypass)."""
        provider = _make_provider()
        mock_resp = _mock_response()

        with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as m:
            m.return_value = mock_resp
            await provider.async_infer(["prompt"])

        call_kwargs = m.call_args.kwargs
        assert "cache" not in call_kwargs

    @pytest.mark.asyncio
    async def test_pass_num_not_forwarded_to_litellm(self):
        """``pass_num`` must never appear in litellm kwargs."""
        provider = _make_provider()
        mock_resp = _mock_response()

        with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as m:
            m.return_value = mock_resp
            await provider.async_infer(["prompt"], pass_num=1)

        call_kwargs = m.call_args.kwargs
        assert "pass_num" not in call_kwargs

    @pytest.mark.asyncio
    async def test_cache_bypass_preserves_instance_kwargs(self):
        """Cache bypass must coexist with provider instance kwargs."""
        provider = _make_provider(temperature=0.7, top_p=0.9)
        mock_resp = _mock_response()

        with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as m:
            m.return_value = mock_resp
            await provider.async_infer(["prompt"], pass_num=1)

        call_kwargs = m.call_args.kwargs
        assert call_kwargs["cache"] == {"no-cache": True}
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_batch_all_get_cache_bypass(self):
        """All prompts in a batch should get cache bypass on pass >= 1."""
        provider = _make_provider()
        mock_resp = _mock_response()

        with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as m:
            m.return_value = mock_resp
            await provider.async_infer(["p1", "p2", "p3"], pass_num=1)

        assert m.await_count == 3
        for call in m.call_args_list:
            assert call.kwargs["cache"] == {"no-cache": True}
