"""Tests for LiteLLM provider plugin registration and factory integration.

Verifies that:
- The provider is registered via the ``langextract.providers`` entry point.
- ``registry.resolve()`` routes ``litellm*`` model IDs correctly.
- Unknown model IDs do **not** resolve to this provider.
- Factory integration creates the correct class.
- Sync ``infer()`` returns expected results (mocked).
"""

from __future__ import annotations

from unittest import mock

import langextract as lx
import pytest
from langextract.providers import registry

from langextract_litellm import LiteLLMLanguageModel

# Ensure plugins are loaded before tests run.
lx.providers.load_plugins_once()


# ── Helpers ──────────────────────────────────────────────────────


def _mock_response(content: str = "ok"):
    """Build a minimal mock that looks like a litellm response."""
    choice = mock.MagicMock()
    choice.message.content = content
    resp = mock.MagicMock()
    resp.choices = [choice]
    resp.usage = None
    return resp


# ── Tests ────────────────────────────────────────────────────────


class TestProviderRegistration:
    """Provider discovery and pattern matching."""

    def test_litellm_prefix_resolves(self) -> None:
        """``litellm-*`` model IDs should resolve to our provider."""
        cls = registry.resolve("litellm-azure/gpt-4o")
        assert cls.__name__ == "LiteLLMLanguageModel"

    def test_litellm_slash_prefix_resolves(self) -> None:
        """``litellm/…`` model IDs should resolve to our provider."""
        cls = registry.resolve("litellm/gpt-4o")
        assert cls.__name__ == "LiteLLMLanguageModel"

    def test_unknown_model_does_not_resolve(self) -> None:
        """Non-litellm model IDs should not resolve to us."""
        with pytest.raises(Exception):
            registry.resolve("unknown-model")


class TestInference:
    """Basic inference smoke tests (mocked)."""

    def test_sync_infer_returns_results(self) -> None:
        """Sync infer should yield one result per prompt."""
        provider = LiteLLMLanguageModel(model_id="litellm/gpt-4o")

        with mock.patch("litellm.completion") as m:
            m.return_value = _mock_response("Hello!")
            results = list(provider.infer(["prompt1", "prompt2"]))

        assert len(results) == 2
        assert results[0][0].score == 1.0
        assert results[0][0].output == "Hello!"
        assert m.call_count == 2

    @pytest.mark.asyncio
    async def test_async_infer_returns_results(self) -> None:
        """Async infer should return one result per prompt."""
        provider = LiteLLMLanguageModel(model_id="litellm/gpt-4o")

        with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as m:
            m.return_value = _mock_response("World!")
            results = await provider.async_infer(["prompt1"])

        assert len(results) == 1
        assert results[0][0].output == "World!"


class TestFactoryIntegration:
    """Integration with langextract.factory."""

    def test_factory_creates_provider(self) -> None:
        """Factory should instantiate LiteLLMLanguageModel."""
        from langextract import factory

        config = factory.ModelConfig(
            model_id="litellm-azure/gpt-4o",
            provider="LiteLLMLanguageModel",
        )
        model = factory.create_model(config)
        assert type(model).__name__ == "LiteLLMLanguageModel"
