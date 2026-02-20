"""Tests for model ID prefix stripping and provider construction.

Verifies that:
- ``litellm/`` prefix is correctly stripped from model IDs.
- ``litellm-`` prefix is correctly stripped.
- Nested prefixes (``litellm/litellm/…``) only strip once.
- IDs without a prefix are passed through unchanged.
- ``original_model_id`` always preserves the input.
- ``api_key`` flows through ``**kwargs`` into ``provider_kwargs``.
"""

from __future__ import annotations

import pytest

from langextract_litellm.provider import LiteLLMLanguageModel


class TestModelIdParsing:
    """Verify litellm/ and litellm- prefix stripping."""

    @pytest.mark.parametrize(
        ("raw_id", "expected_model_id"),
        [
            ("litellm/gpt-4o", "gpt-4o"),
            ("litellm/azure/gpt-4o", "azure/gpt-4o"),
            ("litellm/anthropic/claude-3-opus", "anthropic/claude-3-opus"),
            ("litellm/ollama/llama3", "ollama/llama3"),
            ("litellm-gpt-4", "gpt-4"),
            ("litellm-claude-3-sonnet", "claude-3-sonnet"),
            # Nested prefix — only the first litellm/ is stripped
            ("litellm/litellm/nested", "litellm/nested"),
            # Empty after prefix
            ("litellm/", ""),
            ("litellm-", ""),
            # No prefix — pass through unchanged
            ("gpt-4", "gpt-4"),
            ("some-random-model", "some-random-model"),
        ],
    )
    def test_prefix_stripping(self, raw_id: str, expected_model_id: str) -> None:
        """Model ID prefix should be stripped correctly."""
        provider = LiteLLMLanguageModel(model_id=raw_id)
        assert provider.model_id == expected_model_id
        assert provider.original_model_id == raw_id

    def test_api_key_flows_through_kwargs(self) -> None:
        """``api_key`` passed as kwarg should reach provider_kwargs."""
        provider = LiteLLMLanguageModel(
            model_id="litellm/gpt-4o",
            api_key="sk-test-key",
        )
        assert provider.provider_kwargs["api_key"] == "sk-test-key"

    def test_api_key_in_litellm_kwargs(self) -> None:
        """``api_key`` should appear in the kwargs sent to litellm."""
        provider = LiteLLMLanguageModel(
            model_id="litellm/gpt-4o",
            api_key="sk-test-key",
            temperature=0.5,
        )
        lk = provider._litellm_kwargs
        assert lk["api_key"] == "sk-test-key"
        assert lk["temperature"] == 0.5

    def test_default_max_workers(self) -> None:
        """Default max_workers should be 10."""
        provider = LiteLLMLanguageModel(model_id="litellm/gpt-4o")
        assert provider._max_workers == 10

    def test_custom_max_workers(self) -> None:
        """Custom max_workers should be stored, not forwarded."""
        provider = LiteLLMLanguageModel(model_id="litellm/gpt-4o", max_workers=20)
        assert provider._max_workers == 20
        assert "max_workers" not in provider.provider_kwargs
