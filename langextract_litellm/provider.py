"""Provider implementation for LiteLLM."""

import asyncio
import logging
from collections.abc import Iterator, Sequence
from typing import Any

import langextract as lx
import litellm
from langextract import data, exceptions, schema
from langextract.core.base_model import BaseLanguageModel
from langextract.core.types import ScoredOutput
from langextract.providers import registry

logger = logging.getLogger(__name__)

# Keys consumed internally by the provider and must not be
# forwarded to ``litellm.completion()`` / ``litellm.acompletion()``.
_INTERNAL_KEYS: frozenset[str] = frozenset({"max_workers", "pass_num"})


@lx.providers.registry.register(r"^litellm", priority=10)
class LiteLLMLanguageModel(BaseLanguageModel):
    """LangExtract provider for LiteLLM.

    This provider supports a wide range of models through LiteLLM's unified API,
    including OpenAI GPT models, Anthropic Claude, Google PaLM, and many open-source models.

    Supported model patterns:
    - litellm-* (explicit LiteLLM prefix)
    - gpt-* (OpenAI models)
    - claude-* (Anthropic models)
    - gemini-*, palm-* (Google models)
    - llama*, mistral*, codellama* (Meta/Mistral models)
    - And many more open-source models
    """

    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        """Initialize the LiteLLM provider.

        Args:
            model_id: The model identifier (e.g., 'gpt-4', 'claude-3-opus', 'llama-2-7b-chat').
            api_key: API key for authentication. If not provided, LiteLLM will automatically
                    look for provider-specific environment variables (OPENAI_API_KEY,
                    ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.)
            **kwargs: Any parameters supported by litellm.completion(), including:
                    - api_base: Custom API base URL
                    - temperature: Sampling temperature (0.0-1.0)
                    - max_tokens: Maximum tokens to generate
                    - top_p: Top-p sampling parameter
                    - frequency_penalty: Frequency penalty (-2.0 to 2.0)
                    - presence_penalty: Presence penalty (-2.0 to 2.0)
                    - timeout: Request timeout in seconds
                    - And any other LiteLLM-supported parameters
        """
        super().__init__()

        # Remove litellm prefix for actual model calls
        if model_id.startswith("litellm/"):
            self.model_id = model_id[8:]  # Remove 'litellm/' prefix
        elif model_id.startswith("litellm-"):
            self.model_id = model_id[8:]  # Remove 'litellm-' prefix
        else:
            self.model_id = model_id

        self.original_model_id = model_id

        # Pop internal keys before storing provider kwargs so they
        # are never forwarded to ``litellm.completion()``.
        self._max_workers: int = kwargs.pop("max_workers", 10)
        self.provider_kwargs = kwargs

        # Lazily initialised in ``_get_semaphore`` to avoid binding
        # to an event loop that may not exist yet at construction.
        self._semaphore: asyncio.Semaphore | None = None

        logger.info("Initialized LiteLLM provider for model: %s", self.model_id)

    @property
    def _litellm_kwargs(self) -> dict[str, Any]:
        """Provider kwargs with internal-only keys stripped.

        Ensures keys like ``max_workers`` that are consumed by
        the provider itself are never forwarded to
        ``litellm.completion()`` / ``litellm.acompletion()``.

        Note: ``__init__`` already pops keys listed in
        ``_INTERNAL_KEYS`` from kwargs, so this filter is a
        belt-and-suspenders defence against future direct mutation
        of ``provider_kwargs``.  When adding new internal keys,
        update **both** the ``pop`` in ``__init__`` and the
        ``_INTERNAL_KEYS`` set.
        """
        return {
            k: v for k, v in self.provider_kwargs.items() if k not in _INTERNAL_KEYS
        }

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Return the concurrency-limiting semaphore.

        Lazily initialised so it is bound to the running event
        loop, not whichever loop (if any) existed at
        construction time.
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_workers)
        return self._semaphore

    def infer(
        self, batch_prompts, **kwargs
    ) -> Iterator[Sequence[ScoredOutput]]:
        """Run inference on a batch of prompts.

        Args:
            batch_prompts: List of prompts to process.
            **kwargs: Additional inference parameters that override
                instance defaults.  ``pass_num`` (int) is consumed
                internally: when >= 1 the call includes
                ``cache={"no-cache": True}`` so that repeat
                extraction passes are never served from the LiteLLM
                response cache.

        Yields:
            Lists of ScoredOutput objects, one per prompt.
        """
        pass_num: int = kwargs.pop("pass_num", 0)

        # Build per-call kwargs: instance defaults + cache bypass.
        call_kwargs = dict(self._litellm_kwargs)
        if pass_num >= 1:
            call_kwargs["cache"] = {"no-cache": True}

        for prompt in batch_prompts:
            try:
                logger.info(
                    "Calling LiteLLM completion for model %s",
                    self.model_id,
                )

                # Format prompt as messages for chat models
                messages = [{"role": "user", "content": str(prompt)}]

                response = litellm.completion(
                    model=self.model_id,
                    messages=messages,
                    **call_kwargs,
                )

                # Extract the response content
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                    if content:
                        yield [ScoredOutput(score=1.0, output=content)]
                    else:
                        logger.warning(
                            "Empty response from LiteLLM for model %s",
                            self.model_id,
                        )
                        yield [ScoredOutput(score=0.0, output="")]
                else:
                    logger.error(
                        "No choices in response from LiteLLM for model %s",
                        self.model_id,
                    )
                    yield [ScoredOutput(score=0.0, output="")]

            except Exception as e:
                logger.error(
                    "Error calling LiteLLM completion for model %s: %s",
                    self.model_id,
                    e,
                )
                error_msg = f"LiteLLM API error: {e}"
                yield [ScoredOutput(score=0.0, output=error_msg)]

    async def async_infer(
        self, batch_prompts, **kwargs
    ) -> list[Sequence[ScoredOutput]]:
        """Native async inference using ``litellm.acompletion``.

        Uses a shared ``asyncio.Semaphore`` for concurrency control
        instead of ``ThreadPoolExecutor``, eliminating thread creation
        overhead while providing explicit back-pressure across
        concurrent batches.

        Args:
            batch_prompts: List of prompts to process.
            **kwargs: Additional inference parameters.
                ``pass_num`` (int) is consumed internally: when >= 1
                the call includes ``cache={"no-cache": True}`` so
                that repeat extraction passes bypass the LiteLLM
                response cache.

        Returns:
            List of lists of ScoredOutput objects, one per prompt.
        """
        pass_num: int = kwargs.pop("pass_num", 0)
        semaphore = self._get_semaphore()

        # Build per-call kwargs: instance defaults + cache bypass.
        call_kwargs = dict(self._litellm_kwargs)
        if pass_num >= 1:
            call_kwargs["cache"] = {"no-cache": True}

        async def _process_single(prompt: str) -> list[ScoredOutput]:
            async with semaphore:
                try:
                    logger.info(
                        "Calling LiteLLM acompletion for model %s",
                        self.model_id,
                    )
                    messages = [{"role": "user", "content": str(prompt)}]

                    response = await litellm.acompletion(
                        model=self.model_id,
                        messages=messages,
                        **call_kwargs,
                    )

                    if response.choices and len(response.choices) > 0:
                        content = response.choices[0].message.content
                        if content:
                            return [
                                ScoredOutput(score=1.0, output=content)
                            ]
                        else:
                            logger.warning(
                                "Empty response from LiteLLM for model %s",
                                self.model_id,
                            )
                            return [ScoredOutput(score=0.0, output="")]
                    else:
                        logger.error(
                            "No choices in response from LiteLLM for model %s",
                            self.model_id,
                        )
                        return [ScoredOutput(score=0.0, output="")]

                except Exception as e:
                    logger.error(
                        "Error calling LiteLLM acompletion for model %s: %s",
                        self.model_id,
                        e,
                    )
                    error_msg = f"LiteLLM API error: {e}"
                    return [ScoredOutput(score=0.0, output=error_msg)]

        tasks = [_process_single(prompt) for prompt in batch_prompts]
        results = await asyncio.gather(*tasks)
        return list(results)
