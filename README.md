# LangExtract LiteLLM Provider

A provider plugin for [LangExtract](https://github.com/google/langextract) that supports 100+ LLM models through [LiteLLM](https://docs.litellm.ai/docs/#basic-usage)'s unified API, including OpenAI GPT models, Anthropic Claude, Google PaLM, Azure OpenAI, and many open-source models.

> **Note**: This is a third-party provider plugin for LangExtract. For the main LangExtract library, visit [google/langextract](https://github.com/google/langextract).

## Installation

Install from PyPI:

```bash
pip install langextract-litellm
```

Or install from source for development:

```bash
git clone https://github.com/JustStas/langextract-litellm
cd langextract-litellm
pip install -e .
```

## Supported Models

This provider handles model IDs that start with `litellm` and supports a wide range of models through LiteLLM's unified API:

- **OpenAI models**: `litellm/gpt-4`, `litellm/gpt-4o`, `litellm/gpt-3.5-turbo`, etc.
- **Anthropic models**: `litellm/claude-3-opus`, `litellm/claude-3-sonnet`, `litellm/claude-3-haiku`, etc.
- **Google models**: `litellm/gemini-1.5-pro`, `litellm/palm-2`, etc.
- **Azure OpenAI**: `litellm/azure/your-deployment-name`
- **Open-source models**: `litellm/llama-2-7b-chat`, `litellm/mistral-7b`, `litellm/codellama-34b`, etc.
- **And many more**: See [LiteLLM's supported models](https://docs.litellm.ai/docs/providers)

**Note**: All model IDs must be prefixed with `litellm/` or `litellm-` to be handled by this provider.

## Environment Variables

Configure authentication using LiteLLM's standard environment variable format. Set the appropriate variables based on your provider:

### OpenAI

```bash
export OPENAI_API_KEY="your-api-key"
```

### Anthropic

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

### HuggingFace

```bash
export HUGGINGFACE_API_KEY="your-api-key"
```

### Azure OpenAI

```bash
export AZURE_API_KEY="your-azure-key"
export AZURE_API_BASE="https://your-resource.openai.azure.com/"
export AZURE_API_VERSION="2024-02-01"
```

### Google (VertexAI)

```bash
export VERTEXAI_PROJECT="your-project-id"
export VERTEXAI_LOCATION="us-central1"
# Also run: gcloud auth application-default login
```

### Other Providers

See the [LiteLLM documentation](https://docs.litellm.ai/docs/#basic-usage) for environment variables for other providers like HuggingFace, Cohere, AI21, etc.

## Usage

### Basic Usage

```python
import langextract as lx

# Create model configuration
config = lx.factory.ModelConfig(
    model_id="litellm/azure/gpt-4o",  # or "gpt-4", "claude-3-sonnet", etc.
    provider="LiteLLMLanguageModel",
    provider_kwargs={},  # pass provider-specific kwargs here (e.g. {"api_key": "..."})
)
model = lx.factory.create_model(config)

# Extract entities
result = lx.extract(
    text_or_documents="Lady Juliet gazed longingly at the stars, her heart aching for Romeo",
    model=model,
    prompt_description="Extract characters, emotions, and relationships in order of appearance.",
    examples=[...]
)
```

### Complete Example with Examples

```python
import langextract as lx
import textwrap

# Define extraction prompt
prompt = textwrap.dedent("""\
    Extract characters, emotions, and relationships in order of appearance.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful attributes for each entity to add context.""")

# Provide high-quality examples to guide the model
examples = [
    lx.data.ExampleData(
        text="ROMEO. But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.",
        extractions=[
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="ROMEO",
                attributes={"emotional_state": "wonder"}
            ),
            lx.data.Extraction(
                extraction_class="emotion",
                extraction_text="But soft!",
                attributes={"feeling": "gentle awe"}
            ),
            lx.data.Extraction(
                extraction_class="relationship",
                extraction_text="Juliet is the sun",
                attributes={"type": "metaphor"}
            ),
        ]
    )
]

# Create model configuration
config = lx.factory.ModelConfig(
    model_id="litellm/azure/gpt-4o",
    provider="LiteLLMLanguageModel",
    provider_kwargs={},  # pass provider-specific kwargs here (e.g. {"api_key": "..."})
)
model = lx.factory.create_model(config)

# Extract entities
result = lx.extract(
    text_or_documents="Lady Juliet gazed longingly at the stars, her heart aching for Romeo",
    model=model,
    prompt_description=prompt,
    examples=examples
)

print("✅ Extraction successful!")
print(f"Results: {result}")
```

### Async Usage

The LiteLLM provider supports native async inference via `litellm.acompletion`,
which avoids thread overhead and enables true concurrent I/O:

```python
import asyncio
import langextract as lx

config = lx.factory.ModelConfig(
    model_id="litellm/azure/gpt-4o",
    provider="LiteLLMLanguageModel",
    provider_kwargs={},
)
model = lx.factory.create_model(config)

async def main():
    result = await lx.async_extract(
        text_or_documents="Lady Juliet gazed longingly at the stars...",
        model=model,
        prompt_description="Extract characters and emotions.",
        examples=[...],
    )
    print(result)

asyncio.run(main())
```

When using `async_extract`, the LiteLLM provider:

- Uses `asyncio.Semaphore` instead of `ThreadPoolExecutor` for concurrency control
- Calls `litellm.acompletion()` for non-blocking HTTP requests
- Pipelines inference with alignment for improved throughput

### Multi-Pass Cache Bypass

When LangExtract runs multiple extraction passes (`extraction_passes > 1`), the
first pass is cacheable by LiteLLM's Redis cache, but subsequent passes
automatically bypass the cache. This ensures that each additional pass produces a
fresh LLM response — which is the entire point of multi-pass extraction for
improved recall.

The mechanism is fully automatic: LangExtract threads a `pass_num` keyword
argument through to the provider. When `pass_num >= 1`, the LiteLLM provider
injects `cache={"no-cache": True}` into the `litellm.completion()` /
`litellm.acompletion()` call, telling LiteLLM's cache layer to skip the lookup
and force a live API call.

| Pass | `pass_num` value | Cache behaviour |
|------|------------------|-----------------|
| 1    | `0`              | Normal — may be served from cache |
| 2    | `1`              | Bypass — always calls the LLM |
| 3+   | `2+`             | Bypass — always calls the LLM |

> **Note:** `pass_num` is consumed internally by the provider and is never
> forwarded to LiteLLM. You do not need to set it manually — LangExtract
> passes it automatically during multi-pass extraction.

### Model ID Formats

The model ID must start with `litellm/` or `litellm-` to be handled by this provider.

```python
# Explicit LiteLLM prefix
model_id = "litellm/azure/gpt-4o"
model_id = "litellm/gpt-4"
model_id = "litellm/claude-3-sonnet"

# Alternative prefix formats
model_id = "litellm-gpt-4o"
model_id = "litellm-claude-3-sonnet"
```

### Advanced Configuration

Pass additional parameters supported by LiteLLM using `provider_kwargs`. This is the correct way to supply model-specific settings (API keys, temperature, etc.) to LangExtract:

```python
config = lx.factory.ModelConfig(
    model_id="litellm/gpt-4",
    provider="LiteLLMLanguageModel",
    provider_kwargs={
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
        "timeout": 30,
    },
)
model = lx.factory.create_model(config)
```

#### Internal / Reserved Parameters

The following keyword arguments are consumed internally by the provider and are
**never forwarded** to `litellm.completion()` / `litellm.acompletion()`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_workers` | `int` | `10` | Maximum concurrent async requests (semaphore size) |
| `pass_num` | `int` | `0` | Current extraction pass (0-indexed). Set automatically by LangExtract during multi-pass extraction. When ≥ 1, the provider injects `cache={"no-cache": True}` so that repeat passes always get fresh LLM responses. |

To supply an API key directly instead of via an environment variable:

```python
config = lx.factory.ModelConfig(
    model_id="litellm/gpt-4",
    provider="LiteLLMLanguageModel",
    provider_kwargs={
        "api_key": "sk-...",
    },
)
model = lx.factory.create_model(config)
```

## Expected Output

The extraction will return structured data with precise character intervals:

```python
AnnotatedDocument(
    extractions=[
        Extraction(
            extraction_class='character',
            extraction_text='Lady Juliet',
            char_interval=CharInterval(start_pos=0, end_pos=11),
            alignment_status=<AlignmentStatus.MATCH_EXACT: 'match_exact'>,
            attributes={'emotional_state': 'longing'}
        ),
        Extraction(
            extraction_class='emotion',
            extraction_text='aching',
            char_interval=CharInterval(start_pos=52, end_pos=58),
            alignment_status=<AlignmentStatus.MATCH_FUZZY: 'match_fuzzy'>,
            attributes={'feeling': 'heartfelt yearning'}
        ),
        Extraction(
            extraction_class='relationship',
            extraction_text='her heart aching for Romeo',
            char_interval=CharInterval(start_pos=42, end_pos=68),
            alignment_status=<AlignmentStatus.MATCH_EXACT: 'match_exact'>,
            attributes={'type': 'romantic longing'}
        )
    ],
    text='Lady Juliet gazed longingly at the stars, her heart aching for Romeo'
)
```

## Error Handling

The provider includes robust error handling and will return error messages instead of raising exceptions:

```python
# If API call fails, you'll get:
ScoredOutput(score=0.0, output="LiteLLM API error: [error details]")
```

## Development

1. Install in development mode: `pip install -e .`
2. Run tests: `python test_plugin.py`
3. Build package: `python -m build`
4. Publish to PyPI: `twine upload dist/*`

## Requirements

- `langextract`
- `litellm`

## License

Apache License 2.0
