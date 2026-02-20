# langextract-litellm — Improvement Analysis

> **Date:** 2025-02-20
> **Scope:** Full analysis of the `langextract-litellm` provider package
> **Files reviewed:** `provider.py`, `__init__.py`, `pyproject.toml`, `README.md`, `test_plugin.py`, `test_async_infer.py`, `test_cache_bypass.py`

### Implementation Status

| # | Item | Status |
|---|------|--------|
| 1 | Fix `api_key` bug (§1.1) | ✅ Done |
| 2 | Remove unused imports (§1.2) | ✅ Done |
| 3 | Sanitize error output (§4.1) | ✅ Done |
| 4 | Add type annotations (§2.1) | ✅ Done |
| 5 | Extract `_parse_response` helper (§2.2) | ✅ Done |
| 6 | Narrow exception handling (§2.3) | ✅ Done |
| 7 | Fix logging granularity (§2.4) | ✅ Done |
| 8 | Move tests to `tests/` (§5.1) | ✅ Done |
| 9 | Delete "New folder/" (§6.1) | ✅ Done |
| 10 | Update class docstring (§2.5) | ✅ Done |
| — | `.gitignore` coverage (§6.2) | ✅ Already OK |
| — | `py.typed` marker (§6.3) | Not started |
| — | CI configuration (§6.4) | Not started |
| — | Feature gaps (§7) | Not started |

---

## Summary

| Category | Critical | High | Medium | Low |
|----------|----------|------|--------|-----|
| Bugs | 1 | 1 | — | — |
| Code Quality | — | 2 | 4 | 2 |
| Performance | — | 1 | 1 | 1 |
| Testing | — | — | 2 | 1 |
| Project Structure | — | — | 2 | 2 |
| Security | — | 1 | — | — |
| **Totals** | **1** | **5** | **9** | **6** |

---

## 1. Bugs

### 1.1 `api_key` parameter silently discarded (**Critical**)

`__init__` accepts `api_key` as a named parameter, but it is **never stored or forwarded** to LiteLLM. Because it is a named parameter (not part of `**kwargs`), it does not end up in `provider_kwargs` and is silently lost.

```python
# Current (broken)
def __init__(self, model_id: str, api_key: str = None, **kwargs):
    ...
    self.provider_kwargs = kwargs  # api_key is NOT in kwargs
```

**Fix:** Either store `api_key` explicitly and include it in `_litellm_kwargs`, or remove it as a named parameter and let it pass through `**kwargs`:

```python
# Option A — let it pass through **kwargs (simplest)
def __init__(self, model_id: str, **kwargs):
    ...

# Option B — explicit storage
def __init__(self, model_id: str, api_key: str | None = None, **kwargs):
    ...
    if api_key:
        self.provider_kwargs["api_key"] = api_key
```

### 1.2 Unused imports (**High**)

Three imports from `langextract` are dead code:

```python
from langextract import data, exceptions, schema  # none are used
```

These should be removed. They add unnecessary module load time and are misleading.

---

## 2. Code Quality

### 2.1 Missing type annotations (**High**)

| Location | Current | Suggested |
|----------|---------|-----------|
| `__init__` `api_key` | `str = None` | `str \| None = None` |
| `infer` `batch_prompts` | no annotation | `Sequence[str]` |
| `async_infer` `batch_prompts` | no annotation | `Sequence[str]` |

Per PEP 484, `api_key: str = None` is incorrect — the type hint says `str` but the default is `None`. Modern style is `str | None = None` (Python 3.10+).

### 2.2 Duplicated response-extraction logic (**High**)

The logic for extracting content from a LiteLLM response and converting it to `ScoredOutput` is copy-pasted between `infer()` and `async_infer()`. If a field name or scoring strategy changes, both must be updated in lockstep.

**Suggested refactor:**

```python
@staticmethod
def _parse_response(response) -> list[ScoredOutput]:
    """Convert a LiteLLM response to a list of ScoredOutput."""
    if response.choices and len(response.choices) > 0:
        content = response.choices[0].message.content
        if content:
            return [ScoredOutput(score=1.0, output=content)]
        logger.warning("Empty response from LiteLLM")
        return [ScoredOutput(score=0.0, output="")]
    logger.error("No choices in LiteLLM response")
    return [ScoredOutput(score=0.0, output="")]
```

### 2.3 Overly broad `except Exception` (**Medium**)

Both `infer()` and `async_infer()` catch bare `Exception`. This swallows genuine programming errors (`TypeError`, `AttributeError`, `KeyError`) alongside transient API errors.

**Suggestion:** Catch `litellm.exceptions.APIError` (or the equivalent LiteLLM base exception) for expected failures, and let unexpected errors propagate:

```python
from litellm.exceptions import APIError

try:
    response = litellm.completion(...)
except (APIError, litellm.Timeout) as e:
    # Known transient error
    yield [ScoredOutput(score=0.0, output=f"LiteLLM API error: {e}")]
except Exception:
    logger.exception("Unexpected error during LiteLLM inference")
    raise
```

### 2.4 Excessive per-prompt `logger.info` (**Medium**)

`logger.info("Calling LiteLLM completion for model %s", ...)` fires **inside** the per-prompt loop. For a batch of 100 prompts this produces 100 identical INFO messages.

**Fix:** Log once at `INFO` level for the batch; use `DEBUG` per prompt.

```python
logger.info("Running sync inference for %d prompts on %s",
            len(batch_prompts), self.model_id)
for prompt in batch_prompts:
    logger.debug("Calling LiteLLM completion for model %s", self.model_id)
    ...
```

### 2.5 Docstring claims unregistered patterns (**Medium**)

The class docstring says it supports `gpt-*`, `claude-*`, `gemini-*`, `llama*`, `mistral*`, etc. — but the actual registration is only `r"^litellm"`. Those other model IDs would **not** be routed to this provider without the `litellm/` or `litellm-` prefix.

**Fix:** Update the docstring to accurately describe the registration pattern, or register additional patterns if desired.

### 2.6 `len(response.choices) > 0` is redundant (**Medium**)

`response.choices and len(response.choices) > 0` is equivalent to just `response.choices` since a truthy list is already non-empty.

```python
# Current
if response.choices and len(response.choices) > 0:

# Simplified
if response.choices:
```

### 2.7 `str(prompt)` is unnecessary (**Low**)

`messages = [{"role": "user", "content": str(prompt)}]` — prompts are already strings from the LangExtract pipeline. The `str()` call adds noise. If type safety is important, validate or annotate the input instead.

### 2.8 Redundant `list()` in `async_infer` return (**Low**)

`asyncio.gather(*tasks)` already returns a list (technically a tuple in some contexts, but typed as list). The `return list(results)` wrapping is unnecessary overhead.

---

## 3. Performance

### 3.1 Sequential sync processing (**High**)

`infer()` processes prompts one at a time in a `for` loop. For a batch of N prompts, this means N sequential API round-trips. While the async path handles this well with `gather()`, the sync path has no parallelism.

**Options:**

- Use `concurrent.futures.ThreadPoolExecutor` in `infer()` for parallel HTTP calls
- Document that callers should prefer `async_infer()` for batch performance
- Accept the trade-off if sync usage is uncommon

### 3.2 Double dict creation per call (**Medium**)

Each prompt triggers:

1. `_litellm_kwargs` property → dict comprehension (new dict)
2. `call_kwargs = dict(self._litellm_kwargs)` → another copy

In the sync path, the second copy is created once per prompt batch (before the loop), and the property is only called once, so this is actually **fine**. In `async_infer()` it is also created once before dispatch. **No change needed** — the current code is already correct. However, if the response extraction refactor (§2.2) moves the `call_kwargs` construction, keep it outside the hot loop.

### 3.3 No token usage tracking (**Low**)

LiteLLM responses include `response.usage.prompt_tokens`, `completion_tokens`, and `total_tokens`. These are completely discarded. Tracking them would enable cost monitoring and quota management.

**Suggestion:** Accumulate usage stats on the provider instance or emit them via logging:

```python
if hasattr(response, "usage") and response.usage:
    logger.debug(
        "Token usage: prompt=%d, completion=%d, total=%d",
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        response.usage.total_tokens,
    )
```

---

## 4. Security

### 4.1 Error detail leakage (**High**)

When an API call fails, the full exception message is returned as extraction output:

```python
error_msg = f"LiteLLM API error: {e}"
yield [ScoredOutput(score=0.0, output=error_msg)]
```

LiteLLM exception messages can contain:

- API key fragments
- Internal API endpoint URLs
- Rate-limit headers with account metadata

**Fix:** Return a generic message in the output and log the full error at ERROR level:

```python
logger.error("LiteLLM API error for %s: %s", self.model_id, e)
yield [ScoredOutput(score=0.0, output="LLM inference failed")]
```

---

## 5. Testing

### 5.1 Tests at repo root instead of `tests/` directory (**Medium**)

Test files (`test_plugin.py`, `test_async_infer.py`, `test_cache_bypass.py`) sit at the repo root rather than in a conventional `tests/` directory. This makes the project look unstructured and can cause issues with packaging (tests accidentally included in distributions).

**Fix:** Move test files to `tests/` and update `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

### 5.2 `test_plugin.py` makes live API calls (**Medium**)

`test_plugin.py` calls `provider.infer(prompts)` without any mocking — it makes real API calls. This means:

- Tests fail without valid API credentials
- Tests are non-deterministic
- Tests incur API costs

**Fix:** Mock `litellm.completion` in the inference test (step 3), or clearly separate it as an integration test (e.g., `tests/integration/test_live_api.py`).

### 5.3 No edge-case tests for model ID parsing (**Low**)

No tests verify the `litellm/` and `litellm-` prefix stripping logic. Edge cases to cover:

- `litellm/azure/gpt-4o` → `azure/gpt-4o`
- `litellm-gpt-4` → `gpt-4`
- `litellm/litellm/nested` → `litellm/nested`
- Empty string after prefix: `litellm/` → `""`

---

## 6. Project Structure

### 6.1 Empty "New folder/" directory (**Medium**)

There is an empty directory literally named `New folder/` at the repo root. This appears to be accidentally created and should be deleted.

### 6.2 No `.gitignore` coverage for build artifacts (**Medium**)

Verify that `.gitignore` excludes:

- `dist/`
- `*.egg-info/`
- `build/`
- `__pycache__/`
- `.venv/`

### 6.3 No `py.typed` marker file (**Low**)

For PEP 561 compliance (allowing downstream type checkers to see inline type hints), add an empty `langextract_litellm/py.typed` file.

### 6.4 No CI configuration (**Low**)

No GitHub Actions workflow, tox config, or other CI setup exists. Even a minimal workflow running `pytest` and `ruff` on push would catch regressions.

---

## 7. Feature Gaps (Nice-to-Have)

| Feature | Benefit | Effort |
|---------|---------|--------|
| **Retry logic** | Automatic retries for transient failures (429, 5xx). LiteLLM supports `num_retries` natively — just document/expose it. | Low |
| **Streaming support** | Enable streaming responses via `stream=True` for long extractions. | Medium |
| **Callback hooks** | LiteLLM supports `success_callback` / `failure_callback` for observability (Langfuse, OpenTelemetry, etc.). Document or expose these. | Low |
| **Model cost tracking** | LiteLLM provides `litellm.completion_cost()` — could expose per-call and aggregate costs. | Low |
| **Timeout configuration** | While `timeout` can be passed via `provider_kwargs`, a top-level parameter with a sensible default (e.g., 60s) would improve UX. | Low |

---

## Recommended Priority Order

1. ~~**Fix `api_key` bug** (§1.1)~~ — ✅ Completed
2. ~~**Remove unused imports** (§1.2)~~ — ✅ Completed
3. ~~**Sanitize error output** (§4.1)~~ — ✅ Completed
4. ~~**Add type annotations** (§2.1)~~ — ✅ Completed
5. ~~**Extract `_parse_response` helper** (§2.2)~~ — ✅ Completed
6. ~~**Narrow exception handling** (§2.3)~~ — ✅ Completed
7. ~~**Fix logging granularity** (§2.4)~~ — ✅ Completed (done as part of §2.2 refactor)
8. ~~**Move tests to `tests/`** (§5.1)~~ — ✅ Completed
9. ~~**Delete "New folder/"** (§6.1)~~ — ✅ Completed
10. ~~**Update class docstring** (§2.5)~~ — ✅ Completed

---

## Quick-Win Patch (Items 1, 2, 4, 6, 7, 9)

These can be applied in a single commit with minimal risk:

```python
# provider.py — key changes only (not a complete diff)

# Remove unused imports
- from langextract import data, exceptions, schema
+ # (removed — unused)

# Fix api_key bug + add type hints
- def __init__(self, model_id: str, api_key: str = None, **kwargs):
+ def __init__(self, model_id: str, **kwargs: Any) -> None:

# Fix batch_prompts type hint
- def infer(self, batch_prompts, **kwargs) -> Iterator[...]:
+ def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Iterator[...]:

# Simplify truthiness check
- if response.choices and len(response.choices) > 0:
+ if response.choices:

# Logging — batch level
+ logger.info("Running inference for %d prompts on %s", len(batch_prompts), self.model_id)
  for prompt in batch_prompts:
-     logger.info("Calling LiteLLM completion for model %s", self.model_id)
+     logger.debug("Calling LiteLLM completion for model %s", self.model_id)
```
