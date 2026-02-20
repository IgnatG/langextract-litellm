"""LangExtract provider plugin for LiteLLM.

Supports native async inference via ``litellm.acompletion`` when
used with LangExtract's ``async_extract`` / ``async_infer`` API.
"""

from langextract_litellm.provider import LiteLLMLanguageModel

__all__ = ["LiteLLMLanguageModel"]
__version__ = "0.1.0"
