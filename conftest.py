"""Root conftest — set environment before litellm is imported."""

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
