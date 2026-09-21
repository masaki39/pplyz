"""Configuration settings for LLM Analyser."""

import os
from pathlib import Path

# API Configuration
DEFAULT_MODEL_ENV_VAR = "PPLYZ_DEFAULT_MODEL"
DEFAULT_INPUT_COLUMNS_ENV_VAR = "PPLYZ_DEFAULT_INPUT"
DEFAULT_OUTPUT_FIELDS_ENV_VAR = "PPLYZ_DEFAULT_OUTPUT"
PREVIEW_ROWS_ENV_VAR = "PPLYZ_PREVIEW_ROWS"
DEFAULT_PREVIEW_ROWS = 3
DEFAULT_MODEL_FALLBACK = "openrouter/google/gemini-2.5-flash-lite"

# Jev (TypeSafe's decision-only model) via OpenRouter's decisions endpoint
JEV_MODEL_ENV_VAR = "PPLYZ_JEV_MODEL"
DEFAULT_JEV_MODEL_FALLBACK = "typesafe/jev-1.13"
DEFAULT_JEV_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
JEV_API_KEY_ENV_VARS = ["OPENROUTER_API_KEY"]
JEV_REQUEST_TIMEOUT = 60  # seconds


API_BASE_ENV_VAR = "PPLYZ_API_BASE"
OLLAMA_API_BASE_ENV_VAR = "OLLAMA_API_BASE"


def get_default_model() -> str:
    """Return the default model, allowing override via environment variable."""
    return os.getenv(DEFAULT_MODEL_ENV_VAR, DEFAULT_MODEL_FALLBACK)


def get_default_jev_model() -> str:
    """Return the default Jev model, allowing override via environment variable."""
    return os.getenv(JEV_MODEL_ENV_VAR, DEFAULT_JEV_MODEL_FALLBACK)


def get_default_api_base() -> str | None:
    """Return the API base URL override, if configured via environment."""
    return os.getenv(API_BASE_ENV_VAR) or os.getenv(OLLAMA_API_BASE_ENV_VAR) or None


DEFAULT_MODEL = get_default_model()

# Providers that run locally / self-hosted and do not require an API key.
# LiteLLM exposes both an "ollama" (generate API) and "ollama_chat" (chat API) provider.
NO_API_KEY_PROVIDERS = {"ollama", "ollama_chat"}

# Multi-provider API key environment variables (per LiteLLM docs)
# Each provider entry lists accepted env var names in priority order.
API_KEY_ENV_VARS = {
    "gemini": ["GEMINI_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "claude": ["ANTHROPIC_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "mistralai": ["MISTRAL_API_KEY"],
    "cohere": ["COHERE_API_KEY"],
    "replicate": ["REPLICATE_API_KEY"],
    "huggingface": ["HUGGINGFACE_API_KEY"],
    "together_ai": ["TOGETHERAI_API_KEY", "TOGETHER_AI_TOKEN"],
    "perplexity": ["PERPLEXITY_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "xai": ["XAI_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "azure": ["AZURE_OPENAI_API_KEY", "AZURE_API_KEY"],
    "bedrock": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
    "sagemaker": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
    "vertex_ai": ["GOOGLE_APPLICATION_CREDENTIALS"],
    "vertex_ai_beta": ["GOOGLE_APPLICATION_CREDENTIALS"],
    "watsonx": ["WATSONX_API_KEY", "WATSONX_APIKEY"],
    "databricks": ["DATABRICKS_TOKEN", "DATABRICKS_KEY"],
    "cohere_chat": ["COHERE_API_KEY"],
    "fireworks_ai": ["FIREWORKS_API_KEY", "FIREWORKSAI_API_KEY"],
    "cloudflare": ["CLOUDFLARE_API_KEY"],
}

# Retry Configuration
RETRY_BACKOFF_SCHEDULE = [1, 2, 3, 5, 10, 10, 10, 10, 10]  # seconds
MAX_RETRIES = len(RETRY_BACKOFF_SCHEDULE) + 1  # initial attempt + retries

# Rate Limiting
RATE_LIMIT_CODES = [429]  # HTTP status codes that trigger rate limit retry
TRANSIENT_ERROR_CODES = [500, 502, 503, 504]  # Transient errors to retry

# Processing Configuration
DEFAULT_BATCH_SIZE = 1  # Process one row at a time to respect API limits
REQUEST_DELAY = 0.5  # seconds between requests to avoid rate limiting

# JSON Mode Configuration
USE_JSON_MODE = True  # Force JSON output via LiteLLM

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
