"""LLM client with retry logic using LiteLLM for multi-provider support."""

import json
import logging
import os
import time
from typing import Any, Dict, Optional, Type

import litellm
from litellm import completion, supports_response_schema
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    BadRequestError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from pydantic import BaseModel
from .config import (
    API_KEY_ENV_VARS,
    MAX_RETRIES,
    NO_API_KEY_PROVIDERS,
    REQUEST_DELAY,
    RETRY_BACKOFF_SCHEDULE,
    USE_JSON_MODE,
    get_default_api_base,
    get_default_model,
)
from .keychain import resolve_env_secrets
from .utils import format_error_message

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with multiple LLM providers via LiteLLM with retry logic."""

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        """Initialize the LLM client.

        Args:
            model_name: Name of the model to use (e.g., "gemini/gemini-2.5-flash-lite", "gpt-4o",
                "ollama_chat/llama3.1"). Defaults to the value of PPLYZ_DEFAULT_MODEL env var
                (or Gemini Flash Lite).
            api_key: Optional API key. If None, LiteLLM will use standard environment variables.
            api_base: Optional base URL for the provider's API. Useful for Ollama and other
                OpenAI-compatible local servers. If None, falls back to PPLYZ_API_BASE /
                OLLAMA_API_BASE env vars.

        Raises:
            ValueError: If required API key is not found in environment.
        """
        self.model_name = model_name or get_default_model()
        self.api_base = api_base or get_default_api_base()
        self.last_request_time = 0

        # Determine provider from model name
        self.provider = self._detect_provider(self.model_name)

        self._silence_litellm_logs()
        self._ensure_api_key(api_key)

        # Configure LiteLLM
        litellm.drop_params = True  # Drop unsupported params instead of erroring
        litellm.set_verbose = False  # Disable verbose logging
        litellm.suppress_debug_info = True  # Hide the "Give Feedback / Get Help" banner
        litellm.enable_json_schema_validation = True  # Enable client-side validation

        # Check if model supports response schema / JSON mode
        self.supports_schema = supports_response_schema(self.model_name)
        self.supports_json_mode = self._detect_json_mode_support()

    def _detect_provider(self, model_name: str) -> str:
        """Detect the provider from the model name."""
        try:
            _, provider, _, custom_provider = litellm.get_llm_provider(model_name)
            if provider:
                return provider
            if custom_provider:
                return custom_provider
        except Exception:
            pass

        # Check explicit provider prefix (provider/model-name)
        if "/" in model_name:
            prefix = model_name.split("/", 1)[0]
            if prefix in API_KEY_ENV_VARS:
                return prefix

        if model_name.startswith("gemini/"):
            return "gemini"
        if model_name.startswith("gpt-") or model_name.startswith("openai/"):
            return "openai"
        if model_name.startswith("claude-") or model_name.startswith("anthropic/"):
            return "anthropic"

        for provider in API_KEY_ENV_VARS.keys():
            if model_name.startswith(f"{provider}/"):
                return provider
        for provider in NO_API_KEY_PROVIDERS:
            if model_name.startswith(f"{provider}/"):
                return provider
        return "unknown"

    def _detect_json_mode_support(self) -> bool:
        """Return True when the provider accepts response_format (JSON mode).

        Some providers (e.g. Ollama) do not support full JSON *schema* enforcement
        but still honour ``response_format={"type": "json_object"}``.
        """
        if self.supports_schema:
            return True
        try:
            params = litellm.get_supported_openai_params(
                model=self.model_name, custom_llm_provider=self.provider
            )
        except Exception:
            return False
        return bool(params) and "response_format" in params

    def _set_api_key(self, api_key: str) -> None:
        """Set the API key for the detected provider.

        Args:
            api_key: The API key to set.
        """
        env_vars = API_KEY_ENV_VARS.get(self.provider)
        if not env_vars:
            return
        if isinstance(env_vars, str):
            env_vars = [env_vars]
        target_var = env_vars[0] if env_vars else None
        if target_var:
            os.environ[target_var] = api_key

    def _verify_api_key(self) -> None:
        """Verify that the required API key is set in environment."""
        if self.provider in NO_API_KEY_PROVIDERS:
            logger.debug(
                "Provider '%s' runs locally; no API key required.", self.provider
            )
            return

        env_vars = API_KEY_ENV_VARS.get(self.provider)
        if not env_vars:
            logger.warning(
                "Unknown provider '%s' for model '%s'. API key verification skipped.",
                self.provider,
                self.model_name,
            )
            return

        if isinstance(env_vars, str):
            env_vars = [env_vars]

        if not any(os.getenv(var) for var in env_vars):
            env_hint = ", ".join(env_vars)
            raise ValueError(
                f"API key for {self.provider} not found. "
                f"Please set one of the following environment variables: {env_hint}\n"
                f"Example: export {env_vars[0]}='your-api-key-here'"
            )

    def _ensure_api_key(self, api_key: str | None) -> None:
        """Set or verify API key presence."""
        if self.provider == "unknown":
            raise ValueError(
                f"Unknown model/provider '{self.model_name}'. "
                "Use a supported provider prefix (e.g., gemini/, gpt-4o, claude-, "
                "ollama_chat/) or configure a custom provider in LiteLLM before using "
                "this model."
            )

        if api_key:
            self._set_api_key(api_key)
        else:
            resolve_env_secrets(API_KEY_ENV_VARS.get(self.provider, []))
            self._verify_api_key()

    def _silence_litellm_logs(self) -> None:
        """Reduce LiteLLM logging noise for end users."""
        for logger_name in ("litellm", "LiteLLM"):
            litellm_logger = logging.getLogger(logger_name)
            litellm_logger.setLevel(logging.ERROR)
            litellm_logger.propagate = False

    def _rate_limit_delay(self) -> None:
        """Implement rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self.last_request_time = time.time()

    def _build_completion_params(
        self,
        messages: list,
        response_model: Optional[Type[BaseModel]],
        use_schema_format: bool = True,
    ) -> dict:
        """Prepare completion parameters with schema settings."""
        completion_params = {
            "model": self.model_name,
            "messages": messages,
        }

        if self.api_base:
            completion_params["api_base"] = self.api_base

        if use_schema_format and response_model is not None and self.supports_schema:
            completion_params["response_format"] = response_model
        elif use_schema_format and USE_JSON_MODE and self.supports_json_mode:
            # Providers without full schema support (e.g. Ollama) still honour
            # plain JSON-object mode; combined with the schema hints in the prompt
            # this keeps outputs parseable.
            completion_params["response_format"] = {"type": "json_object"}

        return completion_params

    def _build_messages(
        self,
        prompt: str,
        input_data: Dict[str, Any],
        response_model: Optional[Type[BaseModel]],
    ) -> list[dict]:
        """Construct system/user messages for the LLM."""
        data_str = json.dumps(input_data, ensure_ascii=False, indent=2)

        if response_model is not None:
            schema_fields = list(response_model.model_fields.keys())
            system_message = (
                "You are a data analyst. You must respond ONLY with valid JSON matching the required schema. "
                f"Required fields: {', '.join(schema_fields)}. "
                "Do not include any markdown formatting, explanations, or text outside the JSON object."
            )
        else:
            system_message = (
                "You are a data analyst. You must respond ONLY with valid JSON. "
                "Do not include any markdown formatting, explanations, or text outside the JSON object."
            )

        user_message = f"""Based on the following data and task description, generate a structured output in valid JSON format.

Task: {prompt}

Input Data:
{data_str}

Instructions:
- Analyze the input data according to the task description
- Return ONLY a valid JSON object with your analysis results
- The JSON should contain relevant fields based on the task
- Do not include any explanation or markdown formatting

Output (JSON only):"""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Normalize and parse the JSON returned by the LLM."""
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("Response must be a JSON object (dictionary)")
        return parsed

    def _validate_response(
        self, parsed: Dict[str, Any], response_model: Type[BaseModel]
    ) -> Dict[str, Any]:
        """Validate parsed JSON against the provided schema."""
        validated = response_model(**parsed)
        return validated.model_dump()

    def _generate_with_retry(
        self, messages: list, response_model: Optional[Type[BaseModel]] = None
    ) -> str:
        """Generate content with automatic retry logic."""
        attempt = 0
        use_schema_format = True

        while attempt < MAX_RETRIES:
            self._rate_limit_delay()
            completion_params = self._build_completion_params(
                messages, response_model, use_schema_format
            )

            try:
                response = completion(**completion_params)
                return response.choices[0].message.content
            except BadRequestError as exc:
                # Some providers report schema/JSON-mode support but reject response_format at runtime.
                if use_schema_format and self._is_response_format_error(exc):
                    logger.warning(
                        "Model rejected response_format/JSON mode; retrying without schema enforcement."
                    )
                    self.supports_schema = False
                    self.supports_json_mode = False
                    use_schema_format = False
                    continue
                raise
            except APIConnectionError as exc:
                raise ConnectionError(self._connection_error_hint(exc)) from exc
            except (RateLimitError, ServiceUnavailableError, APIError, Timeout) as exc:
                if attempt >= len(RETRY_BACKOFF_SCHEDULE):
                    raise

                delay = RETRY_BACKOFF_SCHEDULE[attempt]
                logger.warning(
                    "LLM request failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    MAX_RETRIES,
                    format_error_message(exc, limit=300),
                    delay,
                )
                time.sleep(delay)
                attempt += 1

        raise RuntimeError("LLM retry logic exhausted unexpectedly")

    def _connection_error_hint(self, exc: Exception) -> str:
        """Build an actionable message for connection failures."""
        target = self.api_base or "the configured endpoint"
        base = f"Could not connect to {self.provider} at {target}: {exc}"
        if self.provider in NO_API_KEY_PROVIDERS:
            return (
                f"{base}\n"
                "Make sure the Ollama server is running ('ollama serve') and the model "
                f"is pulled ('ollama pull {self.model_name.split('/', 1)[-1]}'). "
                "Override the URL with --api-base or the OLLAMA_API_BASE env var."
            )
        return base

    def _is_response_format_error(self, exc: Exception) -> bool:
        """Detect provider errors caused by response_format/JSON mode."""
        message = str(exc).lower()
        return any(
            keyword in message
            for keyword in (
                "response_format",
                "json mode is not enabled",
                "json_mode",
                "invalid json output config",
            )
        )

    def generate_structured_output(
        self,
        prompt: str,
        input_data: Dict[str, Any],
        response_model: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        """Generate structured output from input data.

        Args:
            prompt: The user-provided prompt describing the task.
            input_data: Dictionary containing the selected column data.
            response_model: Optional Pydantic model defining the output schema.

        Returns:
            Dictionary containing the generated structured output.

        Raises:
            ValueError: If the response cannot be parsed as JSON or validated.
        """
        messages = self._build_messages(prompt, input_data, response_model)
        try:
            response_text = self._generate_with_retry(messages, response_model)
            parsed = self._parse_json_response(response_text)
            if response_model is not None:
                return self._validate_response(parsed, response_model)
            return parsed
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse LLM response as JSON: {exc}\n"
                f"Response was: {response_text[:200]}..."
            ) from exc
        except Exception as exc:
            raise ValueError(f"Error generating structured output: {exc}") from exc
