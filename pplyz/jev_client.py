"""Client for Jev, TypeSafe's decision-only model, called via OpenRouter.

Jev does not generate text: it takes a ``state`` (the row data) and typed
``questions`` and returns choices with probabilities. It is served from a
dedicated endpoint rather than chat completions, so it bypasses LiteLLM.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import (
    DEFAULT_JEV_ENDPOINT,
    JEV_API_KEY_ENV_VARS,
    JEV_REQUEST_TIMEOUT,
    MAX_RETRIES,
    RETRY_BACKOFF_SCHEDULE,
    get_default_jev_model,
)
from .judge import Question, output_values
from .keychain import resolve_env_secrets
from .utils import format_error_message

logger = logging.getLogger(__name__)

# HTTP statuses worth retrying: rate limit, transient server errors, overload.
RETRYABLE_STATUSES = {429, 500, 502, 503, 504, 529}


class JevRequestError(RuntimeError):
    """Raised when the decisions endpoint returns an error."""

    def __init__(self, status: int | None, message: str):
        super().__init__(message)
        self.status = status


class JevClient:
    """Ask Jev a fixed set of questions about each row.

    Thread-safe: `pplyz judge` calls it from several worker threads. There is no
    fixed delay between requests; rate limits (429) are handled by retrying.
    """

    def __init__(
        self,
        questions: List[Question],
        model_name: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        with_probability: bool = False,
    ):
        """Initialize the Jev client.

        Args:
            questions: Questions asked for every row.
            model_name: Jev model id on OpenRouter (default: PPLYZ_JEV_MODEL or jev-1.13).
            endpoint: Decisions endpoint URL (default: OpenRouter's alpha endpoint).
            api_key: Optional API key. Defaults to OPENROUTER_API_KEY
                (``keychain:`` references are resolved).
            with_probability: Also emit ``<column>_p`` for choice/scale answers.

        Raises:
            ValueError: If no questions are given or the API key is missing.
        """
        if not questions:
            raise ValueError("At least one question is required.")
        self.questions = questions
        self.model_name = model_name or get_default_jev_model()
        self.endpoint = endpoint or DEFAULT_JEV_ENDPOINT
        self.with_probability = with_probability
        self.provider = "openrouter"
        self.api_key = api_key or self._load_api_key()

    def _load_api_key(self) -> str:
        resolve_env_secrets(JEV_API_KEY_ENV_VARS)
        for var in JEV_API_KEY_ENV_VARS:
            value = os.getenv(var)
            if value:
                return value
        raise ValueError(
            "API key for Jev not found. Set OPENROUTER_API_KEY, e.g. store it in the "
            "Keychain with 'pplyz auth set openrouter' and add "
            'OPENROUTER_API_KEY = "keychain:openrouter" to the [env] table of config.toml.'
        )

    def build_request(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build the decisions request body for one row."""
        return {
            "model": self.model_name,
            "state": _to_json_safe(input_data),
            # Column names may contain spaces or non-ASCII, so use positional ids.
            "questions": {
                _question_id(i): q.to_jev() for i, q in enumerate(self.questions)
            },
        }

    def _post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=JEV_REQUEST_TIMEOUT
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise JevRequestError(
                exc.code, f"HTTP {exc.code} from {self.endpoint}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise JevRequestError(
                None, f"Could not reach {self.endpoint}: {exc}"
            ) from exc

    def _post_with_retry(self, body: Dict[str, Any]) -> Dict[str, Any]:
        attempt = 0
        while True:
            try:
                return self._post(body)
            except JevRequestError as exc:
                retryable = exc.status is None or exc.status in RETRYABLE_STATUSES
                if not retryable or attempt >= len(RETRY_BACKOFF_SCHEDULE):
                    raise
                delay = RETRY_BACKOFF_SCHEDULE[attempt]
                logger.warning(
                    "Jev request failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    MAX_RETRIES,
                    format_error_message(exc, limit=300),
                    delay,
                )
                time.sleep(delay)
                attempt += 1

    def generate_structured_output(
        self,
        prompt: str,
        input_data: Dict[str, Any],
        response_model: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Ask all questions about one row and return column values.

        ``prompt`` and ``response_model`` exist for interface compatibility with
        ``LLMClient``; the questions carry the instructions.
        """
        response = self._post_with_retry(self.build_request(input_data))
        answers = response.get("answers") if isinstance(response, dict) else None
        if not isinstance(answers, dict):
            raise ValueError(f"Unexpected Jev response: {str(response)[:200]}")

        output: Dict[str, Any] = {}
        for i, question in enumerate(self.questions):
            answer = answers.get(_question_id(i))
            if not isinstance(answer, dict):
                raise ValueError(f"Jev returned no answer for '{question.column}'")
            output.update(output_values(question, answer, self.with_probability))
        return output


def _question_id(index: int) -> str:
    return f"q{index + 1}"


def _to_json_safe(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert pandas/numpy cell values into plain JSON values."""
    safe: Dict[str, Any] = {}
    for key, value in data.items():
        if pd.api.types.is_scalar(value) and pd.isna(value):
            value = None
        elif hasattr(value, "item"):
            value = value.item()
        safe[str(key)] = value
    return safe
