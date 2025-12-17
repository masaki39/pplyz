"""Tests for LLM client module."""

import json
from unittest.mock import MagicMock, patch

import pytest
from litellm.exceptions import BadRequestError, RateLimitError
from pydantic import BaseModel

from pplyz import config
from pplyz.llm_client import LLMClient


class TestLLMClientInitialization:
    """Test LLM client initialization."""

    def test_init_with_gemini_model(self, mock_env_vars):
        """Test initialization with Gemini model."""
        client = LLMClient(model_name="gemini/gemini-2.5-flash-lite")
        assert client.model_name == "gemini/gemini-2.5-flash-lite"
        assert client.provider == "gemini"

    def test_init_with_openai_model(self, mock_env_vars):
        """Test initialization with OpenAI model."""
        client = LLMClient(model_name="gpt-4o")
        assert client.model_name == "gpt-4o"
        assert client.provider == "openai"

    def test_init_with_anthropic_model(self, mock_env_vars):
        """Test initialization with Anthropic model."""
        client = LLMClient(model_name="claude-3-5-sonnet-20241022")
        assert client.model_name == "claude-3-5-sonnet-20241022"
        assert client.provider == "anthropic"

    def test_init_without_api_key_raises_error(self, monkeypatch):
        """Test initialization without API key raises ValueError."""
        # Clear Gemini API env vars
        for env_var in config.API_KEY_ENV_VARS.get("gemini", []):
            monkeypatch.delenv(env_var, raising=False)

        with pytest.raises(ValueError, match="API key.*not found"):
            LLMClient(model_name="gemini/gemini-2.5-flash-lite")

    def test_init_with_explicit_api_key(self):
        """Test initialization with explicit API key."""
        client = LLMClient(
            model_name="gemini/gemini-2.5-flash-lite", api_key="explicit-test-key"
        )
        assert client.model_name == "gemini/gemini-2.5-flash-lite"

    def test_init_uses_env_default_model(self, mock_env_vars, monkeypatch):
        """LLMClient should pick up env override when model_name omitted."""
        custom_default = "groq/llama-3.1-8b-instant"
        monkeypatch.setenv("PPLYZ_DEFAULT_MODEL", custom_default)
        client = LLMClient()
        assert client.model_name == custom_default
        assert client.provider == "groq"
        monkeypatch.delenv("PPLYZ_DEFAULT_MODEL", raising=False)


class TestProviderDetection:
    """Test provider detection from model names."""

    def test_detect_gemini_provider(self, mock_env_vars):
        """Test Gemini provider detection."""
        client = LLMClient(model_name="gemini/gemini-1.5-pro")
        assert client.provider == "gemini"

    def test_detect_openai_provider_gpt(self, mock_env_vars):
        """Test OpenAI provider detection with gpt- prefix."""
        client = LLMClient(model_name="gpt-4o-mini")
        assert client.provider == "openai"

    def test_detect_openai_provider_explicit(self, mock_env_vars):
        """Test OpenAI provider detection with openai/ prefix."""
        client = LLMClient(model_name="openai/gpt-3.5-turbo")
        assert client.provider == "openai"

    def test_detect_anthropic_provider_claude(self, mock_env_vars):
        """Test Anthropic provider detection with claude- prefix."""
        client = LLMClient(model_name="claude-3-haiku-20240307")
        assert client.provider == "anthropic"

    def test_detect_groq_provider(self, mock_env_vars):
        """Test Groq provider detection with explicit prefix."""
        client = LLMClient(model_name="groq/llama-3.1-8b-instant")
        assert client.provider == "groq"

    def test_detect_unknown_provider(self, mock_env_vars):
        """Test unknown provider detection."""
        with pytest.raises(ValueError, match="Unknown model/provider"):
            LLMClient(model_name="unknown-model", api_key="test")


class TestJSONParsing:
    """Test JSON parsing and cleanup."""

    def test_parse_clean_json(self, mock_env_vars, sample_llm_response):
        """Test parsing clean JSON response."""
        client = LLMClient(model_name="gemini/gemini-2.5-flash-lite")

        with patch.object(client, "_generate_with_retry") as mock_generate:
            mock_generate.return_value = json.dumps(sample_llm_response)

            result = client.generate_structured_output(
                prompt="Test prompt", input_data={"test": "data"}
            )

            assert result == sample_llm_response

    def test_parse_json_with_markdown(
        self, mock_env_vars, sample_llm_response_with_markdown, sample_llm_response
    ):
        """Test parsing JSON wrapped in markdown code block."""
        client = LLMClient(model_name="gemini/gemini-2.5-flash-lite")

        with patch.object(client, "_generate_with_retry") as mock_generate:
            mock_generate.return_value = sample_llm_response_with_markdown

            result = client.generate_structured_output(
                prompt="Test prompt", input_data={"test": "data"}
            )

            assert result == sample_llm_response

    def test_parse_invalid_json_raises_error(self, mock_env_vars):
        """Test that invalid JSON raises ValueError."""
        client = LLMClient(model_name="gemini/gemini-2.5-flash-lite")

        with patch.object(client, "_generate_with_retry") as mock_generate:
            mock_generate.return_value = "This is not JSON"

            with pytest.raises(ValueError, match="Failed to parse.*JSON"):
                client.generate_structured_output(
                    prompt="Test prompt", input_data={"test": "data"}
                )

    def test_parse_json_array_raises_error(self, mock_env_vars):
        """Test that JSON array (not object) raises ValueError."""
        client = LLMClient(model_name="gemini/gemini-2.5-flash-lite")

        with patch.object(client, "_generate_with_retry") as mock_generate:
            mock_generate.return_value = json.dumps(["item1", "item2"])

            with pytest.raises(ValueError, match="must be a JSON object"):
                client.generate_structured_output(
                    prompt="Test prompt", input_data={"test": "data"}
                )


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_delay_is_applied(self, mock_env_vars):
        """Test that rate limit delay is applied between requests."""
        client = LLMClient(model_name="gemini/gemini-2.5-flash-lite")

        import time

        start_time = time.time()

        # Simulate two consecutive calls
        client._rate_limit_delay()
        client._rate_limit_delay()

        elapsed = time.time() - start_time

        # Should have at least REQUEST_DELAY between calls
        from pplyz.config import REQUEST_DELAY

        assert elapsed >= REQUEST_DELAY


@pytest.mark.unit
class TestRetryLogic:
    """Test retry logic for API errors."""

    def test_retry_on_rate_limit_error(self, mock_env_vars, sample_llm_response):
        """Test retry on rate limit error."""
        client = LLMClient(model_name="gemini/gemini-2.5-flash-lite")

        with patch("pplyz.llm_client.completion") as mock_completion:
            # First call raises RateLimitError, second succeeds
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json.dumps(sample_llm_response)

            mock_completion.side_effect = [
                RateLimitError(
                    message="Rate limit exceeded", llm_provider="test", model="test"
                ),
                mock_response,
            ]

            result = client.generate_structured_output(
                prompt="Test prompt", input_data={"test": "data"}
            )

            # Should succeed after retry
            assert result == sample_llm_response
            assert mock_completion.call_count == 2

    def test_fallback_when_response_format_rejected(self, mock_env_vars):
        """BadRequest about response_format should disable schema format and retry."""
        client = LLMClient(model_name="gemini/gemini-2.5-flash-lite")

        class DummyModel(BaseModel):
            """Simple schema for validation."""

            category: str

        with patch("pplyz.llm_client.completion") as mock_completion:
            bad_request = BadRequestError(
                message="JSON mode is not enabled for this model",
                llm_provider="test",
                model="test",
            )
            success_response = MagicMock()
            success_response.choices = [MagicMock()]
            success_response.choices[0].message.content = json.dumps({"category": "ok"})

            mock_completion.side_effect = [bad_request, success_response]

            result = client.generate_structured_output(
                prompt="Test prompt",
                input_data={"test": "data"},
                response_model=DummyModel,
            )

            assert result == {"category": "ok"}
            assert mock_completion.call_count == 2
            assert client.supports_schema is False
