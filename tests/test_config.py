"""Tests for configuration module."""

import importlib

import pplyz.config as config


def reload_config():
    """Reload config module so env overrides take effect."""
    global config  # noqa: PLW0603 - needed to update module reference for tests
    config = importlib.reload(config)
    return config


class TestConfig:
    """Test configuration values."""

    def test_default_model(self, monkeypatch):
        """Test default model is set correctly."""
        monkeypatch.delenv("PPLYZ_DEFAULT_MODEL", raising=False)
        cfg = reload_config()
        assert cfg.DEFAULT_MODEL == cfg.DEFAULT_MODEL_FALLBACK

    def test_default_model_env_override(self, monkeypatch):
        """Test default model can be overridden via environment variable."""
        custom_default = "groq/llama-3.1-8b-instant"
        monkeypatch.setenv("PPLYZ_DEFAULT_MODEL", custom_default)
        cfg = reload_config()
        assert cfg.DEFAULT_MODEL == custom_default
        monkeypatch.delenv("PPLYZ_DEFAULT_MODEL", raising=False)
        reload_config()

    def test_api_key_env_vars(self):
        """Test API key environment variable mappings."""
        assert "gemini" in config.API_KEY_ENV_VARS
        assert "openai" in config.API_KEY_ENV_VARS
        assert "anthropic" in config.API_KEY_ENV_VARS
        assert "groq" in config.API_KEY_ENV_VARS
        assert "mistral" in config.API_KEY_ENV_VARS

        assert config.API_KEY_ENV_VARS["gemini"] == ["GEMINI_API_KEY"]
        assert config.API_KEY_ENV_VARS["openai"] == ["OPENAI_API_KEY"]
        assert config.API_KEY_ENV_VARS["anthropic"] == ["ANTHROPIC_API_KEY"]
        assert "GROQ_API_KEY" in config.API_KEY_ENV_VARS["groq"]
        assert "MISTRAL_API_KEY" in config.API_KEY_ENV_VARS["mistral"]

    def test_retry_config(self):
        """Test retry configuration values."""
        assert config.RETRY_BACKOFF_SCHEDULE == [1, 2, 3, 5, 10, 10, 10, 10, 10]
        assert config.MAX_RETRIES == len(config.RETRY_BACKOFF_SCHEDULE) + 1

    def test_rate_limit_codes(self):
        """Test rate limit HTTP codes."""
        assert 429 in config.RATE_LIMIT_CODES

    def test_transient_error_codes(self):
        """Test transient error HTTP codes."""
        assert 500 in config.TRANSIENT_ERROR_CODES
        assert 502 in config.TRANSIENT_ERROR_CODES
        assert 503 in config.TRANSIENT_ERROR_CODES
        assert 504 in config.TRANSIENT_ERROR_CODES

    def test_processing_config(self):
        """Test processing configuration."""
        assert config.DEFAULT_BATCH_SIZE == 1
        assert config.REQUEST_DELAY == 0.5

    def test_json_mode_enabled(self):
        """Test JSON mode is enabled."""
        assert config.USE_JSON_MODE is True

    def test_project_paths(self):
        """Test project paths are set."""
        assert config.PROJECT_ROOT.exists()
        assert config.DATA_DIR.exists()
