"""Configuration loading helpers for pplyz.

This module resolves settings from multiple sources in priority order:

1. Explicit environment variables (already set).
2. Project-local TOML file (for development).
3. User config directory (e.g., `~/.config/pplyz/config.toml`).
"""

from __future__ import annotations

import os
from pathlib import Path

from pplyz.config import (
    DEFAULT_INPUT_COLUMNS_ENV_VAR,
    DEFAULT_OUTPUT_FIELDS_ENV_VAR,
    DEFAULT_MODEL_ENV_VAR,
    PREVIEW_ROWS_ENV_VAR,
)

try:  # Python 3.11+ provides tomllib
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

CONFIG_DIR_ENV = "PPLYZ_CONFIG_DIR"
CONFIG_TOML_NAME = "config.toml"
LOCAL_CONFIG_NAME = "pplyz.local.toml"
PROJECT_ROOT = Path(__file__).parent.parent


def _is_placeholder(value: str) -> bool:
    """Return True when a config/env value looks like a placeholder."""
    normalized = value.strip().lower()
    return normalized.startswith("your-") and normalized.endswith("-here")


_PPYLZ_ENV_MAP = {
    "default_model": DEFAULT_MODEL_ENV_VAR,
    "default_input": DEFAULT_INPUT_COLUMNS_ENV_VAR,
    "default_output": DEFAULT_OUTPUT_FIELDS_ENV_VAR,
    "preview_rows": PREVIEW_ROWS_ENV_VAR,
}


def load_runtime_configuration() -> None:
    """Load settings from project-local and per-user config files."""
    local_config = PROJECT_ROOT / LOCAL_CONFIG_NAME
    if local_config.exists():
        _load_toml_config(local_config)

    config_dir = determine_config_dir()
    config_path = config_dir / CONFIG_TOML_NAME
    if config_path.exists():
        _load_toml_config(config_path)


def determine_config_dir() -> Path:
    """Return the directory containing persistent pplyz settings."""
    if CONFIG_DIR_ENV in os.environ:
        return Path(os.environ[CONFIG_DIR_ENV]).expanduser()

    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        return Path(xdg_config).expanduser() / "pplyz"

    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "pplyz"

    return Path.home() / ".config" / "pplyz"


def _load_toml_config(path: Path) -> None:
    """Load TOML configuration and apply values to the environment."""
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(
            f"Failed to load pplyz config file at {path}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        return

    env_section = data.get("env", {})
    if isinstance(env_section, dict):
        for key, value in env_section.items():
            if isinstance(key, str) and value is not None:
                _set_env_if_missing(key.strip(), str(value))

    pplyz_section = data.get("pplyz", {})
    if isinstance(pplyz_section, dict):
        for key, env_var in _PPYLZ_ENV_MAP.items():
            value = pplyz_section.get(key)
            if value:
                _set_env_if_missing(env_var, str(value))


def _set_env_if_missing(name: str, value: str) -> None:
    """Set environment variable if it is currently unset or empty."""
    current = os.environ.get(name)
    if current and not _is_placeholder(current):
        return
    if _is_placeholder(value):
        return
    os.environ[name] = value
