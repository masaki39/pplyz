"""LLM Analyser - CSV data processing with LLM-powered structured output generation."""

from importlib import metadata

# Keep __version__ in sync with pyproject by asking importlib.metadata.
try:  # pragma: no cover - importlib metadata lookup
    __version__ = metadata.version("pplyz")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
