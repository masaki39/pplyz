"""LLM Analyser - CSV data processing with LLM-powered structured output generation."""

import argparse
import logging
import os
import sys
from pathlib import Path

from pplyz.config import (
    API_KEY_ENV_VARS,
    DATA_DIR,
    DEFAULT_MODEL_FALLBACK,
    DEFAULT_INPUT_COLUMNS_ENV_VAR,
    DEFAULT_OUTPUT_FIELDS_ENV_VAR,
    DEFAULT_PREVIEW_ROWS,
    PREVIEW_ROWS_ENV_VAR,
    get_default_api_base,
    get_default_model,
)
from pplyz.keychain import (
    KEYCHAIN_PREFIX,
    KEYCHAIN_SERVICE,
    SecretResolutionError,
    store_keychain,
)
from pplyz.llm_client import LLMClient
from pplyz.processor import CSVProcessor
from pplyz.schemas import create_output_model_from_string
from pplyz.settings import determine_config_dir, load_runtime_configuration
from pplyz.utils import color_text

DOCS_URL = "https://github.com/masaki39/pplyz#readme"

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.history import FileHistory, InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings
except ImportError:  # pragma: no cover - optional dependency
    PromptSession = None
    AutoSuggestFromHistory = None
    FileHistory = None
    InMemoryHistory = None
    KeyBindings = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # Simple format for user-facing output
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

SECTION_LINE = "=" * 56


class CompactHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Help formatter without metavar duplication in option listing."""

    def _format_action_invocation(self, action):
        if not action.option_strings:
            return super()._format_action_invocation(action)
        # Display short flags first for consistent column width
        option_strings = sorted(action.option_strings, key=len)
        return ", ".join(option_strings)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    default_model = get_default_model()

    parser = argparse.ArgumentParser(
        usage="pplyz [INPUT] [options]",
        formatter_class=CompactHelpFormatter,
        epilog=(
            "Subcommands:\n"
            "  pplyz judge INPUT   Fast yes/no, choice and scale decisions with Jev\n"
            "  pplyz auth set NAME Store an API key in the macOS Keychain"
        ),
    )

    parser.add_argument(
        "input_path",
        nargs="?",
        metavar="INPUT",
        help="Positional input CSV path",
    )

    parser.add_argument(
        "--input",
        "-i",
        dest="input_columns",
        type=str,
        help="Comma-separated list of columns to use as LLM input (e.g., 'title,abstract,keywords')",
    )

    output_help = (
        'Output column definitions (e.g., "relevant:bool,summary:str"). '
        "Supported types: bool, int, float, str."
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        dest="output_fields",
        help=output_help,
    )

    parser.add_argument(
        "--preview",
        "-p",
        action="store_true",
        help="Preview results on sample rows without saving",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=default_model,
        help=f"LLM model name (default: {default_model})",
    )

    parser.add_argument(
        "--api-base",
        dest="api_base",
        type=str,
        default=get_default_api_base(),
        help=(
            "Base URL for the LLM API (for Ollama or other OpenAI-compatible "
            "local servers). Defaults to PPLYZ_API_BASE / OLLAMA_API_BASE."
        ),
    )

    parser.add_argument(
        "--force",
        "-f",
        dest="resume",
        action="store_false",
        default=True,
        help="Force reprocessing of all rows and overwrite previous output (resume is default)",
    )

    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        type=str,
        help="Inline prompt text (skips interactive prompt entry)",
    )
    prompt_group.add_argument(
        "--prompt-file",
        type=str,
        help="Path to a prompt text file (skips interactive prompt entry)",
    )

    def _colorize_help(text: str) -> str:
        """Apply light ANSI colors to section headers when TTY."""
        if not sys.stdout.isatty():
            return text
        replacements = {
            "usage:": color_text("usage:", "yellow"),
            "positional arguments:": color_text("positional arguments:", "cyan"),
            "options:": color_text("options:", "cyan"),
            "Defaults:": color_text("Defaults:", "magenta"),
        }
        for key, val in replacements.items():
            text = text.replace(key, val)
        return text

    def _print_help(file=None):  # type: ignore[override]
        text = parser.format_help()
        default_input = os.getenv(DEFAULT_INPUT_COLUMNS_ENV_VAR) or "(none)"
        default_output = os.getenv(DEFAULT_OUTPUT_FIELDS_ENV_VAR) or "(none)"
        defaults_block = (
            "\n\nDefaults:\n"
            f"  model : {default_model}\n"
            f"  input : {default_input}\n"
            f"  output: {default_output}\n"
        )
        text = text.rstrip() + defaults_block + f"\nDocs & issues: {DOCS_URL}\n"
        target = sys.stdout if file is None else file
        parser._print_message(_colorize_help(text), target)

    parser.print_help = _print_help  # type: ignore[attr-defined]

    return parser.parse_args()


def resolve_preview_rows() -> int:
    """Resolve preview row count from environment/config."""
    value = os.environ.get(PREVIEW_ROWS_ENV_VAR)
    if value is None:
        return DEFAULT_PREVIEW_ROWS

    try:
        rows = int(value)
        if rows <= 0:
            raise ValueError
        return rows
    except ValueError:
        logger.warning(
            "Invalid preview row count '%s'. Falling back to %d rows.",
            value,
            DEFAULT_PREVIEW_ROWS,
        )
        return DEFAULT_PREVIEW_ROWS


def get_user_prompt() -> str:
    """Get the analysis prompt from user interactively.

    Returns:
        The user-provided prompt.
    """
    print("\n" + SECTION_LINE)
    print("Prompt Input")
    print(SECTION_LINE)
    print(
        "\nPlease enter the prompt you want the LLM to run for each row.\n"
        "The LLM will receive the selected columns and generate structured output.\n"
    )
    print("Examples:")
    print('  - "Classify the sentiment as positive, negative, or neutral"')
    print('  - "Extract key findings and methodology from the abstract"')
    print('  - "Summarize the main topic in 1-2 sentences"\n')

    prompt_session = _build_prompt_session()

    try:
        if prompt_session is not None:
            prompt = prompt_session.prompt("Enter your prompt: ").strip()
        else:
            prompt = input("Enter your prompt: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nPrompt entry cancelled. Exiting.")
        sys.exit(1)

    if not prompt:
        print("Error: Prompt cannot be empty")
        sys.exit(1)

    return prompt


def get_prompt_from_args(args: argparse.Namespace) -> str:
    """Resolve prompt from CLI args or interactive input."""
    if args.prompt is not None:
        if not args.prompt.strip():
            print("Error: Prompt cannot be empty")
            sys.exit(1)
        return args.prompt

    if args.prompt_file is not None:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            print(f"Error: Prompt file not found: {prompt_path}")
            sys.exit(1)
        if prompt_path.is_dir():
            print(f"Error: Prompt path must be a file, not a directory: {prompt_path}")
            sys.exit(1)
        try:
            content = prompt_path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"Error: Unable to read prompt file: {exc}")
            sys.exit(1)
        if not content.strip():
            print("Error: Prompt cannot be empty")
            sys.exit(1)
        return content

    return get_user_prompt()


def _build_prompt_session():
    """Create a prompt_toolkit session when available for better line editing."""
    if PromptSession is None:
        return None

    auto_suggest = AutoSuggestFromHistory()
    history = InMemoryHistory()

    try:
        history_dir = determine_config_dir()
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = history_dir / "prompt_history"
        history = FileHistory(str(history_path))
    except OSError:
        history = InMemoryHistory()

    kb = KeyBindings()

    @kb.add("tab")
    def _(event):  # type: ignore[override]
        """Accept autosuggest with Tab when available."""
        buf = event.app.current_buffer
        if buf.suggestion:
            buf.insert_text(buf.suggestion.text)
        else:
            buf.insert_text("\t")

    @kb.add("right")
    def _(event):  # type: ignore[override]
        """Disable accepting autosuggest with Right; just move cursor."""
        event.app.current_buffer.cursor_right()

    # Enable tab acceptance of history suggestions
    return PromptSession(
        history=history,
        auto_suggest=auto_suggest,
        complete_while_typing=True,
        enable_history_search=True,
        key_bindings=kb,
    )


def _other_provider_key_set() -> bool:
    """True when an API key for a provider other than OpenRouter is configured."""
    return any(
        os.getenv(var)
        for provider, env_vars in API_KEY_ENV_VARS.items()
        if provider != "openrouter"
        for var in env_vars
    )


def resolve_input_path(raw_path: str) -> Path:
    """Resolve and validate the input CSV path, exiting on error."""
    input_path = Path(raw_path)
    if not input_path.is_absolute():
        # Try relative to DATA_DIR first
        data_path = DATA_DIR / input_path
        if data_path.exists():
            input_path = data_path
        else:
            # Use as relative to current directory
            input_path = input_path.resolve()

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    if input_path.is_dir():
        print(f"Error: Input path must be a CSV file, not a directory: {input_path}")
        sys.exit(1)

    if input_path.suffix.lower() != ".csv":
        print(f"Error: Input file must have a .csv extension: {input_path}")
        sys.exit(1)

    return input_path


def auth_main(argv: list[str]) -> None:
    """Store an API key in the macOS Keychain (`pplyz auth set <name>`)."""
    parser = argparse.ArgumentParser(
        prog="pplyz auth",
        description="Store API keys in the macOS Keychain instead of config files.",
    )
    sub = parser.add_subparsers(dest="action", required=True)
    set_parser = sub.add_parser("set", help="Store (or replace) a key")
    set_parser.add_argument(
        "name", help="Key name, usually the provider (e.g. openrouter)"
    )
    args = parser.parse_args(argv)

    name = args.name.strip()
    if not name or "/" in name:
        print(
            "Error: name must be non-empty and must not contain '/' "
            "(e.g. pplyz auth set openrouter)."
        )
        sys.exit(1)
    print(f"Enter the API key for '{name}' (input is hidden).")
    try:
        store_keychain(name)
    except SecretResolutionError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    env_vars = API_KEY_ENV_VARS.get(name)
    env_var = env_vars[0] if env_vars else "<PROVIDER>_API_KEY"
    config_path = determine_config_dir() / "config.toml"
    print(f"✓ Stored in Keychain (service={KEYCHAIN_SERVICE}, account={name}).")
    print(f"Reference it from the [env] table of {config_path}:")
    print("\n  [env]")
    print(f'  {env_var} = "{KEYCHAIN_PREFIX}{name}"\n')


def main() -> None:
    """Main entry point for the LLM Analyser CLI."""
    # Load environment/configuration files
    load_runtime_configuration()

    subcommand = sys.argv[1] if len(sys.argv) > 1 else None
    if subcommand == "judge":
        from pplyz.judge_cli import judge_main

        judge_main(sys.argv[2:])
        return
    if subcommand == "auth":
        auth_main(sys.argv[2:])
        return

    # Parse arguments
    args = parse_arguments()

    positional_input = getattr(args, "input_path", None)

    default_columns = os.environ.get(DEFAULT_INPUT_COLUMNS_ENV_VAR)
    default_fields = os.environ.get(DEFAULT_OUTPUT_FIELDS_ENV_VAR)

    resolved_input = positional_input
    resolved_columns = args.input_columns or default_columns
    resolved_fields = args.output_fields or default_fields

    # Validate required arguments (only if not using --list)
    missing_args = []
    if not resolved_input:
        missing_args.append("INPUT (positional)")
    if not resolved_columns:
        missing_args.append("--input/-i")
    if not resolved_fields:
        missing_args.append("--output/-o")

    if missing_args:
        print("Error: the following arguments are required: " + ", ".join(missing_args))
        print("Use --help for more information")
        sys.exit(1)

    # Parse columns
    columns = [col.strip() for col in resolved_columns.split(",")]

    if not columns:
        print("Error: At least one column must be specified")
        sys.exit(1)

    input_path = resolve_input_path(resolved_input)

    # Get user prompt (inline/file/interactive)
    prompt = get_prompt_from_args(args)

    # Create response model from the requested output schema when provided
    response_model = None
    if resolved_fields:
        try:
            response_model = create_output_model_from_string(resolved_fields)
        except Exception as e:
            print(f"Error parsing fields: {e}")
            sys.exit(1)

    # Initialize LLM client
    logger.info("Initializing LLM client (model: %s)...", args.model)
    try:
        llm_client = LLMClient(model_name=args.model, api_base=args.api_base)
        logger.info("✓ LLM client initialized (provider: %s)", llm_client.provider)
    except ValueError as e:
        sample_keys = [envs[0] for envs in API_KEY_ENV_VARS.values() if envs]
        sample_text = ", ".join(sample_keys[:5])
        logger.error("LLM client initialization failed: %s", e)
        if args.model == DEFAULT_MODEL_FALLBACK and _other_provider_key_set():
            logger.error(
                "Note: since pplyz 0.2.0 the default model is %s (OpenRouter). "
                "To keep using another provider, pass --model "
                "(e.g. --model gemini/gemini-2.5-flash-lite) or set default_model "
                "in config.toml.",
                DEFAULT_MODEL_FALLBACK,
            )
        logger.error(
            "Set the API key env var for your provider (e.g., %s) or add it to the TOML config.",
            sample_text,
        )
        logger.error("See the README for the full list of supported provider keys.")
        sys.exit(1)

    # Initialize processor
    processor = CSVProcessor(llm_client)

    try:
        if args.preview:
            preview_rows = resolve_preview_rows()
            # Preview mode
            processor.preview_sample(
                input_path=input_path,
                columns=columns,
                prompt=prompt,
                num_rows=preview_rows,
                response_model=response_model,
            )
        else:
            # Full processing mode (always overwrite input file unless backup is handled externally)
            processor.process_csv(
                input_path=input_path,
                output_path=input_path,
                columns=columns,
                prompt=prompt,
                response_model=response_model,
                resume=args.resume,
            )

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
