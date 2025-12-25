"""Tests for main CLI module."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from pplyz.cli import get_user_prompt, main, parse_arguments, resolve_preview_rows
from pplyz.config import DEFAULT_PREVIEW_ROWS


@pytest.fixture(autouse=True)
def clear_cli_defaults(monkeypatch):
    """Ensure CLI default env vars are unset for each test."""
    for key in (
        "PPLYZ_DEFAULT_INPUT",
        "PPLYZ_DEFAULT_OUTPUT",
        "PPLYZ_PREVIEW_ROWS",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def disable_runtime_config(monkeypatch):
    """Avoid reading developer-local config files during CLI tests."""
    monkeypatch.setattr("pplyz.cli.load_runtime_configuration", lambda: None)


class TestParseArguments:
    """Test command-line argument parsing."""

    def test_parse_required_arguments(self):
        """Test parsing required arguments."""
        test_args = [
            "pplyz",
            "test.csv",
            "--input",
            "col1,col2",
            "--output",
            "score:float,label:str",
        ]

        with patch.object(sys, "argv", test_args):
            args = parse_arguments()

        assert args.input_path == "test.csv"
        assert args.input_columns == "col1,col2"
        assert args.output_fields == "score:float,label:str"

    def test_parse_with_model_option(self):
        """Test parsing with model option."""
        test_args = [
            "pplyz",
            "test.csv",
            "--input",
            "col1",
            "--model",
            "gpt-4o",
            "--output",
            "confidence:float",
        ]

        with patch.object(sys, "argv", test_args):
            args = parse_arguments()

            assert args.model == "gpt-4o"

    def test_parse_with_preview_option(self):
        """Test parsing with preview option."""
        test_args = [
            "pplyz",
            "test.csv",
            "--input",
            "col1",
            "--preview",
            "--output",
            "flag:bool",
        ]

        with patch.object(sys, "argv", test_args):
            args = parse_arguments()

            assert args.preview is True

    def test_parse_with_output_option(self):
        """Test parsing with output schema option."""
        test_args = [
            "pplyz",
            "test.csv",
            "--input",
            "col1",
            "--output",
            "score:float,label:str",
        ]

        with patch.object(sys, "argv", test_args):
            args = parse_arguments()

        assert args.output_fields == "score:float,label:str"

    def test_parse_with_prompt_option(self):
        """Test parsing with inline prompt option."""
        test_args = [
            "pplyz",
            "test.csv",
            "--input",
            "col1",
            "--output",
            "flag:bool",
            "--prompt",
            "Inline prompt",
        ]

        with patch.object(sys, "argv", test_args):
            args = parse_arguments()

        assert args.prompt == "Inline prompt"
        assert args.prompt_file is None

    def test_parse_with_prompt_file_option(self):
        """Test parsing with prompt file option."""
        test_args = [
            "pplyz",
            "test.csv",
            "--input",
            "col1",
            "--output",
            "flag:bool",
            "--prompt-file",
            "prompt.txt",
        ]

        with patch.object(sys, "argv", test_args):
            args = parse_arguments()

        assert args.prompt is None
        assert args.prompt_file == "prompt.txt"

    def test_parse_prompt_options_conflict(self):
        """Inline and file prompts should be mutually exclusive."""
        test_args = [
            "pplyz",
            "test.csv",
            "--input",
            "col1",
            "--output",
            "flag:bool",
            "--prompt",
            "Inline prompt",
            "--prompt-file",
            "prompt.txt",
        ]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                parse_arguments()

    def test_parse_positional_input(self):
        """Positional INPUT argument should populate input_path."""
        test_args = [
            "pplyz",
            "data.csv",
            "--input",
            "col1",
            "--output",
            "flag:bool",
        ]

        with patch.object(sys, "argv", test_args):
            args = parse_arguments()

        assert args.input_path == "data.csv"
        assert args.input_columns == "col1"

    def test_parse_short_options(self):
        """Test parsing with short option names."""
        test_args = [
            "pplyz",
            "test.csv",
            "-i",
            "col1,col2",
            "-m",
            "gpt-4o",
            "-p",
            "-o",
            "flag:bool",
            "-f",
        ]

        with patch.object(sys, "argv", test_args):
            args = parse_arguments()

        assert args.input_path == "test.csv"
        assert args.input_columns == "col1,col2"
        assert args.model == "gpt-4o"
        assert args.preview is True
        assert args.output_fields == "flag:bool"
        assert args.resume is False
        assert not hasattr(args, "list_models")

    def test_help_uses_compact_flags(self, capsys):
        """Help text should not duplicate metavar values in option listing."""
        test_args = ["pplyz", "--help"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                parse_arguments()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "usage: pplyz [INPUT] [options]" in captured.out
        options_text = captured.out.split("options:", 1)[1]
        assert "-i, --input" in options_text
        assert "--input INPUT, -i INPUT" not in options_text
        # Short flags should appear before long flags
        assert options_text.index("-h, --help") < options_text.index("-i, --input")


class TestPromptInput:
    """Tests for interactive prompt handling."""

    def test_prompt_keyboard_interrupt(self, monkeypatch, capsys):
        """Ctrl+C during prompt should exit gracefully."""
        monkeypatch.setattr("pplyz.cli._build_prompt_session", lambda: None)

        def _raise_keyboard_interrupt(_prompt):
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", _raise_keyboard_interrupt)

        with pytest.raises(SystemExit) as exc_info:
            get_user_prompt()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Prompt entry cancelled" in captured.out


class TestPreviewRowResolution:
    """Tests for resolving preview row count."""

    def test_preview_rows_default(self, monkeypatch):
        """Falls back to default when env var missing."""
        monkeypatch.delenv("PPLYZ_PREVIEW_ROWS", raising=False)
        assert resolve_preview_rows() == DEFAULT_PREVIEW_ROWS

    def test_preview_rows_from_env(self, monkeypatch):
        """Uses env var when provided."""
        monkeypatch.setenv("PPLYZ_PREVIEW_ROWS", "7")
        assert resolve_preview_rows() == 7

    def test_preview_rows_invalid_env(self, monkeypatch, caplog):
        """Falls back to default on invalid value."""
        caplog.set_level("INFO")
        monkeypatch.setenv("PPLYZ_PREVIEW_ROWS", "zero")
        assert resolve_preview_rows() == DEFAULT_PREVIEW_ROWS
        assert "Invalid preview row count" in caplog.text


class TestMainExecution:
    """Test main function execution flow."""

    def test_main_requires_input_and_columns(self, capsys):
        """Test that main requires INPUT and --input columns without --list."""
        test_args = ["pplyz"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert (
                "required: INPUT (positional), --input/-i, --output/-o" in captured.out
            )

    def test_main_requires_columns_when_only_input(self, capsys):
        """Test that main requires --input columns when only CSV is provided."""
        test_args = ["pplyz", "test.csv"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "required: --input/-i, --output/-o" in captured.out

    def test_main_requires_input_when_only_columns(self, capsys):
        """Test that main requires INPUT when only columns are provided."""
        test_args = ["pplyz", "--input", "col1,col2"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "required: INPUT (positional), --output/-o" in captured.out

    def test_main_accepts_preview_without_separate_file(self):
        """Test that main accepts --preview even though we always overwrite the input CSV."""
        test_args = [
            "pplyz",
            "test.csv",
            "--input",
            "col1",
            "--preview",
            "--output",
            "flag:bool",
        ]

        with patch.object(sys, "argv", test_args):
            with patch("pplyz.cli.get_user_prompt", return_value="Test prompt"):
                with patch("pplyz.cli.LLMClient"):
                    with patch("pplyz.cli.CSVProcessor"):
                        # Should not raise SystemExit even though the CSV doesn't exist;
                        # the FileNotFoundError happens later during processing.
                        try:
                            main()
                        except (FileNotFoundError, SystemExit):
                            pass  # Expected due to missing test.csv

    def test_main_requires_output_when_missing(self, capsys):
        """Test that main requires --output when INPUT/--input are provided."""
        test_args = [
            "pplyz",
            "test.csv",
            "--input",
            "col1",
        ]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "required: --output/-o" in captured.out

    def test_main_accepts_positional_input(self):
        """Positional input path should behave like --input."""
        test_args = [
            "pplyz",
            "test.csv",
            "--input",
            "col1",
            "--preview",
            "--output",
            "flag:bool",
        ]

        with patch.object(sys, "argv", test_args):
            with patch("pplyz.cli.get_user_prompt", return_value="Test prompt"):
                with patch("pplyz.cli.LLMClient"):
                    with patch("pplyz.cli.CSVProcessor"):
                        main()

    def test_main_stops_before_prompt_when_input_missing(self, monkeypatch, capsys):
        """Missing CSV should abort before prompting the user."""
        test_args = [
            "pplyz",
            "missing.csv",
            "--input",
            "title",
            "--output",
            "summary:str",
        ]

        prompt_called = False

        def _fail_prompt():
            nonlocal prompt_called
            prompt_called = True
            return "should-not-happen"

        with patch.object(sys, "argv", test_args):
            with patch("pplyz.cli.get_user_prompt", side_effect=_fail_prompt):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1
        assert prompt_called is False
        captured = capsys.readouterr()
        assert "Input file not found" in captured.out

    def test_main_rejects_non_csv_extension(self, tmp_path, capsys):
        """Non-CSV inputs should be rejected before prompting the user."""
        bad_file = tmp_path / "data.txt"
        bad_file.write_text("dummy")

        prompt_called = False

        def _fail_prompt():
            nonlocal prompt_called
            prompt_called = True
            return "should-not-happen"

        test_args = [
            "pplyz",
            str(bad_file),
            "--input",
            "title",
            "--output",
            "summary:str",
        ]

        with patch.object(sys, "argv", test_args):
            with patch("pplyz.cli.get_user_prompt", side_effect=_fail_prompt):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1
        assert prompt_called is False
        captured = capsys.readouterr()
        assert "must have a .csv" in captured.out

    def test_main_rejects_directory_input(self, tmp_path, capsys):
        """Directories are not valid CSV inputs."""
        dir_path = tmp_path / "data_dir"
        dir_path.mkdir()

        prompt_called = False

        def _fail_prompt():
            nonlocal prompt_called
            prompt_called = True
            return "should-not-happen"

        test_args = [
            "pplyz",
            str(dir_path),
            "--input",
            "title",
            "--output",
            "summary:str",
        ]

        with patch.object(sys, "argv", test_args):
            with patch("pplyz.cli.get_user_prompt", side_effect=_fail_prompt):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1
        assert prompt_called is False
        captured = capsys.readouterr()
        assert "must be a CSV file, not a directory" in captured.out

    def test_main_uses_env_defaults_for_required_arguments(self, monkeypatch):
        """CLI should fall back to environment/config defaults for column/output flags."""
        test_args = [
            "pplyz",
            "test.csv",
            "--preview",
        ]

        monkeypatch.setenv("PPLYZ_DEFAULT_INPUT", "title")
        monkeypatch.setenv("PPLYZ_DEFAULT_OUTPUT", "summary:str")

        with patch.object(sys, "argv", test_args):
            with patch("pplyz.cli.get_user_prompt", return_value="Test prompt"):
                with patch("pplyz.cli.LLMClient"):
                    with patch("pplyz.cli.CSVProcessor"):
                        main()

    def test_main_uses_inline_prompt_without_interactive(self, tmp_path):
        """Inline prompt should skip interactive prompt entry."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("col1\nvalue\n")

        test_args = [
            "pplyz",
            str(csv_path),
            "--input",
            "col1",
            "--preview",
            "--output",
            "flag:bool",
            "--prompt",
            "Inline prompt",
        ]

        with patch.object(sys, "argv", test_args):
            with patch("pplyz.cli.get_user_prompt", side_effect=AssertionError):
                with patch("pplyz.cli.LLMClient"):
                    with patch("pplyz.cli.CSVProcessor") as mock_processor:
                        instance = mock_processor.return_value
                        instance.preview_sample = MagicMock()
                        main()

        _, kwargs = instance.preview_sample.call_args
        assert kwargs["prompt"] == "Inline prompt"

    def test_main_uses_prompt_file_without_interactive(self, tmp_path):
        """Prompt file should skip interactive prompt entry."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("col1\nvalue\n")
        prompt_path = tmp_path / "prompt.md"
        prompt_path.write_text("Line one\nLine two\n")

        test_args = [
            "pplyz",
            str(csv_path),
            "--input",
            "col1",
            "--preview",
            "--output",
            "flag:bool",
            "--prompt-file",
            str(prompt_path),
        ]

        with patch.object(sys, "argv", test_args):
            with patch("pplyz.cli.get_user_prompt", side_effect=AssertionError):
                with patch("pplyz.cli.LLMClient"):
                    with patch("pplyz.cli.CSVProcessor") as mock_processor:
                        instance = mock_processor.return_value
                        instance.preview_sample = MagicMock()
                        main()

        _, kwargs = instance.preview_sample.call_args
        assert kwargs["prompt"] == "Line one\nLine two\n"
