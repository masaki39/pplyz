"""`pplyz judge` - fast yes/no, choice and scale decisions with Jev."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
except ImportError:  # pragma: no cover - optional dependency
    PromptSession = None
    Completer = object  # type: ignore[assignment,misc]
    Completion = None

from pplyz.cli import CompactHelpFormatter, resolve_input_path, resolve_preview_rows
from pplyz.config import (
    DEFAULT_INPUT_COLUMNS_ENV_VAR,
    DEFAULT_JEV_WORKERS,
    get_default_jev_model,
)
from pplyz.jev_client import JevClient
from pplyz.judge import (
    JudgeDefinition,
    find_definition,
    load_definition,
    output_fields,
    record_last_run,
    record_written_columns,
    run_wizard,
    save_definition,
    summarize,
)
from pplyz.processor import CSVProcessor, read_csv_preserving
from pplyz.schemas import create_output_model
from pplyz.utils import FatalAPIError

logger = logging.getLogger(__name__)

# Wide CSVs: list only the first columns and rely on Tab completion for the rest.
MAX_LISTED_COLUMNS = 30


def parse_judge_arguments(argv: List[str]) -> argparse.Namespace:
    """Parse `pplyz judge` arguments."""
    default_model = get_default_jev_model()
    parser = argparse.ArgumentParser(
        prog="pplyz judge",
        usage="pplyz judge INPUT [options]",
        formatter_class=CompactHelpFormatter,
        description=(
            "Answer yes/no, choice or scale questions for every row with Jev "
            "(via OpenRouter).\nRun without --questions to define questions "
            "interactively; they are saved next to the CSV for reuse.\n"
            "Results are added as new columns to INPUT (in place) unless -o is given."
        ),
        epilog=(
            "examples:\n"
            "  pplyz judge papers.csv              first run asks the questions\n"
            "  pplyz judge papers.csv -p           preview a few rows only\n"
            "  pplyz judge papers.csv -o out.csv   keep papers.csv unchanged\n"
            "  pplyz judge papers.csv -f           recompute every row"
        ),
    )
    parser.add_argument("input_path", metavar="INPUT", help="Input CSV path")
    parser.add_argument(
        "--input",
        "-i",
        dest="input_columns",
        type=str,
        help="Comma-separated columns shown to Jev (asked interactively if omitted)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Write results to this CSV instead of updating INPUT in place",
    )
    parser.add_argument(
        "--questions",
        "-q",
        type=str,
        help="Questions file (default: <INPUT stem>.judge.toml next to the CSV)",
    )
    parser.add_argument(
        "--preview",
        "-p",
        action="store_true",
        help="Preview results on sample rows without saving",
    )
    parser.add_argument(
        "--force",
        "-f",
        dest="resume",
        action="store_false",
        default=True,
        help="Reprocess all rows (resume is default)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=default_model,
        help=f"Jev model on OpenRouter (default: {default_model})",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="Decisions endpoint URL (default: OpenRouter's alpha endpoint)",
    )
    parser.add_argument(
        "--with-probability",
        action="store_true",
        help="Also write <column>_p (probability of the chosen label) for choice/scale",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=_positive_int,
        default=DEFAULT_JEV_WORKERS,
        help=f"Rows processed in parallel (default: {DEFAULT_JEV_WORKERS})",
    )
    return parser.parse_args(argv)


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return number


def _is_interactive() -> bool:
    return sys.stdin.isatty()


_prompt_session = None


def _read_line(prompt: str, completer=None) -> str:
    """Read one line with full line editing when attached to a terminal.

    Plain ``input()`` leaves editing to the terminal driver, where Backspace
    breaks multi-byte (e.g. Japanese) characters and arrow keys print escape
    codes. prompt_toolkit handles both; tests and pipes fall back to ``input()``.
    """
    global _prompt_session
    if PromptSession is None or not (sys.stdin.isatty() and sys.stdout.isatty()):
        return input(prompt)
    if _prompt_session is None:
        _prompt_session = PromptSession()
    return _prompt_session.prompt(
        prompt, completer=completer, complete_while_typing=completer is not None
    )


def _confirm(message: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = _read_line(f"{message} {suffix}: ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


class ColumnCompleter(Completer):
    """Complete the column name being typed after the last comma.

    Matches by prefix first, then by substring, so wide CSVs can be searched by
    any part of a column name. Columns already entered are not suggested again.
    """

    def __init__(self, columns: List[str]):
        self.columns = columns

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        *entered, token = text.split(",")
        token = token.lstrip()
        chosen = {col.strip() for col in entered}
        needle = token.lower()
        candidates = [col for col in self.columns if col not in chosen]
        prefix = [col for col in candidates if col.lower().startswith(needle)]
        substring = [col for col in candidates if needle and needle in col.lower()[1:]]
        for col in prefix + [col for col in substring if col not in prefix]:
            yield Completion(col, start_position=-len(token))


def _ask_input_columns(available: List[str]) -> List[str]:
    print(f"\nColumns in the CSV ({len(available)}):")
    print("  " + ", ".join(available[:MAX_LISTED_COLUMNS]))
    if len(available) > MAX_LISTED_COLUMNS:
        hidden = len(available) - MAX_LISTED_COLUMNS
        print(f"  ... and {hidden} more (type any part of a name and press Tab)")
    while True:
        raw = _read_line(
            "Which columns should Jev read? (comma-separated, Tab to complete): ",
            completer=ColumnCompleter(available),
        )
        columns = [col.strip() for col in raw.split(",") if col.strip()]
        missing = [col for col in columns if col not in available]
        if columns and not missing:
            return columns
        if missing:
            print(f"  Not found: {', '.join(missing)}")


def _resolve_definition(
    args: argparse.Namespace, input_path: Path, csv_columns: List[str]
) -> tuple[JudgeDefinition, Path, bool]:
    """Load saved questions or build them with the wizard.

    Returns:
        (definition, questions path, created) where ``created`` is True when the
        questions were just entered interactively.
    """
    path, exists = find_definition(input_path, args.questions)
    previous: JudgeDefinition | None = None
    if exists:
        use_saved = (
            args.questions is not None
            or not _is_interactive()
            or _confirm(f"Use saved questions from {path.name}? (n = create new ones)")
        )
        if use_saved:
            definition = load_definition(path)
            logger.info("Loaded questions from %s", path)
            return definition, path, False
        try:
            previous = load_definition(path)
        except ValueError:
            previous = None
    elif args.questions is not None:
        raise ValueError(f"Questions file not found: {path}")

    if not _is_interactive():
        raise ValueError(
            "No questions defined. Run 'pplyz judge' in a terminal once to create "
            f"them, or pass --questions (expected at {path})."
        )

    input_columns = _split_columns(args.input_columns)
    _check_columns_exist(input_columns, csv_columns)
    if not input_columns:
        input_columns = _ask_input_columns(csv_columns)

    # Columns written by the questions being replaced may be reused; the new
    # definition inherits them so they still count as pplyz judge output.
    owned: List[str] = []
    if previous is not None:
        owned = list(previous.written_columns)
        owned += [
            col
            for col in output_fields(previous.questions, True)
            if col in csv_columns and col not in owned
        ]
    reserved = [col for col in csv_columns if col not in owned]
    questions = run_wizard(reserved_columns=reserved, ask=_read_line)
    definition = JudgeDefinition(
        questions=questions, input_columns=input_columns, written_columns=owned
    )

    if exists:
        backup = path.with_name(path.name + ".bak")
        path.replace(backup)
        print(f"\n→ Previous questions moved to {backup.name}")
    save_definition(definition, path)
    print(f"\n→ Saved questions to {path} (edit it or rerun to reuse)")
    return definition, path, True


def _split_columns(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [col.strip() for col in raw.split(",") if col.strip()]


def _check_columns_exist(columns: List[str], csv_columns: List[str]) -> None:
    missing = [col for col in columns if col not in csv_columns]
    if missing:
        raise ValueError(
            f"Columns not found in the CSV: {', '.join(missing)}\n"
            f"Available columns: {', '.join(csv_columns)}"
        )


def _check_output_columns(
    output_columns: List[str],
    input_columns: List[str],
    csv_columns: List[str],
    owned: List[str],
) -> None:
    """Refuse to overwrite source data (e.g. `column = "title"` in the file)."""
    overlap = [col for col in output_columns if col in input_columns]
    if overlap:
        raise ValueError(
            f"Output column(s) {', '.join(overlap)} are also input columns. "
            "Rename the question's `column` in the questions file."
        )
    foreign = [col for col in output_columns if col in csv_columns and col not in owned]
    if foreign:
        raise ValueError(
            f"Column(s) {', '.join(foreign)} already exist in the CSV and were not "
            "written by pplyz judge. Rename the question's `column` in the questions "
            "file so the original data is not overwritten. If pplyz judge did create "
            f"them, add `written = {json.dumps(foreign, ensure_ascii=False)}` to the "
            "top of the questions file."
        )


def _has_answers(csv_path: Path, output_columns: List[str]) -> bool:
    if not csv_path.exists():
        return False
    df = read_csv_preserving(csv_path)
    present = [col for col in output_columns if col in df.columns]
    return bool(present) and bool(df[present].notna().any().any())


def _record(recorder, definition: JudgeDefinition, path: Path, value) -> None:
    """Update the questions file; a read-only file only loses the bookkeeping."""
    try:
        recorder(definition, path, value)
    except OSError as exc:
        logger.warning("Could not update %s: %s", path, exc)


def _resolve_output_path(raw: str | None, input_path: Path) -> Path:
    if not raw:
        return input_path
    path = Path(raw).expanduser().resolve()
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Output file must have a .csv extension: {path}")
    return path


def judge_main(argv: List[str]) -> None:
    """Entry point for `pplyz judge`."""
    args = parse_judge_arguments(argv)
    input_path = resolve_input_path(args.input_path)

    try:
        output_path = _resolve_output_path(args.output, input_path)
        csv_columns = list(pd.read_csv(input_path, nrows=0).columns)
        definition, questions_path, created = _resolve_definition(
            args, input_path, csv_columns
        )
        columns = (
            _split_columns(args.input_columns)
            or definition.input_columns
            or _split_columns(os.environ.get(DEFAULT_INPUT_COLUMNS_ENV_VAR))
        )
        if not columns:
            raise ValueError(
                "no input columns. Pass --input/-i or add `input = [...]` "
                "to the questions file."
            )
        _check_columns_exist(columns, csv_columns)
        fields = output_fields(definition.questions, args.with_probability)
        target_path = (
            output_path
            if args.resume and output_path.exists() and output_path != input_path
            else input_path
        )
        target_columns = list(pd.read_csv(target_path, nrows=0).columns)
        _check_output_columns(
            list(fields), columns, target_columns, definition.written_columns
        )

        # Answers already in the CSV may come from other questions: recreated
        # questions reusing a column, edited questions, or an unknown history.
        resume = args.resume
        stale = (
            resume
            and _has_answers(output_path, list(fields))
            and (created or not definition.matches_last_run(columns))
        )
        if stale:
            if _is_interactive():
                resume = not _confirm(
                    "The CSV already has answers from different or edited questions. "
                    "Recompute those rows?"
                )
            else:
                logger.warning(
                    "The CSV already has answers from different or edited questions; "
                    "they are kept. Rerun with -f to recompute them."
                )
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    questions = definition.questions
    response_model = create_output_model(fields, "JudgeOutput")
    summary = "Jev questions: " + summarize(questions)

    try:
        client = JevClient(
            questions,
            model_name=args.model,
            endpoint=args.endpoint,
            with_probability=args.with_probability,
        )
    except ValueError as exc:
        logger.error("Jev client initialization failed: %s", exc)
        sys.exit(1)

    processor = CSVProcessor(client)
    preview_rows = resolve_preview_rows()

    try:
        if args.preview or created:
            processor.preview_sample(
                input_path=input_path,
                columns=columns,
                prompt=summary,
                num_rows=preview_rows,
                response_model=response_model,
            )
            if args.preview:
                return
            total = len(read_csv_preserving(input_path))
            target = (
                f"{output_path.name} (in place)"
                if output_path == input_path
                else output_path.name
            )
            if not _confirm(
                f"\nProcess all {total:,} rows and write results into {target}?"
            ):
                print("Stopped. Rerun the same command to process all rows.")
                return

        _record(record_written_columns, definition, questions_path, list(fields))
        if resume and not stale:
            # Answers in the CSV (old and new) all match these questions.
            _record(record_last_run, definition, questions_path, columns)
        processor.process_csv(
            input_path=input_path,
            output_path=output_path,
            columns=columns,
            prompt=summary,
            response_model=response_model,
            resume=resume,
            workers=args.workers,
        )
        if not resume:
            # Every row was recomputed with these questions.
            _record(record_last_run, definition, questions_path, columns)
    except FatalAPIError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except (KeyboardInterrupt, EOFError):
        print("\n\nProcessing interrupted by user (progress saved)")
        sys.stdout.flush()
        # Worker threads may still be waiting on a request; don't join them.
        os._exit(130)
