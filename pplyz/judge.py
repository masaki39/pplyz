"""Question definitions for `pplyz judge` (Jev decisions).

Users never see Jev's own vocabulary (noul/choice/score). Questions are described
with three plain answer types and saved to a small TOML file so a run can be
repeated without re-entering them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Type

try:  # Python 3.11+ provides tomllib
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

YES_NO = "yes_no"
CHOICE = "choice"
SCALE = "scale"
ANSWER_TYPES = (YES_NO, CHOICE, SCALE)

# Map plain answer types to Jev question types.
_JEV_TYPES = {YES_NO: "noul", CHOICE: "choice", SCALE: "score"}

MAX_CHOICE_OPTIONS = 255
MIN_SCALE_LEVELS = 2
MAX_SCALE_LEVELS = 10
PROBABILITY_SUFFIX = "_p"
QUESTIONS_FILE_SUFFIX = ".judge.toml"


@dataclass
class Question:
    """One question asked about every row; its answer fills one column."""

    column: str
    ask: str
    answer: str = YES_NO
    options: List[str] = field(default_factory=list)

    def validate(self) -> None:
        """Raise ValueError when the definition cannot be sent to Jev."""
        if not self.column.strip():
            raise ValueError("Column name cannot be empty.")
        if not self.ask.strip():
            raise ValueError(f"Question for '{self.column}' cannot be empty.")
        if self.answer not in ANSWER_TYPES:
            raise ValueError(
                f"Unknown answer type '{self.answer}' for '{self.column}'. "
                f"Use one of: {', '.join(ANSWER_TYPES)}."
            )
        if self.answer == YES_NO:
            return
        if len(set(self.options)) != len(self.options):
            raise ValueError(f"Options for '{self.column}' must be unique.")
        if any(not option.strip() for option in self.options):
            raise ValueError(f"Options for '{self.column}' cannot be empty.")
        if self.answer == CHOICE and not 2 <= len(self.options) <= MAX_CHOICE_OPTIONS:
            raise ValueError(
                f"'{self.column}' needs 2-{MAX_CHOICE_OPTIONS} options "
                f"(got {len(self.options)})."
            )
        if self.answer == SCALE and not (
            MIN_SCALE_LEVELS <= len(self.options) <= MAX_SCALE_LEVELS
        ):
            raise ValueError(
                f"'{self.column}' needs {MIN_SCALE_LEVELS}-{MAX_SCALE_LEVELS} levels "
                f"(got {len(self.options)})."
            )

    def to_jev(self) -> Dict[str, Any]:
        """Return the Jev question payload."""
        payload: Dict[str, Any] = {
            "type": _JEV_TYPES[self.answer],
            "instructions": self.ask,
        }
        if self.answer == CHOICE:
            payload["criteria"] = {option: option for option in self.options}
        elif self.answer == SCALE:
            # Ordered lowest -> highest; Jev scores are indexes from zero.
            payload["criteria"] = list(self.options)
        return payload

    def describe(self) -> str:
        """Short human-readable summary, e.g. ``level (high|mid|low)``."""
        if self.answer == YES_NO:
            return f"{self.column} (yes/no probability)"
        separator = " < " if self.answer == SCALE else "|"
        return f"{self.column} ({separator.join(self.options)})"


def validate_questions(questions: List[Question]) -> None:
    """Validate a question list, including column-name uniqueness."""
    if not questions:
        raise ValueError("At least one question is required.")
    seen: set[str] = set()
    for question in questions:
        question.validate()
        if question.column in seen:
            raise ValueError(f"Duplicate column name '{question.column}'.")
        seen.add(question.column)


def output_fields(
    questions: List[Question], with_probability: bool = False
) -> Dict[str, Type]:
    """Return output column names and types, in question order.

    Yes/no columns hold the probability of "yes" (0-1). Choice and scale columns
    hold the chosen label; ``with_probability`` adds ``<column>_p`` with the
    probability of that label.
    """
    fields: Dict[str, Type] = {}
    for question in questions:
        if question.answer == YES_NO:
            fields[question.column] = float
            continue
        fields[question.column] = str
        if with_probability:
            fields[question.column + PROBABILITY_SUFFIX] = float
    return fields


def output_values(
    question: Question, answer: Dict[str, Any], with_probability: bool = False
) -> Dict[str, Any]:
    """Convert one Jev answer into column values.

    Raises:
        ValueError: If the answer does not match the question.
    """
    try:
        if question.answer == YES_NO:
            return {question.column: float(answer["noul"])}
        if question.answer == CHOICE:
            value = answer["choice"]
            if value not in question.options:
                raise ValueError(f"unexpected option '{value}'")
            key = value
        else:
            index = int(answer["score"])
            if not 0 <= index < len(question.options):
                raise ValueError(f"score {index} out of range")
            value = question.options[index]
            key = str(index)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid Jev answer for '{question.column}': {exc} ({answer})"
        ) from exc

    values: Dict[str, Any] = {question.column: value}
    if with_probability:
        probability = (answer.get("probabilities") or {}).get(key)
        if probability is None:
            probability = answer.get("confidence")
        values[question.column + PROBABILITY_SUFFIX] = (
            None if probability is None else float(probability)
        )
    return values


def default_questions_path(csv_path: Path) -> Path:
    """Return the questions file stored next to a CSV (``data.judge.toml``)."""
    return csv_path.with_name(csv_path.stem + QUESTIONS_FILE_SUFFIX)


@dataclass
class JudgeDefinition:
    """Questions plus the input columns they were written for."""

    questions: List[Question]
    input_columns: List[str] = field(default_factory=list)


def load_definition(path: Path) -> JudgeDefinition:
    """Load and validate a questions file.

    Raises:
        ValueError: If the file cannot be read or is invalid.
    """
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Failed to read questions file {path}: {exc}") from exc

    raw_questions = data.get("questions")
    if not isinstance(raw_questions, list):
        raise ValueError(f"{path} has no [[questions]] entries.")

    questions = []
    for entry in raw_questions:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid question entry in {path}: {entry}")
        questions.append(
            Question(
                column=str(entry.get("column", "")).strip(),
                ask=str(entry.get("ask", "")).strip(),
                answer=str(entry.get("answer", YES_NO)).strip(),
                options=[str(option).strip() for option in entry.get("options", [])],
            )
        )
    validate_questions(questions)

    input_columns = data.get("input", [])
    if isinstance(input_columns, str):
        input_columns = [col.strip() for col in input_columns.split(",")]
    return JudgeDefinition(
        questions=questions,
        input_columns=[str(col) for col in input_columns if str(col).strip()],
    )


def _toml_string(value: str) -> str:
    # JSON string escapes are valid TOML basic-string escapes; DEL must be escaped too.
    return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")


def _toml_array(values: Iterable[str]) -> str:
    return "[" + ", ".join(_toml_string(v) for v in values) + "]"


def dump_definition(definition: JudgeDefinition) -> str:
    """Serialize a definition to TOML."""
    lines = [
        "# Questions for `pplyz judge`. Edit freely, then rerun: pplyz judge <csv>",
        "# answer: yes_no | choice (options) | scale (options, lowest -> highest)",
    ]
    if definition.input_columns:
        lines.append(f"input = {_toml_array(definition.input_columns)}")
    for question in definition.questions:
        lines += [
            "",
            "[[questions]]",
            f"column = {_toml_string(question.column)}",
            f"ask = {_toml_string(question.ask)}",
            f"answer = {_toml_string(question.answer)}",
        ]
        if question.answer != YES_NO:
            lines.append(f"options = {_toml_array(question.options)}")
    return "\n".join(lines) + "\n"


def save_definition(definition: JudgeDefinition, path: Path) -> None:
    """Write a definition to ``path``."""
    path.write_text(dump_definition(definition), encoding="utf-8")


_ANSWER_MENU = (
    (YES_NO, "Yes / No (saved as the probability of yes, 0-1)"),
    (CHOICE, "Pick one of several options"),
    (SCALE, "Rate on a scale (lowest -> highest)"),
)


def run_wizard(
    reserved_columns: Iterable[str] = (),
    ask: Optional[Callable[[str], str]] = None,
    say: Optional[Callable[[str], None]] = None,
) -> List[Question]:
    """Interactively build questions without any Jev-specific knowledge.

    Args:
        reserved_columns: Names that must not be used as output columns
            (e.g. columns already in the CSV).
        ask: Function used to read a line of input (default: ``input``).
        say: Function used to print a line (default: ``print``).

    Returns:
        Validated questions in entry order.
    """
    ask = ask or input
    say = say or print
    reserved = set(reserved_columns)
    questions: List[Question] = []

    while True:
        number = len(questions) + 1
        say("")
        say(f"Question {number}: What should be judged for each row?")
        if number == 1:
            say('  e.g. "Is this paper about cancer research?"')
        text = _ask_nonempty(ask, "> ")

        say("How should it be answered?")
        for i, (_, label) in enumerate(_ANSWER_MENU, start=1):
            say(f"  {i}) {label}")
        answer = _ask_menu(ask, say, len(_ANSWER_MENU))
        answer_type = _ANSWER_MENU[answer - 1][0]

        options: List[str] = []
        if answer_type == CHOICE:
            options = _ask_options(
                ask,
                say,
                'Options, separated by "|" (e.g. positive|neutral|negative): ',
                2,
                MAX_CHOICE_OPTIONS,
            )
        elif answer_type == SCALE:
            options = _ask_options(
                ask,
                say,
                'Levels from lowest to highest, separated by "|" (e.g. low|medium|high): ',
                MIN_SCALE_LEVELS,
                MAX_SCALE_LEVELS,
            )

        used = reserved | {q.column for q in questions}
        column = _ask_column(ask, say, _default_column(number, used), used)
        question = Question(
            column=column, ask=text, answer=answer_type, options=options
        )
        question.validate()
        questions.append(question)

        more = ask("Add another question? [y/N]: ").strip().lower()
        if more not in ("y", "yes"):
            return questions


def _ask_nonempty(ask: Callable[[str], str], prompt: str) -> str:
    while True:
        value = ask(prompt).strip()
        if value:
            return value


def _ask_menu(ask: Callable[[str], str], say: Callable[[str], None], size: int) -> int:
    while True:
        value = ask("Choose [1]: ").strip() or "1"
        if value.isdigit() and 1 <= int(value) <= size:
            return int(value)
        say(f"  Please enter a number from 1 to {size}.")


def _ask_options(
    ask: Callable[[str], str],
    say: Callable[[str], None],
    prompt: str,
    minimum: int,
    maximum: int,
) -> List[str]:
    while True:
        options = [part.strip() for part in ask(prompt).split("|") if part.strip()]
        if len(set(options)) != len(options):
            say("  Options must be unique.")
        elif not minimum <= len(options) <= maximum:
            say(f"  Please enter {minimum}-{maximum} options.")
        else:
            return options


def _default_column(number: int, used: set[str]) -> str:
    candidate = f"q{number}"
    suffix = number
    while candidate in used:
        suffix += 1
        candidate = f"q{suffix}"
    return candidate


def _ask_column(
    ask: Callable[[str], str],
    say: Callable[[str], None],
    default: str,
    used: set[str],
) -> str:
    while True:
        column = ask(f"Output column name [{default}]: ").strip() or default
        if column in used:
            say(f"  '{column}' is already used. Choose another name.")
            continue
        return column


def summarize(questions: List[Question]) -> str:
    """One-line summary used in logs."""
    return ", ".join(q.describe() for q in questions)


def find_definition(csv_path: Path, explicit: Optional[str]) -> tuple[Path, bool]:
    """Return (questions path, exists) for an explicit or default location."""
    if explicit:
        path = Path(explicit).expanduser()
        return path, path.exists()
    path = default_questions_path(csv_path)
    return path, path.exists()
