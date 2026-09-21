"""End-to-end tests for `pplyz judge` and `pplyz auth` (API calls mocked)."""

import sys
from unittest.mock import patch

import pandas as pd
import pytest

from pplyz import judge_cli
from pplyz.cli import main
from pplyz.jev_client import JevClient
from pplyz.judge import CHOICE, JudgeDefinition, Question, save_definition


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setattr("pplyz.cli.load_runtime_configuration", lambda: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.delenv("PPLYZ_DEFAULT_INPUT", raising=False)


@pytest.fixture
def fake_jev(monkeypatch):
    """Answer every row: yes for the first question, 'low' for the second."""
    bodies = []

    def fake_post(self, body):
        bodies.append(body)
        return {
            "answers": {
                "q1": {"type": "noul", "noul": 0.9},
                "q2": {
                    "type": "choice",
                    "choice": "low",
                    "probabilities": {"high": 0.4, "low": 0.6},
                },
            }
        }

    monkeypatch.setattr(JevClient, "_post", fake_post)
    return bodies


def _save_questions(csv_path):
    definition = JudgeDefinition(
        questions=[
            Question("is_relevant", "Relevant?"),
            Question("level", "Level?", CHOICE, ["high", "low"]),
        ],
        input_columns=["title"],
    )
    path = csv_path.with_name(csv_path.stem + ".judge.toml")
    save_definition(definition, path)
    return path


def _run(args):
    with patch.object(sys, "argv", ["pplyz", *args]):
        main()


@pytest.mark.parametrize("workers", ["1", "4"])
def test_judge_uses_saved_questions(sample_csv_file, fake_jev, monkeypatch, workers):
    monkeypatch.setattr(judge_cli, "_is_interactive", lambda: False)
    _save_questions(sample_csv_file)

    _run(["judge", str(sample_csv_file), "--with-probability", "-w", workers])

    df = pd.read_csv(sample_csv_file)
    assert df["is_relevant"].tolist() == [0.9, 0.9, 0.9]
    assert df["level"].tolist() == ["low", "low", "low"]
    assert df["level_p"].tolist() == [0.6, 0.6, 0.6]
    assert sorted(body["state"]["title"] for body in fake_jev) == sorted(
        df["title"].tolist()
    )


@pytest.mark.parametrize("workers", ["1", "4"])
def test_judge_resume_skips_processed_rows(
    sample_csv_file, fake_jev, monkeypatch, workers
):
    monkeypatch.setattr(judge_cli, "_is_interactive", lambda: False)
    _save_questions(sample_csv_file)

    _run(["judge", str(sample_csv_file), "-w", workers])
    _run(["judge", str(sample_csv_file), "-w", workers])

    assert len(fake_jev) == 3


def test_judge_without_questions_non_interactive(sample_csv_file, monkeypatch, capsys):
    monkeypatch.setattr(judge_cli, "_is_interactive", lambda: False)
    with pytest.raises(SystemExit):
        _run(["judge", str(sample_csv_file)])
    assert "No questions defined" in capsys.readouterr().out


def test_judge_wizard_saves_previews_then_processes(
    sample_csv_file, fake_jev, monkeypatch
):
    monkeypatch.setattr(judge_cli, "_is_interactive", lambda: True)
    replies = iter(
        [
            "title,abstract",  # input columns
            "Relevant?",
            "1",
            "is_relevant",
            "y",
            "Level?",
            "2",
            "high|low",
            "level",
            "n",
            "",  # confirm the questions
            "y",  # process all rows after preview
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(replies))
    monkeypatch.setenv("PPLYZ_PREVIEW_ROWS", "1")

    _run(["judge", str(sample_csv_file)])

    saved = sample_csv_file.with_name(sample_csv_file.stem + ".judge.toml")
    assert 'input = ["title", "abstract"]' in saved.read_text()
    df = pd.read_csv(sample_csv_file)
    assert df["level"].tolist() == ["low", "low", "low"]
    # 1 preview row + 3 processed rows
    assert len(fake_jev) == 4


def test_judge_preview_does_not_write(sample_csv_file, fake_jev, monkeypatch):
    monkeypatch.setattr(judge_cli, "_is_interactive", lambda: False)
    _save_questions(sample_csv_file)

    _run(["judge", str(sample_csv_file), "--preview"])

    assert "is_relevant" not in pd.read_csv(sample_csv_file).columns


def test_auth_set_prints_config_hint(monkeypatch, capsys):
    stored = []
    monkeypatch.setattr("pplyz.cli.store_keychain", stored.append)

    _run(["auth", "set", "openrouter"])

    assert stored == ["openrouter"]
    assert 'OPENROUTER_API_KEY = "keychain:openrouter"' in capsys.readouterr().out


def _complete(columns, text):
    from prompt_toolkit.document import Document

    completer = judge_cli.ColumnCompleter(columns)
    return [c.text for c in completer.get_completions(Document(text), None)]


def test_column_completer_prefix_then_substring():
    columns = ["title", "abstract", "subtitle", "year"]
    assert _complete(columns, "ti") == ["title", "subtitle"]


def test_column_completer_completes_after_comma_and_skips_chosen():
    columns = ["title", "abstract", "author"]
    assert _complete(columns, "title, a") == ["abstract", "author"]
    assert _complete(columns, "abstract,") == ["title", "author"]


def test_judge_auth_error_stops_without_touching_csv(
    sample_csv_file, monkeypatch, capsys
):
    from pplyz.jev_client import JevFatalError

    monkeypatch.setattr(judge_cli, "_is_interactive", lambda: False)
    _save_questions(sample_csv_file)
    before = sample_csv_file.read_bytes()
    calls = []

    def fake_post(self, body):
        calls.append(body)
        raise JevFatalError("OpenRouter rejected the API key.")

    monkeypatch.setattr(JevClient, "_post", fake_post)

    with pytest.raises(SystemExit):
        _run(["judge", str(sample_csv_file), "-w", "1"])

    assert "rejected the API key" in capsys.readouterr().out
    assert len(calls) == 1
    assert sample_csv_file.read_bytes() == before


def test_judge_output_option_keeps_input(sample_csv_file, fake_jev, monkeypatch):
    monkeypatch.setattr(judge_cli, "_is_interactive", lambda: False)
    _save_questions(sample_csv_file)
    before = sample_csv_file.read_bytes()
    out = sample_csv_file.with_name("judged.csv")

    _run(["judge", str(sample_csv_file), "-o", str(out)])

    assert sample_csv_file.read_bytes() == before
    assert pd.read_csv(out)["level"].tolist() == ["low", "low", "low"]


def test_judge_rejects_output_column_equal_to_input(
    sample_csv_file, monkeypatch, capsys
):
    monkeypatch.setattr(judge_cli, "_is_interactive", lambda: False)
    path = sample_csv_file.with_name(sample_csv_file.stem + ".judge.toml")
    save_definition(JudgeDefinition([Question("title", "Relevant?")], ["title"]), path)
    with pytest.raises(SystemExit):
        _run(["judge", str(sample_csv_file)])
    assert "also input columns" in capsys.readouterr().out


def test_judge_asks_to_recompute_after_questions_change(
    sample_csv_file, fake_jev, monkeypatch
):
    monkeypatch.setattr(judge_cli, "_is_interactive", lambda: False)
    path = _save_questions(sample_csv_file)
    _run(["judge", str(sample_csv_file)])
    assert "last_run =" in path.read_text()

    path.write_text(path.read_text().replace("Relevant?", "Really relevant?"))
    monkeypatch.setattr(judge_cli, "_is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")

    _run(["judge", str(sample_csv_file)])

    assert len(fake_jev) == 6  # all rows recomputed


def test_judge_preserves_original_values(tmp_path, fake_jev, monkeypatch):
    monkeypatch.setattr(judge_cli, "_is_interactive", lambda: False)
    csv_path = tmp_path / "ids.csv"
    csv_path.write_text("id,title,note\n001,A,NA\n010,B,1.10\n")
    _save_questions(csv_path)

    _run(["judge", str(csv_path)])

    lines = csv_path.read_text().splitlines()
    assert lines[1].startswith("001,A,NA,")
    assert lines[2].startswith("010,B,1.10,")


def test_auth_set_rejects_slash(monkeypatch, capsys):
    monkeypatch.setattr("pplyz.cli.store_keychain", lambda name: None)
    with pytest.raises(SystemExit):
        _run(["auth", "set", "a/b"])
    assert "must not contain '/'" in capsys.readouterr().out
