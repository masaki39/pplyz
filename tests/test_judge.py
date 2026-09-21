"""Tests for pplyz.judge question definitions."""

import pytest

from pplyz import judge
from pplyz.judge import CHOICE, SCALE, YES_NO, JudgeDefinition, Question


class TestQuestion:
    def test_yes_no_to_jev(self):
        q = Question(column="is_rel", ask="Relevant?")
        assert q.to_jev() == {"type": "noul", "instructions": "Relevant?"}

    def test_choice_to_jev(self):
        q = Question("level", "Level?", CHOICE, ["high", "low"])
        assert q.to_jev() == {
            "type": "choice",
            "instructions": "Level?",
            "criteria": {"high": "high", "low": "low"},
        }

    def test_scale_to_jev_keeps_order(self):
        q = Question("s", "Score?", SCALE, ["low", "mid", "high"])
        assert q.to_jev()["criteria"] == ["low", "mid", "high"]
        assert q.to_jev()["type"] == "score"

    @pytest.mark.parametrize(
        "question",
        [
            Question("", "x"),
            Question("c", ""),
            Question("c", "x", "maybe"),
            Question("c", "x", CHOICE, ["only"]),
            Question("c", "x", CHOICE, ["a", "a"]),
            Question("c", "x", SCALE, [str(i) for i in range(11)]),
        ],
    )
    def test_invalid_questions(self, question):
        with pytest.raises(ValueError):
            question.validate()

    def test_duplicate_columns_rejected(self):
        with pytest.raises(ValueError, match="Duplicate"):
            judge.validate_questions([Question("a", "x"), Question("a", "y")])


class TestOutputValues:
    def test_yes_no_is_raw_probability(self):
        q = Question("rel", "Relevant?")
        assert judge.output_values(q, {"noul": 0.83}) == {"rel": 0.83}
        # No extra column for yes/no: the value already is the probability.
        assert judge.output_values(q, {"noul": 0.2}, with_probability=True) == {
            "rel": 0.2
        }

    def test_choice_with_probability(self):
        q = Question("level", "Level?", CHOICE, ["high", "low"])
        answer = {
            "choice": "low",
            "probabilities": {"high": 0.25, "low": 0.75},
            "confidence": 0.9,
        }
        assert judge.output_values(q, answer) == {"level": "low"}
        assert judge.output_values(q, answer, with_probability=True) == {
            "level": "low",
            "level_p": 0.75,
        }

    def test_probability_falls_back_to_confidence(self):
        q = Question("level", "Level?", CHOICE, ["high", "low"])
        values = judge.output_values(
            q, {"choice": "low", "confidence": 0.6}, with_probability=True
        )
        assert values["level_p"] == 0.6

    def test_choice_unknown_option(self):
        q = Question("level", "Level?", CHOICE, ["high", "low"])
        with pytest.raises(ValueError, match="unexpected option"):
            judge.output_values(q, {"choice": "other"})

    def test_scale_uses_most_likely_level_not_truncated_score(self):
        """Regression: Jev's score is a weighted mean (1.93), not an index."""
        q = Question("s", "Score?", SCALE, ["low", "mid", "high"])
        answer = {"score": 1.93, "probabilities": {"0": 0, "1": 0.06, "2": 0.94}}
        assert judge.output_values(q, answer, with_probability=True) == {
            "s": "high",
            "s_p": 0.94,
        }
        answer = {"score": 0.74, "probabilities": {"0": 0.33, "1": 0.6, "2": 0.07}}
        assert judge.output_values(q, answer) == {"s": "mid"}

    def test_scale_without_probabilities_rounds_score(self):
        q = Question("s", "Score?", SCALE, ["low", "mid", "high"])
        assert judge.output_values(q, {"score": 1.6}) == {"s": "high"}

    def test_scale_out_of_range(self):
        q = Question("s", "Score?", SCALE, ["low", "high"])
        with pytest.raises(ValueError):
            judge.output_values(q, {"score": 5})

    def test_output_fields(self):
        questions = [
            Question("rel", "?"),
            Question("level", "?", CHOICE, ["a", "b"]),
        ]
        assert judge.output_fields(questions, with_probability=True) == {
            "rel": float,
            "level": str,
            "level_p": float,
        }


class TestDefinitionFile:
    def test_round_trip(self, tmp_path):
        definition = JudgeDefinition(
            questions=[
                Question("関連", 'DISH研究に "関連" するか？\n改行'),
                Question("level", "Level?", SCALE, ["低", "中", "高"]),
            ],
            input_columns=["title", "abstract"],
        )
        path = tmp_path / "data.judge.toml"
        judge.save_definition(definition, path)

        loaded = judge.load_definition(path)

        assert loaded == definition

    def test_invalid_file(self, tmp_path):
        path = tmp_path / "q.toml"
        path.write_text('[[questions]]\ncolumn = "a"\nask = ""\n')
        with pytest.raises(ValueError):
            judge.load_definition(path)

    def test_options_string_is_split(self, tmp_path):
        path = tmp_path / "q.toml"
        path.write_text(
            '[[questions]]\ncolumn = "a"\nask = "x"\nanswer = "choice"\n'
            'options = "yes|no|maybe"\n'
        )
        assert judge.load_definition(path).questions[0].options == [
            "yes",
            "no",
            "maybe",
        ]

    def test_record_completed_run_keeps_user_edits(self, tmp_path):
        path = tmp_path / "q.toml"
        path.write_text(
            '# my notes\ninput = ["title"]\n\n[[questions]]\n'
            'column = "a"  # keep me\nask = "x"\n'
        )
        definition = judge.load_definition(path)
        assert definition.last_run_fingerprint is None
        assert not definition.changed_since_last_run()

        judge.record_completed_run(definition, path)

        text = path.read_text()
        assert "# keep me" in text
        reloaded = judge.load_definition(path)
        assert reloaded.last_run_fingerprint == definition.fingerprint()
        assert not reloaded.changed_since_last_run()

        path.write_text(text.replace('ask = "x"', 'ask = "changed"'))
        assert judge.load_definition(path).changed_since_last_run()

        judge.record_completed_run(judge.load_definition(path), path)
        assert path.read_text().count("last_run =") == 1

    def test_default_path_is_next_to_csv(self, tmp_path):
        csv_path = tmp_path / "papers.csv"
        assert judge.default_questions_path(csv_path) == tmp_path / "papers.judge.toml"


class TestWizard:
    def _run(self, answers, reserved=()):
        replies = iter(answers)
        output = []
        questions = judge.run_wizard(
            reserved_columns=reserved,
            ask=lambda _prompt: next(replies),
            say=output.append,
        )
        return questions, output

    def test_builds_all_answer_types(self):
        questions, _ = self._run(
            [
                "Is it relevant?",
                "1",
                "is_relevant",
                "y",
                "Which topic?",
                "2",
                "a|b|c",
                "",
                "y",
                "How strong?",
                "3",
                "low|high",
                "strength",
                "n",
                "",  # confirm
            ]
        )
        assert questions == [
            Question("is_relevant", "Is it relevant?", YES_NO, []),
            Question("which_topic", "Which topic?", CHOICE, ["a", "b", "c"]),
            Question("strength", "How strong?", SCALE, ["low", "high"]),
        ]

    def test_reprompts_on_invalid_input(self):
        questions, output = self._run(
            ["", "Relevant?", "9", "2", "only", "x|x", "a|b", "title", "ok", "", ""],
            reserved=["title"],
        )
        assert questions == [Question("ok", "Relevant?", CHOICE, ["a", "b"])]
        assert any("already used" in line for line in output)
        assert any("unique" in line for line in output)
