"""Tests for pplyz.jev_client."""

import numpy as np
import pytest

from pplyz import jev_client
from pplyz.jev_client import JevClient, JevRequestError
from pplyz.judge import CHOICE, Question

QUESTIONS = [
    Question("関連", "Relevant?"),
    Question("level", "Level?", CHOICE, ["high", "low"]),
]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(jev_client.time, "sleep", lambda _s: None)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    return JevClient(QUESTIONS)


class TestInit:
    def test_uses_openrouter_key(self, client):
        assert client.api_key == "or-key"
        assert client.endpoint == "https://openrouter.ai/api/alpha/decisions"

    def test_missing_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="pplyz auth set openrouter"):
            JevClient(QUESTIONS)

    def test_resolves_keychain_reference(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "keychain:openrouter")
        monkeypatch.setattr(
            "pplyz.keychain.read_keychain", lambda service, account: "from-keychain"
        )
        assert JevClient(QUESTIONS).api_key == "from-keychain"


class TestRequest:
    def test_build_request(self, client):
        body = client.build_request(
            {"title": "A", "year": np.int64(2024), "note": float("nan")}
        )
        assert body["model"] == client.model_name
        assert body["state"] == {"title": "A", "year": 2024, "note": None}
        # Positional ids keep non-ASCII column names out of the API.
        assert list(body["questions"]) == ["q1", "q2"]
        assert body["questions"]["q2"]["type"] == "choice"

    def test_generate_structured_output_maps_answers(self, client, monkeypatch):
        monkeypatch.setattr(
            client,
            "_post",
            lambda body: {
                "answers": {
                    "q1": {"type": "noul", "noul": 0.9},
                    "q2": {"type": "choice", "choice": "low", "confidence": 0.7},
                }
            },
        )
        assert client.generate_structured_output("", {"title": "A"}) == {
            "関連": True,
            "level": "low",
        }

    def test_missing_answer(self, client, monkeypatch):
        monkeypatch.setattr(
            client, "_post", lambda body: {"answers": {"q1": {"noul": 0.1}}}
        )
        with pytest.raises(ValueError, match="no answer for 'level'"):
            client.generate_structured_output("", {"title": "A"})


class TestRetry:
    def test_retries_rate_limit(self, client, monkeypatch):
        responses = [JevRequestError(429, "rate"), JevRequestError(529, "busy")]

        def fake_post(body):
            if responses:
                raise responses.pop(0)
            return {"answers": {}}

        monkeypatch.setattr(client, "_post", fake_post)
        assert client._post_with_retry({}) == {"answers": {}}

    def test_does_not_retry_auth_error(self, client, monkeypatch):
        calls = []

        def fake_post(body):
            calls.append(body)
            raise JevRequestError(401, "bad key")

        monkeypatch.setattr(client, "_post", fake_post)
        with pytest.raises(JevRequestError):
            client._post_with_retry({})
        assert len(calls) == 1
