"""Tests for pplyz.keychain."""

import subprocess

import pytest

from pplyz import keychain
from pplyz.keychain import SecretResolutionError


@pytest.fixture
def on_macos(monkeypatch):
    monkeypatch.setattr(keychain.sys, "platform", "darwin")


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


class TestParseKeychainRef:
    def test_plain_value_is_not_a_reference(self):
        assert keychain.parse_keychain_ref("sk-123") is None
        assert keychain.parse_keychain_ref(None) is None

    def test_account_only_uses_pplyz_service(self):
        assert keychain.parse_keychain_ref("keychain:openrouter") == (
            "pplyz",
            "openrouter",
        )

    def test_service_and_account(self):
        assert keychain.parse_keychain_ref("keychain:my-svc/me") == ("my-svc", "me")

    @pytest.mark.parametrize("value", ["keychain:", "keychain:/x", "keychain:x/"])
    def test_invalid_reference(self, value):
        with pytest.raises(SecretResolutionError):
            keychain.parse_keychain_ref(value)


class TestReadKeychain:
    def test_reads_secret(self, on_macos, monkeypatch):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _completed(stdout="secret-value\n")

        monkeypatch.setattr(keychain.subprocess, "run", fake_run)

        assert keychain.read_keychain("pplyz", "openrouter") == "secret-value"
        assert calls[0][1:] == [
            "find-generic-password",
            "-s",
            "pplyz",
            "-a",
            "openrouter",
            "-w",
        ]

    def test_missing_item_suggests_auth_command(self, on_macos, monkeypatch):
        monkeypatch.setattr(
            keychain.subprocess,
            "run",
            lambda *a, **k: _completed(returncode=keychain.ITEM_NOT_FOUND_STATUS),
        )
        with pytest.raises(SecretResolutionError, match="pplyz auth set openrouter"):
            keychain.read_keychain("pplyz", "openrouter")

    def test_other_failure(self, on_macos, monkeypatch):
        monkeypatch.setattr(
            keychain.subprocess,
            "run",
            lambda *a, **k: _completed(returncode=51, stderr="denied"),
        )
        with pytest.raises(SecretResolutionError, match="denied"):
            keychain.read_keychain("pplyz", "openrouter")

    def test_non_macos_is_rejected(self, monkeypatch):
        monkeypatch.setattr(keychain.sys, "platform", "linux")
        with pytest.raises(SecretResolutionError, match="only supported on macOS"):
            keychain.read_keychain("pplyz", "openrouter")


class TestResolveEnvSecrets:
    def test_replaces_only_references(self, on_macos, monkeypatch):
        monkeypatch.setenv("REF_KEY", "keychain:openrouter")
        monkeypatch.setenv("PLAIN_KEY", "plain")
        monkeypatch.delenv("UNSET_KEY", raising=False)
        monkeypatch.setattr(
            keychain.subprocess, "run", lambda *a, **k: _completed(stdout="resolved\n")
        )

        keychain.resolve_env_secrets(["REF_KEY", "PLAIN_KEY", "UNSET_KEY"])

        assert keychain.os.environ["REF_KEY"] == "resolved"
        assert keychain.os.environ["PLAIN_KEY"] == "plain"
        assert "UNSET_KEY" not in keychain.os.environ


class TestStoreKeychain:
    def test_secret_is_prompted_not_passed_as_argument(self, on_macos, monkeypatch):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _completed()

        monkeypatch.setattr(keychain.subprocess, "run", fake_run)

        keychain.store_keychain("openrouter")

        # `-w` must be last so `security` reads the secret from the TTY.
        assert calls[0][-1] == "-w"
        assert "-U" in calls[0]
