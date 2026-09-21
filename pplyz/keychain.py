"""Resolve API keys stored in the macOS Keychain instead of plain-text config.

A config value (or environment variable) of the form ``keychain:<account>`` is
looked up as a generic password with service ``pplyz``; ``keychain:<service>/<account>``
selects an arbitrary service. References are resolved lazily, only for the
provider actually in use, so unrelated keys never trigger a Keychain prompt.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Iterable

KEYCHAIN_PREFIX = "keychain:"
KEYCHAIN_SERVICE = "pplyz"
SECURITY_BIN = "/usr/bin/security"
# `security find-generic-password` exit status when the item does not exist.
ITEM_NOT_FOUND_STATUS = 44


class SecretResolutionError(ValueError):
    """Raised when a secret reference cannot be resolved."""


def parse_keychain_ref(value: str | None) -> tuple[str, str] | None:
    """Return (service, account) for a ``keychain:`` reference, else None."""
    if not value or not value.startswith(KEYCHAIN_PREFIX):
        return None
    target = value[len(KEYCHAIN_PREFIX) :].strip()
    if not target:
        raise SecretResolutionError(
            f"Empty keychain reference '{value}'. Use 'keychain:<name>' "
            "(e.g. keychain:openrouter)."
        )
    if "/" in target:
        service, account = target.split("/", 1)
        if not service or not account:
            raise SecretResolutionError(
                f"Invalid keychain reference '{value}'. "
                "Use 'keychain:<name>' or 'keychain:<service>/<account>'."
            )
        return service, account
    return KEYCHAIN_SERVICE, target


def _require_macos() -> None:
    if sys.platform != "darwin":
        raise SecretResolutionError(
            "keychain: references are only supported on macOS. "
            "Set the API key via an environment variable instead."
        )


def read_keychain(service: str, account: str) -> str:
    """Read a generic password from the macOS Keychain.

    Raises:
        SecretResolutionError: If the item is missing or cannot be read.
    """
    _require_macos()
    try:
        result = subprocess.run(
            [SECURITY_BIN, "find-generic-password", "-s", service, "-a", account, "-w"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise SecretResolutionError(f"Unable to run 'security': {exc}") from exc

    if result.returncode == ITEM_NOT_FOUND_STATUS:
        hint = (
            f"pplyz auth set {account}"
            if service == KEYCHAIN_SERVICE
            else f"security add-generic-password -s {service} -a {account} -w"
        )
        raise SecretResolutionError(
            f"Keychain item not found (service={service}, account={account}). "
            f"Store it with: {hint}"
        )
    if result.returncode != 0:
        # stderr never contains the secret itself.
        raise SecretResolutionError(
            f"Failed to read Keychain item (service={service}, account={account}): "
            f"{result.stderr.strip() or f'exit status {result.returncode}'}"
        )

    secret = result.stdout.rstrip("\n")
    if not secret:
        raise SecretResolutionError(
            f"Keychain item is empty (service={service}, account={account})."
        )
    return secret


def resolve_env_secrets(env_vars: Iterable[str]) -> None:
    """Replace ``keychain:`` references in the given env vars with real secrets."""
    for name in env_vars:
        ref = parse_keychain_ref(os.environ.get(name))
        if ref is None:
            continue
        os.environ[name] = read_keychain(*ref)


def store_keychain(account: str, service: str = KEYCHAIN_SERVICE) -> None:
    """Store a secret in the Keychain, letting ``security`` prompt for it.

    Passing ``-w`` last makes ``security`` read the secret from the TTY, so it
    never appears in process arguments or shell history.
    """
    _require_macos()
    result = subprocess.run(
        [
            SECURITY_BIN,
            "add-generic-password",
            "-U",
            "-s",
            service,
            "-a",
            account,
            "-w",
        ],
        check=False,
    )
    if result.returncode != 0:
        raise SecretResolutionError(
            f"Failed to store Keychain item (service={service}, account={account})."
        )
