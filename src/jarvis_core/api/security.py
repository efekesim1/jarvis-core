"""Who may talk to the API.

The API exposes everything the runtime knows: what it has spent, what is
queued, what is waiting for a human. An unauthenticated port doing that on a
machine with any network at all is a data leak with a REST interface.

So there are two rules and neither is configurable down.

**No token, no server.** ``JARVIS_API_TOKEN`` has to be set, and it has to be
long enough to be worth having. The alternative — binding to localhost and
calling that security — is wrong on any machine with a container, a VPN, or a
second user, and it fails silently on all of them.

**No token in the repository.** It comes from the environment, like every
other credential here, and the error when it is missing says how to make one
rather than suggesting a default.
"""

from __future__ import annotations

import hmac
import os
import secrets

from ..logging import AuditEvent, AuditLog, get_logger

log = get_logger("api.security")

# Short enough to type, long enough that guessing it is not a strategy.
MIN_TOKEN_LENGTH = 32


class MissingToken(RuntimeError):
    """The API was asked to start with no way to tell callers apart."""


def required_token() -> str:
    """The token the API will accept, or an error explaining how to make one."""
    token = (os.environ.get("JARVIS_API_TOKEN") or "").strip()
    if not token:
        raise MissingToken(
            "JARVIS_API_TOKEN is not set, so the API would accept anyone. It exposes "
            "what has been spent, what is queued and what is waiting for "
            "approval.\n\n"
            'Make one:  python -c "import secrets; print(secrets.token_urlsafe(32))"\n'
            "Then:      export JARVIS_API_TOKEN=<that value>\n\n"
            "Binding to localhost instead is not a substitute: it is wrong on any "
            "machine with a container, a VPN or a second user, and it fails silently."
        )
    if len(token) < MIN_TOKEN_LENGTH:
        raise MissingToken(
            f"JARVIS_API_TOKEN is {len(token)} characters; {MIN_TOKEN_LENGTH} is the "
            "minimum. A short token is a token someone guesses."
        )
    return token


def new_token() -> str:
    """A token worth using, for the setup message to suggest."""
    return secrets.token_urlsafe(32)


def matches(presented: str, expected: str, *, audit: AuditLog | None = None) -> bool:
    """Whether a presented token is the right one.

    Constant-time, because a timing side channel on a bearer token is a real
    way to recover it one character at a time, and `==` on strings in Python
    short-circuits on the first difference.
    """
    ok = hmac.compare_digest(presented.encode(), expected.encode())
    if not ok and audit is not None:
        # The token itself is never logged — a rejected credential in an audit
        # trail is still a credential in a file.
        audit.record(
            AuditEvent(
                action="security.api_rejected",
                actor="api",
                subject="bearer token",
                reason="the presented token did not match",
            )
        )
    return ok
