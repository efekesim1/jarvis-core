"""Recognising a credential wherever it turns up.

Credentials leak sideways. One lands in a tool result, another in something
the operator typed while pasting a config, a third in a memory written from an agent's
output. Each of those paths grew its own check at one point, which meant three
patterns to keep current and three chances to miss one.

There is one list now, and it lives here — below everything that needs it, so
the tool gateway, the memory layers and the digital twin all refuse the same
things. Adding a pattern in one place fixes every path.
"""

from __future__ import annotations

# The patterns live in `logging`, one layer up, because the audit trail needs
# them and cannot import downwards. Re-exported here so callers in and above
# `security` keep the name they already use.
from ..logging import SECRET_MASK, SECRET_PATTERNS  # noqa: E402

DEFAULT_MASK = SECRET_MASK


def find_secrets(text: str) -> list[str]:
    """Describe every credential-shaped thing in ``text``. Empty means clean."""
    return [description for pattern, description in SECRET_PATTERNS if pattern.search(text)]


def looks_like_credential(text: str) -> bool:
    """Whether ``text`` contains something that should never have been written down."""
    return any(pattern.search(text) for pattern, _ in SECRET_PATTERNS)


def redact_secrets(text: str, *, mask: str = DEFAULT_MASK) -> tuple[str, list[str]]:
    """Remove anything that looks like a credential, and say what was found."""
    found: list[str] = []
    cleaned = text
    for pattern, description in SECRET_PATTERNS:
        cleaned, count = pattern.subn(mask, cleaned)
        if count:
            found.append(description)
    return cleaned, found
