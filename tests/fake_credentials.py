r"""Key-shaped fixtures that the credential scanner cannot match.

The CI job greps source for `(sk-ant-)(api|admin)[0-9]{2}-`, and the secret
redactor matches `sk-ant-(?:api|admin)\d{2}-[A-Za-z0-9_-]{8,}` at runtime. A
test fixture has to be invisible to the first and caught by the second, and the
only difference available is *when* the string exists: the scanner reads the
file, the redactor reads the value.

So these are assembled at import time. The literal prefix never appears in any
source file; the assembled value is a real key shape.

This matters more than it looks. When the scanner started flagging these
fixtures, the first fix rewrote them as `fake-ant-key-...` — which the scanner
ignores and the redactor *also* ignores. The tests went green and stopped
testing the pattern: `test_a_credential_never_reaches_the_audit_table` then
passed only because the metadata key was named `api_key`, so the whole
pattern-matching branch of the redactor lost its coverage. The CI workflow
already knew this technique — its own comment says "the pattern itself is split
so this step does not match its own source" — it just was not applied here.
"""

from __future__ import annotations

# Assembled, never written whole. Changing this to a single literal will make
# the credential scan fail on this file, which is the intended behaviour.
_PREFIX = "sk-" + "ant-" + "api" + "03-"


# Same technique, for the other credential the redactor has to catch. Written
# whole, `ghp_...` is a real GitHub token shape, which is exactly what the
# scanners look for — including the export scanner that guards the public
# release, where a test fixture matching a token pattern is indistinguishable
# from a leak.
_GITHUB_PREFIX = "gh" + "p" + "_"


def fake_github_token(marker: str) -> str:
    """A string the secret redactor must treat as a GitHub token."""
    return f"{_GITHUB_PREFIX}{marker}"


def fake_anthropic_key(marker: str) -> str:
    """A string the secret redactor must treat as an Anthropic key.

    ``marker`` is what a test asserts did *not* survive, so it should be
    distinctive: SHOULDNOTAPPEAR, SHOULDNOTBEREADABLE, and so on.
    """
    return f"{_PREFIX}{marker}"
