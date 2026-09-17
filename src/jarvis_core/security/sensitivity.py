"""How sensitive a piece of remembered information is.

Permissions say what an agent may *do*. This says what an agent may *see*, and
more importantly what may leave the machine. They are different questions: an
agent can legitimately hold ``NETWORK`` and still have no business putting
the operator's medical history in a web search.

The ladder is ordered. Clearance at one level grants everything below it, so a
single comparison answers "may this reach that": ``item.sensitivity <=
audience.clearance``.

``RESTRICTED`` is the level nothing automatic reads. It exists for the
categories where being wrong is not a bug but a harm — health, beliefs,
politics, sexuality, ethnicity, anything a person is entitled to keep. The twin
will not infer anything at this level, ever, and will not render it into a
prompt.
"""

from __future__ import annotations

from enum import IntEnum


class Sensitivity(IntEnum):
    """Ordered from freely shareable to never leaves the machine."""

    PUBLIC = 0  # already public, or the operator would say it to a stranger
    INTERNAL = 1  # system state, telemetry, how the OS itself works
    BUSINESS = 2  # clients, revenue, contracts, plans
    PERSONAL = 3  # the operator's habits, preferences, schedule, relationships
    PRIVATE = 4  # things the operator would not want repeated
    RESTRICTED = 5  # protected categories; never inferred, never rendered

    @property
    def label(self) -> str:
        return self.name.lower()


# What each surface is cleared to see, unless something narrower is set.
#
# The defaults are deliberately mean. An external tool call is the way data
# leaves, so it starts at PUBLIC and has to be granted more explicitly.
DEFAULT_CLEARANCE: dict[str, Sensitivity] = {
    "operator": Sensitivity.RESTRICTED,
    "jarvis": Sensitivity.PRIVATE,
    "agent": Sensitivity.BUSINESS,
    "sandbox": Sensitivity.INTERNAL,
    "external_tool": Sensitivity.PUBLIC,
}

# The one level that is not merely "high". Nothing derives it, nothing renders
# it, and the guard that enforces that is tested by name.
NEVER_INFERRED = Sensitivity.RESTRICTED


def parse_sensitivity(
    raw: str | int | Sensitivity | None, default: Sensitivity = Sensitivity.PERSONAL
) -> Sensitivity:
    """Read a sensitivity from configuration or storage.

    Unrecognised input becomes the default rather than the lowest level: a
    typo in a label must never end up meaning "safe to publish".
    """
    if raw is None:
        return default
    if isinstance(raw, Sensitivity):
        return raw
    if isinstance(raw, int):
        try:
            return Sensitivity(raw)
        except ValueError:
            return default
    try:
        return Sensitivity[str(raw).strip().upper()]
    except KeyError:
        return default


def may_release(item: Sensitivity, audience: Sensitivity) -> bool:
    """Whether something at ``item`` may be shown to someone cleared to ``audience``."""
    return item <= audience
