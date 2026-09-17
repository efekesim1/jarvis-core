"""One error taxonomy for everything that can go wrong reaching a tool.

External failures arrive in a hundred shapes: a socket dies, a server returns
nonsense, a token expired, a rate limit bites. Collapsing all of that into one
generic exception loses exactly the information an agent needs to decide what
to do next, so every failure is normalised into one of these.

Each error says whether retrying could plausibly help. Nothing retries a
permission denial, and nothing retries a write that may already have landed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ToolErrorCode(StrEnum):
    """Every way a tool call can fail, named once."""

    TOOL_NOT_FOUND = "tool_not_found"
    SERVER_NOT_FOUND = "server_not_found"
    SERVER_DISABLED = "server_disabled"
    SERVER_UNHEALTHY = "server_unhealthy"
    TOOL_DISABLED = "tool_disabled"
    PERMISSION_DENIED = "permission_denied"
    RISK_DENIED = "risk_denied"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_REJECTED = "approval_rejected"
    APPROVAL_EXPIRED = "approval_expired"
    BUDGET_EXCEEDED = "budget_exceeded"
    RATE_LIMITED = "rate_limited"
    INVALID_ARGUMENTS = "invalid_arguments"
    INVALID_RESULT = "invalid_result"
    RESULT_TOO_LARGE = "result_too_large"
    TIMEOUT = "timeout"
    AUTH_FAILURE = "auth_failure"
    MCP_PROTOCOL_ERROR = "mcp_protocol_error"
    EXTERNAL_ERROR = "external_error"
    SECURITY_BLOCKED = "security_blocked"
    DUPLICATE_REQUEST = "duplicate_request"
    NOT_IMPLEMENTED = "not_implemented"


# Failures where the same call, tried again, could plausibly succeed. Every
# other code is final: retrying a permission denial only wastes a turn.
RETRYABLE: frozenset[ToolErrorCode] = frozenset(
    {
        ToolErrorCode.TIMEOUT,
        ToolErrorCode.SERVER_UNHEALTHY,
        ToolErrorCode.EXTERNAL_ERROR,
        ToolErrorCode.MCP_PROTOCOL_ERROR,
        ToolErrorCode.RATE_LIMITED,
    }
)


class ToolError(Exception):
    """A normalised tool failure."""

    def __init__(
        self,
        code: ToolErrorCode,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.retryable = code in RETRYABLE if retryable is None else retryable

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ToolError({self.code.value}: {self.message})"


@dataclass(slots=True)
class ToolFailure:
    """A failure as data, for an agent to read rather than catch.

    The gateway returns this instead of raising, because an agent deciding
    what to do about a rate limit should not need an exception handler.
    """

    code: ToolErrorCode
    message: str
    retryable: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_error(cls, error: ToolError) -> ToolFailure:
        return cls(
            code=error.code,
            message=error.message,
            retryable=error.retryable,
            details=error.details,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }
