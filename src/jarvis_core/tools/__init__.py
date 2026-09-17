"""The tool fabric: one gateway between every agent and the outside world.

Agents call ``ToolGateway.call`` and nothing else. Permissions, risk, budgets,
rate limits, approvals, validation and audit all happen inside it, in a fixed
order, with no way around.
"""

from .approval import ApprovalEngine, ApprovalStatus, ToolApproval
from .gateway import GatewayStats, ToolGateway
from .policy import (
    AgentToolPolicy,
    CapabilityCoverage,
    PolicyEngine,
    PolicyVerdict,
    capability_coverage,
    default_policies,
    policies_from_registry,
)
from .ratelimit import LimitVerdict, RateLimiter
from .registry import ToolRegistry
from .setup import build_tool_fabric
from .validation import (
    ValidationReport,
    redact_secrets,
    validate_arguments,
    validate_result,
    wrap_untrusted,
)

__all__ = [
    "AgentToolPolicy",
    "CapabilityCoverage",
    "ApprovalEngine",
    "ApprovalStatus",
    "GatewayStats",
    "LimitVerdict",
    "PolicyEngine",
    "PolicyVerdict",
    "RateLimiter",
    "ToolApproval",
    "ToolGateway",
    "ToolRegistry",
    "ValidationReport",
    "build_tool_fabric",
    "capability_coverage",
    "default_policies",
    "policies_from_registry",
    "redact_secrets",
    "validate_arguments",
    "validate_result",
    "wrap_untrusted",
]
