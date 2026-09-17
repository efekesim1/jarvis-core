"""Telling an agent that exists from an agent that works."""

from .audit import AuditReport, Finding, Severity, audit_agent, audit_roster
from .claims import unsupported_claims
from .failures import FailureKind, classify, schema_gaps, summarise
from .matrix import RUNGS, AgentRow, Cell, build, render, totals
from .readiness import SUBSYSTEMS, Readiness, ReadinessReport, Subsystem, assess
from .status import (
    MIN_MEAN_SCORE,
    MIN_RUNS_FOR_PRODUCTION,
    AgentStatus,
    AgentVerification,
    VerificationRun,
)
from .verifier import BORDERLINE_MARGIN, AgentVerifier, VerificationBudget

__all__ = [
    "MIN_MEAN_SCORE",
    "AuditReport",
    "BORDERLINE_MARGIN",
    "FailureKind",
    "AgentRow",
    "Cell",
    "Readiness",
    "ReadinessReport",
    "SUBSYSTEMS",
    "Subsystem",
    "assess",
    "RUNGS",
    "build",
    "render",
    "totals",
    "classify",
    "schema_gaps",
    "summarise",
    "MIN_RUNS_FOR_PRODUCTION",
    "AgentStatus",
    "AgentVerification",
    "Finding",
    "Severity",
    "AgentVerifier",
    "VerificationBudget",
    "VerificationRun",
    "audit_agent",
    "audit_roster",
    "unsupported_claims",
]
