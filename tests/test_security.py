"""Permissions, risk and the approval gate."""

from __future__ import annotations

import pytest

from jarvis_core.config import get_settings
from jarvis_core.security.approval import (
    ApprovalDecision,
    ApprovalRequest,
    AutoApproveGate,
    DenyAllGate,
    QueueApprovalGate,
)
from jarvis_core.security.guardian import SecurityGuardian, SecurityViolation
from jarvis_core.security.permissions import Permission, RiskLevel, parse_permissions


def test_permissions_parse_from_names():
    parsed = parse_permissions(["read", "NETWORK"])
    assert parsed == frozenset({Permission.READ, Permission.NETWORK})


def test_an_unknown_permission_fails_loudly():
    with pytest.raises(ValueError, match="unknown permission"):
        parse_permissions(["read", "teleport"])


def test_risk_levels_order():
    assert RiskLevel.LOW < RiskLevel.MEDIUM < RiskLevel.HIGH


def test_an_agent_without_the_permission_is_refused(guardian):
    verdict = guardian.check(
        action="send money",
        granted=frozenset({Permission.READ}),
        required=frozenset({Permission.FINANCIAL}),
        agent="rogue_agent",
    )
    assert not verdict.allowed
    assert Permission.FINANCIAL in verdict.missing_permissions
    with pytest.raises(SecurityViolation):
        verdict.raise_if_denied()


def test_a_granted_low_risk_action_needs_no_approval(guardian):
    verdict = guardian.check(
        action="search the web",
        granted=frozenset({Permission.READ, Permission.NETWORK}),
        required=frozenset({Permission.READ, Permission.NETWORK}),
        risk=RiskLevel.LOW,
        agent="web_researcher",
    )
    assert verdict.allowed
    assert not verdict.required_approval


def test_money_deploy_and_admin_always_need_approval(guardian):
    for permission in (Permission.FINANCIAL, Permission.DEPLOY, Permission.ADMIN):
        assert guardian.needs_approval(RiskLevel.LOW, frozenset({permission}))


def test_the_sandbox_refuses_dangerous_permissions_even_when_granted(guardian, settings):
    assert settings.environment == "sandbox"
    verdict = guardian.check(
        action="deploy to production",
        granted=frozenset({Permission.DEPLOY}),
        required=frozenset({Permission.DEPLOY}),
        agent="deployment_agent",
    )
    assert not verdict.allowed
    assert "sandbox" in verdict.reason


async def test_the_default_gate_denies_when_nobody_is_watching(audit):
    guardian = SecurityGuardian(settings=get_settings(), audit=audit, approval_gate=DenyAllGate())
    verdict = await guardian.authorize(
        action="modify a workflow",
        granted=frozenset({Permission.WRITE}),
        required=frozenset({Permission.WRITE}),
        risk=RiskLevel.MEDIUM,
        agent="automation_agent",
    )
    assert not verdict.allowed
    assert "denied" in verdict.reason


async def test_an_approved_action_proceeds(audit):
    guardian = SecurityGuardian(
        settings=get_settings(), audit=audit, approval_gate=AutoApproveGate()
    )
    verdict = await guardian.authorize(
        action="modify a workflow",
        granted=frozenset({Permission.WRITE}),
        required=frozenset({Permission.WRITE}),
        risk=RiskLevel.MEDIUM,
        agent="automation_agent",
    )
    assert verdict.allowed
    assert verdict.required_approval


async def test_the_queue_gate_holds_a_request_until_it_is_answered():
    import asyncio

    gate = QueueApprovalGate(timeout_s=5)
    request = ApprovalRequest(action="wire money", reason="pay a supplier", risk="high")

    task = asyncio.create_task(gate.request(request))
    await asyncio.sleep(0)
    assert [r.request_id for r in gate.pending] == [request.request_id]

    gate.resolve(request.request_id, ApprovalDecision.APPROVED)
    assert await task is ApprovalDecision.APPROVED


async def test_the_queue_gate_times_out_rather_than_hanging():
    gate = QueueApprovalGate(timeout_s=0.05)
    decision = await gate.request(ApprovalRequest(action="x", reason="y", risk="high"))
    assert decision is ApprovalDecision.TIMED_OUT


def test_every_check_lands_in_the_audit_log(guardian, audit):
    guardian.check(action="read a file", granted=frozenset({Permission.READ}), agent="a")
    assert any(e.action == "guardian.check" for e in audit.events)


# -- a credential in a free-text field ---------------------------------------
#
# A real hole, found by writing a test that used a key-shaped fixture in a field
# nobody had named `api_key`. Redaction was *name-based only* — `redact()` looked
# at dict keys and never at values — and it was applied to `metadata` alone. So
# a key in `reason` went into the audit trail intact, and `reason` is written
# from `str(exc)` in a dozen places.
#
# Worse, the two sinks disagreed: `to_json()` protected the JSON-lines file
# while the *raw* event went on to the database sink, which writes fields as
# columns. The file on disk was clean and the table beside it was not.
#
# The layering is why it was missed: `logging` sits above `security`, so the
# audit trail could not import the secret patterns. They now live in `logging`
# and `security.secrets` imports them upwards, so there is still one copy.


def _fake_key(marker: str) -> str:
    # Assembled so the CI credential scan does not match this file.
    return "sk-" + "ant-" + "api" + "03-" + marker


@pytest.mark.parametrize("field_name", ["reason", "subject", "result"])
def test_a_credential_in_any_free_text_field_is_redacted(field_name):
    from jarvis_core.logging import AuditEvent

    event = AuditEvent(
        action="model.call", **{field_name: f"failed on {_fake_key('LEAKMEFIELD01')}"}
    )
    assert "LEAKMEFIELD01" not in event.to_json()
    assert "LEAKMEFIELD01" not in getattr(event.scrubbed(), field_name)


def test_the_scrubbed_event_is_what_reaches_a_sink():
    """The file and the table must not disagree about what was stored."""

    class Recorder:
        def __init__(self):
            self.events = []

        def record(self, event):
            self.events.append(event)

    from jarvis_core.logging import AuditEvent, AuditLog

    sink = Recorder()
    log = AuditLog(sink=sink)
    log.record(
        AuditEvent(action="model.call_failed", reason=f"key {_fake_key('LEAKMESINK01')} rejected")
    )
    assert sink.events
    assert "LEAKMESINK01" not in (sink.events[0].reason or "")


def test_the_in_memory_tail_is_redacted_too():
    """`jarvis status` and the dashboard read it."""
    import json

    from jarvis_core.logging import AuditEvent, AuditLog

    log = AuditLog()
    log.record(AuditEvent(action="model.call", reason=f"used {_fake_key('LEAKMETAIL01')}"))
    assert "LEAKMETAIL01" not in json.dumps(log.tail(1), default=str)


def test_redaction_still_works_by_field_name_for_values_no_pattern_matches():
    """The name-based half must not be lost to the value-based half."""
    from jarvis_core.logging import AuditEvent

    event = AuditEvent(action="x", metadata={"api_key": "short-and-unpatterned"})
    assert "short-and-unpatterned" not in event.to_json()


@pytest.mark.parametrize(
    "secret",
    [
        "nvapi-" + "A" * 24,
        "xai-" + "B" * 24,
        "AIza" + "C" * 32,
        "sk-proj-" + "D" * 24,
    ],
)
def test_the_newer_vendors_key_shapes_are_recognised(secret):
    """Four providers were added to the system; their key shapes were not."""
    from jarvis_core.logging import scrub

    assert secret not in scrub(f"config: {secret}")
