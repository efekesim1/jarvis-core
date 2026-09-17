"""The HTTP interface, and the four things it will not do.

What is asserted here is the posture rather than the plumbing: that nothing
answers without a token, that the one endpoint which spends money is off until
it is switched on, that no endpoint accepts a command, and that an unhandled
error does not return a stack trace to whoever asked.

Offline, against the in-process provider: no key, no network, nothing billed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from jarvis_core.api.security import (  # noqa: E402
    MIN_TOKEN_LENGTH,
    MissingToken,
    new_token,
    required_token,
)
from jarvis_core.api.server import StartRun, create_app  # noqa: E402
from jarvis_core.core.runtime import JarvisRuntime  # noqa: E402

TOKEN = "t" * MIN_TOKEN_LENGTH


@pytest.fixture
def client(router, audit, settings) -> TestClient:
    runtime = JarvisRuntime(router=router, audit=audit, settings=settings, use_model_planner=False)
    return TestClient(create_app(runtime, token=TOKEN))


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


# -- who may talk to it -----------------------------------------------------


def test_no_token_no_server(monkeypatch):
    """The refusal is the feature: an unauthenticated port is a data leak."""
    monkeypatch.delenv("JARVIS_API_TOKEN", raising=False)
    with pytest.raises(MissingToken) as caught:
        required_token()
    message = str(caught.value)
    assert "JARVIS_API_TOKEN" in message
    assert "localhost" in message, "the error did not explain why localhost is not a substitute"


def test_a_short_token_is_refused(monkeypatch):
    monkeypatch.setenv("JARVIS_API_TOKEN", "short")
    with pytest.raises(MissingToken):
        required_token()
    assert len(new_token()) >= MIN_TOKEN_LENGTH


@pytest.mark.parametrize(
    "path", ["/status", "/agents", "/tasks", "/cost", "/memory", "/tools", "/approvals"]
)
def test_every_view_needs_a_token(client, path):
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get(path, headers=auth()).status_code in (200, 503)


def test_health_is_open_and_says_nothing(client):
    """A load balancer should not need a credential, or learn anything."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body == {"ok": True, "service": "jarvis-core"}


def test_a_rejected_token_is_audited_but_never_recorded(client, audit):
    client.get("/status", headers={"Authorization": "Bearer wrongtokenwrongtokenwrongtoken"})
    events = [e for e in audit.events if e.action == "security.api_rejected"]
    assert events, "a rejected credential was not audited"
    dumped = "\n".join(e.to_json() for e in audit.events)
    assert "wrongtokenwrongtoken" not in dumped, "the presented token reached the audit log"


# -- what it will not accept ------------------------------------------------


def test_there_is_no_generic_execution_endpoint():
    """Rule, asserted by name: the run body is one field and no more.

    A field set that grows a `command`, a `tool`, a `model` or a `budget` is
    how an HTTP surface becomes remote code execution with a schema.
    """
    assert set(StartRun.model_fields) == {"request"}


def test_an_unknown_field_is_refused_rather_than_ignored(client):
    response = client.post(
        "/runs", headers=auth(), json={"request": "do the thing", "tool": "shell"}
    )
    assert response.status_code == 422


def test_running_costs_money_so_it_is_off_until_it_is_switched_on(client, monkeypatch):
    monkeypatch.delenv("JARVIS_API_ALLOW_RUNS", raising=False)
    response = client.post("/runs", headers=auth(), json={"request": "research the market"})
    assert response.status_code == 403
    assert "JARVIS_API_ALLOW_RUNS" in response.json()["detail"]


def test_with_runs_allowed_it_answers_and_reports_the_cost(client, monkeypatch):
    monkeypatch.setenv("JARVIS_API_ALLOW_RUNS", "1")
    response = client.post("/runs", headers=auth(), json={"request": "what is two plus two"})
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert body["cost_usd"] >= 0.0
    assert "plan" in body and "report" in body


# -- what it shows ----------------------------------------------------------


def test_the_roster_is_visible_with_its_permissions(client):
    body = client.get("/agents", headers=auth()).json()
    assert body["agents"], "no agents were listed"
    first = body["agents"][0]
    assert {"id", "name", "swarm", "permissions", "tool_capabilities", "allowed_models"} <= set(
        first
    )


def test_an_unknown_agent_is_a_404(client):
    assert client.get("/agents/nobody", headers=auth()).status_code == 404


def test_approvals_say_plainly_when_there_is_no_fabric(client):
    """Not an empty list: "nothing is attached" and "nothing is pending" are
    different answers, and a caller acting on the first needs to know."""
    response = client.get("/approvals", headers=auth())
    assert response.status_code == 503
    assert "tool fabric" in response.json()["detail"]


def test_a_decision_without_a_database_is_refused_not_pretended(client):
    response = client.post("/approvals/anything", headers=auth(), json={"approve": True})
    assert response.status_code == 409
    assert "database" in response.json()["detail"]


def test_the_openapi_document_has_no_write_beyond_the_two(client):
    """Every path that changes something, counted.

    Two: a decision on an approval, and a run. A third appearing here is a
    review question, not a silent addition.
    """
    paths = client.app.openapi()["paths"]
    writes = {
        path for path, methods in paths.items() if {"post", "put", "patch", "delete"} & set(methods)
    }
    assert writes == {"/runs", "/approvals/{approval_id}"}
