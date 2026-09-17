"""The HTTP interface.

Every endpoint here is a view onto state the runtime already holds, behind a
bearer token, plus two things that change something: a decision on a pending
approval, and — only when it has been switched on explicitly — starting a run.

Three decisions worth stating, because each is a deliberate limit rather than
an omission:

* **Reads by default.** A GET cannot spend money, call a tool or write to
  storage. Anything that can is named below and counted on one hand.
* **No generic endpoint.** There is no path that takes a command, a shell
  string, a tool name, a model override or a budget. `POST /runs` takes one
  field, and a test asserts that field set by name. An HTTP surface that
  accepts a tool name is a remote code execution feature with a schema.
* **Running costs money, so it is off until it is turned on.** A browser
  click is one keystroke away from an idle cursor, and a public core that
  bills the reader by default would be a bad default. `JARVIS_API_ALLOW_RUNS`
  turns it on; without it the endpoint exists, answers 403, and says how.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from ..core.runtime import JarvisRuntime, build_runtime
from ..logging import get_logger
from .security import matches, required_token

log = get_logger("api.server")

bearer = HTTPBearer(auto_error=False)

# The environment variable that admits paid work over HTTP.
ALLOW_RUNS = "JARVIS_API_ALLOW_RUNS"


class StartRun(BaseModel):
    """The only body that starts work, and the whole of it.

    One field. No command, no shell, no tool name, no model, no budget, no
    agent list — the plan comes from the planner and the ceiling from the
    budget engine, both on this side of the wire.
    """

    model_config = {"extra": "forbid"}

    request: str = Field(min_length=1, max_length=4000)


class Decision(BaseModel):
    """A human's answer to one approval request."""

    model_config = {"extra": "forbid"}

    approve: bool
    note: str = Field(default="", max_length=500)


def runs_allowed() -> bool:
    return (os.environ.get(ALLOW_RUNS) or "").strip().lower() in {"1", "true", "yes", "on"}


def create_app(runtime: JarvisRuntime | None = None, *, token: str | None = None) -> FastAPI:
    """Build the API.

    ``token`` is read from the environment when not given, and the app refuses
    to be built without one — see :mod:`jarvis_core.api.security` for why
    binding to localhost is not an alternative.
    """
    expected = token or required_token()
    system = runtime or build_runtime()

    app = FastAPI(
        title="jarvis-core",
        summary="A multi-agent runtime with permissions, budgets and human approvals.",
        version="0.1.0",
    )

    async def authorised(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ) -> None:
        if credentials is None or not matches(
            credentials.credentials, expected, audit=system.audit
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="a valid bearer token is required",
                headers={"WWW-Authenticate": "Bearer"},
            )

    guarded: list[Any] = [Depends(authorised)]

    # -- liveness ---------------------------------------------------------

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, Any]:
        """Unauthenticated on purpose, and it says nothing.

        A load balancer needs to know the process is up without holding a
        credential. The answer carries no state, no versions of anything
        installed, and no counts.
        """
        return {"ok": True, "service": "jarvis-core"}

    # -- reads ------------------------------------------------------------

    @app.get("/status", tags=["system"], dependencies=guarded)
    async def get_status() -> dict[str, Any]:
        return system.status()

    @app.get("/agents", tags=["agents"], dependencies=guarded)
    async def list_agents() -> dict[str, Any]:
        return {
            "agents": [
                {
                    "id": spec.id,
                    "name": spec.name,
                    "role": spec.role,
                    "swarm": spec.swarm,
                    "allowed_models": list(spec.allowed_models),
                    "tool_capabilities": list(spec.tool_capabilities),
                    "permissions": [p.value for p in spec.permissions],
                    "max_attempts": spec.retry_policy.max_attempts,
                    "max_tool_risk": spec.max_tool_risk,
                }
                for spec in system.registry.all()
            ],
            "swarms": system.registry.swarms(),
        }

    @app.get("/agents/{agent_id}", tags=["agents"], dependencies=guarded)
    async def get_agent(agent_id: str) -> dict[str, Any]:
        spec = system.registry.get(agent_id)
        if spec is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"no agent {agent_id!r}")
        return spec.as_dict()

    @app.get("/tasks", tags=["tasks"], dependencies=guarded)
    async def list_tasks() -> dict[str, Any]:
        return system.tasks.snapshot()

    @app.get("/cost", tags=["system"], dependencies=guarded)
    async def get_cost() -> dict[str, Any]:
        return system.cost.summary()

    @app.get("/memory", tags=["memory"], dependencies=guarded)
    async def search_memory(q: str = "", limit: int = 10) -> dict[str, Any]:
        limit = max(1, min(limit, 50))
        items = system.memory.recall(q) if q else []
        return {"query": q, "items": [item.as_dict() for item in items[:limit]]}

    @app.get("/tools", tags=["tools"], dependencies=guarded)
    async def list_tools() -> dict[str, Any]:
        """What the gateway publishes, and what the policy says about it.

        Listing a tool is not granting it: the capabilities an agent may
        actually reach are the tool policy's answer, which is why the policy
        summary travels with the list.
        """
        if system.tools is None:
            return {"attached": False, "tools": []}
        return system.tools.summary()

    # -- the human in the loop --------------------------------------------

    @app.get("/approvals", tags=["approvals"], dependencies=guarded)
    async def list_approvals() -> dict[str, Any]:
        if system.tools is None:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "no tool fabric is attached, so nothing can be waiting for approval",
            )
        return {"pending": [a.as_dict() for a in system.tools.approvals.pending]}

    @app.post("/approvals/{approval_id}", tags=["approvals"], dependencies=guarded)
    async def decide(approval_id: str, body: Decision) -> dict[str, Any]:
        """Grant or refuse one pending approval.

        The decision goes to the shared database, which is what lets an
        approval raised by an unattended run overnight be answered from
        somewhere else in the morning. With no database configured an approval
        is only visible to the process that raised it, and this says so
        instead of pretending to have recorded something.
        """
        if system.storage is None:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "no database is configured, so approvals are only visible to the process "
                "that raised them; set a database path to decide them from here",
            )
        verb = "approved" if body.approve else "rejected"
        ok, reason = system.storage.approvals.decide(approval_id, verb, by="api", note=body.note)
        if not ok:
            raise HTTPException(status.HTTP_409_CONFLICT, reason)
        return {"approval_id": approval_id, "decision": verb}

    # -- the one thing that spends -----------------------------------------

    @app.post("/runs", tags=["runs"], dependencies=guarded, status_code=status.HTTP_200_OK)
    async def start_run(body: StartRun) -> dict[str, Any]:
        """Plan and run one request, and return the answer with its cost.

        Off unless `JARVIS_API_ALLOW_RUNS` is set. The ceiling is the budget
        engine's, computed here; nothing about the limits is accepted from the
        caller.
        """
        if not runs_allowed():
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"running over HTTP is off; set {ALLOW_RUNS}=1 to allow it. It spends "
                "real money on every call, so it is not on by default.",
            )
        response = await system.handle(body.request)
        return response.as_dict()

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        """Log the detail, return a generic message.

        A stack trace in a response body tells an unauthenticated prober the
        library versions and the file layout. It goes to the log, where it is
        useful and not public.
        """
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "something went wrong; see the server log"},
        )

    return app
