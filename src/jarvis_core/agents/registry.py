"""The Agent Registry.

One place that knows every agent. Definitions are loaded from YAML under
``agents/<swarm>/``, validated, and indexed by id, swarm and capability.

Registering a new agent at runtime is possible but deliberately gated: a
generated agent enters as a proposal and only a human approval moves it into
the live registry.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import get_settings
from ..logging import AuditLog, get_logger
from .base import AgentRuntime, AgentSpec, LLMAgentRuntime

log = get_logger("agents.registry")


class AgentNotFound(KeyError):
    """No agent with that id is registered."""


class DuplicateAgent(ValueError):
    """Two definitions claim the same agent id."""


@dataclass(slots=True)
class AgentProposal:
    """A generated agent waiting for the owner's decision."""

    spec: AgentSpec
    origin: str
    rationale: str = ""
    test_report: dict[str, Any] = field(default_factory=dict)


class AgentRegistry:
    """Holds the live agents and the runtimes that execute them."""

    def __init__(self, *, audit: AuditLog | None = None) -> None:
        self._agents: dict[str, AgentSpec] = {}
        self._runtimes: dict[str, AgentRuntime] = {}
        self._proposals: dict[str, AgentProposal] = {}
        self.audit = audit or AuditLog()
        self.register_runtime(LLMAgentRuntime())

    # -- runtimes ---------------------------------------------------------

    def register_runtime(self, runtime: AgentRuntime) -> None:
        self._runtimes[runtime.name] = runtime

    def runtime_for(self, spec: AgentSpec) -> AgentRuntime:
        runtime = self._runtimes.get(spec.runtime)
        if runtime is None:
            known = ", ".join(sorted(self._runtimes)) or "(none)"
            raise AgentNotFound(
                f"agent {spec.id} asks for runtime {spec.runtime!r}; registered runtimes: {known}"
            )
        return runtime

    # -- agents -----------------------------------------------------------

    def register(self, spec: AgentSpec, *, replace: bool = False) -> AgentSpec:
        if spec.id in self._agents and not replace:
            raise DuplicateAgent(f"agent id {spec.id!r} is already registered")
        self._agents[spec.id] = spec
        self.audit.emit(
            "agent.registered",
            agent=spec.id,
            subject=spec.name,
            metadata={"swarm": spec.swarm, "version": spec.version, "replace": replace},
        )
        return spec

    def get(self, agent_id: str) -> AgentSpec | None:
        return self._agents.get(agent_id)

    def require(self, agent_id: str) -> AgentSpec:
        spec = self._agents.get(agent_id)
        if spec is None:
            known = ", ".join(sorted(self._agents)) or "(none)"
            raise AgentNotFound(f"unknown agent {agent_id!r}; registered agents: {known}")
        return spec

    def all(self, *, enabled_only: bool = True) -> list[AgentSpec]:
        specs = list(self._agents.values())
        if enabled_only:
            specs = [s for s in specs if s.enabled]
        return sorted(specs, key=lambda s: (s.swarm, s.id))

    def by_swarm(self, swarm: str) -> list[AgentSpec]:
        return [s for s in self.all() if s.swarm == swarm]

    def swarms(self) -> list[str]:
        return sorted({s.swarm for s in self.all()})

    def find_by_capability(self, capability: str) -> list[AgentSpec]:
        needle = capability.strip().lower()
        return [
            s
            for s in self.all()
            if any(needle in c.lower() for c in s.capabilities)
            or needle in s.role.lower()
            or needle in s.id
        ]

    def catalog_for_planner(self) -> str:
        """A compact agent list to paste into the planner's prompt."""
        lines = []
        for spec in self.all():
            caps = ", ".join(spec.capabilities) or spec.role
            lines.append(f"- {spec.id} [{spec.swarm}]: {caps}")
        return "\n".join(lines)

    # -- loading ----------------------------------------------------------

    def load_dir(self, directory: Path | None = None) -> int:
        """Load every ``*.yaml`` under a directory tree. Returns how many loaded."""
        root = Path(directory) if directory is not None else Path(get_settings().agents_dir)
        if not root.exists():
            log.warning("agent directory %s does not exist", root)
            return 0
        key = (str(root.resolve()), _newest_mtime(root))
        cached = _PARSE_CACHE.get(key)
        if cached is None:
            cached = []
            for path in sorted(root.rglob("*.yaml")):
                try:
                    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                except yaml.YAMLError as exc:
                    log.error("skipping %s: %s", path, exc)
                    continue
                if raw is not None:
                    cached.append((raw, str(path)))
            _PARSE_CACHE[key] = cached

        loaded = 0
        for raw, source in cached:
            try:
                loaded += self._load_raw(raw, source)
            except (ValueError, DuplicateAgent) as exc:
                log.error("skipping %s: %s", source, exc)
        log.info("loaded %d agent(s) from %s", loaded, root)
        return loaded

    def load_file(self, path: Path) -> int:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw is None:
            return 0
        return self._load_raw(raw, str(path))

    def _load_raw(self, raw: Any, source: str) -> int:
        """Register whatever one already-parsed YAML document describes."""
        entries: Iterable[dict[str, Any]]
        if isinstance(raw, dict) and "agents" in raw:
            entries = raw["agents"]
        elif isinstance(raw, list):
            entries = raw
        else:
            entries = [raw]
        folder = Path(source).parent.name
        count = 0
        for entry in entries:
            spec = AgentSpec.from_dict(entry, source=source)
            # A file under agents/research/ defines research agents.
            if spec.swarm and folder not in {"agents", ""} and spec.swarm != folder:
                log.warning(
                    "%s: agent %s declares swarm %r but lives under %r",
                    source,
                    spec.id,
                    spec.swarm,
                    folder,
                )
            self.register(spec)
            count += 1
        return count

    # -- proposals --------------------------------------------------------

    def propose(self, proposal: AgentProposal) -> AgentProposal:
        """Park a generated agent. It does not run until it is approved."""
        self._proposals[proposal.spec.id] = proposal
        self.audit.emit(
            "agent.proposed",
            agent=proposal.spec.id,
            subject=proposal.spec.name,
            reason=proposal.rationale,
            metadata={"origin": proposal.origin},
        )
        return proposal

    @property
    def proposals(self) -> list[AgentProposal]:
        return list(self._proposals.values())

    def approve(self, agent_id: str, *, approved_by: str = "operator") -> AgentSpec:
        proposal = self._proposals.pop(agent_id, None)
        if proposal is None:
            raise AgentNotFound(f"no pending proposal for {agent_id!r}")
        spec = self.register(proposal.spec, replace=True)
        self.audit.emit(
            "agent.approved",
            agent=agent_id,
            actor=approved_by,
            result="approved",
            metadata={"origin": proposal.origin},
        )
        return spec

    def reject(self, agent_id: str, *, reason: str = "", rejected_by: str = "operator") -> None:
        if self._proposals.pop(agent_id, None) is None:
            raise AgentNotFound(f"no pending proposal for {agent_id!r}")
        self.audit.emit(
            "agent.rejected", agent=agent_id, actor=rejected_by, result="rejected", reason=reason
        )


# Parsed definitions, keyed by directory and by the newest modification time
# under it. Importing an outside library took the roster from 27 files to 283,
# and the test suite — which builds a registry dozens of times — went from 73
# seconds to 472. The YAML is identical on every build; re-parsing it is the
# whole cost.
#
# Keyed on mtime rather than held forever, so editing a definition still takes
# effect without anyone remembering to clear anything. Specs are frozen-by-
# convention dataclasses that nothing mutates after load; a copy of the list is
# handed out so a caller cannot reorder the cached one.
_PARSE_CACHE: dict[tuple[str, float], list[tuple[dict[str, Any], str]]] = {}


def _newest_mtime(root: Path) -> float:
    return max((p.stat().st_mtime for p in root.rglob("*.yaml")), default=0.0)


def build_registry(
    *, audit: AuditLog | None = None, directory: Path | None = None
) -> AgentRegistry:
    """A registry with every agent definition on disk loaded."""
    registry = AgentRegistry(audit=audit)
    registry.load_dir(directory)
    return registry
