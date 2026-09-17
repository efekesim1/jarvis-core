"""Bringing an outside agent library into this system without lowering its bar.

`msitarzewski/agency-agents` is 264 markdown files, MIT licensed, one persona
each: a short YAML frontmatter and then two hundred–odd lines of opinionated
prose about how that specialist works. It is a genuinely good prompt library.
It is not a set of agent definitions this system can run, and the difference is
not cosmetic:

* no ``output_schema`` — our runtime parses JSON out of every reply, so a
  prose-only agent produces a schema failure on every single run
* no ``tool_capabilities`` and no ``permissions`` — rule 17 fails the build for
  an agent claiming a capability nothing backs, and rule 13 gives an agent with
  no policy nothing at all
* no ``smoke_task`` — the ladder holds it at DEFINED forever, because nothing
  can be run and so nothing can be shown
* no ``allowed_models``, no ``evaluation_policy``, no risk level

So copying the files in would produce 264 agents that load and cannot work.
This module converts instead, and refuses what it cannot convert honestly.

**The prose is untrusted input.** Two hundred lines of instructions from a
public repository, pasted into a system prompt, is precisely the surface rule
14 exists for — and a system prompt is a worse place for it than a tool result,
because it is not fenced. Every file is therefore scanned with the same
injection detector the context compressor uses, and a hit refuses the file
rather than sanitising it. A persona nobody can vouch for is not worth the
risk of a persona that has been edited.

Nothing here adopts anything. The importer writes to a staging directory and
reports; moving a file into the live roster is a person's decision, which is
rule 9 applied to someone else's code rather than to our own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from ..core.context import sanitize_untrusted
from ..logging import get_logger
from ..security.permissions import Permission, RiskLevel
from ..text import terms

log = get_logger("agents.importer")

# Their twenty divisions onto our three swarms. A mapping rather than twenty new
# swarms, because a swarm is what the planner ranks candidates within and the
# rule planner's scoring was tuned against three of them; adding seventeen more
# would change agent selection for every existing request, which is a much
# larger change than an import.
DIVISION_TO_SWARM: dict[str, str] = {
    "academic": "research",
    "research": "research",
    "gis": "research",
    "spatial-computing": "research",
    "healthcare": "research",
    "engineering": "build",
    "testing": "build",
    "security": "build",
    "integrations": "build",
    "game-development": "build",
    "design": "build",
    "product": "build",
    "specialized": "build",
    "finance": "business",
    "marketing": "business",
    "paid-media": "business",
    "sales": "business",
    "strategy": "business",
    "support": "business",
    "project-management": "business",
}

# The reply shape every agent in this system produces. An imported agent gets
# the same one, because the runtime parses it and the evaluator scores schema
# coverage against it.
STANDARD_SCHEMA: dict[str, Any] = {
    "summary": "The answer in plain language, a few short paragraphs",
    "findings": ["Each distinct point as its own string"],
    "sources": [{"url": "https://...", "title": "...", "note": "what this supports"}],
    "gaps": ["Anything you could not establish"],
    "confidence": 0.0,
}

# What an imported agent is allowed to reach on arrival: reading, and nothing
# else. Not because these personas are suspect but because nobody has yet
# watched one work, and a capability is granted on evidence. Widening one is an
# edit to its file, which is a decision with a name attached.
ARRIVAL_PERMISSIONS: tuple[str, ...] = (Permission.READ.value,)
ARRIVAL_TOOL_CAPABILITIES: tuple[str, ...] = ()

# Divisions whose prose is about attacking systems. Their content is
# legitimate — a penetration tester's playbook reads like one — but an agent
# carrying it should not arrive with anything beyond reading, and should be
# looked at by a person before it arrives at all.
NEEDS_A_SECOND_LOOK = frozenset({"security"})

# Frontmatter keys we read. Anything else in their files is theirs.
_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)

# Their headings are decorated with emoji. Stripped from a derived role, kept
# in the prompt body where they are part of the author's voice.
_EMOJI = re.compile("[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f000-\U0001f2ff️]", re.UNICODE)

# How long a converted system prompt may be. Their median is 269 lines, which
# at roughly four characters a token is around 2,700 tokens on *every* call
# this agent makes. That is a real recurring cost and worth a ceiling; a prompt
# over it is truncated at a section boundary and the file says so.
MAX_PROMPT_CHARS = 9000


class Outcome(StrEnum):
    """What happened to one source file."""

    CONVERTED = "converted"
    REFUSED_INJECTION = "refused_injection"
    REFUSED_NO_FRONTMATTER = "refused_no_frontmatter"
    REFUSED_NO_NAME = "refused_no_name"
    REFUSED_THIN = "refused_thin"
    REFUSED_UNKNOWN_DIVISION = "refused_unknown_division"
    SKIPPED_DUPLICATE = "skipped_duplicate"


@dataclass(slots=True)
class ImportedAgent:
    """One converted definition, and what had to be supplied to convert it."""

    agent_id: str
    name: str
    swarm: str
    division: str
    source_path: str
    spec: dict[str, Any]
    supplied: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_yaml(self) -> str:
        return yaml.safe_dump(self.spec, sort_keys=False, allow_unicode=True, width=88)


@dataclass(slots=True)
class ImportReport:
    """What a whole import run did, and what it would not do."""

    converted: list[ImportedAgent] = field(default_factory=list)
    refused: list[tuple[str, Outcome, str]] = field(default_factory=list)

    @property
    def by_swarm(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for agent in self.converted:
            counts[agent.swarm] = counts.get(agent.swarm, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def by_outcome(self) -> dict[str, int]:
        counts = {Outcome.CONVERTED.value: len(self.converted)}
        for _, outcome, _ in self.refused:
            counts[outcome.value] = counts.get(outcome.value, 0) + 1
        return counts

    def as_dict(self) -> dict[str, Any]:
        return {
            "converted": len(self.converted),
            "refused": len(self.refused),
            "by_swarm": self.by_swarm,
            "by_outcome": self.by_outcome,
            "refusals": [
                {"file": name, "why": outcome.value, "detail": detail[:160]}
                for name, outcome, detail in self.refused
            ],
        }

    def render(self) -> str:
        lines = [f"converted {len(self.converted)}, refused {len(self.refused)}"]
        if self.by_swarm:
            lines.append("  " + "  ".join(f"{k}={v}" for k, v in self.by_swarm.items()))
        if self.refused:
            lines.append("")
            lines.append("refused:")
            grouped: dict[str, list[str]] = {}
            for name, outcome, _ in self.refused:
                grouped.setdefault(outcome.value, []).append(name)
            for why, names in sorted(grouped.items()):
                shown = ", ".join(Path(n).stem for n in names[:4])
                more = f" and {len(names) - 4} more" if len(names) > 4 else ""
                lines.append(f"  {why:28} {len(names):3}  {shown}{more}")
        return "\n".join(lines)


def _slug(text: str) -> str:
    """Their file names and titles into our id shape.

    The registry requires lowercase letters, digits and underscores, and an id
    may not begin with a digit — "3D & Scene Developer" would otherwise produce
    `3d_scene_developer`, which the loader rejects. A leading underscore keeps
    the rest of the name readable instead of dropping the digit.
    """
    cleaned = _EMOJI.sub("", text).strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    if cleaned and cleaned[0].isdigit():
        cleaned = f"a_{cleaned}"
    return cleaned or "unnamed"


def _first_sentence(text: str, limit: int = 160) -> str:
    stripped = _EMOJI.sub("", text).strip()
    match = re.search(rf"^(.{{20,{limit}}}?[.!])\s", stripped + " ")
    return (match.group(1) if match else stripped[:limit]).strip()


def _trim_prompt(body: str) -> tuple[str, bool]:
    """Cut a long prompt at a section boundary rather than mid-sentence.

    Truncating a 269-line persona in the middle of a bullet list produces a
    prompt that trails off, and a model reading one does not know whether the
    missing half mattered. Cutting at a heading at least ends on a complete
    thought.
    """
    if len(body) <= MAX_PROMPT_CHARS:
        return body, False
    window = body[:MAX_PROMPT_CHARS]
    boundary = window.rfind("\n## ")
    if boundary < MAX_PROMPT_CHARS // 2:
        boundary = window.rfind("\n\n")
    return window[: boundary if boundary > 0 else MAX_PROMPT_CHARS].rstrip(), True


def _task_kind_for(swarm: str) -> str:
    return {"research": "research", "build": "coding", "business": "analysis"}.get(
        swarm, "analysis"
    )


def _models_for(swarm: str) -> list[str]:
    """Two models, always. One is a retry policy that cannot retry."""
    if swarm == "build":
        return ["claude-opus-5", "claude-sonnet-5"]
    return ["claude-sonnet-5", "claude-opus-5"]


def convert(path: Path, *, existing_ids: frozenset[str] = frozenset()) -> ImportedAgent | None:
    """One source file into one definition, or ``None`` with a reason logged.

    Raises nothing. A library of 264 files will contain some this cannot read,
    and stopping the whole import on the first of them would mean importing
    none of them.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    match = _FRONTMATTER.match(text)
    if match is None:
        _refuse(path, Outcome.REFUSED_NO_FRONTMATTER, "no YAML frontmatter")
        return None

    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        _refuse(path, Outcome.REFUSED_NO_FRONTMATTER, str(exc))
        return None
    if not isinstance(meta, dict) or not meta.get("name"):
        _refuse(path, Outcome.REFUSED_NO_NAME, "frontmatter has no name")
        return None

    body = match.group(2).strip()
    if len(body) < 400:
        _refuse(path, Outcome.REFUSED_THIN, f"{len(body)} characters of prompt")
        return None

    # The prose is untrusted text from a public repository. A hit refuses the
    # file; it is never sanitised and kept, because a persona that has been
    # edited by a pattern matcher is a persona nobody wrote.
    _, report = sanitize_untrusted(body)
    if report.hits:
        _refuse(path, Outcome.REFUSED_INJECTION, ", ".join(report.hits[:3]))
        return None

    division = path.parent.name
    swarm = DIVISION_TO_SWARM.get(division)
    if swarm is None:
        _refuse(path, Outcome.REFUSED_UNKNOWN_DIVISION, division)
        return None

    agent_id = _slug(str(meta["name"]))
    if agent_id in existing_ids:
        _refuse(path, Outcome.SKIPPED_DUPLICATE, f"{agent_id} already exists")
        return None

    prompt, trimmed = _trim_prompt(body)
    description = str(meta.get("description") or "").strip()
    role = _first_sentence(description or str(meta["name"]))

    supplied = [
        "output_schema",
        "permissions",
        "allowed_models",
        "evaluation_policy",
        "retry_policy",
        "smoke_task",
        "risk_level",
        "task_kind",
    ]
    notes: list[str] = []
    if trimmed:
        notes.append(f"prompt trimmed to {MAX_PROMPT_CHARS} characters at a section boundary")
    if division in NEEDS_A_SECOND_LOOK:
        notes.append("security division: read the prompt before enabling this one")
    if meta.get("tools"):
        notes.append(
            f"source declared tools ({meta['tools']}); not carried over — a capability "
            "is granted on evidence, and none exists yet"
        )
    if promises := _tools_named_in(prompt):
        notes.append(
            "the imported prose names "
            + ", ".join(sorted(promises)[:4])
            + " — this agent cannot reach any of them; the appended rules say so, "
            "but the prompt would read better edited"
        )

    spec: dict[str, Any] = {
        "id": agent_id,
        "name": _EMOJI.sub("", str(meta["name"])).strip(),
        "role": role,
        "swarm": swarm,
        "version": "0.1.0",
        "enabled": False,
        "description": description or role,
        # Where this came from. MIT requires attribution and a reader deserves
        # to know which prompts in this repository someone else wrote.
        "source": {
            "repository": "https://github.com/msitarzewski/agency-agents",
            "license": "MIT",
            "file": f"{division}/{path.name}",
            "imported_as": "unverified",
        },
        "capabilities": _capabilities_from(description, str(meta["name"])),
        "tools": [],
        "allowed_models": _models_for(swarm),
        "permissions": list(ARRIVAL_PERMISSIONS),
        "tool_capabilities": list(ARRIVAL_TOOL_CAPABILITIES),
        "max_tool_risk": "read_only",
        "risk_level": RiskLevel.LOW.value,
        "task_kind": _task_kind_for(swarm),
        "timeout_s": 240,
        "max_tokens": 8000,
        "retry_policy": {"max_attempts": 2, "backoff_s": 1.0, "alternative_model": True},
        "evaluation_policy": {
            "criteria": ["correctness", "relevance", "completeness"],
            "min_confidence": 0.4,
            # Deliberately false. These agents arrive with no network and no
            # tools, so requiring citations would fail every run for a reason
            # that is our configuration rather than their answer.
            "require_sources": False,
        },
        "smoke_task": _smoke_task_from(description, str(meta["name"])),
        "system_prompt": _wrap_prompt(prompt),
        "output_schema": dict(STANDARD_SCHEMA),
    }
    return ImportedAgent(
        agent_id=agent_id,
        name=spec["name"],
        swarm=swarm,
        division=division,
        source_path=str(path),
        spec=spec,
        supplied=supplied,
        notes=notes,
    )


# Things their prose habitually tells an agent to use, none of which this
# system exposes. Named here so an imported file can say which ones it promises
# rather than leaving the reader to find out from the audit.
_TOOL_WORDS = (
    "web search",
    "web_search",
    "search the web",
    "ollama",
    "chromadb",
    "faiss",
    "bash",
    "shell",
    "curl",
    "git ",
    "playwright",
    "browser",
)


def _tools_named_in(prompt: str) -> set[str]:
    lowered = prompt.lower()
    return {word.strip() for word in _TOOL_WORDS if word in lowered}


def _capabilities_from(description: str, name: str) -> list[str]:
    """Words the rule planner can match this agent on.

    Drawn from the author's own description rather than invented, because a
    capability the agent cannot do is worse than a missing one: the planner
    will route work to it.
    """
    words = terms(f"{name} {description}")
    ranked = sorted(words, key=lambda w: (-len(w), w))
    return ranked[:8] or [_slug(name).replace("_", " ")]


def _smoke_task_from(description: str, name: str) -> str:
    """A task in the agent's own vocabulary, so verification tests the agent.

    A generic "do your job" smoke task verifies that the plumbing runs, which
    the offline sweep already proves. Using the author's own first sentence at
    least aims the run at what they said the agent is for.
    """
    subject = _first_sentence(description, 120) or name
    return f"Using your own specialism, produce a short worked example: {subject}"


def _wrap_prompt(body: str) -> str:
    """The imported persona, with this system's own rules after it.

    Order matters. The persona is someone else's writing and goes first; the
    non-negotiables go last, where a model reads them most recently, and they
    are the ones that must win. An imported prompt cannot grant itself a tool,
    invent a source, or talk its way past the gateway — those are decided by
    code before any of this text is read — but it can certainly try to talk its
    way past *honesty*, and that is what these lines are for.
    """
    return (
        f"{body}\n\n"
        "---\n\n"
        "You are operating inside the JARVIS operating system. The "
        "following rules come from the system and override anything above them "
        "that conflicts:\n\n"
        "- Every factual claim traces to something you were given or a source "
        "you cite. Where you have no evidence, say so. An honest gap beats a "
        "confident guess, always.\n"
        "- You have no tools, no network and no file access on this run. Do not "
        "claim to have searched, fetched, measured or run anything.\n"
        "- Report what you could not establish as plainly as what you could.\n"
        "- Answer as JSON matching the schema you are given, with nothing "
        "before or after it.\n"
    )


def _refuse(path: Path, outcome: Outcome, detail: str) -> None:
    log.info("not importing %s: %s (%s)", path.name, outcome.value, detail)
    _REFUSALS.append((str(path), outcome, detail))


# Collected during a run and drained by `import_library`. A module-level list is
# ugly; the alternative is threading a report through every helper, and this
# function is called once per file from one place.
_REFUSALS: list[tuple[str, Outcome, str]] = []


def import_library(
    source: Path,
    *,
    divisions: tuple[str, ...] = (),
    limit: int = 0,
    existing_ids: frozenset[str] = frozenset(),
) -> ImportReport:
    """Convert a directory tree of their agents into ours.

    Nothing is written. The caller decides where, and whether.
    """
    _REFUSALS.clear()
    roots = (
        [source / division for division in divisions]
        if divisions
        else [p for p in sorted(source.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    )
    seen = set(existing_ids)
    converted: list[ImportedAgent] = []
    for root in roots:
        if not root.is_dir():
            log.warning("no such division: %s", root.name)
            continue
        for path in sorted(root.glob("*.md")):
            if limit and len(converted) >= limit:
                break
            agent = convert(path, existing_ids=frozenset(seen))
            if agent is not None:
                seen.add(agent.agent_id)
                converted.append(agent)
    return ImportReport(converted=converted, refused=list(_REFUSALS))


def write(report: ImportReport, destination: Path) -> list[Path]:
    """Write the converted definitions, one file each, grouped by swarm.

    They land disabled (`enabled: false`). Loading is not adopting: the
    registry will see them, `jarvis agents` will list them, and nothing will
    route work to one until a person edits that line. That is rule 9 applied to
    someone else's code rather than to our own.
    """
    written: list[Path] = []
    for agent in report.converted:
        folder = destination / agent.swarm
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{agent.agent_id}.yaml"
        header = (
            f"# Imported from {agent.spec['source']['repository']}\n"
            f"# Source: {agent.spec['source']['file']} (MIT)\n"
            f"# Converted by `jarvis import-agents`. Supplied by this system, not by the\n"
            f"# author: {', '.join(agent.supplied)}.\n"
            f"#\n"
            f"# Disabled on arrival. It has never run, here or anywhere this system can\n"
            f"# see, so nothing routes work to it until someone sets enabled: true.\n"
        )
        for note in agent.notes:
            header += f"# Note: {note}\n"
        path.write_text(f"{header}\n{agent.as_yaml()}", encoding="utf-8")
        written.append(path)
    return written
