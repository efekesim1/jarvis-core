"""Structured logging and the append-only audit log.

Two distinct streams:

* the ordinary application log, for humans reading a terminal;
* the audit log, a JSON-lines record of every decision that matters
  (who, what, when, why, which agent, which model, which tool, which
  permission, what result).
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import get_settings

_configured = False
_audit_lock = threading.Lock()

# Substrings that must never reach a log line.
_SECRET_HINTS = ("api_key", "apikey", "secret", "token", "password", "authorization")

# Things that look like a credential whatever produced them, and whatever field
# they landed in.
#
# These live here rather than in `security/secrets.py` because of the layering:
# `logging` sits above `security`, so the audit trail cannot import downwards
# to reach them — and that is exactly why the audit log used to redact by field
# *name* only. A key in a free-text `reason` reached the table intact, and
# `reason` is written from `str(exc)` in a dozen places. `security.secrets`
# imports these from here, so there is still one copy.
SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"sk-ant-(?:api|admin)\d{2}-[A-Za-z0-9_-]{8,}"), "an Anthropic key"),
    (re.compile(r"sk-proj-[A-Za-z0-9_-]{20,}"), "an OpenAI key"),
    (re.compile(r"nvapi-[A-Za-z0-9_-]{20,}"), "an NVIDIA key"),
    (re.compile(r"xai-[A-Za-z0-9_-]{20,}"), "an xAI key"),
    (re.compile(r"AIza[0-9A-Za-z_-]{30,}"), "a Google API key"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{16,}"), "a GitHub token"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "an AWS access key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "a private key"),
    (
        re.compile(
            r"\b\w*(?:API_KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL)\w*\s*[:=]\s*"
            r"['\"]?[A-Za-z0-9_\-]{12,}"
        ),
        "an assignment that looks like a credential",
    ),
)

SECRET_MASK = "[redacted: a credential was removed]"


def scrub(text: str) -> str:
    """Blank out anything in ``text`` that looks like a credential."""
    for pattern, _ in SECRET_PATTERNS:
        text = pattern.sub(SECRET_MASK, text)
    return text


def utcnow() -> datetime:
    return datetime.now(UTC)


def configure_logging(level: str | None = None) -> None:
    """Install a single stderr handler. Safe to call more than once."""
    global _configured
    if _configured:
        return
    settings = get_settings()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger("jarvis_core")
    root.setLevel(level or settings.log_level)
    root.addHandler(handler)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"jarvis_core.{name}")


def redact(value: Any) -> Any:
    """Recursively blank out anything that looks like a credential.

    Two independent checks, because either alone leaks. A field *named* like a
    secret is redacted whatever it holds; a value that *looks* like a key is
    redacted whatever it is called. The second was missing, so a key in a
    free-text field went into the audit table intact.
    """
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if any(hint in str(key).lower() for hint in _SECRET_HINTS):
                out[key] = "***redacted***"
            else:
                out[key] = redact(item)
        return out
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return scrub(value)
    return value


@dataclass(slots=True)
class AuditEvent:
    """One entry in the audit trail."""

    action: str
    actor: str = "jarvis"
    subject: str | None = None
    reason: str | None = None
    agent: str | None = None
    model: str | None = None
    tool: str | None = None
    permission: str | None = None
    task_id: str | None = None
    result: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    at: str = field(default_factory=lambda: utcnow().isoformat())

    # Fields a caller fills with free text. `reason` is the dangerous one: it is
    # written from `str(exc)` all over the system, and a provider error can
    # quote the request that produced it.
    _FREE_TEXT = ("subject", "reason", "result")

    def to_json(self) -> str:
        payload = asdict(self)
        payload["metadata"] = redact(payload["metadata"])
        for field_name in self._FREE_TEXT:
            if isinstance(payload.get(field_name), str):
                payload[field_name] = scrub(payload[field_name])
        return json.dumps(payload, ensure_ascii=False, default=str)

    def scrubbed(self) -> AuditEvent:
        """A copy safe to hand to a sink that stores fields separately.

        `to_json` protects the JSON-lines file. A sink that writes columns never
        sees it, which is how a key reached the audit *table* while the file on
        disk beside it was clean.
        """
        return replace(
            self,
            subject=scrub(self.subject) if self.subject else self.subject,
            reason=scrub(self.reason) if self.reason else self.reason,
            result=scrub(self.result) if self.result else self.result,
            metadata=redact(self.metadata),
        )


class AuditLog:
    """Append-only JSON-lines audit log.

    Writes to disk when a path is available and always keeps an in-memory tail
    so tests and the dashboard can read recent activity without file access.
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        memory_limit: int = 1000,
        sink: Any | None = None,
    ) -> None:
        settings = get_settings()
        self.path = path if path is not None else settings.data_dir / "audit.jsonl"
        self._memory_limit = memory_limit
        self._events: list[AuditEvent] = []
        # An optional durable sink. The file and the in-memory tail stay
        # whatever happens; the sink makes the trail queryable and survive a
        # restart.
        self.sink = sink
        self._log = get_logger("audit")

    def record(self, event: AuditEvent) -> AuditEvent:
        # Scrub once, then use the scrubbed copy everywhere. The file used to be
        # protected by `to_json` while the raw event went on to the sink, so a
        # credential in a free-text field reached the audit *table* with the
        # JSON-lines file beside it clean. Same for the in-memory tail, which
        # the dashboard and `jarvis status` read.
        event = event.scrubbed()
        with _audit_lock:
            self._events.append(event)
            if len(self._events) > self._memory_limit:
                del self._events[0 : len(self._events) - self._memory_limit]
        line = event.to_json()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:  # disk full, read-only mount, ...
            self._log.warning("audit log write failed (%s); event kept in memory only", exc)
        if self.sink is not None:
            try:
                self.sink.record(event)
            except Exception as exc:
                # A failing sink must never stop the thing being audited.
                self._log.warning("audit sink write failed: %s", exc)
        self._log.debug("audit %s", line)
        return event

    def emit(self, action: str, **kwargs: Any) -> AuditEvent:
        return self.record(AuditEvent(action=action, **kwargs))

    @property
    def events(self) -> list[AuditEvent]:
        return list(self._events)

    def tail(self, count: int = 20) -> list[AuditEvent]:
        return list(self._events[-count:])
