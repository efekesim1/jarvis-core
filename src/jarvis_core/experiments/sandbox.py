"""Running code nobody has read yet.

Three things are blocked on this: the knowledge pipeline refuses every
executable candidate because there is nowhere to run one, self-improvement
cannot test a change it proposes, and the agent factory cannot validate what it
generates.

**What this is.** A subprocess with hard limits: its own temporary directory,
an environment stripped of every credential, a CPU ceiling, a memory ceiling, a
wall-clock timeout, no ability to fork, and output truncated so a runaway print
cannot fill a disk. It answers "does this code do something useful, quickly,
without eating the machine".

**What this is not.** A security boundary. A determined attacker with code
execution can get out of a subprocess. Container isolation is the real answer
and needs an image this environment cannot build. So the pipeline in front of
this still refuses anything with a dangerous pattern, and the gate behind it
still requires a human before anything is adopted. The sandbox narrows the
blast radius; it does not make hostile code safe to run.
"""

from __future__ import annotations

import contextlib
import os
import resource
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..logging import get_logger

log = get_logger("experiments.sandbox")

# What a candidate may consume before it is stopped.
DEFAULT_TIMEOUT_S = 20.0
DEFAULT_MEMORY_MB = 512
DEFAULT_CPU_S = 15
MAX_OUTPUT_CHARS = 20_000
MAX_CODE_CHARS = 200_000

# The only environment variables that survive. Everything else, and every
# credential in particular, is left behind.
_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "TZ")


@dataclass(slots=True)
class SandboxResult:
    """What happened when the code ran."""

    ok: bool
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    timed_out: bool = False
    killed: bool = False
    reason: str = ""
    truncated: bool = False

    @property
    def failed(self) -> bool:
        return not self.ok

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_s": round(self.duration_s, 3),
            "timed_out": self.timed_out,
            "killed": self.killed,
            "reason": self.reason,
            "truncated": self.truncated,
        }


@dataclass(slots=True)
class SandboxLimits:
    """The ceilings one run may not cross."""

    timeout_s: float = DEFAULT_TIMEOUT_S
    memory_mb: int = DEFAULT_MEMORY_MB
    cpu_s: int = DEFAULT_CPU_S
    max_output_chars: int = MAX_OUTPUT_CHARS
    # Itself and nothing else; a fork bomb is the easy failure mode.
    #
    # ``None`` means the ceiling is not enforced, and it is spelled `None`
    # rather than `0` on purpose: `0` is a meaningful value for a process
    # ceiling — deny every fork — and a security limit where zero silently
    # means *unlimited* is a footgun waiting for whoever writes
    # `max_processes=0` intending to lock it down. Only the process-group
    # cleanup test sets it, because that test has to create a descendant before
    # it can prove the timeout kills the whole group.
    max_processes: int | None = 1
    extra_env: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "timeout_s": self.timeout_s,
            "memory_mb": self.memory_mb,
            "cpu_s": self.cpu_s,
            "max_output_chars": self.max_output_chars,
            "max_processes": self.max_processes,
        }


def _apply_limits(limits: SandboxLimits):
    """Build the function the child runs before exec.

    Everything here happens in the child, after fork and before the code runs,
    so a limit cannot be raised by the code it constrains.
    """

    def preexec() -> None:  # pragma: no cover - runs in the child process
        # A new session, so a signal to the parent's group does not reach the
        # child and the child cannot signal the parent's group either.
        os.setsid()
        memory_bytes = limits.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_s, limits.cpu_s))
        # ``None`` disables the ceiling; zero still means zero processes. Used
        # only by the process-group cleanup test, which must create one
        # descendant before it can verify the timeout kills the whole group.
        if limits.max_processes is not None:
            resource.setrlimit(resource.RLIMIT_NPROC, (limits.max_processes, limits.max_processes))
        # No core dumps, and a cap on how large a file it may write.
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))

    return preexec


def _kill_group(process: subprocess.Popen[str]) -> None:
    """Stop the child and everything it spawned.

    The child called setsid, so it leads its own process group; signalling the
    group is what reaches a process that forked before the timeout.
    """
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        # It already exited, or the group is gone. Fall back to the child.
        with contextlib.suppress(ProcessLookupError):
            process.kill()


def privileged() -> bool:
    """Whether this process runs as root, which weakens the sandbox.

    A privileged process is not bound by the process-count limit, so a
    candidate could fork. Saying so is better than a limit that quietly does
    not apply.
    """
    return hasattr(os, "geteuid") and os.geteuid() == 0


def _clean_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    """An environment with no credential in it.

    An allowlist rather than a denylist: a new secret named something nobody
    predicted still does not get through.
    """
    env = {name: os.environ[name] for name in _ENV_ALLOWLIST if name in os.environ}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    # Say plainly where the code is running, in case it asks.
    env["JARVIS_SANDBOX"] = "1"
    env.update(extra or {})
    return env


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + f"\n[truncated at {limit} characters]", True


class Sandbox:
    """Runs a piece of Python under hard limits, in a directory of its own."""

    def __init__(self, limits: SandboxLimits | None = None) -> None:
        self.limits = limits or SandboxLimits()
        if privileged():
            log.warning(
                "the sandbox is running as root, so the process-count limit does not "
                "bind; run as an unprivileged user, or in a container, for real limits"
            )

    def run_code(self, code: str, *, files: dict[str, str] | None = None) -> SandboxResult:
        """Run a snippet. Never raises; a failure is a result.

        ``files`` are written alongside it, so a candidate can bring its own
        module and a test that imports it.
        """
        if not code or not code.strip():
            return SandboxResult(ok=False, reason="there was no code to run")
        if len(code) > MAX_CODE_CHARS:
            return SandboxResult(
                ok=False, reason=f"the code is larger than the {MAX_CODE_CHARS} character limit"
            )

        with tempfile.TemporaryDirectory(prefix="operator-sandbox-") as work:
            workdir = Path(work)
            for name, content in (files or {}).items():
                # A candidate may not write outside its own directory, so a
                # path that tries is refused before anything runs.
                target = (workdir / name).resolve()
                if workdir not in target.parents:
                    return SandboxResult(
                        ok=False, reason=f"the file {name!r} would be written outside the sandbox"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")

            entry = workdir / "_candidate.py"
            entry.write_text(textwrap.dedent(code), encoding="utf-8")
            return self._execute([sys.executable, str(entry)], workdir)

    def run_module(
        self, module: str, args: tuple[str, ...] = (), *, cwd: Path | None = None
    ) -> SandboxResult:
        """Run a Python module, for a candidate's own test suite."""
        with tempfile.TemporaryDirectory(prefix="operator-sandbox-") as work:
            workdir = Path(cwd) if cwd is not None else Path(work)
            return self._execute([sys.executable, "-m", module, *args], workdir)

    def _execute(self, argv: list[str], workdir: Path) -> SandboxResult:
        started = time.perf_counter()
        process = None
        try:
            process = subprocess.Popen(  # noqa: S603 - argv is built here, no shell
                argv,
                cwd=str(workdir),
                env=_clean_environment(self.limits.extra_env),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=_apply_limits(self.limits),  # noqa: PLW1509 - the point
            )
        except (OSError, ValueError) as exc:
            return SandboxResult(
                ok=False,
                duration_s=time.perf_counter() - started,
                reason=f"the sandbox could not start it: {exc}",
            )

        try:
            raw_out, raw_err = process.communicate(timeout=self.limits.timeout_s)
        except subprocess.TimeoutExpired:
            # The child runs in its own session, so killing the process alone
            # would leave anything it spawned running. Kill the whole group.
            _kill_group(process)
            raw_out, raw_err = process.communicate()
            duration = time.perf_counter() - started
            stdout, truncated = _truncate(raw_out or "", self.limits.max_output_chars)
            log.info("a sandbox run hit its %.1fs timeout", self.limits.timeout_s)
            return SandboxResult(
                ok=False,
                stdout=stdout,
                duration_s=duration,
                timed_out=True,
                truncated=truncated,
                reason=f"it did not finish within {self.limits.timeout_s:g}s",
            )

        completed = subprocess.CompletedProcess(argv, process.returncode, raw_out, raw_err)

        duration = time.perf_counter() - started
        stdout, out_truncated = _truncate(completed.stdout, self.limits.max_output_chars)
        stderr, err_truncated = _truncate(completed.stderr, self.limits.max_output_chars)
        # A negative code means a signal, which here means a limit was hit.
        killed = completed.returncode is not None and completed.returncode < 0
        ok = completed.returncode == 0

        reason = ""
        if killed:
            reason = f"it was stopped by signal {-completed.returncode}, most likely a limit"
        elif not ok:
            reason = f"it exited with code {completed.returncode}"

        return SandboxResult(
            ok=ok,
            exit_code=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_s=duration,
            killed=killed,
            reason=reason,
            truncated=out_truncated or err_truncated,
        )

    def describe(self) -> dict[str, Any]:
        """What this sandbox does and does not protect against."""
        return {
            "kind": "subprocess",
            "limits": self.limits.as_dict(),
            "isolates": [
                "its own temporary directory",
                "an environment with no credentials",
                "cpu, memory and file size",
                *(["process count"] if self.limits.max_processes is not None else []),
                "a wall-clock timeout",
                "output size",
            ],
            # Surfaced the same way `privileged` is: a limit that is off must
            # say so, not be inferred from reading the constructor call.
            "process_ceiling_enforced": self.limits.max_processes is not None,
            "privileged": privileged(),
            "does_not_isolate": [
                "the network: outbound access is not blocked here",
                "the filesystem outside the working directory, for reads",
                "a determined escape; this narrows blast radius, it is not a security boundary",
                *(
                    [
                        "process count, because this runs as root and a "
                        "privileged process is not bound by that limit"
                    ]
                    if privileged()
                    else []
                ),
            ],
            "real_isolation": "container execution, which needs an image this "
            "environment cannot build",
        }
