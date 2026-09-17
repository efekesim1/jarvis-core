"""The experiment lab.

Where code nobody has read yet gets run, measured against what it would
replace, and judged. Nothing here adopts anything: the lab produces a verdict
and evidence, and a person decides.

The sandbox is a subprocess with hard limits, not a security boundary. What it
does and does not protect against is stated in `Sandbox.describe`, and the
gates in front of and behind it exist because of what it does not.
"""

from .lab import Experiment, ExperimentLab, ExperimentReport, Measurement, Verdict
from .sandbox import Sandbox, SandboxLimits, SandboxResult

__all__ = [
    "Experiment",
    "ExperimentLab",
    "ExperimentReport",
    "Measurement",
    "Sandbox",
    "SandboxLimits",
    "SandboxResult",
    "Verdict",
]
