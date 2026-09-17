# The experiment lab

## What it unblocks

Three things were waiting on this. The knowledge pipeline refused every
executable candidate because there was nowhere to run one. Self-improvement
could not test a change it proposed. The agent factory could not validate what
it generated.

## The sandbox

A subprocess with hard ceilings:

| Limit | Default | Why |
|---|---|---|
| Wall clock | 20s | Code that hangs is stopped |
| CPU | 15s | A busy loop cannot outlast the timeout |
| Memory | 512 MB | An allocation grab fails rather than swapping the machine |
| Processes | 1 | A fork bomb is the easy failure mode |
| File size | 10 MB | Nothing fills a disk |
| Output | 20,000 chars | A runaway print is truncated |

Plus its own temporary directory, thrown away afterwards, and an environment
built from an allowlist so no credential reaches it. An allowlist rather than a
denylist, because a secret named something nobody predicted still must not get
through.

On timeout the whole process group is killed, not just the child. The child
runs in its own session, so a process it spawned before the timeout would
otherwise survive it. A test asserts that nothing does.

## What it is not

**It is not a security boundary.** A determined attacker with code execution
can leave a subprocess. Container isolation is the real answer and needs an
image this environment cannot build.

That is why the gate in front of it still refuses anything matching a dangerous
pattern, and the gate behind it still requires a human before anything is
adopted. The sandbox narrows the blast radius; it does not make hostile code
safe to run.

**Running as root weakens it further.** A privileged process is not bound by
the process-count limit, so a candidate could fork. `Sandbox.describe()` says so
when it applies and a warning is logged at construction, because a limit that
quietly does not apply is worse than no limit.

```python
sandbox.describe()["privileged"]   # True when this applies
```

## The lab

An experiment asks one question: is the candidate better than what it would
replace?

```
does it run          -> no: FAILED, and nothing is timed
do its own tests pass -> no: FAILED, because a faster wrong answer is not better
measure both, n times -> median, not mean, so one slow run does not decide
compare              -> BETTER / EQUAL / WORSE
```

The threshold is 1.15x. Below that the difference is the machine doing
something else, and calling it an improvement would be noise dressed as a
finding. A candidate with no baseline is `INCONCLUSIVE`, not `BETTER`: it can be
judged on correctness, not on being better than something that was never named.

## Nothing is adopted

`ExperimentLab.run` returns a verdict and evidence. There is no method that
adopts anything, and every experiment is audited with `adopted: false`.

A `BETTER` verdict is a proposal. A person decides.

## Using it

```bash
make lab     # the sandbox, the lab, and the pipeline, offline
```

```python
lab = ExperimentLab(sandbox=Sandbox(SandboxLimits(timeout_s=10)))
report = lab.run(Experiment(
    name="the change",
    candidate_code=new_code,
    baseline_code=old_code,
    test_code=tests,
    runs=3,
))
print(report.render())
```
