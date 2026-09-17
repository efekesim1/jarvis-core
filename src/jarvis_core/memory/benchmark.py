"""How often retrieval finds a memory a person would call relevant.

"Retrieval is lexical" sat in the known-issues list as a sentence for weeks.
A sentence is not a number and cannot be argued with, so nobody fixed it. This
module is the number.

Two fixtures, and the second is what makes the first honest:

* ``SHOULD_FIND`` — queries against memories a person would expect to surface.
  Recall is how many of them do.
* ``SHOULD_NOT_FIND`` — pairs that share no topic at all. Recall bought by
  matching everything is worth nothing, and without this half a more aggressive
  matcher is indistinguishable from a broken one.

The fixtures are here rather than in the test file because the CLI reports the
same measurement, and two copies of a benchmark drift until they disagree about
what was measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..text import terms

# The bar `MemoryManager.recall` applies. Kept in step with `RetrievalBudget`
# by a test rather than by hope.
DEFAULT_BAR = 0.15


@dataclass(frozen=True, slots=True)
class Case:
    """One query, one memory, and what makes the pair hard."""

    query: str
    memory: str
    difficulty: str


# Turkish is here in earnest. It is agglutinative — "projelerimi" is
# proje + ler + im + i — and the operator writes in it, so a matcher that handles only
# English plurals cannot find half of what he says.
SHOULD_FIND: tuple[Case, ...] = (
    Case("arabam ne zaman servise girecek", "araba servis randevusu mart ayında", "possessive"),
    Case("hedefim ne durumda", "hedef aylık 5000 dolar gelir", "possessive"),
    Case("projelerimi listele", "proje jarvis ai os", "possessive plural"),
    Case("bütçemiz ne kadar", "bütçe aylık 200 dolar", "possessive plural"),
    Case("raporu kısa tut", "raporlar kısa olmalı", "accusative against plural"),
    Case(
        "müşterilerden gelen geri bildirim",
        "müşteri geri bildirimi olumlu",
        "ablative plural",
    ),
    Case("deployment plan", "we deploy on fridays", "nominalisation"),
    Case("pricing strategy", "we price at 29 dollars a month", "nominalisation"),
    Case("running the tests", "the test suite takes 70 seconds", "participle"),
    Case("bookkeeping cost", "bookkeeping costs 400 dollars a year", "plain"),
    Case("jarvis architecture", "jarvis architecture is layered", "plain"),
    Case("hedef gelir", "hedef aylık 5000 dolar gelir", "plain turkish"),
    # The four below need a synonym rather than a suffix, and are expected to
    # miss. They are in the fixture on purpose: a benchmark that only contains
    # what already passes measures nothing, and these are the exact cases an
    # embedding layer would have to earn its dependency by fixing.
    Case("what does he drive", "operator owns a car", "synonym"),
    Case("how much does it cost", "the price is 200 dollars", "synonym"),
    Case("who buys this", "our customers are small businesses", "synonym"),
    Case("meeting notes", "met with the accountant on tuesday", "irregular verb"),
)

SHOULD_NOT_FIND: tuple[Case, ...] = (
    Case("what font should the invoice use", "hedef aylık 5000 dolar gelir", "unrelated"),
    Case("nasıl bir yazı tipi kullanmalıyım", "araba servis randevusu mart ayında", "unrelated"),
    Case("how do I write a unit test", "müşteri geri bildirimi olumlu", "unrelated"),
    Case("bugün hava nasıl", "bütçe aylık 200 dolar", "unrelated"),
    Case("what is the capital of france", "we deploy on fridays", "unrelated"),
    Case("kediler ne yer", "proje jarvis ai os", "unrelated"),
)

# Where the measurement stood when it was written. Floors, not targets: an
# improvement must never be a failing test, and a regression always must be.
RECALL_FLOOR = 12
PRECISION_FLOOR = 1.0


def matches(query: str, memory: str, *, bar: float = DEFAULT_BAR) -> float:
    """The overlap score retrieval would give this pair.

    The same arithmetic `MemoryManager.recall` uses — overlap against the
    query's own terms — so the benchmark measures retrieval rather than a
    convenient stand-in for it.
    """
    query_terms, memory_terms = terms(query), terms(memory)
    if not query_terms or not memory_terms:
        return 0.0
    return len(query_terms & memory_terms) / len(query_terms)


@dataclass(slots=True)
class BenchmarkResult:
    """What the fixtures showed, with the misses named rather than counted."""

    found: list[Case]
    missed: list[Case]
    false_positives: list[Case]
    rejected: int

    @property
    def recall(self) -> float:
        total = len(self.found) + len(self.missed)
        return len(self.found) / total if total else 0.0

    @property
    def precision(self) -> float:
        """Of everything returned, how much should have been.

        With these fixtures a false positive can only come from the
        should-not-find half, which is what makes this number meaningful at all
        on a set this small.
        """
        returned = len(self.found) + len(self.false_positives)
        return len(self.found) / returned if returned else 0.0

    @property
    def by_difficulty(self) -> dict[str, tuple[int, int]]:
        """found / total for each kind of hard, so a fix can be aimed."""
        counts: dict[str, tuple[int, int]] = {}
        for case in self.found:
            hit, total = counts.get(case.difficulty, (0, 0))
            counts[case.difficulty] = (hit + 1, total + 1)
        for case in self.missed:
            hit, total = counts.get(case.difficulty, (0, 0))
            counts[case.difficulty] = (hit, total + 1)
        return dict(sorted(counts.items()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "recall": round(self.recall, 3),
            "precision": round(self.precision, 3),
            "found": len(self.found),
            "missed": len(self.missed),
            "false_positives": len(self.false_positives),
            "unrelated_correctly_rejected": self.rejected,
            "by_difficulty": {k: f"{h}/{t}" for k, (h, t) in self.by_difficulty.items()},
            "misses": [
                {"query": c.query, "memory": c.memory, "needs": c.difficulty} for c in self.missed
            ],
        }

    def render(self) -> str:
        lines = [
            f"recall    {len(self.found)}/{len(self.found) + len(self.missed)}"
            f"  ({self.recall:.0%})",
            f"precision {self.precision:.0%}"
            f"  ({self.rejected}/{self.rejected + len(self.false_positives)} unrelated rejected)",
            "",
            "by difficulty:",
        ]
        lines += [f"  {name:26} {hit}/{total}" for name, (hit, total) in self.by_difficulty.items()]
        if self.missed:
            lines += ["", "still missed:"]
            lines += [f"  {c.query!r} -> {c.memory!r}  ({c.difficulty})" for c in self.missed]
        return "\n".join(lines)


def run(*, bar: float = DEFAULT_BAR) -> BenchmarkResult:
    """Measure retrieval against both fixtures."""
    found = [case for case in SHOULD_FIND if matches(case.query, case.memory) >= bar]
    missed = [case for case in SHOULD_FIND if case not in found]
    false_positives = [case for case in SHOULD_NOT_FIND if matches(case.query, case.memory) >= bar]
    return BenchmarkResult(
        found=found,
        missed=missed,
        false_positives=false_positives,
        rejected=len(SHOULD_NOT_FIND) - len(false_positives),
    )
