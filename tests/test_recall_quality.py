"""How often retrieval finds a memory a person would call relevant.

Retrieval is word overlap. That is a design decision with a cost, and the cost
was never measured — "retrieval is lexical" sat in KNOWN_ISSUES as a sentence,
which is not a number and cannot be argued with.

So: a fixture of realistic queries against memories that should match, and a
second fixture of pairs that must not. The first says how much is being missed;
the second is the one that stops the fix for the first from being "match
everything". Both are checked at a floor rather than an exact figure, so an
improvement is never a failing test — but a regression is.

The measurement is why the stemmer handles Turkish agglutination. the operator writes
"hedefim" and "projelerimi"; a matcher that knew only English plurals found
nothing for either, and five of the six worst misses were that one cause.

What remains after the fix needs a synonym, not a suffix: "drive" against
"car", "cost" against "price". Closing that needs an embedding model, which
means a dependency, a credential and a network call on every write and every
query. That trade is recorded in DECISIONS and KNOWN_ISSUES rather than taken
quietly here.
"""

from __future__ import annotations

import pytest

from jarvis_core.memory.base import MemoryScope
from jarvis_core.memory.benchmark import (
    DEFAULT_BAR,
    PRECISION_FLOOR,
    RECALL_FLOOR,
    SHOULD_NOT_FIND,
    matches,
    run,
)
from jarvis_core.memory.manager import MemoryManager, RetrievalBudget
from jarvis_core.text import stem


def found(query: str, memory: str) -> bool:
    """Whether ``memory`` clears the retrieval bar for ``query``."""
    return matches(query, memory) >= DEFAULT_BAR


def test_recall_on_realistic_queries_does_not_regress():
    result = run()
    missed = [f"{c.difficulty}: {c.query!r}" for c in result.missed]
    assert len(result.found) >= RECALL_FLOOR, (
        f"recall fell to {len(result.found)}; missing {missed}"
    )


def test_every_remaining_miss_needs_a_synonym_rather_than_a_suffix():
    """The evidence that the morphology work is done and the rest is not.

    If a suffix case ever appears here again, the stemmer regressed. If this
    test has to be relaxed to admit a new kind, that kind is the argument for
    whatever is being proposed next.
    """
    assert {case.difficulty for case in result_misses()} <= {"synonym", "irregular verb"}


def result_misses():
    return run().missed


def test_no_unrelated_pair_is_matched():
    """Precision, on the fixture that stops recall being bought with noise."""
    result = run()
    assert result.false_positives == []
    assert result.precision >= PRECISION_FLOOR


def test_turkish_possessives_reach_their_stem():
    """Five of the six worst misses were this one cause."""
    for inflected, base in [
        ("arabam", "araba"),
        ("hedefim", "hedef"),
        ("projelerimi", "proje"),
        ("bütçemiz", "bütçe"),
        ("müşterilerden", "müşteri"),
        ("raporu", "raporlar"),
    ]:
        assert stem(inflected) == stem(base), f"{inflected} did not meet {base}"


def test_english_nominalisations_reach_their_verb():
    assert stem("deployment") == stem("deploy")
    assert stem("pricing") == stem("price")


@pytest.mark.parametrize("case", SHOULD_NOT_FIND, ids=lambda c: c.query[:24])
def test_an_unrelated_memory_is_still_not_returned(case):
    """The counterweight. Recall bought by matching everything is worth nothing."""
    assert not found(case.query, case.memory)


def test_a_short_word_is_never_eaten_down_to_a_prefix():
    """Without the length guard every three-letter word collides with every other."""
    for word in ("bus", "less", "iş", "veri", "para", "kısa", "test", "cost"):
        assert len(stem(word)) >= min(len(word), 3)


def test_stemming_is_stable_under_repetition():
    """A stem that keeps shrinking would make two runs of the same query differ."""
    for word in ("projelerimi", "müşterilerden", "deployment", "bookkeeping"):
        once = stem(word)
        assert stem(once) == once


def test_the_manager_finds_an_inflected_turkish_memory():
    """End to end, through the real retrieval path rather than the arithmetic."""
    memory = MemoryManager()
    memory.remember(
        "hedef aylık 5000 dolar gelir",
        scope=MemoryScope.USER,
        source="operator",
        importance=0.9,
    )
    recalled = memory.recall("hedefim ne durumda", budget=RetrievalBudget())
    assert recalled, "an inflected query found nothing"
    assert "hedef" in recalled[0].item.content


def test_the_manager_still_returns_nothing_for_an_unrelated_question():
    memory = MemoryManager()
    memory.remember(
        "hedef aylık 5000 dolar gelir",
        scope=MemoryScope.USER,
        source="operator",
        importance=0.9,
    )
    assert memory.recall("what font should the invoice use", budget=RetrievalBudget()) == []
