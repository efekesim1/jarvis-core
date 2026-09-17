"""Pulling the meaningful words out of a piece of text.

Both memory retrieval and the digital twin decide relevance by word overlap,
and both had their own copy of this. That was fine until a bug showed up in
one of them: an unrelated question matched a stored preference because they
shared the word "the". Fixing it in one place would have left the other wrong,
which is the argument for there being one place.

The stopword list covers English and Turkish, because the operator writes in both and a
term list that drops "the" but keeps "bir" is only half a filter.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"\w{3,}", re.UNICODE)

# Words too common to say anything about relevance. Kept as prose rather than
# as a list literal because a hundred quoted strings is not readable, and this
# list is meant to be edited by whoever notices a bad match.
_ENGLISH_STOPWORDS = """
the and but for that this these those with from was were are being have has
had not yes what which who whom whose when where why how its than then all any
can could did does doing done each few here into more most much nor now off
once only other our out over own same should some such very will would you your
they them their there through under until while about above after again against
because been before below between both during further
"""

_TURKISH_STOPWORDS = """
ama için ile veya değil daha çok kadar gibi ise bir bu şu şey olan olarak nasıl
neden hangi kim ne var yok ancak fakat sonra önce diye böyle şöyle çünkü ayrıca
zaten sadece hem her bazı kendi
"""

STOPWORDS: frozenset[str] = frozenset(_ENGLISH_STOPWORDS.split() + _TURKISH_STOPWORDS.split())


# Suffixes stripped before comparing two words. This is not a stemmer and does
# not try to be one: it exists because "how long should the report be" found
# nothing against a trait recorded as "short reports", which is a silly way to
# miss.
#
# Turkish is here in earnest rather than as an afterthought, because the operator writes
# in it and Turkish is agglutinative: "projelerimi" is proje + ler + im + i,
# four morphemes deep, and a matcher that only knows English plurals finds
# nothing for it. Suffixes are therefore stripped *repeatedly* rather than once.
# Measured on a fixture of seventeen realistic queries, this took recall from
# 6 to 12; the five it still misses need a synonym, not a suffix, and that is
# recorded in KNOWN_ISSUES rather than papered over here.
#
# Each entry carries the shortest word it may be taken from. The guard matters
# more than the list: without it "bus" becomes "bu" and every three-letter word
# collides with every other. Single-vowel suffixes carry a higher guard still,
# because they are the ones that would otherwise merge unrelated short words.
_SUFFIXES: tuple[tuple[str, int], ...] = (
    # English
    ("ments", 7),
    ("ment", 6),
    ("ness", 6),
    ("ies", 5),
    ("ing", 5),
    ("ed", 5),
    ("es", 5),
    # Turkish: plural, then possessive, then case. Longest first within a
    # group so "lerin" is not read as "in" over a stem ending in "ler".
    ("larımız", 9),
    ("lerimiz", 9),
    ("ları", 6),
    ("leri", 6),
    ("ların", 7),
    ("lerin", 7),
    ("lardan", 8),
    ("lerden", 8),
    ("larda", 7),
    ("lerde", 7),
    ("lar", 5),
    ("ler", 5),
    ("ımız", 6),
    ("imiz", 6),
    ("umuz", 6),
    ("ümüz", 6),
    ("mız", 5),
    ("miz", 5),
    ("muz", 5),
    ("müz", 5),
    ("ınız", 6),
    ("iniz", 6),
    ("dan", 6),
    ("den", 6),
    ("tan", 6),
    ("ten", 6),
    ("nın", 6),
    ("nin", 6),
    ("nun", 6),
    ("nün", 6),
    ("da", 5),
    ("de", 5),
    ("ta", 5),
    ("te", 5),
    ("ın", 5),
    ("in", 5),
    ("un", 5),
    ("ün", 5),
    ("ım", 5),
    ("im", 5),
    ("um", 5),
    ("üm", 5),
    ("sı", 5),
    ("si", 5),
    ("su", 5),
    ("sü", 5),
    ("yı", 5),
    ("yi", 5),
    ("ya", 5),
    ("ye", 5),
    # Single characters last, and only from a word long enough that dropping
    # one cannot make it another word.
    ("m", 6),
    ("s", 5),
    ("ı", 6),
    ("i", 6),
    ("u", 6),
    ("ü", 6),
    ("a", 5),
    ("e", 5),
)

# How many times to go round. Turkish stacks three or four morphemes in
# ordinary speech ("projelerimi"); beyond four the stem is being eaten rather
# than uncovered.
_MAX_ROUNDS = 4


def stem(word: str) -> str:
    """A crude common form for ``word``, for comparing it with another word.

    Applied to both sides of every comparison, so what matters is that it is
    consistent, not that it is linguistically correct. "araba" and "arabam"
    both reducing to "arab" is a success even though "arab" is not a word.
    """
    for _ in range(_MAX_ROUNDS):
        for suffix, floor in _SUFFIXES:
            if len(word) >= floor and word.endswith(suffix):
                # "ies" -> "y": stories, categories, dependencies.
                word = word[: -len(suffix)]
                if suffix == "ies":
                    word += "y"
                break
        else:
            break
    return word


def terms(text: str) -> set[str]:
    """The words in ``text`` that could carry meaning, in a comparable form."""
    return {stem(w) for w in (m.lower() for m in _WORD.findall(text)) if w not in STOPWORDS}


def overlap_ratio(a: str, b: str) -> float:
    """How much two pieces of text have in common, 0 to 1.

    Normalised by the smaller of the two term sets, so a short phrase can match
    a long one fully. Normalising by the query alone punished long questions
    for being long, which is not what relevance means.
    """
    left, right = terms(a), terms(b)
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))
