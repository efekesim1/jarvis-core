"""Finding claims an answer cannot support.

"Hallucination" is not a thing you can test for in general. What you *can*
test for is narrower and still worth a great deal: an answer that states a
specific, checkable fact and cites nothing.

That is what this looks for. A percentage, a price, a date, a named quantity —
anything a reader would take as researched — appearing in an answer that
brought back no sources. It is a deterministic check over the text, not a model
judging another model, which means it is cheap, repeatable, and wrong in
predictable ways.

What it will miss: a confident wrong sentence with no numbers in it. What it
will not do: guess. A flagged claim is a sentence the reader can go and look at,
never a score.
"""

from __future__ import annotations

import re

# Shapes a reader treats as researched rather than as opinion.
_CHECKABLE = (
    (re.compile(r"\b\d+(?:[.,]\d+)?\s?%"), "a percentage"),
    (re.compile(r"[$€£₺]\s?\d[\d,.]*\s?(?:[kmb]|million|billion|bin|milyon)?\b", re.I), "a price"),
    (
        re.compile(
            r"\b\d[\d,.]*\s?(?:million|billion|thousand|users|customers|milyon|bin)\b", re.I
        ),
        "a quantity",
    ),
    (re.compile(r"\b(?:19|20)\d{2}\b"), "a year"),
    (re.compile(r"\bversion\s+\d+(?:\.\d+)*\b", re.I), "a version number"),
)

# Sentences that carry a number but are not claims about the world.
_HEDGED = re.compile(
    r"\b(?:unknown|not found|could not|unable to|no data|unverified|estimate[ds]?|"
    r"assuming|roughly|approximately|bilinmiyor|bulunamadı|tahmin)\b",
    re.I,
)

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def unsupported_claims(text: str, *, sources: int, limit: int = 5) -> list[str]:
    """Sentences making a checkable claim in an answer that cited nothing.

    With at least one source the check does not fire: judging *which* sentence
    a citation supports needs to read the source, which this deliberately does
    not do. Zero sources and a hard number is the case worth flagging, and it
    is the case that actually happens.
    """
    if sources > 0 or not text.strip():
        return []

    found: list[str] = []
    for sentence in _SENTENCE.split(text):
        sentence = sentence.strip()
        if not sentence or _HEDGED.search(sentence):
            continue
        for pattern, kind in _CHECKABLE:
            if pattern.search(sentence):
                found.append(f"{kind}, uncited: {sentence[:140]}")
                break
        if len(found) >= limit:
            break
    return found
