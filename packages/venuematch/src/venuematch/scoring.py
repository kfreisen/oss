"""Deciding how alike two records are.

The default scorer is a weighted composite, and the term that does the most work
is the one people leave out: a **penalty on disagreeing proper nouns**.

Generic string similarity rates "Iowa primary winner" and "New Hampshire primary
winner" as nearly identical — they share almost every token, and the tokens they
do not share are the only ones that carry the meaning. Any purely
frequency-agnostic metric gets this wrong, and it gets it wrong in the worst
direction, merging two distinct entities into one cluster.

Subtracting for the symmetric difference of capitalized words fixes the common
case cheaply, because in most sources the distinguishing part of a description is
a name.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rapidfuzz import fuzz

if TYPE_CHECKING:
    from collections.abc import Iterable

    from venuematch.records import Record

__all__ = [
    "CompositeScorer",
    "PairScore",
    "jaccard",
    "normalize",
    "proper_nouns",
    "tokenize",
]

DEFAULT_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "will",
        "by",
        "in",
        "on",
        "of",
        "be",
        "is",
        "to",
        "for",
        "and",
        "or",
        "vs",
        "at",
        "with",
        "from",
    }
)

_PUNCTUATION = str.maketrans(dict.fromkeys(string.punctuation, " "))

# A capitalized word of four or more letters. Read off the *original* text, since
# normalization destroys the capitalization this depends on.
_PROPER_NOUN = re.compile(r"\b([A-Z][A-Za-z]{3,})\b")

# Capitalized words that begin a question or sentence and are not names. Without
# excluding these, "Will Acme ship?" and "Does Acme ship?" are penalized for
# disagreeing on "Will" versus "Does", which is exactly backwards.
SENTENCE_OPENERS = frozenset(
    {
        "Will",
        "Does",
        "What",
        "When",
        "Where",
        "Who",
        "How",
        "Should",
        "Are",
        "Have",
        "Was",
        "Were",
        "Did",
        "Could",
        "Would",
        "Might",
        "Shall",
        "Has",
        "The",
        "Over",
        "Under",
        "Yes",
        "This",
        "That",
        "Their",
        "There",
    }
)


def normalize(text: str | None, stopwords: frozenset[str] = DEFAULT_STOPWORDS) -> str:
    """Lowercase, strip punctuation, drop stopwords, collapse whitespace."""
    if not text:
        return ""
    cleaned = text.translate(_PUNCTUATION).lower()
    return " ".join(t for t in cleaned.split() if t and t not in stopwords)


def tokenize(text: str | None, min_length: int = 2) -> frozenset[str]:
    """Token set from normalized text, dropping very short tokens."""
    if not text:
        return frozenset()
    return frozenset(t for t in normalize(text).split() if len(t) >= min_length)


def proper_nouns(text: str | None, openers: frozenset[str] = SENTENCE_OPENERS) -> frozenset[str]:
    """Capitalized words that look like names, read off the original text."""
    if not text:
        return frozenset()
    return frozenset(t for t in _PROPER_NOUN.findall(text) if t not in openers)


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Intersection over union of two token sets. Zero if both are empty."""
    left, right = frozenset(a), frozenset(b)
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


@dataclass(frozen=True, slots=True)
class PairScore:
    """Why two records were judged alike.

    Every component is reported, not just the total. Tuning weights against a
    single opaque number is guesswork; seeing which term carried a bad match is
    what makes the threshold adjustable on evidence.

    Attributes:
        confidence: Composite score in `[0, 1]`.
        fuzz_ratio: rapidfuzz `token_set_ratio`, 0-100.
        jaccard: Token-set Jaccard.
        proper_noun_diff: Size of the symmetric difference of proper nouns.
        time_delta_days: Absolute difference in timestamps, or None.
    """

    confidence: float
    fuzz_ratio: float
    jaccard: float
    proper_noun_diff: int
    time_delta_days: float | None


@dataclass(frozen=True, slots=True)
class CompositeScorer:
    """The default scorer.

    Cheap prefilters run first and return `None`, so the expensive terms are only
    computed for pairs with a chance. In a block of any size most pairs die on the
    Jaccard prefilter.

    Attributes:
        jaccard_floor: Below this token overlap, reject without scoring.
        fuzz_floor: Below this rapidfuzz ratio, reject.
        max_time_delta_days: Beyond this separation, reject. `None` disables the
            time term entirely, for records with no meaningful timestamp.
        fuzz_weight: Weight on the fuzz ratio.
        jaccard_weight: Weight on token overlap.
        time_weight: Weight on timestamp proximity.
        proper_noun_penalty: Subtracted per disagreeing proper noun. The term
            that does the most work; set it to zero and "Iowa primary" merges
            with "New Hampshire primary".
        value_tolerance: Numeric agreement within this counts as corroboration.
        value_bonus: Added when values agree that closely. Deliberately small —
            two different entities of the same kind often carry similar numbers.
    """

    jaccard_floor: float = 0.30
    fuzz_floor: float = 75.0
    max_time_delta_days: float | None = 14.0
    fuzz_weight: float = 0.60
    jaccard_weight: float = 0.30
    time_weight: float = 0.10
    proper_noun_penalty: float = 0.12
    value_tolerance: float = 0.05
    value_bonus: float = 0.03

    def __call__(self, a: Record, b: Record) -> PairScore | None:
        """Score a pair, or return None if it fails a prefilter."""
        norm_a, norm_b = normalize(a.text), normalize(b.text)
        if not norm_a or not norm_b:
            return None

        overlap = jaccard(norm_a.split(), norm_b.split())
        if overlap < self.jaccard_floor:
            return None

        ratio = float(fuzz.token_set_ratio(norm_a, norm_b))
        if ratio < self.fuzz_floor:
            return None

        delta_days: float | None = None
        time_term = 0.0
        if self.max_time_delta_days is not None:
            moment_a, moment_b = a.utc_timestamp(), b.utc_timestamp()
            if moment_a is None or moment_b is None:
                return None
            delta_days = abs((moment_a - moment_b).total_seconds()) / 86400.0
            if delta_days > self.max_time_delta_days:
                return None
            time_term = 1.0 - delta_days / self.max_time_delta_days

        noun_diff = len(proper_nouns(a.text) ^ proper_nouns(b.text))

        bonus = 0.0
        if (
            a.value is not None
            and b.value is not None
            and abs(a.value - b.value) <= self.value_tolerance
        ):
            bonus = self.value_bonus

        confidence = (
            self.fuzz_weight * (ratio / 100.0)
            + self.jaccard_weight * overlap
            + self.time_weight * time_term
            - self.proper_noun_penalty * noun_diff
            + bonus
        )
        return PairScore(
            confidence=min(1.0, max(0.0, confidence)),
            fuzz_ratio=ratio,
            jaccard=overlap,
            proper_noun_diff=noun_diff,
            time_delta_days=delta_days,
        )
