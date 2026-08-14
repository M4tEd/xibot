"""Guess normalization + rapidfuzz fuzzy matching.

``normalize`` folds a guess or song field to a canonical comparison form:
NFKD unicode-fold to ASCII (Beyoncé -> beyonce, CJK brackets handled),
lowercase, bracketed sections (``(...)``, ``[...]``, ``{...}``, ``【...】``)
removed, punctuation collapsed to whitespace, and noise tokens (``feat`` /
``ft`` / ``featuring`` / ``remaster`` / ``remastered``) dropped.

``match_guess`` compares the normalized guess with rapidfuzz
``fuzz.token_set_ratio`` against the normalized title, artist, and — as a
fallback for poorly-parsed YouTube entries — ``raw_title``. A score of
``MATCH_THRESHOLD`` (85) or higher counts as a match; ``token_set_ratio``
makes word-order, subset ("Bohemian" vs "Bohemian Rhapsody"), and "The "
prefix differences immaterial. Title OR artist match is correct; both in one
guess is the 1.5x-bonus condition (``is_both``).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol

from rapidfuzz import fuzz

__all__ = ["MATCH_THRESHOLD", "MatchResult", "SongLike", "match_guess", "normalize"]

MATCH_THRESHOLD = 85
"""Minimum post-normalization ``token_set_ratio`` for a guess to match."""

_BRACKETED_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]|\{[^}]*\}|【[^】]*】")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_NOISE_TOKENS = frozenset({"feat", "ft", "featuring", "remaster", "remastered"})


class SongLike(Protocol):
    """The fields matching needs (satisfied by catalog `Song` and `SongRow`).

    Read-only properties so frozen dataclasses (whose attributes are not
    settable) satisfy the protocol.
    """

    @property
    def title(self) -> str: ...

    @property
    def artist(self) -> str | None: ...

    @property
    def raw_title(self) -> str: ...


@dataclass(frozen=True)
class MatchResult:
    """How a guess scored against a song.

    ``is_correct`` is title OR artist; ``is_both`` (both in one guess) gates
    the 1.5x bonus. A ``raw_title`` fallback rescue is credited as
    ``matched_title`` only — it never fabricates a both-match bonus.
    """

    matched_title: bool
    matched_artist: bool
    is_correct: bool
    is_both: bool


def normalize(text: str) -> str:
    """Fold ``text`` to its canonical comparison form.

    Steps: NFKD decomposition, bracketed-section removal (before the ASCII
    fold so CJK brackets survive to be removed), ASCII fold (drops combining
    marks and any remaining non-ASCII), lowercase, punctuation to whitespace,
    noise-token removal, whitespace collapse.
    """
    text = unicodedata.normalize("NFKD", text)
    text = _BRACKETED_RE.sub(" ", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = _NON_ALNUM_RE.sub(" ", text.lower())
    tokens = [token for token in text.split() if token not in _NOISE_TOKENS]
    return " ".join(tokens)


def match_guess(
    guess: str, song: SongLike, *, threshold: int = MATCH_THRESHOLD
) -> MatchResult:
    """Score ``guess`` against ``song``'s title, artist, and raw_title.

    The guess and every candidate are normalized first; the threshold applies
    to the normalized strings (``>= threshold``). An empty-after-normalization
    guess never matches. The ``raw_title`` fallback is consulted only when
    neither the parsed title nor the parsed artist matched.
    """
    normalized = normalize(guess)
    if not normalized:
        return MatchResult(
            matched_title=False, matched_artist=False, is_correct=False, is_both=False
        )

    def matches(candidate: str) -> bool:
        return bool(candidate) and fuzz.token_set_ratio(normalized, candidate) >= threshold

    matched_title = matches(normalize(song.title))
    matched_artist = matches(normalize(song.artist or ""))
    if not (matched_title or matched_artist) and matches(normalize(song.raw_title)):
        matched_title = True  # raw_title rescue: credited as a title match
    is_correct = matched_title or matched_artist
    return MatchResult(
        matched_title=matched_title,
        matched_artist=matched_artist,
        is_correct=is_correct,
        is_both=matched_title and matched_artist,
    )
