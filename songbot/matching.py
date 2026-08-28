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

When the whole-string ratio falls short, a per-token fallback applies: a
field also matches when EVERY one of its tokens has a close twin among the
guess tokens (``fuzz.ratio`` >= threshold, or exactly one Levenshtein edit
apart for tokens of ``_SHORT_TOKEN_MIN_LEN``+ characters). ``token_set_ratio``
only rewards exact-token subsets, so without this fallback a single typo in
one token ("Bohemian Rhapsody Quen", "Halp") collapses the score below the
threshold — which used to cost players the both-fields bonus on combined
title+artist guesses and made short titles/artists tolerate zero typos.

``match_guess``'s ``mode`` narrows what counts as correct: ``"either"``
(default) accepts the title OR the artist, ``"title"`` only the title, and
``"artist"`` only the artist. The field-level flags
(``matched_title``/``matched_artist``) always report the raw outcomes; the
mode only re-derives ``is_correct`` and forces ``is_both`` off, so the 1.5x
both-fields bonus exists only in ``"either"`` mode.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal, Protocol

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

__all__ = [
    "MATCH_THRESHOLD",
    "GuessMatchMode",
    "MatchResult",
    "SongLike",
    "match_guess",
    "normalize",
]

GuessMatchMode = Literal["title", "artist", "either"]
"""What counts as a correct guess (configured via ``GUESS_MATCH_MODE``)."""

MATCH_THRESHOLD = 85
"""Minimum post-normalization ``token_set_ratio`` for a guess to match."""

_SHORT_TOKEN_MIN_LEN = 4
"""Minimum token length for the per-token one-edit allowance.

Shorter tokens (1-3 chars, e.g. "U2") stay effectively exact: a single edit
on them changes too much of the token to forgive.
"""

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

    ``matched_title``/``matched_artist`` always report the raw field-level
    outcomes. ``is_correct`` applies the match mode on top (``"either"``:
    title OR artist; ``"title"``: title only; ``"artist"``: artist only).
    ``is_both`` (both fields in one guess, ``"either"`` mode only) gates the
    1.5x bonus. A ``raw_title`` fallback rescue is credited as
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


def _token_close(a: str, b: str, threshold: int) -> bool:
    """Per-token closeness: ``fuzz.ratio`` >= ``threshold``, or one edit apart.

    The one-edit allowance (substitution, insertion, or deletion) only
    applies to tokens of ``_SHORT_TOKEN_MIN_LEN``+ characters — it exists for
    short tokens like "halo"/"halp", whose ratio (75.0) falls below the
    threshold for a single typo while longer tokens pass it.
    """
    if fuzz.ratio(a, b) >= threshold:
        return True
    return (
        min(len(a), len(b)) >= _SHORT_TOKEN_MIN_LEN
        and Levenshtein.distance(a, b) == 1
    )


def _tokens_cover(guess_tokens: list[str], candidate_tokens: list[str], threshold: int) -> bool:
    """Per-token fallback: every candidate token has a close twin in the guess.

    Unlike ``token_set_ratio``'s all-or-nothing exact-token intersection,
    this tolerates a typo in one token of a field — including when the guess
    carries extra tokens (a combined title+artist guess).
    """
    return all(
        any(_token_close(candidate_token, guess_token, threshold) for guess_token in guess_tokens)
        for candidate_token in candidate_tokens
    )


def match_guess(
    guess: str,
    song: SongLike,
    *,
    threshold: int = MATCH_THRESHOLD,
    mode: GuessMatchMode = "either",
) -> MatchResult:
    """Score ``guess`` against ``song``'s title, artist, and raw_title.

    The guess and every candidate are normalized first; the threshold applies
    to the normalized strings (``>= threshold``). An empty-after-normalization
    guess never matches. A candidate that misses the whole-string threshold
    can still match via the per-token fallback (`_tokens_cover`). The
    ``raw_title`` fallback is consulted only when neither the parsed title
    nor the parsed artist matched.

    ``mode`` narrows the verdict without touching the field-level flags:
    ``"either"`` (default) counts the title OR the artist as correct and
    allows the both-fields bonus; ``"title"`` counts only the title and
    ``"artist"`` only the artist, and in both restricted modes ``is_both``
    is always False (no 1.5x bonus).
    """
    normalized = normalize(guess)
    if not normalized:
        return MatchResult(
            matched_title=False, matched_artist=False, is_correct=False, is_both=False
        )

    def matches(candidate: str) -> bool:
        if not candidate:
            return False
        if fuzz.token_set_ratio(normalized, candidate) >= threshold:
            return True
        return _tokens_cover(normalized.split(), candidate.split(), threshold)

    matched_title = matches(normalize(song.title))
    matched_artist = matches(normalize(song.artist or ""))
    if not (matched_title or matched_artist) and matches(normalize(song.raw_title)):
        matched_title = True  # raw_title rescue: credited as a title match
    if mode == "title":
        is_correct = matched_title
    elif mode == "artist":
        is_correct = matched_artist
    else:
        is_correct = matched_title or matched_artist
    is_both = mode == "either" and matched_title and matched_artist
    return MatchResult(
        matched_title=matched_title,
        matched_artist=matched_artist,
        is_correct=is_correct,
        is_both=is_both,
    )
