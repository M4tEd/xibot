"""Artist/title parsing heuristics shared by the catalog providers.

``parse_artist_title`` turns a raw video title or filename stem into an
``(artist, title)`` pair. The heuristics are deliberately simple and
documented as imperfect (the skip-song admin command exists for bad parses):

1. Strip bracket decorations anywhere in the string: ``[...]``, ``(...)`` and
   full-width ``【...】`` (e.g. "[Official]", "(Official Video)", album tags).
2. Split on the FIRST ``" - "`` (spaced hyphen) as ``"Artist - Title"``.
3. Otherwise split on the LAST ``" / "`` (spaced slash) as ``"Title / Artist"``.
4. Otherwise the whole cleaned string is the title and the artist is ``None``.
5. Normalize featuring markers in the results: ``ft``/``ft.``/``feat``/
   ``featuring`` all become ``feat.``.

If decoration stripping would leave nothing, the original raw string (stripped
of surrounding whitespace) is kept so the title is never empty for non-empty
input.
"""

from __future__ import annotations

import re

__all__ = ["parse_artist_title"]

_BRACKET_PATTERNS = (
    re.compile(r"\[[^\[\]]*\]"),
    re.compile(r"\([^()]*\)"),
    re.compile(r"【[^【】]*】"),
)
_WHITESPACE_RE = re.compile(r"\s+")
_FEAT_RE = re.compile(r"\b(?:featuring|feat|ft)\.?(?=\s|$)", re.IGNORECASE)

_DASH_SEPARATOR = " - "
_SLASH_SEPARATOR = " / "


def parse_artist_title(raw: str) -> tuple[str | None, str]:
    """Parse a raw title/filename stem into ``(artist, title)``.

    Returns ``(None, title)`` when no artist heuristic applies. The returned
    title is non-empty whenever ``raw`` contains non-whitespace characters.
    """
    text = _strip_decorations(raw)
    if not text:
        text = raw.strip()

    artist: str | None = None
    title = text
    if _DASH_SEPARATOR in text:
        left, _, right = text.partition(_DASH_SEPARATOR)
        if left.strip() and right.strip():
            artist, title = left.strip(), right.strip()
    elif _SLASH_SEPARATOR in text:
        left, _, right = text.rpartition(_SLASH_SEPARATOR)
        if left.strip() and right.strip():
            artist, title = right.strip(), left.strip()

    title = _normalize_feat(title)
    if artist is not None:
        artist = _normalize_feat(artist)
    return artist, title


def _strip_decorations(text: str) -> str:
    """Remove all bracket groups (repeatedly, for nested ones) and squish space."""
    previous = None
    while previous != text:
        previous = text
        for pattern in _BRACKET_PATTERNS:
            text = pattern.sub(" ", text)
    return _collapse_whitespace(text)


def _normalize_feat(text: str) -> str:
    """Normalize featuring markers (``ft.``/``featuring``/...) to ``feat.``."""
    return _collapse_whitespace(_FEAT_RE.sub("feat.", text))


def _collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()
