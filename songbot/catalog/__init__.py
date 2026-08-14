"""Catalog providers: Song dataclass, CatalogProvider protocol, combined fetch.

This package root defines the shared contract every provider implements and
re-exports the combined ``refresh_catalog`` upsert (implemented in
``songbot.catalog.refresh``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

__all__ = [
    "CatalogProvider",
    "CatalogSource",
    "RefreshResult",
    "Song",
    "SourceRefresh",
    "refresh_catalog",
]

CatalogSource = Literal["local", "youtube"]
"""Known catalog source identifiers."""


@dataclass(frozen=True)
class Song:
    """A single catalog entry, normalized across providers.

    ``source_id`` is provider-scoped and unique per ``source`` (relative path
    for local files, video id for YouTube). ``audio_ref`` is an absolute file
    path (local) or watch URL (youtube). ``raw_title`` is the original
    filename stem / video title, kept for fallback matching. ``artist`` may be
    ``None`` when no heuristic could extract one (bare YouTube titles).
    """

    source: CatalogSource
    source_id: str
    title: str
    artist: str | None
    duration_sec: float
    audio_ref: str
    raw_title: str


@runtime_checkable
class CatalogProvider(Protocol):
    """A source of songs. Implementations must be side-effect-free fetchers."""

    def fetch(self) -> list[Song]:
        """Return the full song list for this source (empty list if none)."""
        ...


# Re-exported last: songbot.catalog.refresh imports Song/CatalogProvider from
# this module, so the names above must exist before the import runs.
from songbot.catalog.refresh import (  # noqa: E402
    RefreshResult,
    SourceRefresh,
    refresh_catalog,
)
