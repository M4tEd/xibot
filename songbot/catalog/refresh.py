"""refresh_catalog: combined multi-provider catalog upsert (pinned decisions #10-12).

Behavior (binding per architecture.md):

- **Enabled providers** (pinned #10): the local provider is enabled iff
  ``settings.local_music_dir`` is set, the YouTube provider iff
  ``settings.youtube_playlist_url`` is set. Disabled providers do not
  participate at all — their existing rows are left untouched, as are rows
  of any source no enabled provider reports on.
- **Upsert** (pinned #12): every fetched song is written with
  ``INSERT ... ON CONFLICT(source, source_id) DO UPDATE``, so row ids stay
  stable and changed metadata (title/artist/duration/audio_ref/raw_title)
  updates in place; ``created_at`` keeps its original value.
- **Removal** (pinned #12): rows whose ``(source, source_id)`` no longer
  appears in a *successfully fetched* provider's list have vanished and are
  deleted — but ONLY if no ``challenges`` row references them; referenced
  rows are retained for history (counted as ``retained``).
- **Failure isolation** (pinned #12): each provider is fetched and stored
  independently, in its own transaction. A failing provider records a clear
  named error (``"<ExceptionName>: <message>"``) in its per-source result and
  never rolls back, deletes, or corrupts another source's rows. A failed
  source's own rows are also left exactly as they were: no removal pass runs
  without a successful fetch.
- **Overrides**: after a source's upsert pass, its `song_overrides` rows
  (admin metadata corrections via /songbot-fixsong) are re-applied inside the
  same transaction, so a refresh never clobbers an admin correction.

Per-guild catalogs: a guild whose ``guild_settings.playlist_url`` is set
(/songbot-playlist) refreshes from THAT YouTube playlist only, into its own
``songs.guild_id`` scope; every other guild refreshes the global env-configured
sources into the shared ``''`` scope. Scopes never touch each other's rows:
upserts key on ``(guild_id, source, source_id)`` and the removal pass only
deletes within the refreshed scope.

The return value is a `RefreshResult` with one `SourceRefresh` per enabled
provider (added/updated/removed/retained counts, or the error).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from songbot.catalog import CatalogProvider, Song
from songbot.catalog.local import LocalDirectoryProvider
from songbot.catalog.youtube import YouTubePlaylistProvider
from songbot.config import Settings
from songbot.db import Database

__all__ = ["RefreshResult", "SourceRefresh", "refresh_catalog"]

logger = logging.getLogger(__name__)

_UPSERT_SQL = """
INSERT INTO songs
    (guild_id, source, source_id, title, artist, duration_sec, audio_ref, raw_title,
     created_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (guild_id, source, source_id) DO UPDATE SET
    title = excluded.title,
    artist = excluded.artist,
    duration_sec = excluded.duration_sec,
    audio_ref = excluded.audio_ref,
    raw_title = excluded.raw_title
"""
# `id` and `created_at` are deliberately absent from the UPDATE: row ids stay
# stable and the original insertion timestamp is preserved.

_REFERENCED_SQL = "SELECT 1 FROM challenges WHERE song_id = ? LIMIT 1"
_DELETE_SQL = "DELETE FROM songs WHERE id = ?"

_OVERRIDES_FOR_SOURCE_SQL = (
    "SELECT source_id, title, artist FROM song_overrides WHERE source = ?"
)
_APPLY_OVERRIDE_SQL = (
    "UPDATE songs SET title = ?, artist = ? WHERE source = ? AND source_id = ?"
)


@dataclass(frozen=True)
class SourceRefresh:
    """The per-source outcome of a catalog refresh.

    ``added``/``updated``/``removed`` count this run's inserts, in-place
    updates, and deletions of vanished songs. ``retained`` counts vanished
    songs kept because a challenge still references them. ``error`` is
    ``"<ExceptionName>: <message>"`` when the source failed (fetch or store);
    all counts are then 0 and the source's stored rows are untouched.
    """

    source: str
    added: int = 0
    updated: int = 0
    removed: int = 0
    retained: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when the source refreshed without an error."""
        return self.error is None


@dataclass(frozen=True)
class RefreshResult:
    """The combined result of `refresh_catalog`: one entry per enabled provider."""

    sources: tuple[SourceRefresh, ...]

    @property
    def ok(self) -> bool:
        """True when every enabled source refreshed without an error."""
        return all(source.ok for source in self.sources)

    def by_source(self, source: str) -> SourceRefresh:
        """Return the result for `source` (raises `KeyError` if not refreshed)."""
        for result in self.sources:
            if result.source == source:
                return result
        raise KeyError(f"source {source!r} was not part of this refresh")


def refresh_catalog(
    db: Database,
    settings: Settings,
    *,
    guild_id: str | None = None,
    providers: Mapping[str, CatalogProvider] | None = None,
) -> RefreshResult:
    """Refresh the ``songs`` table from a guild's effective catalog providers.

    When ``guild_id`` names a guild with a custom ``playlist_url``
    (/songbot-playlist), ONLY that YouTube playlist is refreshed, into the
    guild's own ``songs.guild_id`` scope. Otherwise the providers are built
    from ``settings`` (local directory and/or YouTube playlist, per pinned
    decision #10) and refresh the shared global scope (``''``). Tests may
    inject stub providers keyed by source name via ``providers``. Every
    provider is attempted even when others fail; per-source outcomes are
    reported in the returned `RefreshResult` (failures are returned, not
    raised).
    """
    playlist_url = _guild_playlist_url(db, guild_id)
    scope = guild_id if playlist_url is not None and guild_id is not None else ""
    if providers is not None:
        active = providers
    elif playlist_url is not None:
        active = {"youtube": YouTubePlaylistProvider(playlist_url)}
    else:
        active = _build_providers(settings)
    if not active:
        logger.warning("refresh_catalog: no catalog providers enabled")
    results = tuple(
        _refresh_source(db, source, provider, scope) for source, provider in active.items()
    )
    return RefreshResult(sources=results)


def _guild_playlist_url(db: Database, guild_id: str | None) -> str | None:
    """A guild's custom playlist URL, or None when it uses the global catalog."""
    if guild_id is None:
        return None
    row = db.query_one(
        "SELECT playlist_url FROM guild_settings WHERE guild_id = ?", (guild_id,)
    )
    if row is None or row["playlist_url"] is None or not str(row["playlist_url"]).strip():
        return None
    return str(row["playlist_url"])


def _build_providers(settings: Settings) -> dict[str, CatalogProvider]:
    """Construct the enabled providers from settings (pinned decision #10)."""
    providers: dict[str, CatalogProvider] = {}
    if settings.local_music_dir is not None:
        providers["local"] = LocalDirectoryProvider(settings.local_music_dir)
    if settings.youtube_playlist_url is not None:
        providers["youtube"] = YouTubePlaylistProvider(settings.youtube_playlist_url)
    return providers


def _refresh_source(
    db: Database, source: str, provider: CatalogProvider, scope: str
) -> SourceRefresh:
    """Fetch one provider and store its songs; failures become a named error."""
    try:
        songs = provider.fetch()
    except Exception as exc:
        return _source_error(source, exc)
    try:
        with db.transaction():
            return _upsert_source(db, source, songs, scope)
    except Exception as exc:
        return _source_error(source, exc)


def _source_error(source: str, exc: Exception) -> SourceRefresh:
    error = f"{type(exc).__name__}: {exc}"
    logger.error("Catalog refresh failed for source %r: %s", source, error)
    return SourceRefresh(source=source, error=error)


def _upsert_source(
    db: Database, source: str, songs: Sequence[Song], scope: str
) -> SourceRefresh:
    """Upsert `songs` and delete vanished unreferenced rows for `source`.

    Only rows in the catalog ``scope`` (``''`` = the global pool, otherwise a
    guild id) are touched: another scope's rows for the same source are
    invisible to both the upsert and the removal pass. Must run inside a
    transaction so a source's changes commit or roll back as a unit (failure
    isolation, pinned #12).
    """
    now = datetime.now(UTC).isoformat()
    existing: dict[str, int] = {
        str(row["source_id"]): int(row["id"])
        for row in db.query(
            "SELECT id, source_id FROM songs WHERE source = ? AND guild_id = ?",
            (source, scope),
        )
    }

    fetched: set[str] = set()
    added = updated = 0
    for song in songs:
        if song.source_id in fetched:
            continue  # defensive: providers already dedupe; first occurrence wins
        fetched.add(song.source_id)
        if song.source_id in existing:
            updated += 1
        else:
            added += 1
        db.execute(
            _UPSERT_SQL,
            (
                scope,
                song.source,
                song.source_id,
                song.title,
                song.artist,
                song.duration_sec,
                song.audio_ref,
                song.raw_title,
                now,
            ),
        )

    removed = retained = 0
    for source_id, row_id in existing.items():
        if source_id in fetched:
            continue
        if db.query_one(_REFERENCED_SQL, (row_id,)) is None:
            db.execute(_DELETE_SQL, (row_id,))
            removed += 1
        else:
            retained += 1

    _apply_overrides(db, source)

    result = SourceRefresh(
        source=source, added=added, updated=updated, removed=removed, retained=retained
    )
    logger.info("Catalog source %r refreshed: %s", source, result)
    return result


def _apply_overrides(db: Database, source: str) -> None:
    """Re-apply admin metadata overrides (/songbot-fixsong) for `source`.

    Runs inside the source's refresh transaction, after the upsert pass, so
    the provider's metadata never clobbers an admin correction. Overrides are
    keyed by ``(source, source_id)`` with no catalog scope: a correction is
    provider-truth, so it applies to the matching row in EVERY scope (global
    and per-guild) at once. Overrides for songs absent from the table
    (deleted, or not yet re-added) match zero rows and simply wait for the
    song to re-enter the catalog.
    """
    for row in db.query(_OVERRIDES_FOR_SOURCE_SQL, (source,)):
        cursor = db.execute(
            _APPLY_OVERRIDE_SQL, (row["title"], row["artist"], source, row["source_id"])
        )
        if cursor.rowcount:
            logger.info(
                "Re-applied metadata override for %s/%s", source, row["source_id"]
            )
