"""GameEngine: daily lifecycle, guesses, scoring, streaks, leaderboard.

Pure Python — never imports discord, performs I/O only via db/snippets/catalog.

This module currently implements the DAILY LIFECYCLE (this feature):
``ensure_today_challenge`` (idempotent per (guild, local date), no-repeat
song selection with history reset, deterministic seeded song+offset picks,
catalog auto-bootstrap, snippet cache re-heal), ``get_reveal`` (mark the
previous challenge revealed, return song + winners in solve order), and
``skip_today_song`` (pinned decision #5). The gameplay methods
(``unlock_snippet``, ``submit_guess``, ``leaderboard``) are added by the
engine-gameplay-matching feature on top of this class.

Determinism: song and offset are drawn from ``random.Random`` seeded by
``sha256(date | guild_id | skip_count)`` — stable across processes (never
Python's salted ``hash()``). All wall-clock reads are injected via ``now``;
a naive ``now`` is interpreted as UTC.
"""

from __future__ import annotations

import hashlib
import logging
import random
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal, Protocol, cast
from zoneinfo import ZoneInfo

from songbot.catalog import CatalogSource, Song, refresh_catalog
from songbot.catalog.refresh import RefreshResult
from songbot.config import Settings
from songbot.db import ChallengeRow, ChallengeStatus, Database, SongRow

__all__ = [
    "CatalogEmptyError",
    "Challenge",
    "EngineError",
    "GameEngine",
    "Reveal",
    "SkipRefusedError",
    "SkipRefusedReason",
    "SnippetService",
    "Winner",
]

logger = logging.getLogger(__name__)

SkipRefusedReason = Literal["no_challenge", "revealed", "solved"]
"""Why `skip_today_song` refused (pinned decision #5)."""


class EngineError(Exception):
    """Base class for engine-level failures."""


class CatalogEmptyError(EngineError):
    """No songs available even after the auto-refresh bootstrap (pinned #11).

    The harness maps this to ``{"error": "catalog_empty"}``; the message
    always contains the ``catalog_empty`` marker.
    """


class SkipRefusedError(EngineError):
    """skip_today_song refused with zero state mutation (pinned #5).

    ``reason`` is ``"no_challenge"`` (nothing posted today), ``"revealed"``
    (today's challenge is no longer active) or ``"solved"`` (at least one
    user already solved it).
    """

    def __init__(self, reason: SkipRefusedReason, message: str) -> None:
        super().__init__(message)
        self.reason: SkipRefusedReason = reason


class SnippetService(Protocol):
    """The snippet operations the engine needs (satisfied by SnippetGenerator).

    Declared as a protocol so tests can inject a fast fake without ffmpeg.
    """

    def ensure_snippets(
        self,
        song: Song,
        challenge_id: int | str,
        offset: float,
        lengths: Sequence[float],
    ) -> dict[int, Path]:
        """Idempotently ensure all snippet levels exist; return their paths."""
        ...

    def purge_challenge(self, challenge_id: int | str) -> None:
        """Delete a challenge's snippet cache dir and section intermediates."""
        ...


@dataclass(frozen=True)
class Challenge:
    """A guild's daily challenge with its song and snippet set.

    ``date`` is the ISO calendar date in the configured timezone.
    ``created`` is True iff THIS call inserted the row (False means the
    existing challenge was reused — the harness's idempotent re-post signal).
    """

    id: int
    guild_id: str
    channel_id: str
    date: str
    song: SongRow
    snippet_offset_sec: float
    status: ChallengeStatus
    skip_count: int
    created_at: str
    revealed_at: str | None
    snippet_paths: dict[int, Path]
    created: bool


@dataclass(frozen=True)
class Winner:
    """One solver of a revealed challenge."""

    user_id: str
    guesses_used: int
    points_awarded: int
    solved_at: str


@dataclass(frozen=True)
class Reveal:
    """The reveal of a previous challenge: song identity + winners.

    ``winners`` is in solve order (``solved_at`` ascending). Empty when
    nobody solved ("nobody got it").
    """

    challenge_id: int
    date: str
    song: SongRow
    winners: tuple[Winner, ...]
    revealed_at: str


def _seed(date_str: str, guild_id: str, skip_count: int) -> int:
    """Deterministic seed for a day's picks: hash(date, guild_id, skip_count).

    sha256 (never the salted builtin ``hash()``) so picks are reproducible
    across processes and restarts; bumping ``skip_count`` changes both the
    song and the offset deterministically (pinned #5).
    """
    digest = hashlib.sha256(f"{date_str}|{guild_id}|{skip_count}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _song_for_generator(row: SongRow) -> Song:
    """Adapt a stored song row to the catalog `Song` the generator consumes."""
    return Song(
        source=cast("CatalogSource", row.source),
        source_id=row.source_id,
        title=row.title,
        artist=row.artist,
        duration_sec=row.duration_sec,
        audio_ref=row.audio_ref,
        raw_title=row.raw_title,
    )


class GameEngine:
    """The pure game engine. Owns all game rules; never imports discord.

    Args:
        db: the (migrated) database.
        settings: validated configuration (timezone, snippet ladder, ...).
        snippets: snippet generator (or a test fake satisfying the protocol).
        catalog_refresher: test seam for the pinned-#11 bootstrap; defaults to
            ``refresh_catalog(db, settings)``.
    """

    def __init__(
        self,
        db: Database,
        settings: Settings,
        snippets: SnippetService,
        *,
        catalog_refresher: Callable[[], RefreshResult] | None = None,
    ) -> None:
        self._db = db
        self._settings = settings
        self._snippets = snippets
        self._catalog_refresher: Callable[[], RefreshResult] = (
            catalog_refresher
            if catalog_refresher is not None
            else lambda: refresh_catalog(self._db, self._settings)
        )

    # -- time helpers --------------------------------------------------------

    @property
    def _tz(self) -> ZoneInfo:
        return self._settings.tz

    def _local_date(self, now: datetime) -> date:
        """The calendar date of ``now`` in the configured timezone."""
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        return now.astimezone(self._tz).date()

    @staticmethod
    def _utc_iso(now: datetime) -> str:
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        return now.astimezone(UTC).isoformat()

    # -- row helpers ---------------------------------------------------------

    def _challenge_row(self, guild_id: str, date_str: str) -> ChallengeRow | None:
        row = self._db.query_one(
            "SELECT * FROM challenges WHERE guild_id = ? AND date = ?",
            (guild_id, date_str),
        )
        return ChallengeRow.from_row(row) if row is not None else None

    def _song_row(self, song_id: int) -> SongRow:
        row = self._db.query_one("SELECT * FROM songs WHERE id = ?", (song_id,))
        if row is None:  # FK guarantees presence; defensive only
            raise EngineError(f"challenge references missing song id {song_id}")
        return SongRow.from_row(row)

    def _songs_empty(self) -> bool:
        row = self._db.query_one("SELECT COUNT(*) AS c FROM songs")
        return row is None or int(row["c"]) == 0

    # -- song selection --------------------------------------------------------

    def _select_song(
        self,
        guild_id: str,
        rng: random.Random,
        *,
        exclude_song_id: int | None = None,
    ) -> SongRow | None:
        """Pick a song for `guild_id`: no repeats until exhausted, then reset.

        Eligible = catalog songs never used by this guild; when empty, the
        history resets and all songs are eligible again. ``exclude_song_id``
        (skip-song) additionally bars the just-skipped song from the re-pick
        whenever any alternative exists. Returns None iff the catalog is empty.
        """
        songs = [
            SongRow.from_row(row)
            for row in self._db.query("SELECT * FROM songs ORDER BY id")
        ]
        if not songs:
            return None
        used = {
            int(row["song_id"])
            for row in self._db.query(
                "SELECT DISTINCT song_id FROM challenges WHERE guild_id = ?", (guild_id,)
            )
        }
        eligible = [song for song in songs if song.id not in used]
        if not eligible:
            eligible = songs  # catalog exhausted for this guild: history resets
        if exclude_song_id is not None:
            narrowed = [song for song in eligible if song.id != exclude_song_id]
            if narrowed:
                eligible = narrowed
        return eligible[rng.randrange(len(eligible))]

    def _draw_offset(
        self,
        rng: random.Random,
        song: SongRow,
        *,
        exclude_offset: float | None = None,
    ) -> float:
        """Draw the daily snippet offset: uniform in [0, duration - max_len].

        ``exclude_offset`` (skip-song) re-draws on an exact float collision so
        the replacement challenge always has a different offset (VAL-ADMIN-003);
        the seeded RNG keeps the re-draw deterministic.
        """
        max_len = max(self._settings.snippet_lengths)
        max_offset = max(0.0, song.duration_sec - max_len)
        offset = rng.uniform(0.0, max_offset)
        while exclude_offset is not None and offset == exclude_offset:
            offset = rng.uniform(0.0, max_offset)
        return offset

    # -- daily lifecycle -------------------------------------------------------

    def ensure_today_challenge(
        self, guild_id: str, channel_id: str, now: datetime
    ) -> Challenge:
        """Ensure today's challenge exists for a guild; return it.

        Idempotent via ``UNIQUE(guild_id, date)``: a repeat call returns the
        existing challenge unchanged (``created=False``) and never inserts a
        second row. The date is the LOCAL date of ``now`` in the configured
        timezone. Every call runs ``ensure_snippets`` (pinned #14), so a
        deleted snippet cache is regenerated even for an existing row.

        When the ``songs`` table is empty the catalog is refreshed first
        (pinned #11 bootstrap); a still-empty catalog raises
        ``CatalogEmptyError`` without inserting any row. A snippet-generation
        failure likewise leaves no row behind.
        """
        date_str = self._local_date(now).isoformat()

        existing = self._challenge_row(guild_id, date_str)
        if existing is not None:
            return self._build_challenge(existing, created=False)

        if self._songs_empty():
            result = self._catalog_refresher()
            for source in result.sources:
                if source.ok:
                    logger.info(
                        "catalog bootstrap (%s): +%d added, %d updated",
                        source.source,
                        source.added,
                        source.updated,
                    )
                else:
                    logger.warning("catalog bootstrap (%s) failed: %s", source.source, source.error)

        rng = random.Random(_seed(date_str, guild_id, skip_count=0))
        song = self._select_song(guild_id, rng)
        if song is None:
            raise CatalogEmptyError(
                "catalog_empty: no songs in the catalog (auto-refresh found none)"
            )
        offset = self._draw_offset(rng, song)

        try:
            cursor = self._db.execute(
                "INSERT INTO challenges"
                " (guild_id, channel_id, song_id, date, snippet_offset_sec, status,"
                " created_at, skip_count) VALUES (?, ?, ?, ?, ?, 'active', ?, 0)",
                (guild_id, channel_id, song.id, date_str, offset, self._utc_iso(now)),
            )
        except sqlite3.IntegrityError:
            # Lost a race with a concurrent ensure for the same (guild, date):
            # the UNIQUE constraint held; reuse the winner's row.
            logger.info("challenge for (%s, %s) created concurrently; reusing", guild_id, date_str)
            winner = self._challenge_row(guild_id, date_str)
            if winner is None:  # pragma: no cover - defensive
                raise EngineError("concurrent challenge insert vanished") from None
            return self._build_challenge(winner, created=False)

        assert cursor.lastrowid is not None
        row = self._challenge_row(guild_id, date_str)
        assert row is not None
        try:
            return self._build_challenge(row, created=True)
        except BaseException:
            # Never leave a snippet-less challenge row behind.
            self._db.execute("DELETE FROM challenges WHERE id = ?", (row.id,))
            raise

    def get_reveal(self, guild_id: str, now: datetime) -> Reveal | None:
        """Reveal the previous active challenge for a guild, if any.

        Marks every stale active challenge (local date before today)
        ``revealed`` exactly once and returns a `Reveal` for the most recent
        one: its song plus winners in solve order. Returns None when there is
        nothing to reveal (never reveals today's own challenge, so it is safe
        to call before OR after today's post).
        """
        today = self._local_date(now).isoformat()
        stale = [
            ChallengeRow.from_row(row)
            for row in self._db.query(
                "SELECT * FROM challenges"
                " WHERE guild_id = ? AND status = 'active' AND date < ?"
                " ORDER BY date DESC, id DESC",
                (guild_id, today),
            )
        ]
        if not stale:
            return None

        revealed_at = self._utc_iso(now)
        with self._db.transaction():
            for row in stale:
                self._db.execute(
                    "UPDATE challenges SET status = 'revealed', revealed_at = ?"
                    " WHERE id = ? AND status = 'active'",
                    (revealed_at, row.id),
                )

        target = stale[0]
        winners = tuple(
            Winner(
                user_id=str(row["user_id"]),
                guesses_used=int(row["guesses_used"]),
                points_awarded=int(row["points_awarded"]),
                solved_at=str(row["solved_at"]),
            )
            for row in self._db.query(
                "SELECT user_id, guesses_used, points_awarded, solved_at"
                " FROM challenge_users WHERE challenge_id = ? AND solved = 1"
                " ORDER BY solved_at, user_id",
                (target.id,),
            )
        )
        return Reveal(
            challenge_id=target.id,
            date=target.date,
            song=self._song_row(target.song_id),
            winners=winners,
            revealed_at=revealed_at,
        )

    def skip_today_song(self, guild_id: str, now: datetime) -> Challenge:
        """Replace today's song with a new one (pinned decision #5).

        Refused with zero mutation (``SkipRefusedError``) when there is no
        challenge today, when it is already revealed, or when any user has
        solved it. Otherwise: the challenge row is DELETED (challenge_users
        and guesses cascade-delete), its snippet cache is purged, and a NEW
        row for the same date is created with a deterministically different
        song and offset (seed = hash(date, guild_id, skip_count + 1)).
        """
        date_str = self._local_date(now).isoformat()
        old = self._challenge_row(guild_id, date_str)
        if old is None:
            raise SkipRefusedError(
                "no_challenge", f"no challenge to skip for {date_str} in guild {guild_id}"
            )
        if old.status != "active":
            raise SkipRefusedError(
                "revealed", f"challenge for {date_str} is already revealed; cannot skip"
            )
        solver = self._db.query_one(
            "SELECT 1 FROM challenge_users WHERE challenge_id = ? AND solved = 1 LIMIT 1",
            (old.id,),
        )
        if solver is not None:
            raise SkipRefusedError(
                "solved", f"challenge for {date_str} already has a solver; cannot skip"
            )

        skip_count = old.skip_count + 1
        rng = random.Random(_seed(date_str, guild_id, skip_count))
        song = self._select_song(guild_id, rng, exclude_song_id=old.song_id)
        if song is None:  # pragma: no cover - a posted challenge implies songs exist
            raise CatalogEmptyError("catalog_empty: no songs in the catalog")
        offset = self._draw_offset(rng, song, exclude_offset=old.snippet_offset_sec)

        self._db.execute("DELETE FROM challenges WHERE id = ?", (old.id,))
        self._snippets.purge_challenge(old.id)

        cursor = self._db.execute(
            "INSERT INTO challenges"
            " (guild_id, channel_id, song_id, date, snippet_offset_sec, status,"
            " created_at, skip_count) VALUES (?, ?, ?, ?, ?, 'active', ?, ?)",
            (guild_id, old.channel_id, song.id, date_str, offset, self._utc_iso(now), skip_count),
        )
        assert cursor.lastrowid is not None
        row = self._challenge_row(guild_id, date_str)
        assert row is not None
        try:
            return self._build_challenge(row, created=True)
        except BaseException:
            # The old row is already gone; do not leave a snippet-less row.
            self._db.execute("DELETE FROM challenges WHERE id = ?", (row.id,))
            raise

    def refresh_catalog(self) -> RefreshResult:
        """Passthrough for the admin reload-catalog command."""
        return self._catalog_refresher()

    # -- assembly --------------------------------------------------------------

    def _build_challenge(self, row: ChallengeRow, *, created: bool) -> Challenge:
        """Materialize a `Challenge`: join the song and re-heal the snippet cache."""
        song = self._song_row(row.song_id)
        paths = self._snippets.ensure_snippets(
            _song_for_generator(song),
            row.id,
            row.snippet_offset_sec,
            self._settings.snippet_lengths,
        )
        return Challenge(
            id=row.id,
            guild_id=row.guild_id,
            channel_id=row.channel_id,
            date=row.date,
            song=song,
            snippet_offset_sec=row.snippet_offset_sec,
            status=row.status,
            skip_count=row.skip_count,
            created_at=row.created_at,
            revealed_at=row.revealed_at,
            snippet_paths=paths,
            created=created,
        )
