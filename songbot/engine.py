"""GameEngine: daily lifecycle, guesses, scoring, streaks, leaderboard.

Pure Python — never imports discord, performs I/O only via db/snippets/catalog.

Daily lifecycle: ``ensure_today_challenge`` (idempotent per (guild, local
date), no-repeat song selection with history reset, deterministic seeded
song+offset picks, catalog auto-bootstrap, snippet cache re-heal, bounded
auto-skip of unsnippable fresh picks — issue #11), the
pinned-#17 delivery-coupled reveal split — ``peek_reveal`` (read-only:
compute the stale challenge's song + winners in solve order, zero mutation)
and ``mark_revealed`` (the mutation the caller applies ONLY after the reveal
announcement send succeeds) — ``skip_today_song`` (pinned decision #5), and
``delete_challenge`` (the pinned-#16 delivery-failure rollback).

Gameplay: ``unlock_snippet`` (per-user snippet ladder 0..4 with descending
point potential), ``submit_guess`` (fuzzy matching, scoring with the pinned
round-half-up both-bonus, guess log, wins/streaks — all atomic; guesses
past the daily limit are still processed, and a post-limit solve banks a
flat ``POST_LIMIT_SOLVE_POINTS`` with no win/streak), and ``leaderboard``
(total_points DESC, wins DESC, user_id ASC, scoring users
only via ``total_points > 0``; effective streak computed on read, pinned #7).
The first interaction (hear-more or a processed guess — the pinned-#13
``challenge_users`` upsert point) also registers a zero-valued ``user_stats``
row (VAL-SCORE-005), which the leaderboard filter keeps unlisted until the
first solve. Both gameplay entry points refuse revealed challenges with zero
mutation (the VAL-GUESS-019 lockout: ``challenge_closed`` /
``UnlockRefusedError(reason="closed")``).

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
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal, Protocol, cast
from zoneinfo import ZoneInfo

from songbot.catalog import CatalogSource, Song, refresh_catalog
from songbot.catalog.refresh import RefreshResult
from songbot.config import Settings
from songbot.db import (
    ChallengeRow,
    ChallengeStatus,
    ChallengeUserRow,
    Database,
    GuildSettingsRow,
    PingRoleRow,
    SongRow,
    UserStatsRow,
)
from songbot.matching import match_guess
from songbot.snippets import SnippetError

__all__ = [
    "MAX_AUTO_SKIPS",
    "POST_LIMIT_SOLVE_POINTS",
    "CatalogEmptyError",
    "Challenge",
    "EngineError",
    "FixSongRefusedError",
    "FixSongRefusedReason",
    "GameEngine",
    "GuessOutcome",
    "GuessResult",
    "LeaderboardEntry",
    "Reveal",
    "SkipRefusedError",
    "SkipRefusedReason",
    "SnippetService",
    "SongFix",
    "UnlockRefusedError",
    "UnlockRefusedReason",
    "UnlockResult",
    "Winner",
]

logger = logging.getLogger(__name__)

MAX_AUTO_SKIPS = 3
"""Automatic song replacements per `ensure_today_challenge` call (issue #11).

When snippet generation fails for a FRESHLY created challenge (e.g. a
googlevideo 403 on the day's pick), the next deterministic pick is tried
instead of retrying the identical song+offset on every scheduler tick until
a manual skip. The bound caps the songs one call burns through; if all fail,
the last error propagates and the caller's normal retry cadence applies.
"""

SkipRefusedReason = Literal["no_challenge", "revealed", "solved"]
"""Why `skip_today_song` refused (pinned decision #5)."""

FixSongRefusedReason = Literal["no_challenge", "invalid_date", "blank_title"]
"""Why `fix_song_metadata` refused (zero mutation)."""

UnlockRefusedReason = Literal["solved", "max_level", "closed"]
"""Why `unlock_snippet` refused: the challenge is closed (no longer active),
the user already solved, or the user is at max level."""

POST_LIMIT_SOLVE_POINTS = 10
"""What a correct guess past the daily guess limit banks (flat, no bonus,
no win/streak) — the player keeps playing after ``max_guesses_per_day``."""

GuessOutcome = Literal[
    "correct", "correct_after_limit", "wrong", "already_solved", "empty",
    "challenge_closed",
]
"""The result of a `submit_guess` submission.

``challenge_closed`` is the revealed-challenge lockout (VAL-GUESS-019): the
challenge is no longer ``active``, so the submission is refused with zero
mutation. ``empty`` is the pinned-#15 validation rejection (empty after
stripping): never counted, never logged. The ``already_solved`` rejection is
likewise not logged and does not consume a guess (pinned #13). Guesses past
the daily limit are processed like normal ones: a wrong one is counted and
logged as ``wrong``, and a correct one is ``correct_after_limit`` — it still
marks the user solved (reveal winner, public announcement) but banks a flat
``POST_LIMIT_SOLVE_POINTS`` with no win/streak.
"""


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


class UnlockRefusedError(EngineError):
    """unlock_snippet refused with zero state mutation.

    ``reason`` is ``"closed"`` (the challenge is no longer active — revealed;
    VAL-GUESS-019), ``"solved"`` (the user already solved this challenge) or
    ``"max_level"`` (the user already unlocked the longest snippet).
    """

    def __init__(self, reason: UnlockRefusedReason, message: str) -> None:
        super().__init__(message)
        self.reason: UnlockRefusedReason = reason


class FixSongRefusedError(EngineError):
    """fix_song_metadata refused with zero state mutation.

    ``reason`` is ``"no_challenge"`` (the guild has no challenge to fix —
    none at all, or none on the requested date), ``"invalid_date"`` (the
    ``date`` argument is not an ISO ``YYYY-MM-DD`` date), or
    ``"blank_title"`` (the corrected title is empty after stripping).
    """

    def __init__(self, reason: FixSongRefusedReason, message: str) -> None:
        super().__init__(message)
        self.reason: FixSongRefusedReason = reason


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
class SongFix:
    """The record of an admin metadata correction (`fix_song_metadata`).

    ``old_*``/``new_*`` let the admin ack show exactly what changed. The
    correction targets the song of one of the guild's challenges (latest by
    default, or the ``date``-selected one) and is persisted both on the
    ``songs`` row (effective immediately for new guesses and future reveals)
    and as a ``song_overrides`` row (re-applied by every catalog refresh, so
    the provider's metadata never clobbers it). Already-recorded guesses are
    NOT re-scored.
    """

    song_id: int
    challenge_id: int
    challenge_date: str
    old_title: str
    old_artist: str | None
    new_title: str
    new_artist: str | None


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
    nobody solved ("nobody got it"). ``guild_id``/``channel_id`` come from
    the challenge row itself, so the reveal is delivered to the channel the
    challenge was posted in even if the guild's configured channel changed
    since.
    """

    challenge_id: int
    guild_id: str
    channel_id: str
    date: str
    song: SongRow
    winners: tuple[Winner, ...]
    revealed_at: str


@dataclass(frozen=True)
class UnlockResult:
    """A successful Hear-more: the newly unlocked snippet level.

    ``path`` is the snippet file for ``level``; ``potential_points`` is what
    a solve would pay at this level (the ladder value, before any bonus).
    """

    level: int
    path: Path
    potential_points: int


@dataclass(frozen=True)
class GuessResult:
    """The outcome of one ``submit_guess`` submission.

    ``guesses_used``/``guesses_left`` reflect the state AFTER the submission
    (unchanged for the ``already_solved``/``empty`` rejections);
    ``guesses_left`` clamps at 0 once the daily limit is exhausted.
    ``points_awarded`` is non-zero only for ``correct`` and
    ``correct_after_limit`` (the flat ``POST_LIMIT_SOLVE_POINTS``).
    ``snippet_level`` is the user's snippet level when the submission was
    processed — for ``correct``, the level the solver was actually hearing at
    solve time (the ladder rung that scored), which the public solve
    announcement reports as a snippet length. ``announce`` is True exactly
    for the solving guess — per-user solves are singular, so every correct
    guess is the user's first (pinned: one public announcement per solve).
    """

    outcome: GuessOutcome
    matched_title: bool
    matched_artist: bool
    is_both: bool
    guesses_used: int
    guesses_left: int
    points_awarded: int
    snippet_level: int
    announce: bool


@dataclass(frozen=True)
class LeaderboardEntry:
    """One row of a guild leaderboard.

    ``current_streak`` is the EFFECTIVE streak (pinned #7): 0 when the
    user's last win is more than one calendar day before the read date.
    """

    user_id: str
    total_points: int
    wins: int
    current_streak: int


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

    def _challenge_row_by_id(self, challenge_id: int) -> ChallengeRow:
        row = self._db.query_one(
            "SELECT * FROM challenges WHERE id = ?", (challenge_id,)
        )
        if row is None:
            raise EngineError(f"unknown challenge id {challenge_id}")
        return ChallengeRow.from_row(row)

    def _challenge_user_row(
        self, challenge_id: int, user_id: str
    ) -> ChallengeUserRow | None:
        row = self._db.query_one(
            "SELECT * FROM challenge_users WHERE challenge_id = ? AND user_id = ?",
            (challenge_id, user_id),
        )
        return ChallengeUserRow.from_row(row) if row is not None else None

    def _song_row(self, song_id: int) -> SongRow:
        row = self._db.query_one("SELECT * FROM songs WHERE id = ?", (song_id,))
        if row is None:  # FK guarantees presence; defensive only
            raise EngineError(f"challenge references missing song id {song_id}")
        return SongRow.from_row(row)

    def _songs_empty(self) -> bool:
        row = self._db.query_one("SELECT COUNT(*) AS c FROM songs")
        return row is None or int(row["c"]) == 0

    # -- guild configuration (multi-guild post targets) ------------------------

    def set_guild_channel(
        self, guild_id: str, channel_id: str, *, set_by: str, now: datetime
    ) -> GuildSettingsRow:
        """Upsert a guild's daily-post channel; return the stored row.

        Used by /songbot-setup and by the client's env bootstrap seed. A
        re-configuration keeps the original ``created_at`` and refreshes
        ``updated_at``. Existing challenge rows keep the channel they were
        posted to — only future posts use the new channel.
        """
        iso_now = self._utc_iso(now)
        with self._db.transaction():
            self._db.execute(
                "INSERT INTO guild_settings"
                " (guild_id, channel_id, set_by, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(guild_id) DO UPDATE SET"
                " channel_id = excluded.channel_id, set_by = excluded.set_by,"
                " updated_at = excluded.updated_at",
                (guild_id, channel_id, set_by, iso_now, iso_now),
            )
        row = self.guild_settings(guild_id)
        if row is None:  # pragma: no cover - the upsert just wrote it
            raise EngineError(f"guild_settings upsert lost guild {guild_id}")
        return row

    def guild_settings(self, guild_id: str) -> GuildSettingsRow | None:
        """A guild's configured post target, or None when not set up."""
        row = self._db.query_one(
            "SELECT * FROM guild_settings WHERE guild_id = ?", (guild_id,)
        )
        return GuildSettingsRow.from_row(row) if row is not None else None

    def all_guild_settings(self) -> list[GuildSettingsRow]:
        """Every configured guild's post target (the scheduler's work list)."""
        return [
            GuildSettingsRow.from_row(row)
            for row in self._db.query(
                "SELECT * FROM guild_settings ORDER BY guild_id"
            )
        ]

    def remove_guild_settings(self, guild_id: str) -> None:
        """Drop a guild's configuration (the bot left the guild).

        Challenge/score history is KEPT — re-adding the bot and re-running
        /songbot-setup resumes the guild's game where it left off. The
        guild's `ping_role_settings` row (if any) cascades away with the
        `guild_settings` row.
        """
        self._db.execute("DELETE FROM guild_settings WHERE guild_id = ?", (guild_id,))

    # -- ping-role configuration (reaction-role opt-in) -----------------------

    def set_ping_role(
        self,
        guild_id: str,
        channel_id: str,
        message_id: str,
        role_id: str,
        emoji: str,
        *,
        set_by: str,
        now: datetime,
    ) -> PingRoleRow:
        """Upsert a guild's reaction-role opt-in config; return the stored row.

        Used by /songbot-pingrole after the announcement message is posted
        (``message_id`` is the announcement the reaction listeners watch). A
        re-configuration keeps the original ``created_at``, refreshes
        ``updated_at``, and supersedes the previous announcement (reactions
        on the old message no longer match any watched message id).
        """
        iso_now = self._utc_iso(now)
        with self._db.transaction():
            self._db.execute(
                "INSERT INTO ping_role_settings"
                " (guild_id, channel_id, message_id, role_id, emoji, set_by,"
                " created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(guild_id) DO UPDATE SET"
                " channel_id = excluded.channel_id, message_id = excluded.message_id,"
                " role_id = excluded.role_id, emoji = excluded.emoji,"
                " set_by = excluded.set_by, updated_at = excluded.updated_at",
                (guild_id, channel_id, message_id, role_id, emoji, set_by, iso_now, iso_now),
            )
        row = self.ping_role_settings(guild_id)
        if row is None:  # pragma: no cover - the upsert just wrote it
            raise EngineError(f"ping_role_settings upsert lost guild {guild_id}")
        return row

    def ping_role_settings(self, guild_id: str) -> PingRoleRow | None:
        """A guild's reaction-role opt-in config, or None when not set up."""
        row = self._db.query_one(
            "SELECT * FROM ping_role_settings WHERE guild_id = ?", (guild_id,)
        )
        return PingRoleRow.from_row(row) if row is not None else None

    def ping_role_for_message(self, message_id: str) -> PingRoleRow | None:
        """The opt-in config watching ``message_id``, or None.

        The reaction listeners dispatch on the reacted message's id: only
        reactions on a configured announcement (and only with the configured
        emoji) grant or revoke the role.
        """
        row = self._db.query_one(
            "SELECT * FROM ping_role_settings WHERE message_id = ?", (message_id,)
        )
        return PingRoleRow.from_row(row) if row is not None else None

    def latest_challenge_id(self, guild_id: str) -> int | None:
        """The id of the guild's most recent challenge (persistent-view binding)."""
        row = self._db.query_one(
            "SELECT id FROM challenges WHERE guild_id = ?"
            " ORDER BY date DESC, id DESC LIMIT 1",
            (guild_id,),
        )
        return int(row["id"]) if row is not None else None

    # -- song selection --------------------------------------------------------

    def _select_song(
        self,
        guild_id: str,
        rng: random.Random,
        *,
        exclude_song_ids: Collection[int] = (),
    ) -> SongRow | None:
        """Pick a song for `guild_id`: no repeats until exhausted, then reset.

        Eligible = catalog songs never used by this guild; when empty, the
        history resets and all songs are eligible again. ``exclude_song_ids``
        (skip-song / auto-skip) additionally bars those songs from the pick
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
        if exclude_song_ids:
            narrowed = [song for song in eligible if song.id not in exclude_song_ids]
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

        Auto-skip (issue #11): when snippet generation fails for a FRESHLY
        created row (``SnippetError`` — e.g. a googlevideo 403 on the day's
        pick), the row is deleted and the next deterministic pick
        (``skip_count`` + 1, failed songs excluded) is tried instead, up to
        ``MAX_AUTO_SKIPS`` times per call — the scheduler otherwise retries
        the identical song+offset every 60s until a manual skip. The
        surviving row carries the ``skip_count`` it took, so a later manual
        skip continues the same deterministic chain. Only fresh rows are
        auto-skipped: a pre-existing row's re-heal failure propagates
        untouched (that challenge may already be posted; ``skip_today_song``
        is the remedy there). If every attempt fails, the last
        ``SnippetError`` propagates.
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

        last_error: SnippetError | None = None
        tried: set[int] = set()
        for skip_count in range(MAX_AUTO_SKIPS + 1):
            rng = random.Random(_seed(date_str, guild_id, skip_count))
            song = self._select_song(guild_id, rng, exclude_song_ids=tried)
            if song is None:
                if last_error is None:
                    raise CatalogEmptyError(
                        "catalog_empty: no songs in the catalog (auto-refresh found none)"
                    )
                break  # catalog exhausted by failures; raise the last snippet error
            if song.id in tried:
                # Exclusion is best-effort — a tiny catalog may offer no
                # untried alternative; re-trying a just-failed song in the
                # same call cannot succeed.
                break
            offset = self._draw_offset(rng, song)

            try:
                cursor = self._db.execute(
                    "INSERT INTO challenges"
                    " (guild_id, channel_id, song_id, date, snippet_offset_sec, status,"
                    " created_at, skip_count) VALUES (?, ?, ?, ?, ?, 'active', ?, ?)",
                    (
                        guild_id,
                        channel_id,
                        song.id,
                        date_str,
                        offset,
                        self._utc_iso(now),
                        skip_count,
                    ),
                )
            except sqlite3.IntegrityError:
                # Lost a race with a concurrent ensure for the same (guild, date):
                # the UNIQUE constraint held; reuse the winner's row.
                logger.info(
                    "challenge for (%s, %s) created concurrently; reusing", guild_id, date_str
                )
                winner = self._challenge_row(guild_id, date_str)
                if winner is None:  # pragma: no cover - defensive
                    raise EngineError("concurrent challenge insert vanished") from None
                return self._build_challenge(winner, created=False)

            assert cursor.lastrowid is not None
            row = self._challenge_row(guild_id, date_str)
            assert row is not None
            try:
                return self._build_challenge(row, created=True)
            except SnippetError as exc:
                # Auto-skip: this song cannot be snippeted right now; delete
                # the fresh row and try the next deterministic pick.
                self._db.execute("DELETE FROM challenges WHERE id = ?", (row.id,))
                tried.add(song.id)
                last_error = exc
                logger.warning(
                    "auto-skip %d/%d for guild %s on %s: snippet generation failed "
                    "for song %r; trying the next deterministic pick (%s)",
                    skip_count + 1,
                    MAX_AUTO_SKIPS,
                    guild_id,
                    date_str,
                    song.title,
                    exc,
                )
            except BaseException:
                # Never leave a snippet-less challenge row behind.
                self._db.execute("DELETE FROM challenges WHERE id = ?", (row.id,))
                raise
        assert last_error is not None  # every break/loop-exit path sets it
        raise last_error

    def _stale_active_rows(self, guild_id: str, today: str) -> list[ChallengeRow]:
        """Active challenges dated before ``today`` (ISO), most recent first."""
        return [
            ChallengeRow.from_row(row)
            for row in self._db.query(
                "SELECT * FROM challenges"
                " WHERE guild_id = ? AND status = 'active' AND date < ?"
                " ORDER BY date DESC, id DESC",
                (guild_id, today),
            )
        ]

    def peek_reveal(self, guild_id: str, now: datetime) -> Reveal | None:
        """Compute the previous challenge's reveal WITHOUT marking it (pinned #17).

        The read-only half of the delivery-coupled reveal: returns a `Reveal`
        for the most recent stale active challenge (local date before today)
        — its song plus winners in solve order — and mutates NOTHING, so a
        failed reveal send leaves the challenge active and the next
        tick/advance-day re-peeks the identical reveal. ``revealed_at`` is the
        timestamp `mark_revealed` will persist for the same ``now``. Returns
        None when there is nothing to reveal: already-revealed rows are
        invisible (a delivered reveal is never computed again) and today's own
        challenge is never revealed, so it is safe to call before OR after
        today's post.
        """
        today = self._local_date(now).isoformat()
        stale = self._stale_active_rows(guild_id, today)
        if not stale:
            return None

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
            guild_id=target.guild_id,
            channel_id=target.channel_id,
            date=target.date,
            song=self._song_row(target.song_id),
            winners=winners,
            revealed_at=self._utc_iso(now),
        )

    def mark_revealed(self, guild_id: str, now: datetime) -> None:
        """Mark every stale active challenge revealed (pinned #17 commit half).

        Applied by the reveal flows (scheduler tick, harness advance-day)
        ONLY after the reveal announcement send succeeds. Marks ALL stale
        active rows (local date before today) — including un-announced older
        rows from missed cycles, exactly as the pre-split reveal did — in one
        transaction, with ``revealed_at`` matching the peeked `Reveal` for the
        same ``now``. A no-op when nothing is stale.
        """
        today = self._local_date(now).isoformat()
        revealed_at = self._utc_iso(now)
        with self._db.transaction():
            for row in self._stale_active_rows(guild_id, today):
                self._db.execute(
                    "UPDATE challenges SET status = 'revealed', revealed_at = ?"
                    " WHERE id = ? AND status = 'active'",
                    (revealed_at, row.id),
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
        song = self._select_song(guild_id, rng, exclude_song_ids={old.song_id})
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

    def fix_song_metadata(
        self,
        guild_id: str,
        *,
        title: str,
        artist: str | None = None,
        date_str: str | None = None,
        set_by: str,
        now: datetime,
    ) -> SongFix:
        """Correct the title/artist of a challenge's song (/songbot-fixsong).

        Targets the song of the guild's MOST RECENT challenge, or of the
        ``date_str``-selected challenge (ISO ``YYYY-MM-DD``, the challenge's
        local date) when given. ``artist`` omitted (None) keeps the current
        artist; a blank-after-strip ``artist`` clears it to None. Refused
        with zero mutation (``FixSongRefusedError``) when the title is blank,
        the date is malformed, or the guild has no such challenge.

        The correction writes the ``songs`` row (effective immediately:
        ``submit_guess`` re-reads the row per guess, and future reveals use
        it) and upserts a ``song_overrides`` row in the same transaction, so
        every later catalog refresh re-applies it (refresh.py). Already-logged
        guesses and awarded points are NOT re-scored.
        """
        new_title = title.strip()
        if not new_title:
            raise FixSongRefusedError(
                "blank_title", "corrected title must be non-empty after stripping"
            )
        # artist omitted -> keep the current one; blank-after-strip -> clear.
        stripped_artist = artist.strip() if artist is not None else None

        if date_str is not None:
            try:
                date.fromisoformat(date_str)
            except ValueError:
                raise FixSongRefusedError(
                    "invalid_date", f"invalid date {date_str!r}: expected YYYY-MM-DD"
                ) from None
            challenge = self._challenge_row(guild_id, date_str)
        else:
            row = self._db.query_one(
                "SELECT * FROM challenges WHERE guild_id = ?"
                " ORDER BY date DESC, id DESC LIMIT 1",
                (guild_id,),
            )
            challenge = ChallengeRow.from_row(row) if row is not None else None
        if challenge is None:
            raise FixSongRefusedError(
                "no_challenge",
                f"no challenge to fix in guild {guild_id}"
                + (f" on {date_str}" if date_str is not None else ""),
            )

        song = self._song_row(challenge.song_id)
        new_artist = song.artist if stripped_artist is None else stripped_artist or None
        iso_now = self._utc_iso(now)
        with self._db.transaction():
            self._db.execute(
                "UPDATE songs SET title = ?, artist = ? WHERE id = ?",
                (new_title, new_artist, song.id),
            )
            self._db.execute(
                "INSERT INTO song_overrides"
                " (source, source_id, title, artist, set_by, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(source, source_id) DO UPDATE SET"
                " title = excluded.title, artist = excluded.artist,"
                " set_by = excluded.set_by, updated_at = excluded.updated_at",
                (song.source, song.source_id, new_title, new_artist, set_by, iso_now, iso_now),
            )
        return SongFix(
            song_id=song.id,
            challenge_id=challenge.id,
            challenge_date=challenge.date,
            old_title=song.title,
            old_artist=song.artist,
            new_title=new_title,
            new_artist=new_artist,
        )

    def delete_challenge(self, challenge_id: int) -> None:
        """Roll back a just-created challenge whose channel post failed (pinned #16).

        Mirrors skip_today_song's delete path WITHOUT the recreate: the row is
        DELETEd (challenge_users and guesses cascade-delete — none can exist
        for a just-created challenge) and its snippet cache is purged. The
        daily-post paths (scheduler tick, admin post, harness post) call this
        ONLY for a challenge they created in the same call
        (``Challenge.created``) when sending the channel message fails; a
        pre-existing challenge row is never rolled back. After the rollback a
        retry recreates the identical challenge (deterministic
        date+guild+skip_count seed) and delivers it.
        """
        self._db.execute("DELETE FROM challenges WHERE id = ?", (challenge_id,))
        self._snippets.purge_challenge(challenge_id)

    def refresh_catalog(self) -> RefreshResult:
        """Passthrough for the admin reload-catalog command."""
        return self._catalog_refresher()

    # -- gameplay ----------------------------------------------------------------

    def unlock_snippet(self, challenge_id: int, user_id: str) -> UnlockResult:
        """Unlock the next-longer snippet for a user (the Hear-more button).

        Increments the user's ``snippet_level`` (0 -> 4) and returns the new
        level's snippet path and remaining point potential. Refused with zero
        mutation (``UnlockRefusedError``) when the challenge is no longer
        active (``"closed"`` — the revealed-challenge lockout, VAL-GUESS-019),
        when the user already solved this challenge, or is already at the
        maximum level. The per-user row is upserted on this first interaction
        (pinned #13), and a zero-valued ``user_stats`` row is registered
        alongside it (VAL-SCORE-005). The snippet cache is re-healed on every
        unlock (idempotent ``ensure_snippets``, pinned #14).

        Writes no timestamps, so it needs no injected ``now``.
        """
        challenge = self._challenge_row_by_id(challenge_id)
        if challenge.status != "active":
            # Revealed-challenge lockout (VAL-GUESS-019): the answer is public,
            # so the persistent view on the old message must not unlock audio.
            # Refuse before any upsert or snippet re-heal — zero mutation.
            raise UnlockRefusedError(
                "closed",
                f"challenge {challenge_id} is {challenge.status}; gameplay is closed",
            )
        max_level = len(self._settings.snippet_lengths) - 1
        with self._db.transaction():
            state = self._challenge_user_row(challenge_id, user_id)
            if state is not None and state.solved:
                raise UnlockRefusedError(
                    "solved",
                    f"user {user_id} already solved challenge {challenge_id}",
                )
            level = state.snippet_level if state is not None else 0
            if level >= max_level:
                raise UnlockRefusedError(
                    "max_level",
                    f"user {user_id} is already at max snippet level {max_level}",
                )
            new_level = level + 1
            self._ensure_user_stats_row(challenge.guild_id, user_id)
            self._upsert_challenge_user(
                challenge_id,
                user_id,
                snippet_level=new_level,
                guesses_used=state.guesses_used if state is not None else 0,
                solved=False,
                points_awarded=0,
                solved_at=None,
            )
        # Outside the write transaction: snippet generation can be slow.
        paths = self._snippets.ensure_snippets(
            _song_for_generator(self._song_row(challenge.song_id)),
            challenge.id,
            challenge.snippet_offset_sec,
            self._settings.snippet_lengths,
        )
        return UnlockResult(
            level=new_level,
            path=paths[new_level],
            potential_points=self._settings.snippet_points[new_level],
        )

    def submit_guess(
        self, challenge_id: int, user_id: str, text: str, now: datetime
    ) -> GuessResult:
        """Process one guess; update per-user state, the guess log, and stats.

        Outcomes (see `GuessOutcome`): a submission against a challenge that
        is no longer ``active`` (revealed) is refused as ``challenge_closed``
        with zero mutation (VAL-GUESS-019) — this lockout dominates every
        other refusal, including ``empty`` and ``already_solved``. An
        empty-after-strip guess is a validation rejection — not counted, not
        logged, never matching (pinned #15). Post-solve submissions are
        rejected without state change or log rows (pinned #13). Any other
        submission consumes one of the day's guesses (the winning guess
        included) and is logged verbatim — submissions PAST the daily limit
        are no longer refused: they count and log like normal guesses, and a
        correct one (``correct_after_limit``) still marks the user solved
        and fires the public announcement, but banks a flat
        ``POST_LIMIT_SOLVE_POINTS`` and adds no win/streak. As a first
        interaction a submission also registers a zero-valued ``user_stats``
        row (VAL-SCORE-005). A correct guess within the limit
        banks ``SNIPPET_POINTS[level]`` — round-half-up x1.5 when one guess
        matches BOTH title and artist (pinned #6) — and updates
        ``user_stats`` (points, wins, streaks). All writes happen in one
        transaction.
        """
        challenge = self._challenge_row_by_id(challenge_id)
        max_guesses = self._settings.max_guesses_per_day

        if challenge.status != "active":
            # Revealed-challenge lockout (VAL-GUESS-019): the answer is public,
            # so the persistent view on the old message must not farm points.
            state = self._challenge_user_row(challenge_id, user_id)
            used = state.guesses_used if state is not None else 0
            return GuessResult(
                outcome="challenge_closed",
                matched_title=False,
                matched_artist=False,
                is_both=False,
                guesses_used=used,
                guesses_left=max_guesses - used,
                points_awarded=0,
                snippet_level=state.snippet_level if state is not None else 0,
                announce=False,
            )

        stripped = text.strip()
        if not stripped:
            state = self._challenge_user_row(challenge_id, user_id)
            used = state.guesses_used if state is not None else 0
            return GuessResult(
                outcome="empty",
                matched_title=False,
                matched_artist=False,
                is_both=False,
                guesses_used=used,
                guesses_left=max_guesses - used,
                points_awarded=0,
                snippet_level=state.snippet_level if state is not None else 0,
                announce=False,
            )

        with self._db.transaction():
            state = self._challenge_user_row(challenge_id, user_id)
            if state is not None and state.solved:
                return GuessResult(
                    outcome="already_solved",
                    matched_title=False,
                    matched_artist=False,
                    is_both=False,
                    guesses_used=state.guesses_used,
                    guesses_left=max_guesses - state.guesses_used,
                    points_awarded=0,
                    snippet_level=state.snippet_level,
                    announce=False,
                )
            used = state.guesses_used if state is not None else 0
            over_limit = used >= max_guesses

            song = self._song_row(challenge.song_id)
            match = match_guess(stripped, song)
            new_used = used + 1
            created_at = self._utc_iso(now)
            level = state.snippet_level if state is not None else 0
            solved = match.is_correct
            if solved and over_limit:
                # Post-limit solve: flat award, no both-bonus, no win/streak.
                points = POST_LIMIT_SOLVE_POINTS
            elif solved:
                points = self._points_for_level(level)
                if match.is_both:
                    points = self._apply_bonus(points)
            else:
                points = 0
            solved_at = created_at if solved else None

            self._ensure_user_stats_row(challenge.guild_id, user_id)
            self._upsert_challenge_user(
                challenge_id,
                user_id,
                snippet_level=level,
                guesses_used=new_used,
                solved=solved,
                points_awarded=points,
                solved_at=solved_at,
            )
            self._db.execute(
                "INSERT INTO guesses"
                " (challenge_id, user_id, text, matched_title, matched_artist,"
                " is_correct, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    challenge_id,
                    user_id,
                    text,
                    int(match.matched_title),
                    int(match.matched_artist),
                    int(solved),
                    created_at,
                ),
            )
            if solved and over_limit:
                self._apply_points(challenge.guild_id, user_id, points)
            elif solved:
                self._apply_win(
                    challenge.guild_id, user_id, points, date.fromisoformat(challenge.date)
                )

            return GuessResult(
                outcome=(
                    "correct_after_limit"
                    if solved and over_limit
                    else "correct"
                    if solved
                    else "wrong"
                ),
                matched_title=match.matched_title,
                matched_artist=match.matched_artist,
                is_both=match.is_both,
                guesses_used=new_used,
                guesses_left=max(0, max_guesses - new_used),
                points_awarded=points,
                snippet_level=level,
                announce=solved,
            )

    def leaderboard(
        self, guild_id: str, now: datetime, limit: int = 10
    ) -> list[LeaderboardEntry]:
        """The guild's top players: points DESC, wins DESC, user_id ASC.

        Only scoring users are listed: the ``total_points > 0`` filter keeps
        the zero-valued rows created on first interaction (VAL-SCORE-005) out
        of the leaderboard, so a guild where nobody has scored still reads as
        empty (VAL-SCORE-011/012). ``current_streak`` is the EFFECTIVE streak
        (pinned #7): computed on read from ``last_win_date`` against the
        local date of ``now`` — a gap of more than one calendar day reads as
        0 without touching the stored value (which is only rewritten on the
        next solve).
        """
        today = self._local_date(now)
        rows = self._db.query(
            "SELECT user_id, total_points, wins, current_streak, last_win_date"
            " FROM user_stats WHERE guild_id = ? AND total_points > 0"
            " ORDER BY total_points DESC, wins DESC, user_id ASC LIMIT ?",
            (guild_id, limit),
        )
        entries: list[LeaderboardEntry] = []
        for row in rows:
            last_win_raw = row["last_win_date"]
            stored_streak = int(row["current_streak"])
            effective = 0
            if last_win_raw is not None:
                last_win = date.fromisoformat(str(last_win_raw))
                if (today - last_win).days <= 1:
                    effective = stored_streak
            entries.append(
                LeaderboardEntry(
                    user_id=str(row["user_id"]),
                    total_points=int(row["total_points"]),
                    wins=int(row["wins"]),
                    current_streak=effective,
                )
            )
        return entries

    # -- gameplay helpers --------------------------------------------------------

    def _points_for_level(self, level: int) -> int:
        return self._settings.snippet_points[level]

    def _apply_bonus(self, points: int) -> int:
        """The both-fields bonus: round-half-up (pinned #6). 75 -> 113, 15 -> 23."""
        return int(points * self._settings.both_correct_multiplier + 0.5)

    def _ensure_user_stats_row(self, guild_id: str, user_id: str) -> None:
        """Insert a zero-valued ``user_stats`` row when none exists.

        Companion to the pinned-#13 ``challenge_users` upsert: the first
        interaction with a challenge (hear-more or a processed guess)
        registers the user in the guild's stats table with all-zero values
        and ``last_win_date = NULL``, so even a user who never solves has the
        contract-required row on record (VAL-SCORE-005). ``INSERT OR IGNORE``
        keeps it idempotent; the leaderboard filters these rows out via
        ``total_points > 0`` until the first solve banks points, and
        ``_apply_win`` upgrades the zero row in place on that solve.
        """
        self._db.execute(
            "INSERT OR IGNORE INTO user_stats"
            " (guild_id, user_id, total_points, wins, current_streak, best_streak,"
            " last_win_date) VALUES (?, ?, 0, 0, 0, 0, NULL)",
            (guild_id, user_id),
        )

    def _upsert_challenge_user(
        self,
        challenge_id: int,
        user_id: str,
        *,
        snippet_level: int,
        guesses_used: int,
        solved: bool,
        points_awarded: int,
        solved_at: str | None,
    ) -> None:
        """Insert or update the per-user row for a challenge (pinned #13)."""
        self._db.execute(
            "INSERT INTO challenge_users"
            " (challenge_id, user_id, snippet_level, guesses_used, solved,"
            " points_awarded, solved_at) VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(challenge_id, user_id) DO UPDATE SET"
            " snippet_level = excluded.snippet_level,"
            " guesses_used = excluded.guesses_used,"
            " solved = excluded.solved,"
            " points_awarded = excluded.points_awarded,"
            " solved_at = excluded.solved_at",
            (
                challenge_id,
                user_id,
                snippet_level,
                guesses_used,
                int(solved),
                points_awarded,
                solved_at,
            ),
        )

    def _apply_points(self, guild_id: str, user_id: str, points: int) -> None:
        """Bank points into ``user_stats`` WITHOUT a win/streak.

        The post-limit solve counterpart to ``_apply_win``: a correct guess
        past the daily limit still counts as solved (reveal winner, public
        announcement) but only adds ``total_points`` — wins, streaks, and
        ``last_win_date`` are untouched. The row is guaranteed to exist
        (``_ensure_user_stats_row`` runs earlier in the same transaction).
        """
        self._db.execute(
            "UPDATE user_stats SET total_points = total_points + ?"
            " WHERE guild_id = ? AND user_id = ?",
            (points, guild_id, user_id),
        )

    def _apply_win(
        self, guild_id: str, user_id: str, points: int, win_date: date
    ) -> None:
        """Bank a solve into ``user_stats``: points, wins, and streaks.

        ``win_date`` is the solved challenge's local date. The streak extends
        when it follows ``last_win_date`` by exactly one calendar day, holds
        for a same-day/out-of-order win (never regresses ``last_win_date``),
        and otherwise resets to 1. ``best_streak`` only ever grows. The row
        usually already exists as the zero-valued first-interaction row
        (VAL-SCORE-005); the upsert's additive conflict clause upgrades it in
        place (points/wins accumulate onto the zeros), and a ``None``
        ``last_win_date`` correctly starts the streak at 1.
        """
        row = self._db.query_one(
            "SELECT * FROM user_stats WHERE guild_id = ? AND user_id = ?",
            (guild_id, user_id),
        )
        prev = UserStatsRow.from_row(row) if row is not None else None

        if prev is None or prev.last_win_date is None:
            streak = 1
            last_win = win_date
        else:
            last_win = date.fromisoformat(prev.last_win_date)
            if win_date <= last_win:
                streak = prev.current_streak
            elif (win_date - last_win).days == 1:
                streak = prev.current_streak + 1
            else:
                streak = 1
            last_win = max(last_win, win_date)
        best = max(prev.best_streak if prev is not None else 0, streak)

        self._db.execute(
            "INSERT INTO user_stats"
            " (guild_id, user_id, total_points, wins, current_streak, best_streak,"
            " last_win_date) VALUES (?, ?, ?, 1, ?, ?, ?)"
            " ON CONFLICT(guild_id, user_id) DO UPDATE SET"
            " total_points = total_points + excluded.total_points,"
            " wins = wins + 1,"
            " current_streak = excluded.current_streak,"
            " best_streak = excluded.best_streak,"
            " last_win_date = excluded.last_win_date",
            (guild_id, user_id, points, streak, best, last_win.isoformat()),
        )

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
