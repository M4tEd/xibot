"""Unit tests for the GameEngine daily lifecycle: ensure_today_challenge,
peek_reveal / mark_revealed (the pinned-#17 delivery-coupled reveal split),
skip_today_song.

Deterministic and fast: tmp SQLite DBs, injected `now`, and a fake
SnippetService that creates empty cache files without ffmpeg. Contract
coverage: VAL-DAILY-005 (challenge date in configured timezone),
no-repeat/reset selection (VAL-DAILY-006/007 engine level), pinned decisions
#5 (skip-song), #11 (catalog bootstrap / catalog_empty), #14 (snippet cache
re-heal), and the per-guild isolation of VAL-SCORE-013.

The full VAL-SCORE-013 assertion (submit_guess + leaderboard) lives in
`TestPerGuildIsolation::test_stats_isolated_per_guild` (unskipped by the
engine-gameplay-matching feature).
"""

from __future__ import annotations

from collections.abc import Collection, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from songbot.catalog import Song
from songbot.catalog.refresh import RefreshResult
from songbot.config import Settings
from songbot.db import Database
from songbot.engine import (
    MAX_AUTO_SKIPS,
    CatalogEmptyError,
    GameEngine,
    Reveal,
    SkipRefusedError,
)
from songbot.snippets import SnippetGenerationError

NOW = datetime(2026, 8, 13, 16, 0, 0, tzinfo=UTC)  # 2026-08-13 13:00 ADT
TODAY = "2026-08-13"


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        discord_token="test-token",
        guild_id="guild-1",
        channel_id="channel-1",
        youtube_playlist_url=None,
        local_music_dir=None,
        daily_post_time="12:00",
        timezone="America/Halifax",
        max_guesses_per_day=6,
        snippet_lengths=(1.0, 2.0, 4.0, 8.0, 16.0),
        snippet_points=(100, 75, 50, 30, 15),
        both_correct_multiplier=1.5,
        database_path=tmp_path / "songbot.db",
        snippet_cache_dir=tmp_path / "snippets",
        health_port=3108,
        log_level="INFO",
        discord_api_base="https://discord.com/api/v10",
    )


class FakeSnippets:
    """SnippetService test double: creates empty level files, records calls.

    ``fail`` raises a plain RuntimeError for every call (a non-SnippetError:
    no auto-skip). ``fail_ids`` raises a `SnippetGenerationError` — the
    production failure contract — only for songs whose source_id is in the
    set, so tests can fail specific picks and exercise auto-skip.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        fail: bool = False,
        fail_ids: Collection[str] = (),
    ) -> None:
        self.cache_dir = cache_dir
        self.fail = fail
        self.fail_ids = set(fail_ids)
        self.ensure_calls: list[tuple[str, int | str, float, tuple[float, ...]]] = []
        self.purged: list[int | str] = []

    def ensure_snippets(
        self,
        song: Song,
        challenge_id: int | str,
        offset: float,
        lengths: Sequence[float],
    ) -> dict[int, Path]:
        self.ensure_calls.append((song.source_id, challenge_id, offset, tuple(lengths)))
        if self.fail:
            raise RuntimeError("fake snippet failure")
        if song.source_id in self.fail_ids:
            raise SnippetGenerationError(f"fake snippet failure for {song.source_id}")
        base = self.cache_dir / str(challenge_id)
        base.mkdir(parents=True, exist_ok=True)
        paths: dict[int, Path] = {}
        for level in range(len(lengths)):
            path = base / f"{level}.mp3"
            path.touch(exist_ok=True)
            paths[level] = path
        return paths

    def purge_challenge(self, challenge_id: int | str) -> None:
        self.purged.append(challenge_id)
        import shutil

        shutil.rmtree(self.cache_dir / str(challenge_id), ignore_errors=True)


def _add_song(
    db: Database,
    source_id: str,
    *,
    duration_sec: float = 30.0,
    title: str | None = None,
) -> int:
    cursor = db.execute(
        "INSERT INTO songs (source, source_id, title, artist, duration_sec, audio_ref,"
        " raw_title, created_at) VALUES ('local', ?, ?, ?, ?, ?, ?, ?)",
        (
            source_id,
            title if title is not None else f"Title {source_id}",
            f"Artist {source_id}",
            duration_sec,
            f"/music/{source_id}.mp3",
            f"raw {source_id}",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def _add_songs(db: Database, count: int, *, duration_sec: float = 30.0) -> list[int]:
    return [_add_song(db, f"song-{i}", duration_sec=duration_sec) for i in range(count)]


def _challenge_count(db: Database) -> int:
    row = db.query_one("SELECT COUNT(*) AS c FROM challenges")
    assert row is not None
    return int(row["c"])


def _db_snapshot(db: Database) -> dict[str, list[dict[str, object]]]:
    """Full-content dump of every mutable table, for zero-mutation proofs."""
    return {
        "challenges": [dict(r) for r in db.query("SELECT * FROM challenges ORDER BY id")],
        "challenge_users": [
            dict(r)
            for r in db.query(
                "SELECT * FROM challenge_users ORDER BY challenge_id, user_id"
            )
        ],
        "guesses": [dict(r) for r in db.query("SELECT * FROM guesses ORDER BY id")],
        "user_stats": [
            dict(r)
            for r in db.query("SELECT * FROM user_stats ORDER BY guild_id, user_id")
        ],
    }


def _reveal_previous(engine: GameEngine, guild_id: str, now: datetime) -> Reveal | None:
    """Test-setup helper: the DELIVERED reveal flow (pinned #17 peek + mark).

    The production reveal flows (scheduler tick, harness advance-day) call
    ``peek_reveal``, send the announcement, then ``mark_revealed``. Tests that
    need a revealed challenge as their starting state use this composed form.
    """
    reveal = engine.peek_reveal(guild_id, now)
    if reveal is not None:
        engine.mark_revealed(guild_id, now)
    return reveal


def _make_engine(
    tmp_path: Path,
    db: Database,
    *,
    snippets: FakeSnippets | None = None,
    catalog_refresher: object = None,
) -> tuple[GameEngine, FakeSnippets]:
    fake = snippets if snippets is not None else FakeSnippets(tmp_path / "snippets")
    engine = GameEngine(
        db,
        _settings(tmp_path),
        fake,
        catalog_refresher=catalog_refresher,  # type: ignore[arg-type]
    )
    return engine, fake


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    database = Database.open(tmp_path / "songbot.db")
    yield database
    database.close()


class TestEnsureTodayChallenge:
    def test_creates_row_and_returns_created_challenge(
        self, db: Database, tmp_path: Path
    ) -> None:
        _add_songs(db, 3)
        engine, fake = _make_engine(tmp_path, db)

        challenge = engine.ensure_today_challenge("g1", "c1", NOW)

        assert challenge.created is True
        assert challenge.date == TODAY
        assert challenge.guild_id == "g1"
        assert challenge.channel_id == "c1"
        assert challenge.status == "active"
        assert challenge.skip_count == 0
        assert challenge.song.id in {r["id"] for r in db.query("SELECT id FROM songs")}
        assert len(challenge.snippet_paths) == 5
        assert all(path.exists() for path in challenge.snippet_paths.values())
        assert len(fake.ensure_calls) == 1

        rows = db.query("SELECT * FROM challenges")
        assert len(rows) == 1
        row = rows[0]
        assert row["song_id"] == challenge.song.id
        assert row["snippet_offset_sec"] == challenge.snippet_offset_sec
        assert row["status"] == "active"
        assert row["revealed_at"] is None

    def test_offset_satisfies_constraint(self, db: Database, tmp_path: Path) -> None:
        """VAL-SNIP-005 at engine level: 0 <= offset and offset + 16 <= duration."""
        _add_song(db, "short", duration_sec=30.0)
        _add_song(db, "long", duration_sec=300.0)
        engine, _ = _make_engine(tmp_path, db)

        for day in range(1, 6):
            now = datetime(2026, 8, day, 16, 0, 0, tzinfo=UTC)
            challenge = engine.ensure_today_challenge("g1", "c1", now)
            assert challenge.snippet_offset_sec >= 0
            assert challenge.snippet_offset_sec + 16 <= challenge.song.duration_sec + 1e-6

    def test_idempotent_same_day(self, db: Database, tmp_path: Path) -> None:
        """Second call same day: same song/offset, no second row (VAL-DAILY-004 level)."""
        _add_songs(db, 3)
        engine, fake = _make_engine(tmp_path, db)

        first = engine.ensure_today_challenge("g1", "c1", NOW)
        later = datetime(2026, 8, 13, 22, 0, 0, tzinfo=UTC)  # still the 13th in Halifax
        second = engine.ensure_today_challenge("g1", "c1", later)

        assert second.created is False
        assert second.id == first.id
        assert second.song.id == first.song.id
        assert second.snippet_offset_sec == first.snippet_offset_sec
        assert _challenge_count(db) == 1
        # Pinned #14: ensure_snippets runs on EVERY call (cache re-heal).
        assert len(fake.ensure_calls) == 2

    def test_snippet_cache_reheal_for_existing_challenge(
        self, db: Database, tmp_path: Path
    ) -> None:
        """Pinned #14: a deleted cache dir is regenerated for an existing row."""
        _add_songs(db, 3)
        engine, _fake = _make_engine(tmp_path, db)
        first = engine.ensure_today_challenge("g1", "c1", NOW)

        import shutil

        shutil.rmtree(tmp_path / "snippets" / str(first.id))
        assert not (tmp_path / "snippets" / str(first.id)).exists()

        second = engine.ensure_today_challenge("g1", "c1", NOW)
        assert second.created is False
        assert len(second.snippet_paths) == 5
        assert all(path.exists() for path in second.snippet_paths.values())

    @pytest.mark.parametrize(
        ("now", "expected_date"),
        [
            (datetime(2025, 6, 10, 1, 30, 0, tzinfo=UTC), "2025-06-09"),
            (datetime(2025, 6, 10, 2, 59, 59, tzinfo=UTC), "2025-06-09"),
            (datetime(2025, 6, 10, 3, 0, 0, tzinfo=UTC), "2025-06-10"),
        ],
        ids=["utc-0130->prev-day", "utc-025959->prev-day", "utc-030000->same-day"],
    )
    def test_challenge_date_uses_configured_timezone(
        self, db: Database, tmp_path: Path, now: datetime, expected_date: str
    ) -> None:
        """VAL-DAILY-005: challenge date is the America/Halifax local date, not UTC."""
        _add_songs(db, 1)
        engine, _ = _make_engine(tmp_path, db)

        challenge = engine.ensure_today_challenge("g1", "c1", now)

        assert challenge.date == expected_date
        row = db.query_one("SELECT date FROM challenges")
        assert row is not None
        assert row["date"] == expected_date

    def test_no_repeat_until_exhausted_then_history_reset(
        self, db: Database, tmp_path: Path
    ) -> None:
        """8-song catalog -> 8 distinct daily songs; day 9 resets and succeeds."""
        song_ids = set(_add_songs(db, 8))
        engine, _ = _make_engine(tmp_path, db)

        picked: list[int] = []
        for day in range(1, 10):  # 9 days
            now = datetime(2026, 8, day, 16, 0, 0, tzinfo=UTC)
            picked.append(engine.ensure_today_challenge("g1", "c1", now).song.id)

        assert len(set(picked[:8])) == 8
        assert set(picked[:8]) == song_ids
        assert picked[8] in song_ids  # day 9: reset, any catalog song eligible

    def test_no_repeat_is_per_guild(self, db: Database, tmp_path: Path) -> None:
        """Guild B's history does not shrink guild A's eligible pool."""
        _add_songs(db, 2)
        engine, _ = _make_engine(tmp_path, db)

        for day in range(1, 4):
            now = datetime(2026, 8, day, 16, 0, 0, tzinfo=UTC)
            engine.ensure_today_challenge("gA", "c1", now)
        # gA exhausted its 2-song catalog and reset on day 3; gB posts only day 3
        # and must still see the full catalog (no cross-guild history bleed).
        engine.ensure_today_challenge("gB", "c1", datetime(2026, 8, 3, 16, 0, 0, tzinfo=UTC))
        used_b = {
            r["song_id"]
            for r in db.query("SELECT song_id FROM challenges WHERE guild_id = 'gB'")
        }
        assert len(used_b) == 1  # any song was eligible for gB's first day

    def test_auto_refresh_when_songs_table_empty(
        self, db: Database, tmp_path: Path
    ) -> None:
        """Pinned #11: an empty songs table triggers a catalog refresh first."""
        refresh_calls = 0

        def refresher() -> RefreshResult:
            nonlocal refresh_calls
            refresh_calls += 1
            _add_songs(db, 2)
            return RefreshResult(sources=())

        engine, _ = _make_engine(tmp_path, db, catalog_refresher=refresher)
        challenge = engine.ensure_today_challenge("g1", "c1", NOW)

        assert challenge.created is True
        assert refresh_calls == 1
        # Second call same day: idempotent path, no additional refresh.
        engine.ensure_today_challenge("g1", "c1", NOW)
        assert refresh_calls == 1

    def test_empty_catalog_raises_and_inserts_no_row(
        self, db: Database, tmp_path: Path
    ) -> None:
        """Pinned #11: still-empty catalog after refresh -> catalog_empty, no row."""
        engine, _ = _make_engine(
            tmp_path, db, catalog_refresher=lambda: RefreshResult(sources=())
        )

        with pytest.raises(CatalogEmptyError, match="catalog_empty"):
            engine.ensure_today_challenge("g1", "c1", NOW)

        assert _challenge_count(db) == 0

    def test_snippet_failure_leaves_no_challenge_row(
        self, db: Database, tmp_path: Path
    ) -> None:
        """A failed generation must not persist a snippet-less challenge row."""
        _add_songs(db, 2)
        engine, _ = _make_engine(
            tmp_path, db, snippets=FakeSnippets(tmp_path / "snippets", fail=True)
        )

        with pytest.raises(RuntimeError, match="fake snippet failure"):
            engine.ensure_today_challenge("g1", "c1", NOW)

        assert _challenge_count(db) == 0

    def test_deterministic_selection_across_dbs(self, tmp_path: Path) -> None:
        """Seed = hash(date, guild, skip_count): same inputs -> same song+offset."""
        results: list[tuple[int, float]] = []
        for name in ("a", "b"):
            db_path = tmp_path / name / "songbot.db"
            database = Database.open(db_path)
            try:
                _add_songs(database, 8)
                engine, _ = _make_engine(tmp_path / name, database)
                challenge = engine.ensure_today_challenge("g1", "c1", NOW)
                # song ids differ across DBs; compare source_id + offset instead
                results.append(
                    (_source_id_index(database, challenge.song.id), challenge.snippet_offset_sec)
                )
            finally:
                database.close()
        assert results[0] == results[1]


def _source_id_index(db: Database, song_id: int) -> int:
    row = db.query_one("SELECT source_id FROM songs WHERE id = ?", (song_id,))
    assert row is not None
    return int(str(row["source_id"]).split("-")[1])


class TestAutoSkip:
    """Issue #11: an unsnippable FRESH pick is auto-replaced (bounded).

    Without it, the scheduler retries the identical song+offset (seed
    hash(date|guild|skip=0)) every 60s until a manual /songbot-skip.
    """

    @staticmethod
    def _fail_seed0_pick(
        engine: GameEngine, db: Database, fake: FakeSnippets
    ) -> tuple[int, str]:
        """Learn today's seed-0 pick via a probe post, reset, then fail it."""
        probe = engine.ensure_today_challenge("g1", "c1", NOW)
        db.execute("DELETE FROM challenges WHERE id = ?", (probe.id,))
        fake.fail_ids = {probe.song.source_id}
        return probe.song.id, probe.song.source_id

    def test_failed_pick_is_auto_skipped_to_the_next_seed(
        self, db: Database, tmp_path: Path
    ) -> None:
        _add_songs(db, 4)
        engine, fake = _make_engine(tmp_path, db)
        failed_id, failed_source = self._fail_seed0_pick(engine, db, fake)
        calls_before = len(fake.ensure_calls)

        challenge = engine.ensure_today_challenge("g1", "c1", NOW)

        assert challenge.created is True
        assert challenge.skip_count == 1
        assert challenge.song.id != failed_id
        assert _challenge_count(db) == 1  # only the replacement row survives
        row = db.query_one("SELECT song_id, skip_count FROM challenges")
        assert row is not None
        assert (row["song_id"], row["skip_count"]) == (challenge.song.id, 1)
        # One failed attempt (the seed-0 song) + one success, in that order.
        new_calls = fake.ensure_calls[calls_before:]
        assert [call[0] for call in new_calls] == [failed_source, challenge.song.source_id]

    def test_auto_skip_pick_matches_the_manual_skip_chain(
        self, db: Database, tmp_path: Path
    ) -> None:
        """Auto-skip k lands on the same song+offset as k manual skips (one chain)."""
        _add_songs(db, 8)
        engine, fake = _make_engine(tmp_path, db)
        first = engine.ensure_today_challenge("g1", "c1", NOW)  # seed-0 pick
        manual = engine.skip_today_song("g1", NOW)  # seed-1 pick (manual chain)

        # Replay the day with the seed-0 pick unsnippable: the auto-skip must
        # land on the manual chain's seed-1 pick (same seed, same exclusion).
        db.execute("DELETE FROM challenges WHERE id = ?", (manual.id,))
        fake.fail_ids = {first.song.source_id}
        auto = engine.ensure_today_challenge("g1", "c1", NOW)

        assert auto.skip_count == 1
        assert auto.song.id == manual.song.id
        assert auto.snippet_offset_sec == manual.snippet_offset_sec

    def test_all_picks_failing_raises_and_leaves_no_row(
        self, db: Database, tmp_path: Path
    ) -> None:
        """Every song failing -> the last SnippetError propagates, no row."""
        _add_songs(db, 2)
        engine, fake = _make_engine(tmp_path, db)
        fake.fail_ids = {"song-0", "song-1"}

        with pytest.raises(SnippetGenerationError, match="fake snippet failure"):
            engine.ensure_today_challenge("g1", "c1", NOW)

        assert _challenge_count(db) == 0

    def test_auto_skip_is_bounded(self, db: Database, tmp_path: Path) -> None:
        """One call tries at most MAX_AUTO_SKIPS + 1 songs before giving up."""
        _add_songs(db, 10)
        engine, fake = _make_engine(tmp_path, db)
        fake.fail_ids = {f"song-{i}" for i in range(10)}

        with pytest.raises(SnippetGenerationError):
            engine.ensure_today_challenge("g1", "c1", NOW)

        assert len(fake.ensure_calls) == MAX_AUTO_SKIPS + 1
        assert _challenge_count(db) == 0

    def test_existing_row_reheal_failure_does_not_auto_skip(
        self, db: Database, tmp_path: Path
    ) -> None:
        """A pre-existing row may already be posted: never auto-replace it."""
        _add_songs(db, 4)
        engine, fake = _make_engine(tmp_path, db)
        challenge = engine.ensure_today_challenge("g1", "c1", NOW)
        fake.fail_ids = {challenge.song.source_id}

        with pytest.raises(SnippetGenerationError, match="fake snippet failure"):
            engine.ensure_today_challenge("g1", "c1", NOW)

        row = db.query_one("SELECT song_id, skip_count FROM challenges")
        assert row is not None
        assert (row["song_id"], row["skip_count"]) == (challenge.song.id, 0)


class TestPerGuildIsolation:
    def test_same_date_challenges_in_two_guilds(
        self, db: Database, tmp_path: Path
    ) -> None:
        """VAL-SCORE-013 (challenge level): UNIQUE(guild_id, date) not violated."""
        _add_songs(db, 4)
        engine, _ = _make_engine(tmp_path, db)

        first = engine.ensure_today_challenge("G1", "c1", NOW)
        second = engine.ensure_today_challenge("G2", "c1", NOW)

        assert first.id != second.id
        assert _challenge_count(db) == 2
        rows = db.query("SELECT guild_id, date FROM challenges ORDER BY guild_id")
        assert [(r["guild_id"], r["date"]) for r in rows] == [("G1", TODAY), ("G2", TODAY)]

    def test_reveal_and_skip_in_one_guild_leave_the_other_untouched(
        self, db: Database, tmp_path: Path
    ) -> None:
        _add_songs(db, 4)
        engine, _ = _make_engine(tmp_path, db)
        engine.ensure_today_challenge("G1", "c1", NOW)
        g2 = engine.ensure_today_challenge("G2", "c1", NOW)

        next_day = datetime(2026, 8, 14, 16, 0, 0, tzinfo=UTC)
        reveal = _reveal_previous(engine, "G1", next_day)
        assert reveal is not None

        g2_row = db.query_one(
            "SELECT status FROM challenges WHERE guild_id = 'G2' AND date = ?", (TODAY,)
        )
        assert g2_row is not None
        assert g2_row["status"] == "active"

        skipped = engine.skip_today_song("G2", NOW)
        assert skipped.skip_count == g2.skip_count + 1  # recreated row (rowid may be reused)
        g1_row = db.query_one(
            "SELECT status FROM challenges WHERE guild_id = 'G1' AND date = ?", (TODAY,)
        )
        assert g1_row is not None
        assert g1_row["status"] == "revealed"

    def test_stats_isolated_per_guild(self, db: Database, tmp_path: Path) -> None:
        """VAL-SCORE-013 (full): a solve in G1 must not appear in G2."""
        _add_songs(db, 4)
        engine, _ = _make_engine(tmp_path, db)
        engine.ensure_today_challenge("G1", "c1", NOW)
        engine.ensure_today_challenge("G2", "c1", NOW)

        g1 = db.query_one("SELECT * FROM challenges WHERE guild_id = 'G1'")
        assert g1 is not None
        song = db.query_one("SELECT * FROM songs WHERE id = ?", (g1["song_id"],))
        assert song is not None
        engine.submit_guess(g1["id"], "user-U", str(song["title"]), NOW)

        leaderboard_g1 = engine.leaderboard("G1", NOW)
        leaderboard_g2 = engine.leaderboard("G2", NOW)
        assert [entry.user_id for entry in leaderboard_g1] == ["user-U"]
        assert leaderboard_g2 == []
        assert db.query_one(
            "SELECT * FROM user_stats WHERE guild_id = 'G1' AND user_id = 'user-U'"
        ) is not None
        assert db.query_one(
            "SELECT * FROM user_stats WHERE guild_id = 'G2' AND user_id = 'user-U'"
        ) is None


class TestGuildSettings:
    """The per-guild post-target store backing multi-guild /songbot-setup."""

    def test_set_then_get_round_trip(self, db: Database, tmp_path: Path) -> None:
        engine, _ = _make_engine(tmp_path, db)

        row = engine.set_guild_channel("g1", "c1", set_by="user-1", now=NOW)

        assert row.guild_id == "g1"
        assert row.channel_id == "c1"
        assert row.set_by == "user-1"
        assert row.created_at == row.updated_at
        assert engine.guild_settings("g1") == row

    def test_upsert_updates_channel_and_keeps_created_at(
        self, db: Database, tmp_path: Path
    ) -> None:
        engine, _ = _make_engine(tmp_path, db)
        first = engine.set_guild_channel("g1", "c1", set_by="user-1", now=NOW)
        later = datetime(2026, 8, 14, 16, 0, 0, tzinfo=UTC)

        second = engine.set_guild_channel("g1", "c2", set_by="user-2", now=later)

        assert second.channel_id == "c2"
        assert second.set_by == "user-2"
        assert second.created_at == first.created_at
        assert second.updated_at != first.updated_at
        assert len(engine.all_guild_settings()) == 1  # still one row

    def test_unknown_guild_is_none(self, db: Database, tmp_path: Path) -> None:
        engine, _ = _make_engine(tmp_path, db)
        assert engine.guild_settings("nope") is None

    def test_all_guild_settings_ordered_by_guild_id(
        self, db: Database, tmp_path: Path
    ) -> None:
        engine, _ = _make_engine(tmp_path, db)
        engine.set_guild_channel("g2", "c2", set_by="t", now=NOW)
        engine.set_guild_channel("g1", "c1", set_by="t", now=NOW)

        assert [row.guild_id for row in engine.all_guild_settings()] == ["g1", "g2"]

    def test_remove_guild_settings_drops_only_that_guild(
        self, db: Database, tmp_path: Path
    ) -> None:
        engine, _ = _make_engine(tmp_path, db)
        engine.set_guild_channel("g1", "c1", set_by="t", now=NOW)
        engine.set_guild_channel("g2", "c2", set_by="t", now=NOW)

        engine.remove_guild_settings("g1")

        assert engine.guild_settings("g1") is None
        assert engine.guild_settings("g2") is not None

    def test_latest_challenge_id(self, db: Database, tmp_path: Path) -> None:
        _add_songs(db, 3)
        engine, _ = _make_engine(tmp_path, db)
        assert engine.latest_challenge_id("g1") is None

        first = engine.ensure_today_challenge("g1", "c1", NOW)
        assert engine.latest_challenge_id("g1") == first.id
        # Another guild's challenges never leak into the lookup.
        assert engine.latest_challenge_id("g2") is None

    def test_reveal_carries_the_challenge_row_channel(
        self, db: Database, tmp_path: Path
    ) -> None:
        """The reveal targets the channel the challenge was POSTED to."""
        _add_songs(db, 3)
        engine, _ = _make_engine(tmp_path, db)
        engine.ensure_today_challenge("g1", "c1", NOW)
        next_day = datetime(2026, 8, 14, 16, 0, 0, tzinfo=UTC)

        reveal = engine.peek_reveal("g1", next_day)

        assert reveal is not None
        assert reveal.guild_id == "g1"
        assert reveal.channel_id == "c1"


class TestPingRoleSettings:
    """The per-guild reaction-role opt-in store backing /songbot-pingrole."""

    def _configure(self, engine: GameEngine, guild_id: str) -> None:
        # The FK requires a guild_settings row (the opt-in rides on setup).
        engine.set_guild_channel(guild_id, "c1", set_by="t", now=NOW)

    def test_set_then_get_round_trip(self, db: Database, tmp_path: Path) -> None:
        engine, _ = _make_engine(tmp_path, db)
        self._configure(engine, "g1")

        row = engine.set_ping_role(
            "g1", "c1", "m1", "r1", "🎵", set_by="user-1", now=NOW
        )

        assert row.guild_id == "g1"
        assert row.channel_id == "c1"
        assert row.message_id == "m1"
        assert row.role_id == "r1"
        assert row.emoji == "🎵"
        assert row.set_by == "user-1"
        assert row.created_at == row.updated_at
        assert engine.ping_role_settings("g1") == row

    def test_upsert_replaces_and_keeps_created_at(
        self, db: Database, tmp_path: Path
    ) -> None:
        engine, _ = _make_engine(tmp_path, db)
        self._configure(engine, "g1")
        first = engine.set_ping_role("g1", "c1", "m1", "r1", "🎵", set_by="u1", now=NOW)
        later = datetime(2026, 8, 14, 16, 0, 0, tzinfo=UTC)

        second = engine.set_ping_role(
            "g1", "c1", "m2", "r2", "🔔", set_by="u2", now=later
        )

        assert second.message_id == "m2"
        assert second.role_id == "r2"
        assert second.emoji == "🔔"
        assert second.created_at == first.created_at
        assert second.updated_at != first.updated_at

    def test_unknown_guild_and_message_are_none(
        self, db: Database, tmp_path: Path
    ) -> None:
        engine, _ = _make_engine(tmp_path, db)
        assert engine.ping_role_settings("nope") is None
        assert engine.ping_role_for_message("nope") is None

    def test_ping_role_for_message_dispatches_to_the_right_guild(
        self, db: Database, tmp_path: Path
    ) -> None:
        engine, _ = _make_engine(tmp_path, db)
        for guild, message in (("g1", "m1"), ("g2", "m2")):
            self._configure(engine, guild)
            engine.set_ping_role(guild, "c1", message, f"role-{guild}", "🎵", set_by="t", now=NOW)

        assert engine.ping_role_for_message("m2").guild_id == "g2"  # type: ignore[union-attr]
        assert engine.ping_role_for_message("m1").role_id == "role-g1"  # type: ignore[union-attr]

    def test_remove_guild_settings_cascades_the_ping_role(
        self, db: Database, tmp_path: Path
    ) -> None:
        engine, _ = _make_engine(tmp_path, db)
        self._configure(engine, "g1")
        engine.set_ping_role("g1", "c1", "m1", "r1", "🎵", set_by="t", now=NOW)

        engine.remove_guild_settings("g1")

        assert engine.ping_role_settings("g1") is None
        assert engine.ping_role_for_message("m1") is None


class TestPeekReveal:
    """The read-only half of the delivery-coupled reveal (pinned #17).

    ``peek_reveal`` computes the stale challenge's `Reveal` (song + winners)
    with ZERO mutation, so a failed reveal send leaves the challenge active
    and the retry re-peeks the identical reveal (VAL-DAILY-014).
    """

    def _insert_solver(
        self,
        db: Database,
        challenge_id: int,
        user_id: str,
        *,
        guesses_used: int,
        points: int,
        solved_at: str,
    ) -> None:
        db.execute(
            "INSERT INTO challenge_users"
            " (challenge_id, user_id, snippet_level, guesses_used, solved,"
            " points_awarded, solved_at) VALUES (?, ?, 0, ?, 1, ?, ?)",
            (challenge_id, user_id, guesses_used, points, solved_at),
        )

    def test_peek_returns_winners_in_solve_order_with_zero_mutation(
        self, db: Database, tmp_path: Path
    ) -> None:
        _add_songs(db, 3)
        engine, _ = _make_engine(tmp_path, db)
        day1 = engine.ensure_today_challenge("g1", "c1", NOW)
        self._insert_solver(
            db, day1.id, "alice", guesses_used=1, points=100,
            solved_at="2026-08-13T16:05:00+00:00",
        )
        self._insert_solver(
            db, day1.id, "bob", guesses_used=3, points=75,
            solved_at="2026-08-13T16:02:00+00:00",
        )
        before = _db_snapshot(db)

        next_day = datetime(2026, 8, 14, 16, 0, 0, tzinfo=UTC)
        reveal = engine.peek_reveal("g1", next_day)

        assert reveal is not None
        assert reveal.challenge_id == day1.id
        assert reveal.date == TODAY
        assert reveal.song.id == day1.song.id
        # solve order = solved_at ascending: bob before alice
        assert [w.user_id for w in reveal.winners] == ["bob", "alice"]
        assert reveal.winners[0].guesses_used == 3
        assert reveal.winners[0].points_awarded == 75
        assert reveal.winners[1].guesses_used == 1

        # ZERO mutation: the challenge is still active and nothing else moved.
        assert _db_snapshot(db) == before
        row = db.query_one(
            "SELECT status, revealed_at FROM challenges WHERE id = ?", (day1.id,)
        )
        assert row is not None
        assert row["status"] == "active"
        assert row["revealed_at"] is None

    def test_peek_is_idempotent_and_never_marks(
        self, db: Database, tmp_path: Path
    ) -> None:
        """A failed send followed by a retry peeks the IDENTICAL reveal."""
        _add_songs(db, 3)
        engine, _ = _make_engine(tmp_path, db)
        day1 = engine.ensure_today_challenge("g1", "c1", NOW)
        self._insert_solver(
            db, day1.id, "alice", guesses_used=1, points=100,
            solved_at="2026-08-13T16:05:00+00:00",
        )
        next_day = datetime(2026, 8, 14, 16, 0, 0, tzinfo=UTC)

        first = engine.peek_reveal("g1", next_day)
        second = engine.peek_reveal("g1", next_day)

        assert first is not None
        assert first == second  # same challenge, song, winners, revealed_at
        row = db.query_one("SELECT status FROM challenges WHERE id = ?", (day1.id,))
        assert row is not None
        assert row["status"] == "active"

    def test_peek_none_when_no_challenge(self, db: Database, tmp_path: Path) -> None:
        engine, _ = _make_engine(tmp_path, db)
        assert engine.peek_reveal("g1", NOW) is None

    def test_peek_ignores_todays_challenge(self, db: Database, tmp_path: Path) -> None:
        """peek_reveal never reveals the challenge of the current local day."""
        _add_songs(db, 3)
        engine, _ = _make_engine(tmp_path, db)
        engine.ensure_today_challenge("g1", "c1", NOW)
        assert engine.peek_reveal("g1", NOW) is None
        row = db.query_one("SELECT status FROM challenges")
        assert row is not None
        assert row["status"] == "active"

    def test_peek_with_no_winners(self, db: Database, tmp_path: Path) -> None:
        _add_songs(db, 3)
        engine, _ = _make_engine(tmp_path, db)
        day1 = engine.ensure_today_challenge("g1", "c1", NOW)
        db.execute(
            "INSERT INTO challenge_users"
            " (challenge_id, user_id, snippet_level, guesses_used, solved, points_awarded)"
            " VALUES (?, 'eve', 2, 4, 0, 0)",
            (day1.id,),
        )
        next_day = datetime(2026, 8, 14, 16, 0, 0, tzinfo=UTC)

        reveal = engine.peek_reveal("g1", next_day)
        assert reveal is not None
        assert reveal.winners == ()

    def test_peek_returns_most_recent_stale_without_marking(
        self, db: Database, tmp_path: Path
    ) -> None:
        """Two unrevealed past challenges (missed cycles): peek returns the
        most recent one's reveal and marks NOTHING."""
        _add_songs(db, 3)
        engine, _ = _make_engine(tmp_path, db)
        engine.ensure_today_challenge("g1", "c1", datetime(2026, 8, 11, 16, 0, tzinfo=UTC))
        day2 = engine.ensure_today_challenge(
            "g1", "c1", datetime(2026, 8, 12, 16, 0, tzinfo=UTC)
        )

        reveal = engine.peek_reveal("g1", datetime(2026, 8, 13, 16, 0, tzinfo=UTC))

        assert reveal is not None
        assert reveal.challenge_id == day2.id  # most recent stale active
        statuses = db.query("SELECT status FROM challenges ORDER BY date")
        assert [r["status"] for r in statuses] == ["active", "active"]


class TestMarkRevealed:
    """The mutation half of the delivery-coupled reveal (pinned #17).

    ``mark_revealed`` is applied by the caller ONLY after the reveal send
    succeeds; it marks every stale active row revealed (one transaction), so
    a delivered reveal is never computed or sent again.
    """

    def test_mark_after_peek_reveals_all_stale_rows(
        self, db: Database, tmp_path: Path
    ) -> None:
        """Two unrevealed past challenges (missed cycles) both get revealed."""
        _add_songs(db, 3)
        engine, _ = _make_engine(tmp_path, db)
        engine.ensure_today_challenge("g1", "c1", datetime(2026, 8, 11, 16, 0, tzinfo=UTC))
        day2 = engine.ensure_today_challenge(
            "g1", "c1", datetime(2026, 8, 12, 16, 0, tzinfo=UTC)
        )
        now = datetime(2026, 8, 13, 16, 0, tzinfo=UTC)

        reveal = engine.peek_reveal("g1", now)
        assert reveal is not None
        assert reveal.challenge_id == day2.id  # most recent stale active
        engine.mark_revealed("g1", now)

        statuses = db.query("SELECT status, revealed_at FROM challenges ORDER BY date")
        assert [r["status"] for r in statuses] == ["revealed", "revealed"]
        assert all(r["revealed_at"] is not None for r in statuses)

    def test_mark_persists_the_peeked_revealed_at(
        self, db: Database, tmp_path: Path
    ) -> None:
        """The Reveal's ``revealed_at`` is exactly what mark_revealed writes."""
        _add_songs(db, 3)
        engine, _ = _make_engine(tmp_path, db)
        day1 = engine.ensure_today_challenge("g1", "c1", NOW)
        next_day = datetime(2026, 8, 14, 16, 0, 0, tzinfo=UTC)

        reveal = engine.peek_reveal("g1", next_day)
        assert reveal is not None
        engine.mark_revealed("g1", next_day)

        row = db.query_one(
            "SELECT status, revealed_at FROM challenges WHERE id = ?", (day1.id,)
        )
        assert row is not None
        assert row["status"] == "revealed"
        assert row["revealed_at"] == reveal.revealed_at

    def test_peek_after_mark_returns_none(self, db: Database, tmp_path: Path) -> None:
        """Exactly-once: a delivered reveal is never computed (or sent) again."""
        _add_songs(db, 3)
        engine, _ = _make_engine(tmp_path, db)
        engine.ensure_today_challenge("g1", "c1", NOW)
        next_day = datetime(2026, 8, 14, 16, 0, 0, tzinfo=UTC)

        assert _reveal_previous(engine, "g1", next_day) is not None
        assert engine.peek_reveal("g1", next_day) is None
        row = db.query_one("SELECT COUNT(*) AS c FROM challenges WHERE status = 'revealed'")
        assert row is not None
        assert row["c"] == 1

    def test_mark_with_nothing_stale_is_a_noop(
        self, db: Database, tmp_path: Path
    ) -> None:
        _add_songs(db, 3)
        engine, _ = _make_engine(tmp_path, db)
        engine.ensure_today_challenge("g1", "c1", NOW)
        before = _db_snapshot(db)

        engine.mark_revealed("g1", NOW)  # only today's challenge exists

        assert _db_snapshot(db) == before
        row = db.query_one("SELECT status FROM challenges")
        assert row is not None
        assert row["status"] == "active"


class TestSkipTodaySong:
    def _insert_interaction(
        self, db: Database, challenge_id: int, user_id: str = "todd"
    ) -> None:
        db.execute(
            "INSERT INTO challenge_users"
            " (challenge_id, user_id, snippet_level, guesses_used, solved, points_awarded)"
            " VALUES (?, ?, 2, 1, 0, 0)",
            (challenge_id, user_id),
        )
        db.execute(
            "INSERT INTO guesses"
            " (challenge_id, user_id, text, matched_title, matched_artist, is_correct,"
            " created_at) VALUES (?, ?, 'wrong', 0, 0, 0, ?)",
            (challenge_id, user_id, NOW.isoformat()),
        )

    def test_skip_replaces_song_and_offset_and_resets_state(
        self, db: Database, tmp_path: Path
    ) -> None:
        """Pinned #5: delete + recreate same-date row, new song+offset, cascade."""
        _add_songs(db, 8)
        engine, fake = _make_engine(tmp_path, db)
        old = engine.ensure_today_challenge("g1", "c1", NOW)
        # Pad with a later-rowid challenge in another guild so the skipped row
        # is not the max rowid: the replacement then gets a fresh id, making
        # the old cache-dir purge observable (SQLite reuses freed max rowids).
        engine.ensure_today_challenge("g2", "c1", NOW)
        self._insert_interaction(db, old.id)

        new = engine.skip_today_song("g1", NOW)

        assert new.created is True
        # NB: SQLite may reuse the freed rowid for the new row (no
        # AUTOINCREMENT); recreation is proven by skip_count incrementing.
        assert new.skip_count == old.skip_count + 1
        assert new.date == old.date
        assert new.guild_id == old.guild_id
        assert new.channel_id == old.channel_id
        assert new.song.id != old.song.id
        assert new.snippet_offset_sec != old.snippet_offset_sec
        assert _challenge_count(db) == 2  # g1's replacement + g2's untouched row
        # cascade: per-user state and guess log are gone with the old row
        assert db.query("SELECT * FROM challenge_users") == []
        assert db.query("SELECT * FROM guesses") == []
        # snippet cache purged for the old challenge, regenerated for the new
        assert fake.purged == [old.id]
        assert not (tmp_path / "snippets" / str(old.id)).exists()
        assert all(p.exists() for p in new.snippet_paths.values())

    def test_skip_is_deterministic_across_replays(self, tmp_path: Path) -> None:
        """seed = hash(date, guild_id, skip_count): replays give identical picks."""
        picks: list[tuple[int, float]] = []
        for name in ("a", "b"):
            database = Database.open(tmp_path / name / "songbot.db")
            try:
                _add_songs(database, 8)
                engine, _ = _make_engine(tmp_path / name, database)
                engine.ensure_today_challenge("g1", "c1", NOW)
                new = engine.skip_today_song("g1", NOW)
                picks.append(
                    (_source_id_index(database, new.song.id), new.snippet_offset_sec)
                )
            finally:
                database.close()
        assert picks[0] == picks[1]

    def test_second_skip_replaces_again(self, db: Database, tmp_path: Path) -> None:
        _add_songs(db, 8)
        engine, _ = _make_engine(tmp_path, db)
        first = engine.ensure_today_challenge("g1", "c1", NOW)
        second = engine.skip_today_song("g1", NOW)
        third = engine.skip_today_song("g1", NOW)

        assert third.skip_count == 2
        assert third.song.id != second.song.id
        assert third.snippet_offset_sec != second.snippet_offset_sec
        assert _challenge_count(db) == 1
        assert db.query("SELECT * FROM challenge_users") == []
        assert first.song.id != second.song.id

    def test_skip_refused_when_revealed(self, db: Database, tmp_path: Path) -> None:
        """VAL-ADMIN-005 engine level: revealed challenge -> refusal, zero mutation."""
        _add_songs(db, 8)
        engine, fake = _make_engine(tmp_path, db)
        old = engine.ensure_today_challenge("g1", "c1", NOW)
        db.execute(
            "UPDATE challenges SET status = 'revealed', revealed_at = ? WHERE id = ?",
            (NOW.isoformat(), old.id),
        )
        calls_before = len(fake.ensure_calls)

        with pytest.raises(SkipRefusedError) as excinfo:
            engine.skip_today_song("g1", NOW)

        assert excinfo.value.reason == "revealed"
        assert fake.purged == []
        assert len(fake.ensure_calls) == calls_before
        row = db.query_one("SELECT song_id, snippet_offset_sec, status FROM challenges")
        assert row is not None
        assert (row["song_id"], row["snippet_offset_sec"], row["status"]) == (
            old.song.id,
            old.snippet_offset_sec,
            "revealed",
        )

    def test_skip_refused_after_a_solve(self, db: Database, tmp_path: Path) -> None:
        """VAL-ADMIN-006 engine level: any solver -> refusal, earned state intact."""
        _add_songs(db, 8)
        engine, _ = _make_engine(tmp_path, db)
        old = engine.ensure_today_challenge("g1", "c1", NOW)
        db.execute(
            "INSERT INTO challenge_users"
            " (challenge_id, user_id, snippet_level, guesses_used, solved,"
            " points_awarded, solved_at) VALUES (?, 'uma', 0, 1, 1, 100, ?)",
            (old.id, NOW.isoformat()),
        )

        with pytest.raises(SkipRefusedError) as excinfo:
            engine.skip_today_song("g1", NOW)

        assert excinfo.value.reason == "solved"
        row = db.query_one("SELECT song_id, status FROM challenges WHERE id = ?", (old.id,))
        assert row is not None
        assert row["song_id"] == old.song.id
        assert row["status"] == "active"
        user = db.query_one("SELECT solved, points_awarded FROM challenge_users")
        assert user is not None
        assert (user["solved"], user["points_awarded"]) == (1, 100)

    def test_skip_refused_without_challenge(self, db: Database, tmp_path: Path) -> None:
        engine, _ = _make_engine(tmp_path, db)
        with pytest.raises(SkipRefusedError) as excinfo:
            engine.skip_today_song("g1", NOW)
        assert excinfo.value.reason == "no_challenge"

    def test_skip_unsolved_interactions_do_not_block(
        self, db: Database, tmp_path: Path
    ) -> None:
        """Wrong guesses / hear-more alone (solved=0) must NOT block a skip."""
        _add_songs(db, 8)
        engine, _ = _make_engine(tmp_path, db)
        old = engine.ensure_today_challenge("g1", "c1", NOW)
        self._insert_interaction(db, old.id)  # solved = 0

        new = engine.skip_today_song("g1", NOW)
        assert new.song.id != old.song.id
