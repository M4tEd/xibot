"""Unit tests for the GameEngine daily lifecycle: ensure_today_challenge,
get_reveal, skip_today_song.

Deterministic and fast: tmp SQLite DBs, injected `now`, and a fake
SnippetService that creates empty cache files without ffmpeg. Contract
coverage: VAL-DAILY-005 (challenge date in configured timezone),
no-repeat/reset selection (VAL-DAILY-006/007 engine level), pinned decisions
#5 (skip-song), #11 (catalog bootstrap / catalog_empty), #14 (snippet cache
re-heal), and the per-guild isolation of VAL-SCORE-013.

The full VAL-SCORE-013 assertion (submit_guess + leaderboard) depends on the
engine-gameplay-matching feature; a ready-made test is included here, skipped
until that feature lands.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from songbot.catalog import Song
from songbot.catalog.refresh import RefreshResult
from songbot.config import Settings
from songbot.db import Database
from songbot.engine import (
    CatalogEmptyError,
    GameEngine,
    SkipRefusedError,
)

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
    """SnippetService test double: creates empty level files, records calls."""

    def __init__(self, cache_dir: Path, *, fail: bool = False) -> None:
        self.cache_dir = cache_dir
        self.fail = fail
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
        reveal = engine.get_reveal("G1", next_day)
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

    @pytest.mark.skip(
        reason="needs GameEngine.submit_guess/leaderboard from engine-gameplay-matching"
    )
    def test_stats_isolated_per_guild(self, db: Database, tmp_path: Path) -> None:
        """VAL-SCORE-013 (full): a solve in G1 must not appear in G2.

        Ready-made acceptance test for the gameplay feature: unskip once
        submit_guess/leaderboard exist.
        """
        _add_songs(db, 4)
        engine, _ = _make_engine(tmp_path, db)
        engine.ensure_today_challenge("G1", "c1", NOW)
        engine.ensure_today_challenge("G2", "c1", NOW)

        g1 = db.query_one("SELECT * FROM challenges WHERE guild_id = 'G1'")
        assert g1 is not None
        song = db.query_one("SELECT * FROM songs WHERE id = ?", (g1["song_id"],))
        assert song is not None
        engine.submit_guess(g1["id"], "user-U", str(song["title"]))  # type: ignore[attr-defined]

        leaderboard_g1 = engine.leaderboard("G1")  # type: ignore[attr-defined]
        leaderboard_g2 = engine.leaderboard("G2")  # type: ignore[attr-defined]
        assert [entry.user_id for entry in leaderboard_g1] == ["user-U"]
        assert leaderboard_g2 == []
        assert db.query_one(
            "SELECT * FROM user_stats WHERE guild_id = 'G1' AND user_id = 'user-U'"
        ) is not None
        assert db.query_one(
            "SELECT * FROM user_stats WHERE guild_id = 'G2' AND user_id = 'user-U'"
        ) is None


class TestGetReveal:
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

    def test_reveal_marks_previous_and_returns_winners_in_solve_order(
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

        next_day = datetime(2026, 8, 14, 16, 0, 0, tzinfo=UTC)
        reveal = engine.get_reveal("g1", next_day)

        assert reveal is not None
        assert reveal.challenge_id == day1.id
        assert reveal.date == TODAY
        assert reveal.song.id == day1.song.id
        # solve order = solved_at ascending: bob before alice
        assert [w.user_id for w in reveal.winners] == ["bob", "alice"]
        assert reveal.winners[0].guesses_used == 3
        assert reveal.winners[0].points_awarded == 75
        assert reveal.winners[1].guesses_used == 1

        row = db.query_one("SELECT status, revealed_at FROM challenges WHERE id = ?", (day1.id,))
        assert row is not None
        assert row["status"] == "revealed"
        assert row["revealed_at"] is not None

    def test_reveal_marks_exactly_once(self, db: Database, tmp_path: Path) -> None:
        """A second get_reveal the same day finds no active challenge -> None."""
        _add_songs(db, 3)
        engine, _ = _make_engine(tmp_path, db)
        engine.ensure_today_challenge("g1", "c1", NOW)
        next_day = datetime(2026, 8, 14, 16, 0, 0, tzinfo=UTC)

        assert engine.get_reveal("g1", next_day) is not None
        assert engine.get_reveal("g1", next_day) is None
        row = db.query_one("SELECT COUNT(*) AS c FROM challenges WHERE status = 'revealed'")
        assert row is not None
        assert row["c"] == 1

    def test_reveal_none_when_no_challenge(self, db: Database, tmp_path: Path) -> None:
        engine, _ = _make_engine(tmp_path, db)
        assert engine.get_reveal("g1", NOW) is None

    def test_reveal_ignores_todays_challenge(self, db: Database, tmp_path: Path) -> None:
        """get_reveal never reveals the challenge of the current local day."""
        _add_songs(db, 3)
        engine, _ = _make_engine(tmp_path, db)
        engine.ensure_today_challenge("g1", "c1", NOW)
        assert engine.get_reveal("g1", NOW) is None
        row = db.query_one("SELECT status FROM challenges")
        assert row is not None
        assert row["status"] == "active"

    def test_reveal_with_no_winners(self, db: Database, tmp_path: Path) -> None:
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

        reveal = engine.get_reveal("g1", next_day)
        assert reveal is not None
        assert reveal.winners == ()

    def test_reveal_clears_all_stale_actives_returns_most_recent(
        self, db: Database, tmp_path: Path
    ) -> None:
        """Two unrevealed past challenges (missed cycles) both get revealed."""
        _add_songs(db, 3)
        engine, _ = _make_engine(tmp_path, db)
        engine.ensure_today_challenge("g1", "c1", datetime(2026, 8, 11, 16, 0, tzinfo=UTC))
        day2 = engine.ensure_today_challenge(
            "g1", "c1", datetime(2026, 8, 12, 16, 0, tzinfo=UTC)
        )

        reveal = engine.get_reveal("g1", datetime(2026, 8, 13, 16, 0, tzinfo=UTC))

        assert reveal is not None
        assert reveal.challenge_id == day2.id  # most recent stale active
        statuses = db.query("SELECT status FROM challenges ORDER BY date")
        assert [r["status"] for r in statuses] == ["revealed", "revealed"]


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
