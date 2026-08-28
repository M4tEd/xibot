"""Unit tests for the GameEngine gameplay methods: unlock_snippet,
submit_guess, leaderboard, and streak accounting.

Deterministic and fast: tmp SQLite DBs, injected `now`, and the fake
SnippetService shared with the daily-lifecycle tests. Contract coverage
(engine level): VAL-GUESS-002..011, VAL-GUESS-014, VAL-GUESS-018,
VAL-GUESS-019 (revealed-challenge lockout), VAL-SCORE-001..010, and pinned
decisions #6 (round-half-up bonus), #7 (effective streak on read), #13
(rejected submissions not logged), #15 (empty guesses rejected as
validation).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from songbot.db import Database
from songbot.engine import (
    EngineError,
    GameEngine,
    UnlockRefusedError,
)
from tests.unit.test_engine_daily import (
    NOW,
    TODAY,
    _db_snapshot,
    _make_engine,
    _reveal_previous,
)

TITLE = "Neon Skyline"
ARTIST = "Midnight Circuit"
RAW = "Midnight Circuit - Neon Skyline"
BOTH_GUESS = "Midnight Circuit - Neon Skyline"
WRONG = "zxqv unrelated noise"

DAY1 = datetime(2026, 8, 13, 16, 0, 0, tzinfo=UTC)  # 2026-08-13 13:00 ADT
DAY2 = datetime(2026, 8, 14, 16, 0, 0, tzinfo=UTC)  # 2026-08-14
DAY3 = datetime(2026, 8, 15, 16, 0, 0, tzinfo=UTC)  # 2026-08-15
DAY4 = datetime(2026, 8, 16, 16, 0, 0, tzinfo=UTC)  # 2026-08-16


def _add_song(
    db: Database,
    source_id: str = "song-1",
    *,
    title: str = TITLE,
    artist: str | None = ARTIST,
    raw_title: str = RAW,
    duration_sec: float = 30.0,
) -> int:
    cursor = db.execute(
        "INSERT INTO songs (source, source_id, title, artist, duration_sec, audio_ref,"
        " raw_title, created_at) VALUES ('local', ?, ?, ?, ?, ?, ?, ?)",
        (
            source_id,
            title,
            artist,
            duration_sec,
            f"/music/{source_id}.mp3",
            raw_title,
            "2026-01-01T00:00:00+00:00",
        ),
    )
    assert cursor.lastrowid is not None
    return cursor.lastrowid


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    database = Database.open(tmp_path / "songbot.db")
    yield database
    database.close()


@pytest.fixture
def engine(db: Database, tmp_path: Path) -> GameEngine:
    engine, _ = _make_engine(tmp_path, db)
    return engine


@pytest.fixture
def challenge_id(engine: GameEngine, db: Database) -> int:
    _add_song(db)
    return engine.ensure_today_challenge("g1", "c1", NOW).id


def _challenge_user(db: Database, challenge_id: int, user_id: str) -> dict[str, object]:
    row = db.query_one(
        "SELECT * FROM challenge_users WHERE challenge_id = ? AND user_id = ?",
        (challenge_id, user_id),
    )
    assert row is not None
    return dict(row)


def _user_stats(db: Database, user_id: str, guild_id: str = "g1") -> dict[str, object] | None:
    row = db.query_one(
        "SELECT * FROM user_stats WHERE guild_id = ? AND user_id = ?", (guild_id, user_id)
    )
    return dict(row) if row is not None else None


def _guess_rows(db: Database, challenge_id: int, user_id: str) -> list[dict[str, object]]:
    return [
        dict(row)
        for row in db.query(
            "SELECT * FROM guesses WHERE challenge_id = ? AND user_id = ? ORDER BY id",
            (challenge_id, user_id),
        )
    ]


def _solve(
    engine: GameEngine,
    challenge_id: int,
    user_id: str,
    now: datetime,
    *,
    unlocks: int = 0,
    text: str = TITLE,
) -> None:
    for _ in range(unlocks):
        engine.unlock_snippet(challenge_id, user_id)
    result = engine.submit_guess(challenge_id, user_id, text, now)
    assert result.outcome == "correct"


class TestUnlockSnippet:
    def test_unlock_walks_ladder_with_descending_points(
        self, engine: GameEngine, challenge_id: int, tmp_path: Path
    ) -> None:
        expected = [(1, 75), (2, 50), (3, 30), (4, 15)]
        for level, points in expected:
            result = engine.unlock_snippet(challenge_id, "alice")
            assert result.level == level
            assert result.potential_points == points
            assert result.path == tmp_path / "snippets" / str(challenge_id) / f"{level}.mp3"
            assert result.path.exists()

    def test_unlock_creates_challenge_users_row_on_first_interaction(
        self, engine: GameEngine, db: Database, challenge_id: int
    ) -> None:
        """Pinned #13: the per-user row is upserted on first hear-more."""
        engine.unlock_snippet(challenge_id, "alice")
        row = _challenge_user(db, challenge_id, "alice")
        assert row["snippet_level"] == 1
        assert row["guesses_used"] == 0
        assert row["solved"] == 0
        assert row["points_awarded"] == 0

    def test_unlock_rejected_at_max_level(
        self, engine: GameEngine, db: Database, challenge_id: int
    ) -> None:
        for _ in range(4):
            engine.unlock_snippet(challenge_id, "alice")
        with pytest.raises(UnlockRefusedError) as excinfo:
            engine.unlock_snippet(challenge_id, "alice")
        assert excinfo.value.reason == "max_level"
        assert _challenge_user(db, challenge_id, "alice")["snippet_level"] == 4

    def test_unlock_rejected_after_solve(
        self, engine: GameEngine, db: Database, challenge_id: int
    ) -> None:
        _solve(engine, challenge_id, "alice", NOW)
        with pytest.raises(UnlockRefusedError) as excinfo:
            engine.unlock_snippet(challenge_id, "alice")
        assert excinfo.value.reason == "solved"
        assert _challenge_user(db, challenge_id, "alice")["snippet_level"] == 0

    def test_unlock_is_per_user(self, engine: GameEngine, challenge_id: int) -> None:
        engine.unlock_snippet(challenge_id, "alice")
        engine.unlock_snippet(challenge_id, "alice")
        bob = engine.unlock_snippet(challenge_id, "bob")
        assert bob.level == 1
        assert bob.potential_points == 75

    def test_unlock_unknown_challenge_raises(self, engine: GameEngine) -> None:
        with pytest.raises(EngineError, match="unknown challenge"):
            engine.unlock_snippet(9999, "alice")


class TestSubmitGuessOutcomes:
    def test_correct_title_guess_banks_100(
        self, engine: GameEngine, db: Database, challenge_id: int
    ) -> None:
        """VAL-GUESS-002 / VAL-SCORE-001 at engine level."""
        result = engine.submit_guess(challenge_id, "alice", TITLE, NOW)

        assert result.outcome == "correct"
        assert result.matched_title is True
        assert result.matched_artist is False
        assert result.is_both is False
        assert result.points_awarded == 100
        assert result.guesses_used == 1
        assert result.guesses_left == 5
        assert result.snippet_level == 0
        assert result.announce is True

        row = _challenge_user(db, challenge_id, "alice")
        assert (row["solved"], row["guesses_used"], row["points_awarded"]) == (1, 1, 100)
        assert row["snippet_level"] == 0
        assert row["solved_at"] is not None

        guesses = _guess_rows(db, challenge_id, "alice")
        assert len(guesses) == 1
        assert (guesses[0]["matched_title"], guesses[0]["matched_artist"]) == (1, 0)
        assert guesses[0]["is_correct"] == 1

        stats = _user_stats(db, "alice")
        assert stats is not None
        assert stats["total_points"] == 100
        assert stats["wins"] == 1
        assert stats["current_streak"] == 1
        assert stats["best_streak"] == 1
        assert stats["last_win_date"] == TODAY

    def test_correct_artist_guess(
        self, engine: GameEngine, db: Database, challenge_id: int
    ) -> None:
        """VAL-GUESS-003: artist-only text solves via matched_artist."""
        result = engine.submit_guess(challenge_id, "bob", ARTIST, NOW)
        assert result.outcome == "correct"
        assert result.matched_artist is True
        assert result.matched_title is False
        assert result.points_awarded == 100
        guesses = _guess_rows(db, challenge_id, "bob")
        assert (guesses[0]["matched_title"], guesses[0]["matched_artist"]) == (0, 1)

    def test_both_match_bonus_150(
        self, engine: GameEngine, db: Database, challenge_id: int
    ) -> None:
        """VAL-GUESS-004: one guess matching BOTH fields pays 1.5x."""
        result = engine.submit_guess(challenge_id, "carol", BOTH_GUESS, NOW)
        assert result.outcome == "correct"
        assert result.is_both is True
        assert result.points_awarded == 150
        assert _challenge_user(db, challenge_id, "carol")["points_awarded"] == 150

    def test_bonus_applies_to_current_level(
        self, engine: GameEngine, db: Database, challenge_id: int
    ) -> None:
        """VAL-GUESS-005: level 2 (50 potential) both-match pays 75, not 150."""
        engine.unlock_snippet(challenge_id, "dave")
        engine.unlock_snippet(challenge_id, "dave")
        result = engine.submit_guess(challenge_id, "dave", BOTH_GUESS, NOW)
        assert result.points_awarded == 75
        assert result.snippet_level == 2  # the level actually heard at solve time
        row = _challenge_user(db, challenge_id, "dave")
        assert (row["snippet_level"], row["points_awarded"]) == (2, 75)

    def test_bonus_rounding_half_up_at_every_level(
        self, engine: GameEngine, db: Database, challenge_id: int
    ) -> None:
        """VAL-SCORE-003 / pinned #6: 150, 113, 75, 45, 23 (round-half-up)."""
        expected = [150, 113, 75, 45, 23]
        for level, points in enumerate(expected):
            user = f"b{level}"
            for _ in range(level):
                engine.unlock_snippet(challenge_id, user)
            result = engine.submit_guess(challenge_id, user, BOTH_GUESS, NOW)
            assert result.points_awarded == points
            assert _challenge_user(db, challenge_id, user)["points_awarded"] == points

    def test_wrong_guess_is_free_and_counted(
        self, engine: GameEngine, db: Database, challenge_id: int
    ) -> None:
        """VAL-GUESS-006/007: wrong guess costs nothing, counts toward the 6."""
        result = engine.submit_guess(challenge_id, "eve", WRONG, NOW)
        assert result.outcome == "wrong"
        assert result.matched_title is False
        assert result.matched_artist is False
        assert result.points_awarded == 0
        assert result.guesses_used == 1
        assert result.guesses_left == 5
        assert result.announce is False

        row = _challenge_user(db, challenge_id, "eve")
        assert (row["solved"], row["guesses_used"], row["points_awarded"]) == (0, 1, 0)
        # VAL-SCORE-005: the first processed guess registers a zero-valued
        # user_stats row (VAL-SCORE-006 explicitly allows the row to exist as
        # long as every value is zero/NULL).
        stats = _user_stats(db, "eve")
        assert stats is not None
        assert stats["total_points"] == 0
        assert stats["wins"] == 0
        assert stats["current_streak"] == 0
        assert stats["best_streak"] == 0
        assert stats["last_win_date"] is None

    def test_guess_log_preserves_raw_text_verbatim(
        self, engine: GameEngine, db: Database, challenge_id: int
    ) -> None:
        """VAL-GUESS-014: 2 wrong + 1 correct -> 3 rows, order + flags exact."""
        engine.submit_guess(challenge_id, "alice", "  first wrong  ", NOW)
        engine.submit_guess(challenge_id, "alice", "second wrong", NOW)
        engine.submit_guess(challenge_id, "alice", TITLE, NOW)

        guesses = _guess_rows(db, challenge_id, "alice")
        assert [g["text"] for g in guesses] == ["  first wrong  ", "second wrong", TITLE]
        assert [(g["matched_title"], g["matched_artist"], g["is_correct"]) for g in guesses] == [
            (0, 0, 0),
            (0, 0, 0),
            (1, 0, 1),
        ]

    def test_six_wrong_guesses_then_limit_reached(
        self, engine: GameEngine, db: Database, challenge_id: int
    ) -> None:
        """VAL-GUESS-008/009: 6th wrong exhausts; 7th rejected, zero mutation."""
        for i in range(6):
            result = engine.submit_guess(challenge_id, "eve", f"{WRONG} {i}", NOW)
            assert result.outcome == "wrong"
            assert result.guesses_left == 6 - (i + 1)

        rejected = engine.submit_guess(challenge_id, "eve", TITLE, NOW)  # would-be correct
        assert rejected.outcome == "limit_reached"
        assert rejected.guesses_used == 6
        assert rejected.guesses_left == 0
        assert rejected.announce is False

        row = _challenge_user(db, challenge_id, "eve")
        assert (row["guesses_used"], row["solved"]) == (6, 0)
        assert len(_guess_rows(db, challenge_id, "eve")) == 6  # rejection not logged

    def test_correct_on_sixth_attempt_solves(
        self, engine: GameEngine, db: Database, challenge_id: int
    ) -> None:
        """VAL-GUESS-010: the winning guess counts toward the 6 and still solves."""
        for i in range(5):
            engine.submit_guess(challenge_id, "frank", f"{WRONG} {i}", NOW)
        result = engine.submit_guess(challenge_id, "frank", TITLE, NOW)
        assert result.outcome == "correct"
        assert result.guesses_used == 6
        assert result.guesses_left == 0
        assert result.points_awarded == 100
        row = _challenge_user(db, challenge_id, "frank")
        assert (row["solved"], row["guesses_used"]) == (1, 6)

    def test_already_solved_rejected_without_state_change(
        self, engine: GameEngine, db: Database, challenge_id: int
    ) -> None:
        """VAL-GUESS-011 / pinned #13: post-solve guesses are not logged."""
        _solve(engine, challenge_id, "alice", NOW)

        result = engine.submit_guess(challenge_id, "alice", ARTIST, NOW)
        assert result.outcome == "already_solved"
        assert result.guesses_used == 1
        assert result.announce is False

        row = _challenge_user(db, challenge_id, "alice")
        assert (row["guesses_used"], row["points_awarded"]) == (1, 100)
        assert len(_guess_rows(db, challenge_id, "alice")) == 1  # unchanged
        assert _user_stats(db, "alice")["total_points"] == 100  # no double award

    def test_empty_guess_rejected_as_validation(
        self, engine: GameEngine, db: Database, challenge_id: int
    ) -> None:
        """VAL-GUESS-018 / pinned #15: empty-after-strip is not a playable guess."""
        for text in ("", "   "):
            result = engine.submit_guess(challenge_id, "alice", text, NOW)
            assert result.outcome == "empty"
            assert result.guesses_used == 0
            assert result.guesses_left == 6
            assert result.announce is False

        assert db.query("SELECT * FROM guesses") == []
        assert db.query("SELECT * FROM challenge_users") == []  # not an interaction

    def test_empty_guess_after_wrong_guesses_keeps_count(
        self, engine: GameEngine, db: Database, challenge_id: int
    ) -> None:
        engine.submit_guess(challenge_id, "alice", WRONG, NOW)
        engine.submit_guess(challenge_id, "alice", WRONG, NOW)
        result = engine.submit_guess(challenge_id, "alice", "   ", NOW)
        assert result.outcome == "empty"
        assert result.guesses_used == 2
        assert len(_guess_rows(db, challenge_id, "alice")) == 2

    def test_raw_title_fallback_solves_through_engine(
        self, engine: GameEngine, db: Database, tmp_path: Path
    ) -> None:
        """VAL-GUESS-017 at engine level: a raw_title rescue banks the win."""
        _add_song(
            db,
            "yt-1",
            title="Official Video",
            artist=None,
            raw_title="Real Artist - Real Title (Official Video)",
        )
        challenge = engine.ensure_today_challenge("g1", "c1", NOW)
        assert challenge.song.title == "Official Video"

        result = engine.submit_guess(challenge.id, "alice", "Real Title", NOW)
        assert result.outcome == "correct"
        assert result.matched_title is True
        assert result.points_awarded == 100

    def test_unknown_challenge_raises(self, engine: GameEngine) -> None:
        with pytest.raises(EngineError, match="unknown challenge"):
            engine.submit_guess(9999, "alice", TITLE, NOW)

    def test_stats_failure_rolls_back_all_guess_state(
        self, engine: GameEngine, db: Database, challenge_id: int, monkeypatch: object
    ) -> None:
        """Atomicity: a failure mid-write leaves no partial guess state."""
        original_execute = db.execute

        def failing_execute(sql: str, parameters: object = ()) -> object:
            if "user_stats" in sql:
                raise RuntimeError("injected stats failure")
            return original_execute(sql, parameters)  # type: ignore[arg-type]

        monkeypatch.setattr(db, "execute", failing_execute)  # type: ignore[attr-defined]

        with pytest.raises(RuntimeError, match="injected stats failure"):
            engine.submit_guess(challenge_id, "alice", TITLE, NOW)

        assert db.query("SELECT * FROM challenge_users") == []
        assert db.query("SELECT * FROM guesses") == []
        assert db.query("SELECT * FROM user_stats") == []


class TestStreaks:
    def test_streak_extends_across_consecutive_days(
        self, engine: GameEngine, db: Database
    ) -> None:
        """VAL-SCORE-007: day 1/2/3 solves -> streak 1/2/3, best tracks."""
        _add_song(db)
        for day, streak in ((DAY1, 1), (DAY2, 2), (DAY3, 3)):
            challenge = engine.ensure_today_challenge("g1", "c1", day)
            _solve(engine, challenge.id, "heidi", day)
            stats = _user_stats(db, "heidi")
            assert stats is not None
            assert stats["current_streak"] == streak
            assert stats["best_streak"] == streak
            assert stats["last_win_date"] == challenge.date

    def test_missed_day_resets_streak_but_keeps_best(
        self, engine: GameEngine, db: Database
    ) -> None:
        """VAL-SCORE-008: solve d1+d2, miss d3, solve d4 -> streak 1, best 2."""
        _add_song(db)
        _solve(engine, engine.ensure_today_challenge("g1", "c1", DAY1).id, "ivan", DAY1)
        _solve(engine, engine.ensure_today_challenge("g1", "c1", DAY2).id, "ivan", DAY2)
        # DAY3: ivan does not play (challenge may or may not exist)
        _solve(engine, engine.ensure_today_challenge("g1", "c1", DAY4).id, "ivan", DAY4)

        stats = _user_stats(db, "ivan")
        assert stats is not None
        assert stats["current_streak"] == 1
        assert stats["best_streak"] == 2
        assert stats["last_win_date"] == "2026-08-16"

    def test_effective_streak_computed_on_read(
        self, engine: GameEngine, db: Database
    ) -> None:
        """Pinned #7: after a missed day the leaderboard shows streak 0
        immediately, even though the stored value is only updated on solve."""
        _add_song(db)
        _solve(engine, engine.ensure_today_challenge("g1", "c1", DAY1).id, "ivan", DAY1)
        _solve(engine, engine.ensure_today_challenge("g1", "c1", DAY2).id, "ivan", DAY2)

        # Day 3 (next calendar day): streak still alive.
        (entry,) = engine.leaderboard("g1", DAY3)
        assert entry.current_streak == 2
        # Day 4 (missed day 3): effective streak collapses to 0 on read.
        (entry,) = engine.leaderboard("g1", DAY4)
        assert entry.current_streak == 0
        # The stored value is lazily left alone until the next solve.
        assert _user_stats(db, "ivan")["current_streak"] == 2

    def test_streak_dates_use_halifax_calendar(
        self, engine: GameEngine, db: Database
    ) -> None:
        """Streaks extend by America/Halifax calendar day, not UTC day."""
        _add_song(db)
        late = datetime(2026, 8, 13, 2, 30, 0, tzinfo=UTC)  # 2026-08-12 23:30 ADT
        next_late = datetime(2026, 8, 14, 1, 0, 0, tzinfo=UTC)  # 2026-08-13 22:00 ADT
        c1 = engine.ensure_today_challenge("g1", "c1", late)
        assert c1.date == "2026-08-12"
        c2 = engine.ensure_today_challenge("g1", "c1", next_late)
        assert c2.date == "2026-08-13"

        _solve(engine, c1.id, "heidi", late)
        _solve(engine, c2.id, "heidi", next_late)
        stats = _user_stats(db, "heidi")
        assert stats is not None
        assert stats["current_streak"] == 2
        assert stats["last_win_date"] == "2026-08-13"

    def test_points_and_wins_accumulate_across_days(
        self, engine: GameEngine, db: Database
    ) -> None:
        """VAL-SCORE-004/005: 100 + 75 + 50 across three days -> 225, 3 wins."""
        _add_song(db)
        _solve(engine, engine.ensure_today_challenge("g1", "c1", DAY1).id, "carol", DAY1)
        _solve(
            engine, engine.ensure_today_challenge("g1", "c1", DAY2).id, "carol", DAY2,
            unlocks=1,
        )
        _solve(
            engine, engine.ensure_today_challenge("g1", "c1", DAY3).id, "carol", DAY3,
            unlocks=2,
        )
        stats = _user_stats(db, "carol")
        assert stats is not None
        assert stats["total_points"] == 225
        assert stats["wins"] == 3


class TestLeaderboard:
    def test_ordering_points_desc_wins_desc_user_id_asc(
        self, engine: GameEngine, db: Database, challenge_id: int
    ) -> None:
        """VAL-SCORE-009: pinned tiebreak, deterministic order."""
        for level in range(4):  # p1..p4 at levels 0..3 -> 100/75/50/30
            _solve(engine, challenge_id, f"user-p{level + 1}", NOW, unlocks=level)
        _solve(engine, challenge_id, "user-p5", NOW, unlocks=4)  # 15
        _solve(engine, challenge_id, "user-p6", NOW, unlocks=4)  # 15, tie

        entries = engine.leaderboard("g1", NOW)
        assert [e.user_id for e in entries] == [
            "user-p1",
            "user-p2",
            "user-p3",
            "user-p4",
            "user-p5",
            "user-p6",
        ]
        assert [e.total_points for e in entries] == [100, 75, 50, 30, 15, 15]
        assert all(e.wins == 1 for e in entries)
        assert all(e.current_streak == 1 for e in entries)
        # Repeated invocations return the identical order.
        assert [e.user_id for e in engine.leaderboard("g1", NOW)] == [
            e.user_id for e in entries
        ]

    def test_wins_tiebreak_beats_equal_points(
        self, engine: GameEngine, db: Database
    ) -> None:
        """Equal totals: more wins ranks higher (wins DESC before user_id ASC)."""
        _add_song(db)
        # wx: two 75-point wins across two days -> 150 pts, 2 wins
        _solve(
            engine, engine.ensure_today_challenge("g1", "c1", DAY1).id, "user-wx", DAY1,
            unlocks=1,
        )
        _solve(
            engine, engine.ensure_today_challenge("g1", "c1", DAY2).id, "user-wx", DAY2,
            unlocks=1,
        )
        # wy: one both-match win at level 0 -> 150 pts, 1 win
        _solve(
            engine, engine.ensure_today_challenge("g1", "c1", DAY1).id, "user-wy", DAY1,
            text=BOTH_GUESS,
        )

        entries = engine.leaderboard("g1", DAY2)
        assert [(e.user_id, e.total_points, e.wins) for e in entries] == [
            ("user-wx", 150, 2),
            ("user-wy", 150, 1),
        ]

    def test_leaderboard_capped_at_limit(
        self, engine: GameEngine, db: Database, challenge_id: int
    ) -> None:
        """VAL-SCORE-010: 11 scoring users -> exactly 10 entries, cap after order."""
        for i in range(1, 11):  # q01..q10 at level 0 -> 100 each
            _solve(engine, challenge_id, f"user-q{i:02d}", NOW)
        _solve(engine, challenge_id, "user-q11", NOW, unlocks=1)  # 75

        entries = engine.leaderboard("g1", NOW)
        assert len(entries) == 10
        assert "user-q11" not in {e.user_id for e in entries}
        assert all(e.total_points == 100 for e in entries)

    def test_empty_leaderboard(self, engine: GameEngine, challenge_id: int) -> None:
        """VAL-SCORE-012 engine level: no scorers -> empty list (adapter renders
        the friendly 'no scores yet' message)."""
        assert engine.leaderboard("g1", NOW) == []

    def test_non_scoring_interactions_do_not_enter_leaderboard(
        self, engine: GameEngine, db: Database, challenge_id: int
    ) -> None:
        """Hear-more and wrong guesses create zero-valued user_stats rows
        (VAL-SCORE-005) but never leaderboard entries (0-point filter)."""
        engine.unlock_snippet(challenge_id, "frank")
        engine.submit_guess(challenge_id, "grace", WRONG, NOW)
        assert _user_stats(db, "frank") is not None
        assert _user_stats(db, "grace") is not None
        assert engine.leaderboard("g1", NOW) == []


class TestUserStatsRowOnInteraction:
    """VAL-SCORE-005 regression: the first interaction with a challenge
    (hear-more OR a processed guess — the pinned-#13 challenge_users upsert
    point) also upserts a zero-valued per-guild user_stats row, so a user who
    never solves still has wins=0/total_points=0 on record. The leaderboard
    must keep excluding 0-point users (VAL-SCORE-011/012 preserved), and the
    first solve must upgrade the zero row in place (no lost points, no
    double row).
    """

    def test_hear_more_only_user_has_zero_valued_stats_row(
        self, engine: GameEngine, db: Database
    ) -> None:
        """The VAL-SCORE-005 dave/erin/frank scenario at SQL level."""
        _add_song(db)
        # Day 1: dave solves; erin only hears more and guesses wrong.
        day1 = engine.ensure_today_challenge("g1", "c1", DAY1)
        _solve(engine, day1.id, "dave", DAY1)
        engine.unlock_snippet(day1.id, "erin")
        erin_wrong = engine.submit_guess(day1.id, "erin", WRONG, DAY1)
        assert erin_wrong.outcome == "wrong"
        # Day 2: dave does not play; erin solves.
        day2 = engine.ensure_today_challenge("g1", "c1", DAY2)
        _solve(engine, day2.id, "erin", DAY2)
        # Day 3: dave solves; erin does not play; frank only presses hear-more.
        day3 = engine.ensure_today_challenge("g1", "c1", DAY3)
        _solve(engine, day3.id, "dave", DAY3)
        engine.unlock_snippet(day3.id, "frank")

        rows = db.query(
            "SELECT user_id, total_points, wins, current_streak, best_streak,"
            " last_win_date FROM user_stats WHERE guild_id = 'g1' ORDER BY user_id"
        )
        by_user = {str(row["user_id"]): dict(row) for row in rows}
        assert set(by_user) == {"dave", "erin", "frank"}  # exactly one row each
        assert (by_user["dave"]["wins"], by_user["dave"]["total_points"]) == (2, 200)
        assert (by_user["erin"]["wins"], by_user["erin"]["total_points"]) == (1, 100)
        frank = by_user["frank"]
        assert frank["total_points"] == 0
        assert frank["wins"] == 0
        assert frank["current_streak"] == 0
        assert frank["best_streak"] == 0
        assert frank["last_win_date"] is None

    def test_first_wrong_guess_also_creates_the_zero_row(
        self, engine: GameEngine, db: Database, challenge_id: int
    ) -> None:
        """A processed guess is a first interaction too (pinned #13): the
        zero-valued row appears even without any hear-more press."""
        result = engine.submit_guess(challenge_id, "grace", WRONG, NOW)
        assert result.outcome == "wrong"
        stats = _user_stats(db, "grace")
        assert stats is not None
        assert (stats["total_points"], stats["wins"]) == (0, 0)
        assert stats["last_win_date"] is None

    def test_zero_point_users_never_enter_leaderboard(
        self, engine: GameEngine, db: Database, challenge_id: int
    ) -> None:
        """VAL-SCORE-011 preserved: only solvers are listed; the zero-valued
        rows exist in SQL but are filtered out of the leaderboard."""
        engine.unlock_snippet(challenge_id, "frank")  # hear-more only
        engine.submit_guess(challenge_id, "grace", WRONG, NOW)  # wrong only
        _solve(engine, challenge_id, "alice", NOW)

        entries = engine.leaderboard("g1", NOW)
        assert [e.user_id for e in entries] == ["alice"]
        frank = _user_stats(db, "frank")
        grace = _user_stats(db, "grace")
        assert frank is not None
        assert frank["total_points"] == 0
        assert grace is not None
        assert grace["total_points"] == 0

    def test_guild_with_only_zero_point_users_gets_empty_leaderboard(
        self, engine: GameEngine, challenge_id: int
    ) -> None:
        """VAL-SCORE-012 preserved: a guild where nobody has scored still
        reads empty (the adapter renders the friendly 'no scores yet')."""
        engine.unlock_snippet(challenge_id, "frank")
        engine.submit_guess(challenge_id, "grace", WRONG, NOW)
        assert engine.leaderboard("g1", NOW) == []

    def test_solve_after_hear_more_upgrades_the_zero_row(
        self, engine: GameEngine, db: Database, challenge_id: int
    ) -> None:
        """The first solve updates the pre-existing zero row in place: exact
        points/wins/streak, and still exactly one row (no double insert)."""
        engine.unlock_snippet(challenge_id, "erin")
        engine.unlock_snippet(challenge_id, "erin")  # level 2 -> 50 potential
        before = _user_stats(db, "erin")
        assert before is not None
        assert before["total_points"] == 0

        result = engine.submit_guess(challenge_id, "erin", TITLE, NOW)
        assert result.outcome == "correct"
        assert result.points_awarded == 50

        rows = db.query(
            "SELECT * FROM user_stats WHERE guild_id = 'g1' AND user_id = 'erin'"
        )
        assert len(rows) == 1  # no double row
        stats = dict(rows[0])
        assert stats["total_points"] == 50
        assert stats["wins"] == 1
        assert stats["current_streak"] == 1
        assert stats["best_streak"] == 1
        assert stats["last_win_date"] == TODAY


class TestRevealedChallengeLockout:
    """VAL-GUESS-019 (engine half): a revealed challenge refuses all gameplay.

    After the daily reveal the answer is public, so the persistent view on
    yesterday's message must not farm points: ``submit_guess`` returns
    ``challenge_closed`` and ``unlock_snippet`` raises ``UnlockRefusedError``
    with reason ``"closed"`` — both with ZERO mutation (no guesses rows, no
    points, no challenge_users/user_stats changes, no snippet-level change).
    """

    def test_correct_guess_on_revealed_challenge_refused_with_zero_mutation(
        self, db: Database, tmp_path: Path
    ) -> None:
        engine, _ = _make_engine(tmp_path, db)
        _add_song(db)
        challenge_id = engine.ensure_today_challenge("g1", "c1", NOW).id
        # Pre-reveal activity that must survive the refusal untouched.
        engine.unlock_snippet(challenge_id, "alice")
        engine.submit_guess(challenge_id, "alice", WRONG, NOW)
        reveal = _reveal_previous(engine, "g1", DAY2)  # the delivered day-advance path
        assert reveal is not None
        assert reveal.challenge_id == challenge_id

        before = _db_snapshot(db)
        result = engine.submit_guess(challenge_id, "alice", TITLE, DAY2)  # correct answer

        assert result.outcome == "challenge_closed"
        assert result.matched_title is False
        assert result.matched_artist is False
        assert result.is_both is False
        assert result.points_awarded == 0
        assert result.announce is False
        assert result.guesses_used == 1  # the unchanged pre-reveal count
        assert result.guesses_left == 5
        assert _db_snapshot(db) == before

    def test_unlock_on_revealed_challenge_refused_with_zero_mutation(
        self, db: Database, tmp_path: Path
    ) -> None:
        engine, fake = _make_engine(tmp_path, db)
        _add_song(db)
        challenge_id = engine.ensure_today_challenge("g1", "c1", NOW).id
        engine.unlock_snippet(challenge_id, "alice")  # alice at level 1
        assert _reveal_previous(engine, "g1", DAY2) is not None

        before = _db_snapshot(db)
        ensure_calls_before = len(fake.ensure_calls)
        with pytest.raises(UnlockRefusedError) as excinfo:
            engine.unlock_snippet(challenge_id, "alice")

        assert excinfo.value.reason == "closed"
        assert _db_snapshot(db) == before
        assert _challenge_user(db, challenge_id, "alice")["snippet_level"] == 1
        # The refusal precedes the snippet re-heal: no ensure_snippets call.
        assert len(fake.ensure_calls) == ensure_calls_before

    def test_fresh_user_refused_without_creating_any_rows(
        self, db: Database, tmp_path: Path
    ) -> None:
        """A first-time interactor is refused AND no challenge_users row is
        upserted — a lockout refusal is not an interaction (contrast #13)."""
        engine, _ = _make_engine(tmp_path, db)
        _add_song(db)
        challenge_id = engine.ensure_today_challenge("g1", "c1", NOW).id
        assert _reveal_previous(engine, "g1", DAY2) is not None

        before = _db_snapshot(db)
        result = engine.submit_guess(challenge_id, "bob", TITLE, DAY2)
        assert result.outcome == "challenge_closed"
        assert result.guesses_used == 0
        assert result.guesses_left == 6
        with pytest.raises(UnlockRefusedError) as excinfo:
            engine.unlock_snippet(challenge_id, "bob")
        assert excinfo.value.reason == "closed"

        assert _db_snapshot(db) == before
        assert db.query("SELECT * FROM challenge_users") == []
        assert db.query("SELECT * FROM user_stats") == []

    def test_lockout_dominates_the_already_solved_refusal(
        self, db: Database, tmp_path: Path
    ) -> None:
        """Check order: closed beats already_solved, and the solver's banked
        state (100 points, streak) stays exactly as earned."""
        engine, _ = _make_engine(tmp_path, db)
        _add_song(db)
        challenge_id = engine.ensure_today_challenge("g1", "c1", NOW).id
        _solve(engine, challenge_id, "alice", NOW)
        assert _reveal_previous(engine, "g1", DAY2) is not None

        before = _db_snapshot(db)
        result = engine.submit_guess(challenge_id, "alice", ARTIST, DAY2)
        assert result.outcome == "challenge_closed"
        assert result.guesses_used == 1
        with pytest.raises(UnlockRefusedError) as excinfo:
            engine.unlock_snippet(challenge_id, "alice")
        assert excinfo.value.reason == "closed"

        assert _db_snapshot(db) == before
        stats = _user_stats(db, "alice")
        assert stats is not None
        assert stats["total_points"] == 100

    def test_empty_guess_on_revealed_challenge_reports_closed(
        self, db: Database, tmp_path: Path
    ) -> None:
        """Check order: the closed lockout is evaluated before the pinned-#15
        empty-guess validation — the adapter only needs the closed message."""
        engine, _ = _make_engine(tmp_path, db)
        _add_song(db)
        challenge_id = engine.ensure_today_challenge("g1", "c1", NOW).id
        assert _reveal_previous(engine, "g1", DAY2) is not None

        result = engine.submit_guess(challenge_id, "alice", "   ", DAY2)
        assert result.outcome == "challenge_closed"
        assert db.query("SELECT * FROM guesses") == []
        assert db.query("SELECT * FROM challenge_users") == []
