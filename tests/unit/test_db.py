"""Unit tests for songbot.db: schema, versioned migrations, constraints, row helpers.

Every test uses a tmp-path (or in-memory) database; none touches data/songbot.db.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest

from songbot.db import (
    SCHEMA_VERSION,
    ChallengeRow,
    ChallengeUserRow,
    Database,
    GuessRow,
    SongRow,
    UserStatsRow,
)

NOW = "2026-08-13T12:00:00+00:00"

EXPECTED_TABLES = {
    "schema_migrations",
    "songs",
    "challenges",
    "challenge_users",
    "guesses",
    "user_stats",
    "guild_settings",
}

# Exact PRAGMA table_info tuples (name, type, notnull, dflt_value, pk) per the
# architecture.md data model. INTEGER PRIMARY KEY columns report notnull=0
# (rowid-alias quirk); composite-PK columns are declared NOT NULL explicitly.
EXPECTED_COLUMNS: dict[str, list[tuple[str, str, int, str | None, int]]] = {
    "schema_migrations": [
        ("version", "INTEGER", 0, None, 1),
        ("applied_at", "TEXT", 1, None, 0),
    ],
    "songs": [
        ("id", "INTEGER", 0, None, 1),
        ("source", "TEXT", 1, None, 0),
        ("source_id", "TEXT", 1, None, 0),
        ("title", "TEXT", 1, None, 0),
        ("artist", "TEXT", 0, None, 0),  # nullable: bare YouTube titles may lack an artist
        ("duration_sec", "REAL", 1, None, 0),
        ("audio_ref", "TEXT", 1, None, 0),
        ("raw_title", "TEXT", 1, None, 0),
        ("created_at", "TEXT", 1, None, 0),
    ],
    "challenges": [
        ("id", "INTEGER", 0, None, 1),
        ("guild_id", "TEXT", 1, None, 0),
        ("channel_id", "TEXT", 1, None, 0),
        ("song_id", "INTEGER", 1, None, 0),
        ("date", "TEXT", 1, None, 0),
        ("snippet_offset_sec", "REAL", 1, None, 0),
        ("status", "TEXT", 1, None, 0),
        ("created_at", "TEXT", 1, None, 0),
        ("revealed_at", "TEXT", 0, None, 0),
        ("skip_count", "INTEGER", 1, "0", 0),  # migration 2: skip-song seed (pinned #5)
    ],
    "challenge_users": [
        ("challenge_id", "INTEGER", 1, None, 1),
        ("user_id", "TEXT", 1, None, 2),
        ("snippet_level", "INTEGER", 1, "0", 0),
        ("guesses_used", "INTEGER", 1, "0", 0),
        ("solved", "INTEGER", 1, "0", 0),
        ("points_awarded", "INTEGER", 1, "0", 0),
        ("solved_at", "TEXT", 0, None, 0),
    ],
    "guesses": [
        ("id", "INTEGER", 0, None, 1),
        ("challenge_id", "INTEGER", 1, None, 0),
        ("user_id", "TEXT", 1, None, 0),
        ("text", "TEXT", 1, None, 0),
        ("matched_title", "INTEGER", 1, None, 0),
        ("matched_artist", "INTEGER", 1, None, 0),
        ("is_correct", "INTEGER", 1, None, 0),
        ("created_at", "TEXT", 1, None, 0),
    ],
    "user_stats": [
        ("guild_id", "TEXT", 1, None, 1),
        ("user_id", "TEXT", 1, None, 2),
        ("total_points", "INTEGER", 1, "0", 0),
        ("wins", "INTEGER", 1, "0", 0),
        ("current_streak", "INTEGER", 1, "0", 0),
        ("best_streak", "INTEGER", 1, "0", 0),
        ("last_win_date", "TEXT", 0, None, 0),
    ],
    # migration 3: multi-guild post targets (the /songbot-setup store)
    "guild_settings": [
        ("guild_id", "TEXT", 1, None, 1),
        ("channel_id", "TEXT", 1, None, 0),
        ("set_by", "TEXT", 1, None, 0),
        ("created_at", "TEXT", 1, None, 0),
        ("updated_at", "TEXT", 1, None, 0),
    ],
}


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    """A migrated database at a tmp path, closed after the test."""
    database = Database.open(tmp_path / "songbot.db")
    yield database
    database.close()


def insert_song(
    db: Database,
    *,
    source: str = "local",
    source_id: str = "song-1",
    title: str = "Neon Skyline",
    artist: str | None = "Midnight Circuit",
    duration_sec: float = 30.0,
    created_at: str = NOW,
) -> int:
    cursor = db.execute(
        "INSERT INTO songs (source, source_id, title, artist, duration_sec, audio_ref,"
        " raw_title, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            source,
            source_id,
            title,
            artist,
            duration_sec,
            f"/music/{source_id}.mp3",
            f"{source_id} raw",
            created_at,
        ),
    )
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def insert_challenge(
    db: Database,
    song_id: int,
    *,
    guild_id: str = "g1",
    channel_id: str = "c1",
    date: str = "2026-08-13",
    offset: float = 5.0,
    status: str = "active",
    created_at: str = NOW,
) -> int:
    cursor = db.execute(
        "INSERT INTO challenges (guild_id, channel_id, song_id, date, snippet_offset_sec,"
        " status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (guild_id, channel_id, song_id, date, offset, status, created_at),
    )
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def insert_guess(
    db: Database,
    challenge_id: int,
    user_id: str,
    text: str,
    *,
    matched_title: int = 0,
    matched_artist: int = 0,
    is_correct: int = 0,
) -> int:
    cursor = db.execute(
        "INSERT INTO guesses (challenge_id, user_id, text, matched_title, matched_artist,"
        " is_correct, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (challenge_id, user_id, text, matched_title, matched_artist, is_correct, NOW),
    )
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def count(db: Database, table: str) -> int:
    row = db.query_one(f"SELECT COUNT(*) AS c FROM {table}")
    assert row is not None
    return int(row["c"])


class TestMigration:
    def test_fresh_db_migrates_cleanly(self, db: Database) -> None:
        tables = {
            str(row["name"])
            for row in db.query("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert tables == EXPECTED_TABLES

    def test_migrate_returns_applied_versions_then_nothing(self, tmp_path: Path) -> None:
        database = Database(tmp_path / "songbot.db")
        try:
            assert database.migrate() == [1, 2, 3]
            assert database.migrate() == []
        finally:
            database.close()

    def test_migrate_is_idempotent_across_reopens(self, tmp_path: Path) -> None:
        path = tmp_path / "songbot.db"
        first = Database.open(path)
        first.close()
        second = Database(path)
        try:
            assert second.migrate() == []
            assert second.schema_version() == SCHEMA_VERSION
            assert count(second, "schema_migrations") == 3
        finally:
            second.close()

    def test_schema_migrations_records_version_and_timestamp(self, db: Database) -> None:
        rows = db.query("SELECT version, applied_at FROM schema_migrations ORDER BY version")
        assert [int(row["version"]) for row in rows] == [1, 2, 3]
        for row in rows:
            # applied_at must be a parseable ISO-8601 timestamp
            parsed = datetime.fromisoformat(str(row["applied_at"]))
            assert parsed.tzinfo is not None

    def test_migration_2_adds_skip_count_to_existing_challenges(self, tmp_path: Path) -> None:
        """Upgrading a v1 database: pre-existing challenge rows get skip_count=0."""
        path = tmp_path / "songbot.db"
        database = Database(path)
        try:
            # Simulate a v1-only database: bookkeeping table + migration 1 by hand.
            from songbot.db import MIGRATIONS

            database.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            for statement in MIGRATIONS[0][1]:
                database.execute(statement)
            database.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (1, ?)", (NOW,)
            )
            song_id = insert_song(database)
            challenge_id = insert_challenge(database, song_id)

            assert database.migrate() == [2, 3]
            assert database.schema_version() == 3
            row = database.query_one(
                "SELECT skip_count FROM challenges WHERE id = ?", (challenge_id,)
            )
            assert row is not None
            assert row["skip_count"] == 0
        finally:
            database.close()

    def test_migration_3_adds_guild_settings_to_a_v2_database(
        self, tmp_path: Path
    ) -> None:
        """Upgrading a v2 database: guild_settings appears, game data untouched."""
        path = tmp_path / "songbot.db"
        database = Database(path)
        try:
            # Simulate a v2 database: migrations 1+2 applied by hand.
            from songbot.db import MIGRATIONS

            database.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            for version, statements in MIGRATIONS:
                if version > 2:
                    continue
                for statement in statements:
                    database.execute(statement)
                database.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (version, NOW),
                )
            song_id = insert_song(database)
            challenge_id = insert_challenge(database, song_id)

            assert database.migrate() == [3]
            assert database.schema_version() == 3

            table = database.query_one(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
                " AND name = 'guild_settings'"
            )
            assert table is not None
            assert count(database, "guild_settings") == 0
            # Pre-existing rows survive the upgrade.
            row = database.query_one(
                "SELECT skip_count FROM challenges WHERE id = ?", (challenge_id,)
            )
            assert row is not None
            assert row["skip_count"] == 0
        finally:
            database.close()

    def test_schema_version_zero_before_migrate(self, tmp_path: Path) -> None:
        database = Database(tmp_path / "songbot.db")
        try:
            assert database.schema_version() == 0
            database.migrate()
            assert database.schema_version() == SCHEMA_VERSION
        finally:
            database.close()

    def test_open_classmethod_applies_migrations(self, tmp_path: Path) -> None:
        database = Database.open(tmp_path / "songbot.db")
        try:
            assert database.schema_version() == SCHEMA_VERSION
        finally:
            database.close()

    def test_data_persists_across_reopen(self, tmp_path: Path) -> None:
        path = tmp_path / "songbot.db"
        first = Database.open(path)
        insert_song(first, source_id="persist-me")
        first.close()
        second = Database.open(path)
        try:
            row = second.query_one("SELECT source_id FROM songs")
            assert row is not None
            assert row["source_id"] == "persist-me"
        finally:
            second.close()

    def test_in_memory_database_supported(self) -> None:
        database = Database.open(":memory:")
        try:
            assert database.schema_version() == SCHEMA_VERSION
            insert_song(database)
            assert count(database, "songs") == 1
        finally:
            database.close()


class TestSchemaMatchesArchitecture:
    @pytest.mark.parametrize("table", sorted(EXPECTED_COLUMNS))
    def test_table_columns_exact(self, db: Database, table: str) -> None:
        info = [tuple(row)[1:] for row in db.query(f"PRAGMA table_info({table})")]
        assert info == EXPECTED_COLUMNS[table]

    def test_foreign_keys_match_architecture(self, db: Database) -> None:
        # (table, from, to, on_delete) per foreign_key_list row
        def fks(table: str) -> list[tuple[str, str, str, str]]:
            return [
                (str(row[2]), str(row[3]), str(row[4]), str(row[6]))
                for row in db.query(f"PRAGMA foreign_key_list({table})")
            ]

        assert fks("challenges") == [("songs", "song_id", "id", "NO ACTION")]
        assert fks("challenge_users") == [("challenges", "challenge_id", "id", "CASCADE")]
        assert fks("guesses") == [("challenges", "challenge_id", "id", "CASCADE")]


class TestPragmas:
    def test_wal_journal_mode(self, db: Database) -> None:
        row = db.query_one("PRAGMA journal_mode")
        assert row is not None
        assert str(row[0]).lower() == "wal"

    def test_foreign_keys_enabled(self, db: Database) -> None:
        row = db.query_one("PRAGMA foreign_keys")
        assert row is not None
        assert row[0] == 1

    def test_connection_usable_from_other_thread(self, db: Database) -> None:
        # check_same_thread=False: a query from another thread must not raise.
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                db.query("SELECT COUNT(*) FROM songs")
            except BaseException as exc:  # collected for assertion below
                errors.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        assert errors == []


class TestSongsTable:
    def test_unique_source_source_id_enforced(self, db: Database) -> None:
        insert_song(db, source="youtube", source_id="vid-1")
        with pytest.raises(sqlite3.IntegrityError):
            insert_song(db, source="youtube", source_id="vid-1", title="Duplicate")

    def test_same_source_id_allowed_for_different_source(self, db: Database) -> None:
        insert_song(db, source="local", source_id="same-id")
        insert_song(db, source="youtube", source_id="same-id")
        assert count(db, "songs") == 2

    def test_artist_nullable(self, db: Database) -> None:
        # Bare YouTube titles may yield no artist (VAL-CATALOG-009).
        song_id = insert_song(db, artist=None)
        row = db.query_one("SELECT artist FROM songs WHERE id = ?", (song_id,))
        assert row is not None
        assert row["artist"] is None


class TestChallengesTable:
    def test_unique_guild_date_enforced(self, db: Database) -> None:
        song_id = insert_song(db)
        insert_challenge(db, song_id, guild_id="g1", date="2026-08-13")
        with pytest.raises(sqlite3.IntegrityError):
            insert_challenge(db, song_id, guild_id="g1", date="2026-08-13")

    def test_same_date_allowed_in_different_guilds(self, db: Database) -> None:
        song_id = insert_song(db)
        insert_challenge(db, song_id, guild_id="g1", date="2026-08-13")
        insert_challenge(db, song_id, guild_id="g2", date="2026-08-13")
        assert count(db, "challenges") == 2

    def test_same_guild_allowed_on_different_dates(self, db: Database) -> None:
        song_id = insert_song(db)
        insert_challenge(db, song_id, date="2026-08-12")
        insert_challenge(db, song_id, date="2026-08-13")
        assert count(db, "challenges") == 2

    def test_song_fk_enforced(self, db: Database) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            insert_challenge(db, song_id=9999)

    def test_revealed_at_defaults_to_null(self, db: Database) -> None:
        challenge_id = insert_challenge(db, insert_song(db))
        row = db.query_one("SELECT revealed_at FROM challenges WHERE id = ?", (challenge_id,))
        assert row is not None
        assert row["revealed_at"] is None


class TestChallengeUsersTable:
    def test_composite_pk_enforced(self, db: Database) -> None:
        challenge_id = insert_challenge(db, insert_song(db))
        db.execute(
            "INSERT INTO challenge_users (challenge_id, user_id) VALUES (?, ?)",
            (challenge_id, "u1"),
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO challenge_users (challenge_id, user_id) VALUES (?, ?)",
                (challenge_id, "u1"),
            )

    def test_same_user_allowed_on_different_challenges(self, db: Database) -> None:
        song_id = insert_song(db)
        c1 = insert_challenge(db, song_id, date="2026-08-12")
        c2 = insert_challenge(db, song_id, date="2026-08-13")
        for challenge_id in (c1, c2):
            db.execute(
                "INSERT INTO challenge_users (challenge_id, user_id) VALUES (?, ?)",
                (challenge_id, "u1"),
            )
        assert count(db, "challenge_users") == 2

    def test_defaults(self, db: Database) -> None:
        challenge_id = insert_challenge(db, insert_song(db))
        db.execute(
            "INSERT INTO challenge_users (challenge_id, user_id) VALUES (?, ?)",
            (challenge_id, "u1"),
        )
        row = db.query_one(
            "SELECT snippet_level, guesses_used, solved, points_awarded, solved_at"
            " FROM challenge_users WHERE challenge_id = ? AND user_id = ?",
            (challenge_id, "u1"),
        )
        assert row is not None
        assert tuple(row) == (0, 0, 0, 0, None)

    def test_challenge_fk_enforced(self, db: Database) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO challenge_users (challenge_id, user_id) VALUES (?, ?)",
                (9999, "u1"),
            )


class TestGuessesTable:
    def test_challenge_fk_enforced(self, db: Database) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            insert_guess(db, challenge_id=9999, user_id="u1", text="nope")

    def test_full_log_keeps_every_guess(self, db: Database) -> None:
        challenge_id = insert_challenge(db, insert_song(db))
        for i in range(3):
            insert_guess(db, challenge_id, "u1", f"guess {i}")
        assert count(db, "guesses") == 3


class TestUserStatsTable:
    def test_composite_pk_enforced(self, db: Database) -> None:
        db.execute("INSERT INTO user_stats (guild_id, user_id) VALUES (?, ?)", ("g1", "u1"))
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO user_stats (guild_id, user_id) VALUES (?, ?)", ("g1", "u1")
            )

    def test_same_user_allowed_in_different_guilds(self, db: Database) -> None:
        db.execute("INSERT INTO user_stats (guild_id, user_id) VALUES (?, ?)", ("g1", "u1"))
        db.execute("INSERT INTO user_stats (guild_id, user_id) VALUES (?, ?)", ("g2", "u1"))
        assert count(db, "user_stats") == 2

    def test_defaults(self, db: Database) -> None:
        db.execute("INSERT INTO user_stats (guild_id, user_id) VALUES (?, ?)", ("g1", "u1"))
        row = db.query_one(
            "SELECT total_points, wins, current_streak, best_streak, last_win_date"
            " FROM user_stats WHERE guild_id = ? AND user_id = ?",
            ("g1", "u1"),
        )
        assert row is not None
        assert tuple(row) == (0, 0, 0, 0, None)


class TestCascadeDelete:
    """ON DELETE CASCADE backs skip-song delete+recreate (pinned decision #5)."""

    def test_delete_challenge_cascades_users_and_guesses(self, db: Database) -> None:
        challenge_id = insert_challenge(db, insert_song(db))
        for user in ("u1", "u2"):
            db.execute(
                "INSERT INTO challenge_users (challenge_id, user_id) VALUES (?, ?)",
                (challenge_id, user),
            )
        for i in range(3):
            insert_guess(db, challenge_id, "u1", f"guess {i}")

        db.execute("DELETE FROM challenges WHERE id = ?", (challenge_id,))

        assert count(db, "challenge_users") == 0
        assert count(db, "guesses") == 0
        # The song itself is retained (catalog history is not cascade-deleted).
        assert count(db, "songs") == 1

    def test_cascade_is_scoped_to_the_deleted_challenge(self, db: Database) -> None:
        song_id = insert_song(db)
        kept = insert_challenge(db, song_id, date="2026-08-12")
        deleted = insert_challenge(db, song_id, date="2026-08-13")
        for challenge_id in (kept, deleted):
            db.execute(
                "INSERT INTO challenge_users (challenge_id, user_id) VALUES (?, ?)",
                (challenge_id, "u1"),
            )
            insert_guess(db, challenge_id, "u1", "guess")

        db.execute("DELETE FROM challenges WHERE id = ?", (deleted,))

        assert count(db, "challenge_users") == 1
        assert count(db, "guesses") == 1
        row = db.query_one("SELECT challenge_id FROM challenge_users")
        assert row is not None
        assert row["challenge_id"] == kept

    def test_deleting_referenced_song_is_rejected(self, db: Database) -> None:
        # refresh_catalog may only delete songs never referenced by challenges
        # (pinned decision #12); the FK is the database-level safety net.
        song_id = insert_song(db)
        insert_challenge(db, song_id)
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("DELETE FROM songs WHERE id = ?", (song_id,))

    def test_deleting_unreferenced_song_is_allowed(self, db: Database) -> None:
        song_id = insert_song(db)
        db.execute("DELETE FROM songs WHERE id = ?", (song_id,))
        assert count(db, "songs") == 0


class TestRowHelpers:
    def test_song_row_round_trip(self, db: Database) -> None:
        song_id = insert_song(db, artist=None)
        row = db.query_one("SELECT * FROM songs WHERE id = ?", (song_id,))
        assert row is not None
        song = SongRow.from_row(row)
        assert song == SongRow(
            id=song_id,
            source="local",
            source_id="song-1",
            title="Neon Skyline",
            artist=None,
            duration_sec=30.0,
            audio_ref="/music/song-1.mp3",
            raw_title="song-1 raw",
            created_at=NOW,
        )
        assert isinstance(song.duration_sec, float)

    def test_challenge_row_round_trip_active(self, db: Database) -> None:
        challenge_id = insert_challenge(db, insert_song(db))
        row = db.query_one("SELECT * FROM challenges WHERE id = ?", (challenge_id,))
        assert row is not None
        challenge = ChallengeRow.from_row(row)
        assert challenge.id == challenge_id
        assert challenge.guild_id == "g1"
        assert challenge.channel_id == "c1"
        assert challenge.date == "2026-08-13"
        assert challenge.snippet_offset_sec == 5.0
        assert challenge.status == "active"
        assert challenge.revealed_at is None

    def test_challenge_row_round_trip_revealed(self, db: Database) -> None:
        challenge_id = insert_challenge(db, insert_song(db))
        db.execute(
            "UPDATE challenges SET status = 'revealed', revealed_at = ? WHERE id = ?",
            (NOW, challenge_id),
        )
        row = db.query_one("SELECT * FROM challenges WHERE id = ?", (challenge_id,))
        assert row is not None
        challenge = ChallengeRow.from_row(row)
        assert challenge.status == "revealed"
        assert challenge.revealed_at == NOW

    def test_challenge_user_row_converts_solved_to_bool(self, db: Database) -> None:
        challenge_id = insert_challenge(db, insert_song(db))
        db.execute(
            "INSERT INTO challenge_users (challenge_id, user_id) VALUES (?, ?)",
            (challenge_id, "u1"),
        )
        row = db.query_one("SELECT * FROM challenge_users")
        assert row is not None
        user = ChallengeUserRow.from_row(row)
        assert user.snippet_level == 0
        assert user.guesses_used == 0
        assert user.solved is False
        assert user.points_awarded == 0
        assert user.solved_at is None

        db.execute(
            "UPDATE challenge_users SET solved = 1, solved_at = ? WHERE challenge_id = ?",
            (NOW, challenge_id),
        )
        row = db.query_one("SELECT * FROM challenge_users")
        assert row is not None
        assert ChallengeUserRow.from_row(row).solved is True

    def test_guess_row_converts_flags_to_bool(self, db: Database) -> None:
        challenge_id = insert_challenge(db, insert_song(db))
        guess_id = insert_guess(
            db, challenge_id, "u1", "Midnight Circuit", matched_artist=1, is_correct=1
        )
        row = db.query_one("SELECT * FROM guesses WHERE id = ?", (guess_id,))
        assert row is not None
        guess = GuessRow.from_row(row)
        assert guess.text == "Midnight Circuit"
        assert guess.matched_title is False
        assert guess.matched_artist is True
        assert guess.is_correct is True

    def test_user_stats_row_round_trip(self, db: Database) -> None:
        db.execute(
            "INSERT INTO user_stats (guild_id, user_id, total_points, wins, current_streak,"
            " best_streak, last_win_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("g1", "u1", 225, 3, 2, 4, "2026-08-12"),
        )
        row = db.query_one("SELECT * FROM user_stats")
        assert row is not None
        stats = UserStatsRow.from_row(row)
        assert stats == UserStatsRow(
            guild_id="g1",
            user_id="u1",
            total_points=225,
            wins=3,
            current_streak=2,
            best_streak=4,
            last_win_date="2026-08-12",
        )


class TestTransactions:
    def test_transaction_commits_on_success(self, db: Database) -> None:
        with db.transaction():
            insert_song(db, source_id="a")
            insert_song(db, source_id="b")
        assert count(db, "songs") == 2

    def test_transaction_rolls_back_on_exception(self, db: Database) -> None:
        def run() -> None:
            with db.transaction():
                insert_song(db, source_id="a")
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            run()
        assert count(db, "songs") == 0

    def test_nested_transaction_joins_outer(self, db: Database) -> None:
        with db.transaction():
            insert_song(db, source_id="outer")
            with db.transaction():
                insert_song(db, source_id="inner")
        assert count(db, "songs") == 2

    def test_nested_exception_marks_rollback_only_even_when_caught(self, db: Database) -> None:
        def inner() -> None:
            with db.transaction():
                insert_song(db, source_id="inner")
                raise RuntimeError("boom")

        with db.transaction():
            insert_song(db, source_id="outer")
            with pytest.raises(RuntimeError, match="boom"):
                inner()
        # The inner failure poisons the outer transaction: nothing commits.
        assert count(db, "songs") == 0

    def test_execute_outside_transaction_autocommits(self, tmp_path: Path) -> None:
        # isolation_level=None: writes are immediately visible to other connections.
        path = tmp_path / "songbot.db"
        first = Database.open(path)
        insert_song(first, source_id="visible")
        second = Database(path)
        try:
            assert count(second, "songs") == 1
        finally:
            second.close()
            first.close()


class TestLifecycle:
    def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deeper" / "songbot.db"
        database = Database.open(path)
        try:
            assert path.exists()
            assert database.path == path
        finally:
            database.close()

    def test_context_manager_closes_connection(self, tmp_path: Path) -> None:
        with Database.open(tmp_path / "songbot.db") as database:
            database.migrate()
        with pytest.raises(sqlite3.ProgrammingError):
            database.query("SELECT 1")
