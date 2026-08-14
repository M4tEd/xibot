"""SQLite persistence layer: schema, versioned migrations, connection helper.

Sync sqlite3 (check_same_thread=False, WAL, foreign_keys ON). The engine owns
all SQL through this module — no SQL elsewhere.

Schema (migration 1) implements the architecture.md data model exactly:
`songs`, `challenges`, `challenge_users`, `guesses`, `user_stats`, plus the
`schema_migrations` bookkeeping table. `challenge_users` and `guesses` carry
`ON DELETE CASCADE` so skip-song can delete + recreate a challenge row
(pinned design decision #5).
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, cast

__all__ = [
    "SCHEMA_VERSION",
    "ChallengeRow",
    "ChallengeStatus",
    "ChallengeUserRow",
    "Database",
    "GuessRow",
    "SongRow",
    "UserStatsRow",
]

ChallengeStatus = Literal["active", "revealed"]
"""Valid values for `challenges.status`."""

SCHEMA_VERSION = 2
"""Latest schema version; bump when appending to `MIGRATIONS`."""

_BUSY_TIMEOUT_MS = 5000

_MIGRATION_001_INITIAL: tuple[str, ...] = (
    """
    CREATE TABLE songs (
        id INTEGER PRIMARY KEY,
        source TEXT NOT NULL,
        source_id TEXT NOT NULL,
        title TEXT NOT NULL,
        artist TEXT,
        duration_sec REAL NOT NULL,
        audio_ref TEXT NOT NULL,
        raw_title TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (source, source_id)
    )
    """,
    """
    CREATE TABLE challenges (
        id INTEGER PRIMARY KEY,
        guild_id TEXT NOT NULL,
        channel_id TEXT NOT NULL,
        song_id INTEGER NOT NULL REFERENCES songs(id),
        date TEXT NOT NULL,
        snippet_offset_sec REAL NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        revealed_at TEXT,
        UNIQUE (guild_id, date)
    )
    """,
    """
    CREATE TABLE challenge_users (
        challenge_id INTEGER NOT NULL REFERENCES challenges(id) ON DELETE CASCADE,
        user_id TEXT NOT NULL,
        snippet_level INTEGER NOT NULL DEFAULT 0,
        guesses_used INTEGER NOT NULL DEFAULT 0,
        solved INTEGER NOT NULL DEFAULT 0,
        points_awarded INTEGER NOT NULL DEFAULT 0,
        solved_at TEXT,
        PRIMARY KEY (challenge_id, user_id)
    )
    """,
    """
    CREATE TABLE guesses (
        id INTEGER PRIMARY KEY,
        challenge_id INTEGER NOT NULL REFERENCES challenges(id) ON DELETE CASCADE,
        user_id TEXT NOT NULL,
        text TEXT NOT NULL,
        matched_title INTEGER NOT NULL,
        matched_artist INTEGER NOT NULL,
        is_correct INTEGER NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE user_stats (
        guild_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        total_points INTEGER NOT NULL DEFAULT 0,
        wins INTEGER NOT NULL DEFAULT 0,
        current_streak INTEGER NOT NULL DEFAULT 0,
        best_streak INTEGER NOT NULL DEFAULT 0,
        last_win_date TEXT,
        PRIMARY KEY (guild_id, user_id)
    )
    """,
)

_MIGRATION_002_CHALLENGE_SKIP_COUNT: tuple[str, ...] = (
    # skip_count feeds the deterministic skip-song RNG seed
    # hash(date, guild_id, skip_count) (pinned design decision #5). It lives on
    # the challenge row itself: skip_today_song deletes the row and recreates
    # it for the same date with skip_count + 1, so the count survives the
    # delete+recreate cycle without extra tables.
    """
    ALTER TABLE challenges ADD COLUMN skip_count INTEGER NOT NULL DEFAULT 0
    """,
)

# Versioned migrations, applied in ascending order. Never edit an applied
# migration; append a new (version, statements) entry instead.
MIGRATIONS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (1, _MIGRATION_001_INITIAL),
    (2, _MIGRATION_002_CHALLENGE_SKIP_COUNT),
)


@dataclass(frozen=True)
class SongRow:
    """A row of `songs`: one catalog entry.

    `artist` is nullable: unparseable (bare) YouTube titles may yield no
    artist. `audio_ref` is an absolute file path (local) or watch URL
    (youtube); `raw_title` preserves the original filename/video title.
    """

    id: int
    source: str  # "local" | "youtube"
    source_id: str
    title: str
    artist: str | None
    duration_sec: float
    audio_ref: str
    raw_title: str
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> SongRow:
        """Build a typed `SongRow` from a `SELECT * FROM songs` row."""
        return cls(
            id=row["id"],
            source=row["source"],
            source_id=row["source_id"],
            title=row["title"],
            artist=row["artist"],
            duration_sec=row["duration_sec"],
            audio_ref=row["audio_ref"],
            raw_title=row["raw_title"],
            created_at=row["created_at"],
        )


@dataclass(frozen=True)
class ChallengeRow:
    """A row of `challenges`: one guild's daily challenge.

    `date` is the ISO calendar date in the configured timezone; `status` is
    "active" until the next day's post reveals it (`revealed_at` set).
    `skip_count` (migration 2) counts how many times this date's song has
    been admin-skipped; it seeds the deterministic re-pick (pinned #5).
    """

    id: int
    guild_id: str
    channel_id: str
    song_id: int
    date: str
    snippet_offset_sec: float
    status: ChallengeStatus
    created_at: str
    revealed_at: str | None
    skip_count: int = 0

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> ChallengeRow:
        """Build a typed `ChallengeRow` from a `SELECT * FROM challenges` row."""
        return cls(
            id=row["id"],
            guild_id=row["guild_id"],
            channel_id=row["channel_id"],
            song_id=row["song_id"],
            date=row["date"],
            snippet_offset_sec=row["snippet_offset_sec"],
            status=row["status"],
            created_at=row["created_at"],
            revealed_at=row["revealed_at"],
            skip_count=row["skip_count"],
        )


@dataclass(frozen=True)
class ChallengeUserRow:
    """A row of `challenge_users`: per-user state on one challenge."""

    challenge_id: int
    user_id: str
    snippet_level: int
    guesses_used: int
    solved: bool
    points_awarded: int
    solved_at: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> ChallengeUserRow:
        """Build a typed `ChallengeUserRow` from a `SELECT * FROM challenge_users` row."""
        return cls(
            challenge_id=row["challenge_id"],
            user_id=row["user_id"],
            snippet_level=row["snippet_level"],
            guesses_used=row["guesses_used"],
            solved=bool(row["solved"]),
            points_awarded=row["points_awarded"],
            solved_at=row["solved_at"],
        )


@dataclass(frozen=True)
class GuessRow:
    """A row of `guesses`: one logged guess (rejected submissions are not logged)."""

    id: int
    challenge_id: int
    user_id: str
    text: str
    matched_title: bool
    matched_artist: bool
    is_correct: bool
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> GuessRow:
        """Build a typed `GuessRow` from a `SELECT * FROM guesses` row."""
        return cls(
            id=row["id"],
            challenge_id=row["challenge_id"],
            user_id=row["user_id"],
            text=row["text"],
            matched_title=bool(row["matched_title"]),
            matched_artist=bool(row["matched_artist"]),
            is_correct=bool(row["is_correct"]),
            created_at=row["created_at"],
        )


@dataclass(frozen=True)
class UserStatsRow:
    """A row of `user_stats`: persistent per-guild user totals and streaks."""

    guild_id: str
    user_id: str
    total_points: int
    wins: int
    current_streak: int
    best_streak: int
    last_win_date: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> UserStatsRow:
        """Build a typed `UserStatsRow` from a `SELECT * FROM user_stats` row."""
        return cls(
            guild_id=row["guild_id"],
            user_id=row["user_id"],
            total_points=row["total_points"],
            wins=row["wins"],
            current_streak=row["current_streak"],
            best_streak=row["best_streak"],
            last_win_date=row["last_win_date"],
        )


class Database:
    """Thread-shareable SQLite wrapper with versioned migrations.

    The connection uses `check_same_thread=False` (serialized via an internal
    reentrant lock), WAL journal mode, `foreign_keys=ON`, and autocommit mode
    (`isolation_level=None`): statements outside an explicit `transaction()`
    commit immediately, while `transaction()` groups writes atomically.

    Use `Database.open(path)` at startup to connect AND migrate in one step.
    """

    def __init__(self, path: str | Path) -> None:
        """Open (creating if needed) the database at `path`.

        Missing parent directories are created. The sentinel path
        `":memory:"` opens an in-memory database. Does NOT run migrations —
        call `migrate()` (or use `Database.open`).
        """
        self._path = Path(path)
        if str(path) != ":memory:":
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._transaction_depth = 0
        self._rollback_only = False
        self._conn = sqlite3.connect(
            str(path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; explicit BEGIN via transaction()
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")

    @classmethod
    def open(cls, path: str | Path) -> Database:
        """Create a `Database` and apply all pending migrations (startup path)."""
        db = cls(path)
        db.migrate()
        return db

    @property
    def path(self) -> Path:
        """The database file path (`Path(":memory:")` for in-memory databases)."""
        return self._path

    @property
    def connection(self) -> sqlite3.Connection:
        """The underlying sqlite3 connection (prefer the typed helpers)."""
        return self._conn

    def migrate(self) -> list[int]:
        """Apply pending migrations in version order; idempotent.

        Returns the list of migration versions applied by this call (empty
        when the schema is already up to date). Each migration runs in its
        own transaction and is recorded in `schema_migrations`.
        """
        applied: list[int] = []
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            current = self.schema_version()
            for version, statements in MIGRATIONS:
                if version <= current:
                    continue
                with self.transaction():
                    for statement in statements:
                        self._conn.execute(statement)
                    self._conn.execute(
                        "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                        (version, datetime.now(UTC).isoformat()),
                    )
                applied.append(version)
        return applied

    def schema_version(self) -> int:
        """The highest applied migration version (0 when unmigrated)."""
        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table'"
                " AND name = 'schema_migrations'"
            ).fetchone()
            if exists is None:
                return 0
            row = self._conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
            return int(row[0]) if row is not None and row[0] is not None else 0

    def execute(self, sql: str, parameters: Sequence[Any] = ()) -> sqlite3.Cursor:
        """Execute a single statement; autocommits unless inside `transaction()`."""
        with self._lock:
            return self._conn.execute(sql, parameters)

    def executemany(
        self, sql: str, seq_of_parameters: Iterable[Sequence[Any]]
    ) -> sqlite3.Cursor:
        """Execute a statement against a batch of parameter sequences."""
        with self._lock:
            return self._conn.executemany(sql, seq_of_parameters)

    def query(self, sql: str, parameters: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """Run a query and return all rows."""
        with self._lock:
            return list(self._conn.execute(sql, parameters).fetchall())

    def query_one(self, sql: str, parameters: Sequence[Any] = ()) -> sqlite3.Row | None:
        """Run a query and return the first row, or None."""
        with self._lock:
            # row_factory=sqlite3.Row guarantees Row | None; typeshed says Any.
            return cast("sqlite3.Row | None", self._conn.execute(sql, parameters).fetchone())

    @contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator[None]:
        """Explicit atomic transaction (`BEGIN IMMEDIATE` by default).

        Nested calls join the outer transaction. Any exception marks the
        whole transaction rollback-only: the outermost exit rolls back even
        if an intermediate block caught the exception.
        """
        with self._lock:
            if self._transaction_depth == 0:
                self._conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            self._transaction_depth += 1
            try:
                yield
            except BaseException:
                self._rollback_only = True
                raise
            finally:
                self._transaction_depth -= 1
                if self._transaction_depth == 0:
                    if self._rollback_only:
                        self._conn.rollback()
                    else:
                        self._conn.commit()
                    self._rollback_only = False

    def close(self) -> None:
        """Close the connection. Subsequent use raises `sqlite3.ProgrammingError`."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
