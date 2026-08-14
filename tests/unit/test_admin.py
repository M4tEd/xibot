"""Unit tests for the admin slash command bodies (songbot/bot/admin.py).

Drives the REAL AdminCommands bodies with the unit-test fakes against a REAL
GameEngine on a tmp SQLite DB with the fake snippet service — no discord.py
gateway, no network. Covers the Manage-Guild gate that both discord.py and
the harness flag honor (VAL-ADMIN-009), admin-post idempotency (pinned #4,
VAL-ADMIN-001/002), skip refusal/replacement semantics (pinned #5,
VAL-ADMIN-003..006), the reload per-source summary ack (VAL-ADMIN-007), and
the secrecy invariant (no ack ever names the song, pinned #9).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import discord
import pytest
from discord import app_commands

from songbot.bot.admin import (
    AdminCommands,
    has_manage_guild,
    register_admin_commands,
)
from songbot.bot.embeds import PERMISSION_DENIED_MESSAGE
from songbot.catalog.refresh import RefreshResult, SourceRefresh
from songbot.db import Database
from songbot.engine import Challenge, GameEngine
from tests.unit.interaction_fakes import FakeInteraction
from tests.unit.test_engine_daily import _make_engine, _settings
from tests.unit.test_engine_gameplay import _add_song

DAY1 = datetime(2026, 8, 13, 16, 0, 0, tzinfo=UTC)  # 2026-08-13 13:00 ADT
ADMIN_ID = 9001

TITLE = "Neon Skyline"
ARTIST = "Midnight Circuit"
TITLE2 = "Digital Horizon"
ARTIST2 = "Quantum Drift"


class _RecordingPoster:
    """DailyPostSender test double: records the challenges it was asked to post."""

    def __init__(self) -> None:
        self.posted: list[Challenge] = []

    async def __call__(self, challenge: Challenge) -> None:
        self.posted.append(challenge)


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    database = Database.open(tmp_path / "songbot.db")
    yield database
    database.close()


@pytest.fixture
def engine(db: Database, tmp_path: Path) -> GameEngine:
    made, _ = _make_engine(tmp_path, db)
    return made


@pytest.fixture
def poster() -> _RecordingPoster:
    return _RecordingPoster()


@pytest.fixture
def commands(
    engine: GameEngine, tmp_path: Path, poster: _RecordingPoster
) -> AdminCommands:
    return AdminCommands(
        engine,
        _settings(tmp_path),
        clock=lambda: DAY1,
        post_sender=poster,
    )


def _interaction(*, manage_guild: bool = True) -> FakeInteraction:
    return FakeInteraction.for_user(ADMIN_ID, "admin", manage_guild=manage_guild)


def _row_count(db: Database, table: str) -> int:
    row = db.query_one(f"SELECT COUNT(*) AS c FROM {table}")
    assert row is not None
    return int(row["c"])


def _add_two_songs(db: Database) -> None:
    _add_song(db)  # song-1: Neon Skyline / Midnight Circuit
    _add_song(db, "song-2", title=TITLE2, artist=ARTIST2, raw_title=f"{ARTIST2} - {TITLE2}")


class TestPermissionGate:
    """VAL-ADMIN-009: one check, honored by discord.py and the harness alike."""

    def test_has_manage_guild_reads_member_permissions(self) -> None:
        assert has_manage_guild(_interaction(manage_guild=True)) is True
        assert has_manage_guild(_interaction(manage_guild=False)) is False

    def test_has_manage_guild_denies_users_without_permissions_attr(self) -> None:
        interaction = _interaction(manage_guild=True)

        class BareUser:
            id = 1
            name = "bare"

        interaction.user = BareUser()  # type: ignore[assignment]
        assert has_manage_guild(interaction) is False

    async def test_every_command_denied_without_permission_zero_mutation(
        self, db: Database, tmp_path: Path, poster: _RecordingPoster
    ) -> None:
        refresh_calls = 0

        def _spy_refresher() -> RefreshResult:
            nonlocal refresh_calls
            refresh_calls += 1
            return RefreshResult(sources=())

        engine, _ = _make_engine(tmp_path, db, catalog_refresher=_spy_refresher)
        spied = AdminCommands(
            engine, _settings(tmp_path), clock=lambda: DAY1, post_sender=poster
        )

        for method in (spied.post_now, spied.skip_song, spied.reload_catalog):
            interaction = _interaction(manage_guild=False)
            result = await method(interaction)

            assert result.outcome == "denied"
            assert len(interaction.payloads) == 1
            denial = interaction.payloads[0]
            assert denial.kind == "ephemeral"
            assert denial.recipient == str(ADMIN_ID)
            assert denial.content == PERMISSION_DENIED_MESSAGE
            assert denial.attachment is None

        # Zero state change and no engine/catalog work happened at all.
        assert poster.posted == []
        assert refresh_calls == 0
        for table in ("songs", "challenges", "challenge_users", "guesses", "user_stats"):
            assert _row_count(db, table) == 0


class TestPostNow:
    async def test_posts_challenge_and_acks_ephemerally(
        self, commands: AdminCommands, db: Database, poster: _RecordingPoster
    ) -> None:
        _add_song(db)
        interaction = _interaction()

        result = await commands.post_now(interaction)

        assert result.outcome == "posted"
        # The daily post went through the injected sender exactly once.
        assert len(poster.posted) == 1
        posted = poster.posted[0]
        assert posted.date == "2026-08-13"
        assert posted.status == "active"
        assert posted.created is True
        # Exactly one ephemeral ack to the invoking admin.
        assert len(interaction.payloads) == 1
        ack = interaction.payloads[0]
        assert ack.kind == "ephemeral"
        assert ack.recipient == str(ADMIN_ID)
        assert ack.content is not None
        # The challenge row exists and is active.
        row = db.query_one("SELECT status, date FROM challenges")
        assert row is not None
        assert (row["status"], row["date"]) == ("active", "2026-08-13")

    async def test_repeat_is_already_posted_without_reposting(
        self, commands: AdminCommands, db: Database, poster: _RecordingPoster
    ) -> None:
        _add_song(db)
        await commands.post_now(_interaction())

        interaction = _interaction()
        result = await commands.post_now(interaction)

        assert result.outcome == "already_posted"
        assert len(poster.posted) == 1, "the daily post must not be sent twice"
        assert len(interaction.payloads) == 1
        ack = interaction.payloads[0]
        assert ack.kind == "ephemeral"
        assert ack.recipient == str(ADMIN_ID)
        assert ack.content is not None
        assert "already" in ack.content.lower()
        assert _row_count(db, "challenges") == 1

    async def test_empty_catalog_acks_without_a_challenge(
        self, commands: AdminCommands, db: Database, poster: _RecordingPoster
    ) -> None:
        # No songs and both providers disabled in _settings -> catalog_empty.
        interaction = _interaction()
        result = await commands.post_now(interaction)

        assert result.outcome == "catalog_empty"
        assert poster.posted == []
        assert len(interaction.payloads) == 1
        ack = interaction.payloads[0]
        assert ack.kind == "ephemeral"
        assert ack.content is not None
        assert "catalog" in ack.content.lower()
        assert _row_count(db, "challenges") == 0

    async def test_ack_never_names_the_song(
        self, commands: AdminCommands, db: Database, poster: _RecordingPoster
    ) -> None:
        _add_song(db)
        for interaction in (_interaction(), _interaction()):
            await commands.post_now(interaction)
            for payload in interaction.payloads:
                assert TITLE.lower() not in (payload.content or "").lower()
                assert ARTIST.lower() not in (payload.content or "").lower()


class TestSkipSong:
    async def test_replaces_song_offset_and_resets_user_state(
        self, commands: AdminCommands, db: Database, engine: GameEngine
    ) -> None:
        _add_two_songs(db)
        challenge = engine.ensure_today_challenge("guild-1", "channel-1", DAY1)
        # todd builds per-user state: two hear-more presses + one wrong guess.
        engine.unlock_snippet(challenge.id, "todd")
        engine.unlock_snippet(challenge.id, "todd")
        engine.submit_guess(challenge.id, "todd", "zxqv unrelated noise", DAY1)

        interaction = _interaction()
        result = await commands.skip_song(interaction)

        assert result.outcome == "skipped"
        assert result.reason is None
        row = db.query_one("SELECT * FROM challenges WHERE date = '2026-08-13'")
        assert row is not None
        assert row["song_id"] != challenge.song.id, "skip must pick a different song"
        assert row["snippet_offset_sec"] != challenge.snippet_offset_sec
        assert row["status"] == "active"
        assert row["skip_count"] == 1, "the skip seed survives the row recreate"
        # Per-user challenge state and the guess log cascade-deleted.
        assert _row_count(db, "challenge_users") == 0
        assert _row_count(db, "guesses") == 0
        # Exactly one ephemeral ack; skip emits no channel/announcement payload.
        assert [p.kind for p in interaction.payloads] == ["ephemeral"]
        ack = interaction.payloads[0]
        assert ack.recipient == str(ADMIN_ID)
        assert ack.content is not None
        # Secrecy: the ack names neither the old nor the new song.
        for secret in (TITLE, ARTIST, TITLE2, ARTIST2):
            assert secret.lower() not in ack.content.lower()

    async def test_skip_purges_and_regenerates_the_snippet_cache(
        self, db: Database, tmp_path: Path, poster: _RecordingPoster
    ) -> None:
        _add_two_songs(db)
        engine, fake_snippets = _make_engine(tmp_path, db)
        commands = AdminCommands(
            engine, _settings(tmp_path), clock=lambda: DAY1, post_sender=poster
        )
        challenge = engine.ensure_today_challenge("guild-1", "channel-1", DAY1)

        result = await commands.skip_song(_interaction())

        assert result.outcome == "skipped"
        assert challenge.id in fake_snippets.purged
        # The replacement challenge's snippet set was regenerated (5 levels).
        row = db.query_one("SELECT id FROM challenges WHERE date = '2026-08-13'")
        assert row is not None
        cache_dir = tmp_path / "snippets" / str(row["id"])
        assert sorted(p.name for p in cache_dir.glob("*.mp3")) == [
            "0.mp3",
            "1.mp3",
            "2.mp3",
            "3.mp3",
            "4.mp3",
        ]

    async def test_refused_when_no_challenge_today(
        self, commands: AdminCommands, db: Database
    ) -> None:
        interaction = _interaction()
        result = await commands.skip_song(interaction)

        assert result.outcome == "refused"
        assert result.reason == "no_challenge"
        assert [p.kind for p in interaction.payloads] == ["ephemeral"]
        assert _row_count(db, "challenges") == 0

    async def test_refused_when_challenge_revealed_zero_mutation(
        self, commands: AdminCommands, db: Database, engine: GameEngine
    ) -> None:
        _add_two_songs(db)
        challenge = engine.ensure_today_challenge("guild-1", "channel-1", DAY1)
        db.execute(
            "UPDATE challenges SET status = 'revealed', revealed_at = ? WHERE id = ?",
            ("2026-08-13T16:30:00+00:00", challenge.id),
        )

        interaction = _interaction()
        result = await commands.skip_song(interaction)

        assert result.outcome == "refused"
        assert result.reason == "revealed"
        assert [p.kind for p in interaction.payloads] == ["ephemeral"]
        content = interaction.payloads[0].content or ""
        assert "revealed" in content.lower() or "no longer active" in content.lower()
        row = db.query_one("SELECT * FROM challenges WHERE id = ?", (challenge.id,))
        assert row is not None
        assert row["status"] == "revealed"
        assert row["song_id"] == challenge.song.id
        assert row["skip_count"] == 0

    async def test_refused_after_a_solve_preserves_solver_state(
        self, commands: AdminCommands, db: Database, engine: GameEngine
    ) -> None:
        _add_two_songs(db)
        challenge = engine.ensure_today_challenge("guild-1", "channel-1", DAY1)
        solved = engine.submit_guess(challenge.id, "uma", TITLE, DAY1)
        assert solved.outcome == "correct"

        interaction = _interaction()
        result = await commands.skip_song(interaction)

        assert result.outcome == "refused"
        assert result.reason == "solved"
        assert [p.kind for p in interaction.payloads] == ["ephemeral"]
        content = interaction.payloads[0].content or ""
        assert "solv" in content.lower()
        # Zero mutation: song/offset unchanged, solver rows fully intact.
        row = db.query_one("SELECT * FROM challenges WHERE id = ?", (challenge.id,))
        assert row is not None
        assert row["song_id"] == challenge.song.id
        assert row["snippet_offset_sec"] == challenge.snippet_offset_sec
        assert row["status"] == "active"
        user_row = db.query_one(
            "SELECT solved, points_awarded FROM challenge_users"
            " WHERE challenge_id = ? AND user_id = 'uma'",
            (challenge.id,),
        )
        assert user_row is not None
        assert (user_row["solved"], user_row["points_awarded"]) == (1, 100)
        stats = db.query_one(
            "SELECT total_points, wins, current_streak FROM user_stats"
            " WHERE user_id = 'uma'"
        )
        assert stats is not None
        assert (stats["total_points"], stats["wins"], stats["current_streak"]) == (100, 1, 1)


class TestReloadCatalog:
    async def test_ack_reports_per_source_summary(
        self, db: Database, tmp_path: Path, poster: _RecordingPoster
    ) -> None:
        refresh = RefreshResult(
            sources=(
                SourceRefresh(source="local", added=2, updated=5, removed=1, retained=1),
                SourceRefresh(source="youtube", added=200),
            )
        )
        engine, _ = _make_engine(tmp_path, db, catalog_refresher=lambda: refresh)
        commands = AdminCommands(
            engine, _settings(tmp_path), clock=lambda: DAY1, post_sender=poster
        )

        interaction = _interaction()
        result = await commands.reload_catalog(interaction)

        assert result.outcome == "reloaded"
        assert result.refresh == refresh
        assert [p.kind for p in interaction.payloads] == ["ephemeral"]
        content = interaction.payloads[0].content or ""
        assert "local" in content
        assert "2 added" in content
        assert "5 updated" in content
        assert "1 removed" in content
        assert "1 retained" in content
        assert "youtube" in content
        assert "200 added" in content

    async def test_ack_reports_source_errors(
        self, db: Database, tmp_path: Path, poster: _RecordingPoster
    ) -> None:
        refresh = RefreshResult(
            sources=(
                SourceRefresh(source="local", added=8),
                SourceRefresh(
                    source="youtube", error="YouTubeCatalogError: fetch failed for playlist"
                ),
            )
        )
        engine, _ = _make_engine(tmp_path, db, catalog_refresher=lambda: refresh)
        commands = AdminCommands(
            engine, _settings(tmp_path), clock=lambda: DAY1, post_sender=poster
        )

        result = await commands.reload_catalog(_interaction())

        assert result.outcome == "reloaded"
        content = result.refresh.sources[1].error or ""
        assert "YouTubeCatalogError" in content

    async def test_ack_when_no_sources_configured(
        self, commands: AdminCommands
    ) -> None:
        interaction = _interaction()
        result = await commands.reload_catalog(interaction)

        assert result.outcome == "reloaded"
        assert result.refresh is not None
        assert result.refresh.sources == ()
        content = interaction.payloads[0].content or ""
        assert "no catalog sources" in content.lower()


class TestRegistration:
    def test_registers_three_guild_commands_with_manage_guild_default(
        self, commands: AdminCommands
    ) -> None:
        client = discord.Client(intents=discord.Intents.default())
        tree = app_commands.CommandTree(client)
        guild = discord.Object(id=1234567890)

        register_admin_commands(tree, commands, guild=guild)

        registered = tree.get_commands(guild=guild)
        names = {command.name for command in registered}
        assert names == {"songbot-post", "songbot-skip", "songbot-reload"}
        for command in registered:
            assert isinstance(command, app_commands.Command)
            assert command.default_permissions is not None
            assert command.default_permissions.manage_guild is True
