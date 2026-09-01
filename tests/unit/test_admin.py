"""Unit tests for the admin slash command bodies (songbot/bot/admin.py).

Drives the REAL AdminCommands bodies with the unit-test fakes against a REAL
GameEngine on a tmp SQLite DB with the fake snippet service — no discord.py
gateway, no network. Covers the Manage-Guild gate that both discord.py and
the harness flag honor (VAL-ADMIN-009), admin-post idempotency (pinned #4,
VAL-ADMIN-001/002), skip refusal/replacement semantics (pinned #5,
VAL-ADMIN-003..006), the reload per-source summary ack (VAL-ADMIN-007), the
secrecy invariant (no ack ever names the song, pinned #9 — with ONE scoped
exception: the /songbot-fixsong ephemeral ack shows old -> new metadata to
the Manage-Server admin, since the command is unusable blind), and the
fixsong metadata-correction flow (songs row + durable song_overrides row).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import discord
import pytest
from discord import app_commands

from songbot.bot.admin import (
    AdminCommands,
    has_manage_guild,
    register_admin_commands,
)
from songbot.bot.embeds import PERMISSION_DENIED_MESSAGE
from songbot.bot.fixsong import FixSongModal
from songbot.catalog.refresh import RefreshResult, SourceRefresh
from songbot.db import Database
from songbot.engine import Challenge, GameEngine
from tests.unit.interaction_fakes import FakeInteraction, press
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


class _RecordingAnnouncer:
    """AnnouncementPoster test double: records posts, returns a stable message id."""

    MESSAGE_ID = "announcement-message-1"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.posts: list[tuple[str, str, str]] = []  # (channel_id, content, emoji)

    async def __call__(self, channel_id: str, content: str, emoji: str) -> str:
        self.posts.append((channel_id, content, emoji))
        if self.fail:
            raise RuntimeError("fake announcement send failure")
        return self.MESSAGE_ID


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
def announcer() -> _RecordingAnnouncer:
    return _RecordingAnnouncer()


@pytest.fixture
def commands(
    engine: GameEngine,
    tmp_path: Path,
    poster: _RecordingPoster,
    announcer: _RecordingAnnouncer,
) -> AdminCommands:
    # The fake interactions act on "guild-1"; configure its post channel so
    # the guild-scoped bodies find it (the live client seeds this from the
    # env bootstrap or /songbot-setup).
    engine.set_guild_channel("guild-1", "channel-1", set_by="test", now=DAY1)
    return AdminCommands(
        engine,
        _settings(tmp_path),
        clock=lambda: DAY1,
        post_sender=poster,
        announcement_poster=announcer,
    )


def _interaction(
    *, manage_guild: bool = True, guild_id: str | None = "guild-1"
) -> FakeInteraction:
    return FakeInteraction.for_user(
        ADMIN_ID, "admin", manage_guild=manage_guild, guild_id=guild_id
    )


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
            engine,
            _settings(tmp_path),
            clock=lambda: DAY1,
            post_sender=poster,
            announcement_poster=_RecordingAnnouncer(),
        )

        async def setup_via(interaction: FakeInteraction) -> Any:
            return await spied.setup_channel(interaction, "channel-1", "#channel-1")

        async def fixsong_via(interaction: FakeInteraction) -> Any:
            return await spied.fix_song(interaction)

        async def pingrole_via(interaction: FakeInteraction) -> Any:
            return await spied.setup_ping_role(
                interaction, role_id="role-1", role_mention="<@&role-1>", emoji="🎵"
            )

        for method in (
            spied.post_now,
            spied.skip_song,
            spied.reload_catalog,
            setup_via,
            fixsong_via,
            pingrole_via,
        ):
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
        for table in (
            "songs",
            "challenges",
            "challenge_users",
            "guesses",
            "user_stats",
            "guild_settings",
            "song_overrides",
            "ping_role_settings",
        ):
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
            engine,
            _settings(tmp_path),
            clock=lambda: DAY1,
            post_sender=poster,
            announcement_poster=_RecordingAnnouncer(),
        )
        challenge = engine.ensure_today_challenge("guild-1", "channel-1", DAY1)

        result = await commands.skip_song(_interaction())

        assert result.outcome == "skipped"
        assert challenge.id in fake_snippets.purged
        # The replacement challenge's snippet set was regenerated (5 levels)
        # plus the pre-cached full audio (issue #7).
        row = db.query_one("SELECT id FROM challenges WHERE date = '2026-08-13'")
        assert row is not None
        cache_dir = tmp_path / "snippets" / str(row["id"])
        assert sorted(p.name for p in cache_dir.glob("*.mp3")) == [
            "0.mp3",
            "1.mp3",
            "2.mp3",
            "3.mp3",
            "4.mp3",
            "full.mp3",
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
            engine,
            _settings(tmp_path),
            clock=lambda: DAY1,
            post_sender=poster,
            announcement_poster=_RecordingAnnouncer(),
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
            engine,
            _settings(tmp_path),
            clock=lambda: DAY1,
            post_sender=poster,
            announcement_poster=_RecordingAnnouncer(),
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
    def test_registers_six_guild_commands_with_manage_guild_default(
        self, commands: AdminCommands
    ) -> None:
        client = discord.Client(intents=discord.Intents.default())
        tree = app_commands.CommandTree(client)
        guild = discord.Object(id=1234567890)

        register_admin_commands(tree, commands, guild=guild)

        registered = tree.get_commands(guild=guild)
        names = {command.name for command in registered}
        assert names == {
            "songbot-setup",
            "songbot-post",
            "songbot-skip",
            "songbot-reload",
            "songbot-fixsong",
            "songbot-pingrole",
        }
        for command in registered:
            assert isinstance(command, app_commands.Command)
            assert command.default_permissions is not None
            assert command.default_permissions.manage_guild is True

    def test_setup_command_takes_a_text_channel_option(
        self, commands: AdminCommands
    ) -> None:
        client = discord.Client(intents=discord.Intents.default())
        tree = app_commands.CommandTree(client)
        register_admin_commands(tree, commands, guild=discord.Object(id=1234567890))

        setup = next(
            c
            for c in tree.get_commands(guild=discord.Object(id=1234567890))
            if c.name == "songbot-setup"
        )
        params = setup.parameters
        assert len(params) == 1
        assert params[0].name == "channel"
        assert params[0].required is True
        assert discord.ChannelType.text in params[0].channel_types

    def test_registration_is_per_guild(self, commands: AdminCommands) -> None:
        """Multi-guild: one registration call per joined guild, no global leak."""
        client = discord.Client(intents=discord.Intents.default())
        tree = app_commands.CommandTree(client)
        guild_a = discord.Object(id=111)
        guild_b = discord.Object(id=222)

        register_admin_commands(tree, commands, guild=guild_a)
        register_admin_commands(tree, commands, guild=guild_b)

        assert len(tree.get_commands(guild=guild_a)) == 6
        assert len(tree.get_commands(guild=guild_b)) == 6
        assert tree.get_commands() == []  # nothing global


class TestSetupChannel:
    """Multi-guild configuration: /songbot-setup writes guild_settings."""

    async def test_configures_channel_and_acks_ephemerally(
        self, commands: AdminCommands, db: Database, engine: GameEngine
    ) -> None:
        interaction = _interaction()

        result = await commands.setup_channel(interaction, "999888", "#daily-song")

        assert result.outcome == "configured"
        row = db.query_one("SELECT * FROM guild_settings WHERE guild_id = 'guild-1'")
        assert row is not None
        assert row["channel_id"] == "999888"
        assert row["set_by"] == str(ADMIN_ID)
        assert [p.kind for p in interaction.payloads] == ["ephemeral"]
        ack = interaction.payloads[0]
        assert ack.recipient == str(ADMIN_ID)
        assert ack.content is not None
        assert "#daily-song" in ack.content
        # The ack advertises the global schedule and the manual post path.
        assert "12:00" in ack.content
        assert "/songbot-post" in ack.content

    async def test_re_setup_updates_channel_keeps_created_at(
        self,
        commands: AdminCommands,
        db: Database,
        engine: GameEngine,
        tmp_path: Path,
        poster: _RecordingPoster,
    ) -> None:
        # The commands fixture seeded guild-1 -> channel-1 at DAY1.
        before = db.query_one(
            "SELECT created_at FROM guild_settings WHERE guild_id = 'guild-1'"
        )
        assert before is not None

        later = datetime(2026, 8, 14, 16, 0, 0, tzinfo=UTC)
        later_commands = AdminCommands(
            engine,
            _settings(tmp_path),
            clock=lambda: later,
            post_sender=poster,
            announcement_poster=_RecordingAnnouncer(),
        )
        result = await later_commands.setup_channel(_interaction(), "777", "#new-home")

        assert result.outcome == "configured"
        row = db.query_one(
            "SELECT channel_id, created_at, updated_at FROM guild_settings"
            " WHERE guild_id = 'guild-1'"
        )
        assert row is not None
        assert row["channel_id"] == "777"
        assert row["created_at"] == before["created_at"]
        assert row["updated_at"] != before["created_at"]

    async def test_denied_without_permission_zero_mutation(
        self, db: Database, tmp_path: Path, poster: _RecordingPoster
    ) -> None:
        engine, _ = _make_engine(tmp_path, db)
        commands = AdminCommands(
            engine,
            _settings(tmp_path),
            clock=lambda: DAY1,
            post_sender=poster,
            announcement_poster=_RecordingAnnouncer(),
        )
        interaction = _interaction(manage_guild=False)

        result = await commands.setup_channel(interaction, "999888", "#daily-song")

        assert result.outcome == "denied"
        assert _row_count(db, "guild_settings") == 0

    async def test_post_after_setup_uses_the_new_channel(
        self, commands: AdminCommands, db: Database, poster: _RecordingPoster
    ) -> None:
        _add_song(db)
        await commands.setup_channel(_interaction(), "999888", "#daily-song")

        result = await commands.post_now(_interaction())

        assert result.outcome == "posted"
        assert poster.posted[0].channel_id == "999888"
        row = db.query_one("SELECT channel_id FROM challenges")
        assert row is not None
        assert row["channel_id"] == "999888"


class TestSetupPingRole:
    """/songbot-pingrole: post the opt-in announcement, record the config."""

    async def test_posts_announcement_persists_config_and_acks(
        self,
        commands: AdminCommands,
        db: Database,
        announcer: _RecordingAnnouncer,
    ) -> None:
        interaction = _interaction()

        result = await commands.setup_ping_role(
            interaction, role_id="role-1", role_mention="<@&role-1>", emoji="🎵"
        )

        assert result.outcome == "ping_configured"
        # The announcement went to the guild's CONFIGURED channel, built by
        # the real content builder (emoji + role mention included).
        assert len(announcer.posts) == 1
        channel_id, content, emoji = announcer.posts[0]
        assert channel_id == "channel-1"
        assert emoji == "🎵"
        assert "🎵" in content
        assert "<@&role-1>" in content
        # The message/emoji/role triple is stored for the reaction listeners.
        row = db.query_one(
            "SELECT * FROM ping_role_settings WHERE guild_id = 'guild-1'"
        )
        assert row is not None
        assert row["channel_id"] == "channel-1"
        assert row["message_id"] == _RecordingAnnouncer.MESSAGE_ID
        assert row["role_id"] == "role-1"
        assert row["emoji"] == "🎵"
        assert row["set_by"] == str(ADMIN_ID)
        # Exactly one ephemeral ack to the invoking admin.
        assert [p.kind for p in interaction.payloads] == ["ephemeral"]
        ack = interaction.payloads[0]
        assert ack.recipient == str(ADMIN_ID)
        assert ack.content is not None
        assert "<@&role-1>" in ack.content

    async def test_re_run_supersedes_the_previous_announcement(
        self,
        commands: AdminCommands,
        db: Database,
        engine: GameEngine,
        tmp_path: Path,
        announcer: _RecordingAnnouncer,
    ) -> None:
        await commands.setup_ping_role(
            _interaction(), role_id="role-1", role_mention="<@&role-1>", emoji="🎵"
        )
        before = db.query_one(
            "SELECT created_at FROM ping_role_settings WHERE guild_id = 'guild-1'"
        )
        assert before is not None

        later = datetime(2026, 8, 14, 16, 0, 0, tzinfo=UTC)
        later_commands = AdminCommands(
            engine,
            _settings(tmp_path),
            clock=lambda: later,
            post_sender=_RecordingPoster(),
            announcement_poster=announcer,
        )
        result = await later_commands.setup_ping_role(
            _interaction(), role_id="role-2", role_mention="<@&role-2>", emoji="🔔"
        )

        assert result.outcome == "ping_configured"
        row = db.query_one(
            "SELECT * FROM ping_role_settings WHERE guild_id = 'guild-1'"
        )
        assert row is not None
        assert row["role_id"] == "role-2"
        assert row["emoji"] == "🔔"
        assert row["created_at"] == before["created_at"]
        assert row["updated_at"] != before["created_at"]

    async def test_not_configured_guild_posts_nothing(
        self, commands: AdminCommands, db: Database, announcer: _RecordingAnnouncer
    ) -> None:
        interaction = _interaction(guild_id="guild-never-set-up")

        result = await commands.setup_ping_role(
            interaction, role_id="role-1", role_mention="<@&role-1>", emoji="🎵"
        )

        assert result.outcome == "not_configured"
        assert announcer.posts == []
        assert [p.kind for p in interaction.payloads] == ["ephemeral"]
        content = interaction.payloads[0].content or ""
        assert "/songbot-setup" in content
        assert _row_count(db, "ping_role_settings") == 0

    async def test_announcement_failure_persists_nothing(
        self,
        db: Database,
        engine: GameEngine,
        tmp_path: Path,
        poster: _RecordingPoster,
    ) -> None:
        # The bare engine fixture is unconfigured; seed guild-1 like the
        # commands fixture does so the body gets past the not-configured check.
        engine.set_guild_channel("guild-1", "channel-1", set_by="test", now=DAY1)
        failing_commands = AdminCommands(
            engine,
            _settings(tmp_path),
            clock=lambda: DAY1,
            post_sender=poster,
            announcement_poster=_RecordingAnnouncer(fail=True),
        )
        interaction = _interaction()

        result = await failing_commands.setup_ping_role(
            interaction, role_id="role-1", role_mention="<@&role-1>", emoji="🎵"
        )

        assert result.outcome == "error"
        assert result.error is not None
        assert _row_count(db, "ping_role_settings") == 0
        assert [p.kind for p in interaction.payloads] == ["ephemeral"]
        content = interaction.payloads[0].content or ""
        assert "announcement" in content.lower()


class TestNotConfigured:
    """A guild that never ran /songbot-setup gets the not-configured ack."""

    async def test_post_now_not_configured_zero_mutation(
        self, commands: AdminCommands, db: Database, poster: _RecordingPoster
    ) -> None:
        _add_song(db)
        interaction = _interaction(guild_id="guild-never-set-up")

        result = await commands.post_now(interaction)

        assert result.outcome == "not_configured"
        assert poster.posted == []
        assert [p.kind for p in interaction.payloads] == ["ephemeral"]
        content = interaction.payloads[0].content or ""
        assert "/songbot-setup" in content
        assert _row_count(db, "challenges") == 0

    async def test_skip_in_unconfigured_guild_is_plain_no_challenge_refusal(
        self, commands: AdminCommands, db: Database
    ) -> None:
        interaction = _interaction(guild_id="guild-never-set-up")

        result = await commands.skip_song(interaction)

        assert result.outcome == "refused"
        assert result.reason == "no_challenge"


DAY2 = datetime(2026, 8, 14, 16, 0, 0, tzinfo=UTC)  # 2026-08-14 13:00 ADT


async def _open_edit_modal(interaction: FakeInteraction) -> FixSongModal:
    """Press the show-first ephemeral's Edit button; return the opened modal."""
    assert interaction.payloads[-1].view is not None
    await press(interaction.payloads[-1].view, "songbot:fixsong_edit", interaction)
    payload = interaction.payloads[-1]
    assert payload.kind == "modal"
    modal = payload.modal
    assert isinstance(modal, FixSongModal)
    return modal


async def _submit_edit(
    modal: FixSongModal,
    interaction: FakeInteraction,
    *,
    title: str | None = None,
    artist: str | None = None,
) -> None:
    """Submit the REAL modal, mirroring Discord's payload: an omitted field
    arrives with its pre-filled default, exactly as an untouched input does.
    """
    modal.title_input._value = (
        title if title is not None else (modal.title_input.default or "")
    )
    modal.artist_input._value = (
        artist if artist is not None else (modal.artist_input.default or "")
    )
    await modal.on_submit(interaction)


class TestFixSong:
    """/songbot-fixsong: show the current song metadata, then edit it.

    The command body is the zero-mutation show-first step; the correction
    lands on the modal submit. The show payload and the submit ack name the
    song — the deliberate, scoped exception to the pinned-#9 secrecy rule:
    ephemeral, admin-gated, and the edit form is unusable blind.
    """

    async def test_shows_current_metadata_with_edit_button_zero_mutation(
        self, commands: AdminCommands, db: Database, engine: GameEngine
    ) -> None:
        _add_song(db)  # song-1: Neon Skyline / Midnight Circuit
        challenge = engine.ensure_today_challenge("guild-1", "channel-1", DAY1)

        interaction = _interaction()
        result = await commands.fix_song(interaction)

        assert result.outcome == "shown"
        assert result.reason is None
        target = result.target
        assert target is not None
        assert target.song_id == challenge.song.id
        assert target.challenge_id == challenge.id
        assert target.challenge_date == "2026-08-13"
        assert (target.title, target.artist) == (TITLE, ARTIST)
        # Exactly one ephemeral payload showing the current parameters...
        assert [p.kind for p in interaction.payloads] == ["ephemeral"]
        shown = interaction.payloads[0]
        assert shown.recipient == str(ADMIN_ID)
        content = shown.content or ""
        for text in (TITLE, ARTIST, "song-1", "2026-08-13"):
            assert text in content
        # ...carrying the view whose button opens the edit modal.
        assert shown.view is not None
        # Zero mutation: the correction only lands on the modal submit.
        row = db.query_one("SELECT title, artist FROM songs")
        assert row is not None
        assert (row["title"], row["artist"]) == (TITLE, ARTIST)
        assert _row_count(db, "song_overrides") == 0

    async def test_edit_button_opens_modal_prefilled_with_current_values(
        self, commands: AdminCommands, db: Database, engine: GameEngine
    ) -> None:
        _add_song(db)
        engine.ensure_today_challenge("guild-1", "channel-1", DAY1)
        interaction = _interaction()
        await commands.fix_song(interaction)

        modal = await _open_edit_modal(interaction)

        assert modal.title_input.default == TITLE
        assert modal.title_input.required is True
        assert modal.artist_input.default == ARTIST
        assert modal.artist_input.required is False
        # Opening the form mutates nothing.
        assert _row_count(db, "song_overrides") == 0

    async def test_submit_applies_fix_and_records_override(
        self, commands: AdminCommands, db: Database, engine: GameEngine
    ) -> None:
        _add_song(db)
        challenge = engine.ensure_today_challenge("guild-1", "channel-1", DAY1)
        interaction = _interaction()
        await commands.fix_song(interaction)
        modal = await _open_edit_modal(interaction)

        await _submit_edit(modal, interaction, title="Fixed Title", artist="Fixed Artist")

        fix = modal.result
        assert fix is not None
        assert modal.refused_reason is None
        assert fix.song_id == challenge.song.id
        assert fix.challenge_id == challenge.id
        assert fix.challenge_date == "2026-08-13"
        assert (fix.old_title, fix.old_artist) == (TITLE, ARTIST)
        assert (fix.new_title, fix.new_artist) == ("Fixed Title", "Fixed Artist")
        # The songs row carries the correction...
        row = db.query_one("SELECT title, artist FROM songs WHERE id = ?", (fix.song_id,))
        assert row is not None
        assert (row["title"], row["artist"]) == ("Fixed Title", "Fixed Artist")
        # ...and the durable override row survives catalog reloads.
        override = db.query_one(
            "SELECT title, artist, set_by FROM song_overrides"
            " WHERE source = 'local' AND source_id = 'song-1'"
        )
        assert override is not None
        assert (override["title"], override["artist"]) == ("Fixed Title", "Fixed Artist")
        assert override["set_by"] == str(ADMIN_ID)
        # Show ephemeral, modal, then the ephemeral old -> new ack.
        assert [p.kind for p in interaction.payloads] == ["ephemeral", "modal", "ephemeral"]
        ack = interaction.payloads[-1]
        assert ack.recipient == str(ADMIN_ID)
        content = ack.content or ""
        for text in (TITLE, ARTIST, "Fixed Title", "Fixed Artist"):
            assert text in content

    async def test_submit_with_untouched_fields_keeps_current_values(
        self, commands: AdminCommands, db: Database, engine: GameEngine
    ) -> None:
        _add_song(db)
        engine.ensure_today_challenge("guild-1", "channel-1", DAY1)
        interaction = _interaction()
        await commands.fix_song(interaction)
        modal = await _open_edit_modal(interaction)

        await _submit_edit(modal, interaction)  # both fields left pre-filled

        fix = modal.result
        assert fix is not None
        assert (fix.new_title, fix.new_artist) == (TITLE, ARTIST)
        row = db.query_one("SELECT title, artist FROM songs")
        assert row is not None
        assert (row["title"], row["artist"]) == (TITLE, ARTIST)

    async def test_blank_artist_clears_the_artist(
        self, commands: AdminCommands, db: Database, engine: GameEngine
    ) -> None:
        _add_song(db)
        engine.ensure_today_challenge("guild-1", "channel-1", DAY1)
        interaction = _interaction()
        await commands.fix_song(interaction)
        modal = await _open_edit_modal(interaction)

        await _submit_edit(modal, interaction, title="Fixed Title", artist="   ")

        fix = modal.result
        assert fix is not None
        assert fix.new_artist is None
        row = db.query_one("SELECT artist FROM songs")
        assert row is not None
        assert row["artist"] is None

    async def test_date_targets_an_earlier_challenge_and_default_is_latest(
        self, commands: AdminCommands, db: Database, engine: GameEngine
    ) -> None:
        _add_two_songs(db)
        day1 = engine.ensure_today_challenge("guild-1", "channel-1", DAY1)
        day2 = engine.ensure_today_challenge("guild-1", "channel-1", DAY2)
        assert day2.song.id != day1.song.id  # no repeats until exhausted

        dated_interaction = _interaction()
        dated = await commands.fix_song(dated_interaction, date="2026-08-13")

        assert dated.outcome == "shown"
        assert dated.target is not None
        assert dated.target.challenge_id == day1.id
        # The modal submit fixes exactly the challenge that was displayed.
        modal = await _open_edit_modal(dated_interaction)
        await _submit_edit(modal, dated_interaction, title="Day One Fix")
        assert modal.result is not None
        assert modal.result.challenge_id == day1.id
        row = db.query_one("SELECT title FROM songs WHERE id = ?", (day1.song.id,))
        assert row is not None
        assert row["title"] == "Day One Fix"
        untouched = db.query_one("SELECT title FROM songs WHERE id = ?", (day2.song.id,))
        assert untouched is not None
        assert untouched["title"] != "Day One Fix"

        latest = await commands.fix_song(_interaction())

        assert latest.outcome == "shown"
        assert latest.target is not None
        assert latest.target.challenge_id == day2.id

    async def test_refused_on_invalid_date_zero_mutation(
        self, commands: AdminCommands, db: Database, engine: GameEngine
    ) -> None:
        _add_song(db)
        engine.ensure_today_challenge("guild-1", "channel-1", DAY1)

        interaction = _interaction()
        result = await commands.fix_song(interaction, date="13-08-2026")

        assert result.outcome == "refused"
        assert result.reason == "invalid_date"
        assert [p.kind for p in interaction.payloads] == ["ephemeral"]
        assert interaction.payloads[0].view is None
        row = db.query_one("SELECT title, artist FROM songs")
        assert row is not None
        assert (row["title"], row["artist"]) == (TITLE, ARTIST)
        assert _row_count(db, "song_overrides") == 0

    async def test_refused_without_a_challenge_zero_mutation(
        self, commands: AdminCommands, db: Database
    ) -> None:
        _add_song(db)

        interaction = _interaction()
        result = await commands.fix_song(interaction)

        assert result.outcome == "refused"
        assert result.reason == "no_challenge"
        assert [p.kind for p in interaction.payloads] == ["ephemeral"]
        assert interaction.payloads[0].view is None
        row = db.query_one("SELECT title, artist FROM songs")
        assert row is not None
        assert (row["title"], row["artist"]) == (TITLE, ARTIST)
        assert _row_count(db, "song_overrides") == 0

    async def test_blank_title_refused_at_submit_zero_mutation(
        self, commands: AdminCommands, db: Database, engine: GameEngine
    ) -> None:
        _add_song(db)
        engine.ensure_today_challenge("guild-1", "channel-1", DAY1)
        interaction = _interaction()
        await commands.fix_song(interaction)
        modal = await _open_edit_modal(interaction)

        await _submit_edit(modal, interaction, title="   ")

        assert modal.result is None
        assert modal.refused_reason == "blank_title"
        # Show ephemeral, modal, then the ephemeral refusal.
        assert [p.kind for p in interaction.payloads] == ["ephemeral", "modal", "ephemeral"]
        row = db.query_one("SELECT title FROM songs")
        assert row is not None
        assert row["title"] == TITLE
        assert _row_count(db, "song_overrides") == 0

    async def test_non_admin_cannot_open_or_submit_the_modal(
        self, commands: AdminCommands, db: Database, engine: GameEngine
    ) -> None:
        """The view/modal re-check Manage-Guild (it could be revoked mid-flow)."""
        _add_song(db)
        engine.ensure_today_challenge("guild-1", "channel-1", DAY1)
        admin = _interaction()
        await commands.fix_song(admin)
        view = admin.payloads[0].view
        assert view is not None
        non_admin = _interaction(manage_guild=False)

        await press(view, "songbot:fixsong_edit", non_admin)

        assert [p.kind for p in non_admin.payloads] == ["ephemeral"]
        assert non_admin.payloads[0].content == PERMISSION_DENIED_MESSAGE

        modal = await _open_edit_modal(admin)
        await _submit_edit(modal, non_admin, title="Fixed Title")

        assert modal.result is None
        assert non_admin.payloads[-1].content == PERMISSION_DENIED_MESSAGE
        row = db.query_one("SELECT title FROM songs")
        assert row is not None
        assert row["title"] == TITLE
        assert _row_count(db, "song_overrides") == 0

    async def test_fix_applies_to_new_guesses_but_is_not_retroactive(
        self, commands: AdminCommands, db: Database, engine: GameEngine
    ) -> None:
        _add_song(db)
        challenge = engine.ensure_today_challenge("guild-1", "channel-1", DAY1)
        # Before the fix, the future-correct text is a plain wrong guess.
        before = engine.submit_guess(challenge.id, "uma", "Fixed Title Fixed Artist", DAY1)
        assert before.outcome == "wrong"

        interaction = _interaction()
        await commands.fix_song(interaction)
        modal = await _open_edit_modal(interaction)
        await _submit_edit(modal, interaction, title="Fixed Title", artist="Fixed Artist")
        assert modal.result is not None

        # New guesses match the corrected metadata immediately (both -> bonus).
        after = engine.submit_guess(challenge.id, "uma", "Fixed Title Fixed Artist", DAY1)
        assert after.outcome == "correct"
        assert after.is_both is True
        assert after.points_awarded == 150  # 100 x 1.5, round-half-up

        # The pre-fix guess row keeps its original result (no re-scoring).
        rows = db.query(
            "SELECT is_correct FROM guesses WHERE challenge_id = ? ORDER BY id",
            (challenge.id,),
        )
        assert [bool(row["is_correct"]) for row in rows] == [False, True]
