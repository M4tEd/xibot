"""Unit tests for per-guild custom catalog playlists (/songbot-playlist).

Covers the engine's playlist configuration (set/clear, configured-guild
guard), scoped song selection (a guild with a custom playlist draws ONLY from
its own ``songs`` scope; every other guild draws from the shared global
pool), the guild-aware catalog refresh (scoped upsert/removal, provider
selection from the stored URL), the admin command bodies, and the harness
scenario drivers.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from songbot.bot.admin import AdminCommands
from songbot.bot.embeds import (
    ADMIN_NOT_CONFIGURED_MESSAGE,
    ADMIN_PLAYLIST_BLANK_MESSAGE,
    PERMISSION_DENIED_MESSAGE,
)
from songbot.catalog.refresh import RefreshResult, SourceRefresh, refresh_catalog
from songbot.db import Database
from songbot.engine import CatalogEmptyError, EngineError, GameEngine
from songbot.harness.cli import (
    HarnessContext,
    scenario_admin_playlist,
    scenario_admin_playlist_clear,
)
from tests.unit.interaction_fakes import FakeInteraction
from tests.unit.test_catalog_refresh import StubProvider, _song, _youtube_song
from tests.unit.test_catalog_refresh import _settings as _catalog_settings
from tests.unit.test_engine_daily import _make_engine, _settings
from tests.unit.test_harness import ADMIN, NON_ADMIN, json_roundtrip, kinds

DAY1 = datetime(2026, 8, 13, 16, 0, 0, tzinfo=UTC)  # 2026-08-13 13:00 ADT
NOW_ISO = "2026-01-01T00:00:00+00:00"
PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLcustom"
PLAYLIST_URL_2 = "https://www.youtube.com/playlist?list=PLother"
ADMIN_ID = 9001


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    database = Database.open(tmp_path / "songbot.db")
    yield database
    database.close()


@pytest.fixture
def engine(db: Database, tmp_path: Path) -> GameEngine:
    made, _ = _make_engine(tmp_path, db)
    return made


def _add_scoped_song(
    db: Database,
    scope: str,
    source_id: str,
    *,
    source: str = "local",
    duration_sec: float = 30.0,
) -> int:
    """Insert a song row into a specific catalog scope ("" = global pool)."""
    cursor = db.execute(
        "INSERT INTO songs (guild_id, source, source_id, title, artist, duration_sec,"
        " audio_ref, raw_title, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            scope,
            source,
            source_id,
            f"Title {source_id}",
            f"Artist {source_id}",
            duration_sec,
            f"/music/{source_id}.mp3",
            f"raw {source_id}",
            NOW_ISO,
        ),
    )
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def _configure_guild(
    db: Database, guild_id: str, playlist_url: str | None = None
) -> None:
    db.execute(
        "INSERT INTO guild_settings"
        " (guild_id, channel_id, set_by, created_at, updated_at, playlist_url)"
        " VALUES (?, 'c1', 'test', ?, ?, ?)",
        (guild_id, NOW_ISO, NOW_ISO, playlist_url),
    )


def _scoped_source_ids(db: Database, scope: str) -> set[str]:
    return {
        str(row["source_id"])
        for row in db.query("SELECT source_id FROM songs WHERE guild_id = ?", (scope,))
    }


class _RecordingPoster:
    """DailyPostSender test double: records the challenges it was asked to post."""

    def __init__(self) -> None:
        self.posted: list[Any] = []

    async def __call__(self, challenge: Any) -> None:
        self.posted.append(challenge)


def _interaction(
    *, manage_guild: bool = True, guild_id: str | None = "guild-1"
) -> FakeInteraction:
    return FakeInteraction.for_user(
        ADMIN_ID, "admin", manage_guild=manage_guild, guild_id=guild_id
    )


def _commands(engine: GameEngine, tmp_path: Path) -> AdminCommands:
    engine.set_guild_channel("guild-1", "channel-1", set_by="test", now=DAY1)
    return AdminCommands(
        engine,
        _settings(tmp_path),
        clock=lambda: DAY1,
        post_sender=_RecordingPoster(),
    )


class TestGuildPlaylistConfig:
    """Engine-level playlist_url storage on guild_settings."""

    def test_set_and_clear_round_trip(self, engine: GameEngine, db: Database) -> None:
        engine.set_guild_channel("g1", "c1", set_by="test", now=DAY1)

        row = engine.set_guild_playlist("g1", PLAYLIST_URL, set_by="admin", now=DAY1)
        assert row.playlist_url == PLAYLIST_URL
        assert row.set_by == "admin"

        cleared = engine.clear_guild_playlist("g1", set_by="admin", now=DAY1)
        assert cleared.playlist_url is None
        assert cleared.channel_id == "c1"  # the post target is untouched

    def test_set_strips_surrounding_whitespace(
        self, engine: GameEngine, db: Database
    ) -> None:
        engine.set_guild_channel("g1", "c1", set_by="test", now=DAY1)
        row = engine.set_guild_playlist("g1", f"  {PLAYLIST_URL}  ", set_by="a", now=DAY1)
        assert row.playlist_url == PLAYLIST_URL

    def test_clear_when_unset_is_a_noop(self, engine: GameEngine) -> None:
        engine.set_guild_channel("g1", "c1", set_by="test", now=DAY1)
        row = engine.clear_guild_playlist("g1", set_by="admin", now=DAY1)
        assert row.playlist_url is None

    def test_set_requires_a_configured_guild(self, engine: GameEngine) -> None:
        with pytest.raises(EngineError, match="unconfigured guild"):
            engine.set_guild_playlist("ghost", PLAYLIST_URL, set_by="admin", now=DAY1)

    def test_clear_requires_a_configured_guild(self, engine: GameEngine) -> None:
        with pytest.raises(EngineError, match="unconfigured guild"):
            engine.clear_guild_playlist("ghost", set_by="admin", now=DAY1)


class TestScopedSelection:
    """A guild's challenges draw only from its effective catalog scope."""

    def test_custom_playlist_guild_draws_only_from_its_scope(
        self, engine: GameEngine, db: Database
    ) -> None:
        global_ids = {_add_scoped_song(db, "", f"global-{i}") for i in range(3)}
        custom_ids = {_add_scoped_song(db, "g1", f"custom-{i}") for i in range(3)}
        engine.set_guild_channel("g1", "c1", set_by="test", now=DAY1)
        engine.set_guild_playlist("g1", PLAYLIST_URL, set_by="test", now=DAY1)

        challenge = engine.ensure_today_challenge("g1", "c1", DAY1)

        assert challenge.song.id in custom_ids
        assert challenge.song.id not in global_ids

    def test_guild_without_playlist_draws_only_from_the_global_pool(
        self, engine: GameEngine, db: Database
    ) -> None:
        global_ids = {_add_scoped_song(db, "", f"global-{i}") for i in range(3)}
        custom_ids = {_add_scoped_song(db, "g1", f"custom-{i}") for i in range(3)}
        # g2 never configured a playlist: its pool is the global scope, even
        # though g1's custom rows sit in the same table.
        challenge = engine.ensure_today_challenge("g2", "c1", DAY1)

        assert challenge.song.id in global_ids
        assert challenge.song.id not in custom_ids

    def test_two_guilds_use_different_playlists_simultaneously(
        self, engine: GameEngine, db: Database
    ) -> None:
        a_songs = {_add_scoped_song(db, "gA", f"aaa-{i}") for i in range(2)}
        b_songs = {_add_scoped_song(db, "gB", f"bbb-{i}") for i in range(2)}
        _add_scoped_song(db, "", "global-0")
        for guild, url in (("gA", PLAYLIST_URL), ("gB", PLAYLIST_URL_2)):
            engine.set_guild_channel(guild, "c1", set_by="test", now=DAY1)
            engine.set_guild_playlist(guild, url, set_by="test", now=DAY1)

        challenge_a = engine.ensure_today_challenge("gA", "c1", DAY1)
        challenge_b = engine.ensure_today_challenge("gB", "c1", DAY1)

        assert challenge_a.song.id in a_songs
        assert challenge_b.song.id in b_songs

    def test_empty_custom_scope_never_falls_back_to_the_global_pool(
        self, db: Database, tmp_path: Path
    ) -> None:
        _add_scoped_song(db, "", "global-0")
        calls: list[str] = []

        def refresher(guild_id: str) -> RefreshResult:
            calls.append(guild_id)
            return RefreshResult(sources=())  # adds nothing

        engine, _ = _make_engine(tmp_path, db, catalog_refresher=refresher)
        engine.set_guild_channel("g1", "c1", set_by="test", now=DAY1)
        engine.set_guild_playlist("g1", PLAYLIST_URL, set_by="test", now=DAY1)

        with pytest.raises(CatalogEmptyError, match="catalog_empty"):
            engine.ensure_today_challenge("g1", "c1", DAY1)

        # The bootstrap targeted the guild's own catalog, not the global one.
        assert calls == ["g1"]

    def test_bootstrap_refresh_populates_the_custom_scope(
        self, db: Database, tmp_path: Path
    ) -> None:
        def refresher(guild_id: str) -> RefreshResult:
            assert guild_id == "g1"
            _add_scoped_song(db, guild_id, "custom-0")
            return RefreshResult(sources=(SourceRefresh(source="youtube", added=1),))

        engine, _ = _make_engine(tmp_path, db, catalog_refresher=refresher)
        engine.set_guild_channel("g1", "c1", set_by="test", now=DAY1)
        engine.set_guild_playlist("g1", PLAYLIST_URL, set_by="test", now=DAY1)

        challenge = engine.ensure_today_challenge("g1", "c1", DAY1)

        assert challenge.created is True
        assert challenge.song.source_id == "custom-0"

    def test_clearing_the_playlist_reverts_to_the_global_pool(
        self, engine: GameEngine, db: Database
    ) -> None:
        custom_ids = {_add_scoped_song(db, "g1", f"custom-{i}") for i in range(2)}
        global_ids = {_add_scoped_song(db, "", f"global-{i}") for i in range(2)}
        engine.set_guild_channel("g1", "c1", set_by="test", now=DAY1)
        engine.set_guild_playlist("g1", PLAYLIST_URL, set_by="test", now=DAY1)
        engine.clear_guild_playlist("g1", set_by="test", now=DAY1)

        challenge = engine.ensure_today_challenge("g1", "c1", DAY1)

        assert challenge.song.id in global_ids
        assert challenge.song.id not in custom_ids

    def test_refresh_passthrough_targets_the_guild(
        self, db: Database, tmp_path: Path
    ) -> None:
        calls: list[str] = []

        def refresher(guild_id: str) -> RefreshResult:
            calls.append(guild_id)
            return RefreshResult(sources=())

        engine, _ = _make_engine(tmp_path, db, catalog_refresher=refresher)
        engine.refresh_catalog("g1")
        assert calls == ["g1"]


class TestGuildAwareRefresh:
    """refresh_catalog scoping: custom playlist -> guild scope, else global."""

    def test_custom_playlist_refreshes_into_the_guild_scope(
        self, db: Database, tmp_path: Path
    ) -> None:
        _configure_guild(db, "g1", playlist_url=PLAYLIST_URL)
        _add_scoped_song(db, "", "vid-global")

        result = refresh_catalog(
            db,
            _catalog_settings(tmp_path),
            guild_id="g1",
            providers={"youtube": StubProvider([_youtube_song("vid-custom")])},
        )

        assert result.ok
        assert result.by_source("youtube").added == 1
        assert _scoped_source_ids(db, "g1") == {"vid-custom"}
        assert _scoped_source_ids(db, "") == {"vid-global"}  # untouched

    def test_guild_without_playlist_refreshes_the_global_scope(
        self, db: Database, tmp_path: Path
    ) -> None:
        _configure_guild(db, "g1")  # configured, but no playlist override
        result = refresh_catalog(
            db,
            _catalog_settings(tmp_path),
            guild_id="g1",
            providers={"local": StubProvider([_song("a.mp3")])},
        )

        assert result.ok
        assert _scoped_source_ids(db, "") == {"a.mp3"}
        assert _scoped_source_ids(db, "g1") == set()

    def test_same_source_id_coexists_across_scopes(
        self, db: Database, tmp_path: Path
    ) -> None:
        _configure_guild(db, "g1", playlist_url=PLAYLIST_URL)
        settings = _catalog_settings(tmp_path)
        provider = {"youtube": StubProvider([_youtube_song("vid1")])}
        refresh_catalog(db, settings, providers=provider)
        refresh_catalog(db, settings, guild_id="g1", providers=provider)

        rows = db.query("SELECT guild_id FROM songs WHERE source_id = 'vid1'")
        assert sorted(str(row["guild_id"]) for row in rows) == ["", "g1"]

    def test_removal_is_limited_to_the_refreshed_scope(
        self, db: Database, tmp_path: Path
    ) -> None:
        _configure_guild(db, "g1", playlist_url=PLAYLIST_URL)
        global_id = _add_scoped_song(db, "", "vid1", source="youtube")
        _add_scoped_song(db, "g1", "vid1", source="youtube")

        # The guild's playlist no longer contains vid1: its scoped row is
        # removed; the global pool's identical row survives.
        result = refresh_catalog(
            db, _catalog_settings(tmp_path), guild_id="g1", providers={"youtube": StubProvider()}
        )

        assert result.by_source("youtube").removed == 1
        assert _scoped_source_ids(db, "g1") == set()
        assert _scoped_source_ids(db, "") == {"vid1"}
        row = db.query_one("SELECT id FROM songs WHERE guild_id = ''")
        assert row is not None
        assert int(row["id"]) == global_id

    def test_custom_playlist_builds_the_youtube_provider_from_the_stored_url(
        self, db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _configure_guild(db, "g1", playlist_url=PLAYLIST_URL)
        built: list[str] = []

        def factory(url: str) -> StubProvider:
            built.append(url)
            return StubProvider([_youtube_song("vid9")])

        monkeypatch.setattr("songbot.catalog.refresh.YouTubePlaylistProvider", factory)

        result = refresh_catalog(db, _catalog_settings(tmp_path), guild_id="g1")

        assert built == [PLAYLIST_URL]
        assert result.ok
        assert _scoped_source_ids(db, "g1") == {"vid9"}


class TestPlaylistAdminCommands:
    """The /songbot-playlist and /songbot-playlist-clear bodies."""

    async def test_set_playlist_stores_refreshes_and_acks(
        self, db: Database, tmp_path: Path
    ) -> None:
        calls: list[str] = []
        refresh = RefreshResult(
            sources=(SourceRefresh(source="youtube", added=42),)
        )

        def refresher(guild_id: str) -> RefreshResult:
            calls.append(guild_id)
            return refresh

        engine, _ = _make_engine(tmp_path, db, catalog_refresher=refresher)
        commands = _commands(engine, tmp_path)

        interaction = _interaction()
        result = await commands.set_playlist(interaction, f"  {PLAYLIST_URL}  ")

        assert result.outcome == "playlist_set"
        assert result.refresh == refresh
        row = engine.guild_settings("guild-1")
        assert row is not None
        assert row.playlist_url == PLAYLIST_URL  # stripped
        # The refresh ran against the just-configured guild.
        assert calls == ["guild-1"]
        assert [p.kind for p in interaction.payloads] == ["ephemeral"]
        content = interaction.payloads[0].content or ""
        assert PLAYLIST_URL in content
        assert "42 added" in content

    async def test_set_playlist_ack_reports_a_failed_fetch(
        self, db: Database, tmp_path: Path
    ) -> None:
        refresh = RefreshResult(
            sources=(SourceRefresh(source="youtube", error="YouTubeCatalogError: nope"),)
        )
        engine, _ = _make_engine(tmp_path, db, catalog_refresher=lambda _g: refresh)
        commands = _commands(engine, tmp_path)

        interaction = _interaction()
        result = await commands.set_playlist(interaction, PLAYLIST_URL)

        # The config is kept (retry via /songbot-reload); the ack surfaces it.
        assert result.outcome == "playlist_set"
        content = interaction.payloads[0].content or ""
        assert "⚠️ failed" in content
        assert "YouTubeCatalogError" in content
        row = engine.guild_settings("guild-1")
        assert row is not None
        assert row.playlist_url == PLAYLIST_URL

    async def test_set_playlist_blank_url_refused_with_zero_mutation(
        self, db: Database, tmp_path: Path
    ) -> None:
        calls: list[str] = []

        def refresher(guild_id: str) -> RefreshResult:
            calls.append(guild_id)
            return RefreshResult(sources=())

        engine, _ = _make_engine(tmp_path, db, catalog_refresher=refresher)
        commands = _commands(engine, tmp_path)

        interaction = _interaction()
        result = await commands.set_playlist(interaction, "   ")

        assert result.outcome == "refused"
        assert interaction.payloads[0].content == ADMIN_PLAYLIST_BLANK_MESSAGE
        assert calls == []  # no refresh ran
        row = engine.guild_settings("guild-1")
        assert row is not None
        assert row.playlist_url is None

    async def test_set_playlist_unconfigured_guild(
        self, engine: GameEngine, tmp_path: Path
    ) -> None:
        engine.set_guild_channel("guild-1", "channel-1", set_by="test", now=DAY1)
        commands = AdminCommands(
            engine, _settings(tmp_path), clock=lambda: DAY1, post_sender=_RecordingPoster()
        )

        interaction = _interaction(guild_id="guild-unknown")
        result = await commands.set_playlist(interaction, PLAYLIST_URL)

        assert result.outcome == "not_configured"
        assert interaction.payloads[0].content == ADMIN_NOT_CONFIGURED_MESSAGE

    async def test_set_playlist_non_admin_denied(
        self, db: Database, tmp_path: Path
    ) -> None:
        engine, _ = _make_engine(tmp_path, db)
        commands = _commands(engine, tmp_path)

        interaction = _interaction(manage_guild=False)
        result = await commands.set_playlist(interaction, PLAYLIST_URL)

        assert result.outcome == "denied"
        assert interaction.payloads[0].content == PERMISSION_DENIED_MESSAGE
        row = engine.guild_settings("guild-1")
        assert row is not None
        assert row.playlist_url is None

    async def test_clear_playlist_reverts_and_acks(
        self, db: Database, tmp_path: Path
    ) -> None:
        engine, _ = _make_engine(tmp_path, db)
        commands = _commands(engine, tmp_path)
        engine.set_guild_playlist("guild-1", PLAYLIST_URL, set_by="admin", now=DAY1)

        interaction = _interaction()
        result = await commands.clear_playlist(interaction)

        assert result.outcome == "playlist_cleared"
        row = engine.guild_settings("guild-1")
        assert row is not None
        assert row.playlist_url is None
        assert [p.kind for p in interaction.payloads] == ["ephemeral"]
        assert "default catalog" in (interaction.payloads[0].content or "")

    async def test_clear_playlist_when_unset_still_acks(
        self, db: Database, tmp_path: Path
    ) -> None:
        engine, _ = _make_engine(tmp_path, db)
        commands = _commands(engine, tmp_path)

        result = await commands.clear_playlist(_interaction())

        assert result.outcome == "playlist_cleared"

    async def test_clear_playlist_unconfigured_guild(
        self, engine: GameEngine, tmp_path: Path
    ) -> None:
        engine.set_guild_channel("guild-1", "channel-1", set_by="test", now=DAY1)
        commands = AdminCommands(
            engine, _settings(tmp_path), clock=lambda: DAY1, post_sender=_RecordingPoster()
        )

        interaction = _interaction(guild_id="guild-unknown")
        result = await commands.clear_playlist(interaction)

        assert result.outcome == "not_configured"
        assert interaction.payloads[0].content == ADMIN_NOT_CONFIGURED_MESSAGE


class TestHarnessPlaylistScenarios:
    """The admin-playlist / admin-playlist-clear harness drivers."""

    async def test_admin_playlist_scenario(self, db: Database, tmp_path: Path) -> None:
        refresh = RefreshResult(sources=(SourceRefresh(source="youtube", added=3),))
        engine, _ = _make_engine(tmp_path, db, catalog_refresher=lambda _g: refresh)
        ctx = HarnessContext(settings=_settings(tmp_path), db=db, engine=engine)

        out = json_roundtrip(await scenario_admin_playlist(ctx, ADMIN, PLAYLIST_URL, DAY1))

        assert kinds(out) == ["ephemeral"]
        assert out["state"]["outcome"] == "playlist_set"
        assert out["state"]["playlist_url"] == PLAYLIST_URL
        assert out["state"]["sources"] == [
            {
                "source": "youtube",
                "added": 3,
                "updated": 0,
                "removed": 0,
                "retained": 0,
                "error": None,
            }
        ]
        assert PLAYLIST_URL in out["payloads"][0]["content"]

    async def test_admin_playlist_clear_scenario(
        self, db: Database, tmp_path: Path
    ) -> None:
        engine, _ = _make_engine(tmp_path, db)
        ctx = HarnessContext(settings=_settings(tmp_path), db=db, engine=engine)
        engine.set_guild_playlist(ctx.guild_id, PLAYLIST_URL, set_by="admin", now=DAY1)

        out = json_roundtrip(await scenario_admin_playlist_clear(ctx, ADMIN, DAY1))

        assert kinds(out) == ["ephemeral"]
        assert out["state"]["outcome"] == "playlist_cleared"
        assert out["state"]["playlist_url"] is None

    async def test_admin_playlist_non_admin_denied_without_mutation(
        self, db: Database, tmp_path: Path
    ) -> None:
        calls: list[str] = []

        def refresher(guild_id: str) -> RefreshResult:
            calls.append(guild_id)
            return RefreshResult(sources=())

        engine, _ = _make_engine(tmp_path, db, catalog_refresher=refresher)
        ctx = HarnessContext(settings=_settings(tmp_path), db=db, engine=engine)

        out = json_roundtrip(await scenario_admin_playlist(ctx, NON_ADMIN, PLAYLIST_URL, DAY1))

        assert kinds(out) == ["ephemeral"]
        assert out["state"]["outcome"] == "denied"
        assert out["state"]["playlist_url"] is None
        assert calls == []
