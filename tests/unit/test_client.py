"""Construct-level tests for the live client wiring (songbot/bot/client.py).

ABSOLUTE CONSTRAINT: the live bot never runs in-mission (a network security
agent flags all Discord traffic). These tests therefore verify CONSTRUCTION
and WIRING only — a real (unconnected) ``discord.Client`` subclass is
instantiated with the
production stack seams faked (post/reveal transports, health starter, command
syncer), and the scheduler logic is driven one tick at a time against a real
GameEngine on a tmp SQLite DB with the fake snippet service. ``client.run()``,
``client.login()``, and the gateway are NEVER touched; ``Route.BASE`` tests
restore the module global.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import runpy
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest import mock

import discord
import pytest

from songbot.bot import client as client_module
from songbot.bot.client import (
    SongBotClient,
    apply_api_base_override,
    main,
    run_bot,
)
from songbot.bot.views import DailyChallengeView
from songbot.config import ConfigError, Settings
from songbot.db import Database
from songbot.engine import Challenge, GameEngine, Reveal
from tests.unit.test_engine_daily import FakeSnippets, _add_song

GUILD_ID = "1234567890"
CHANNEL_ID = "555666777"
GUILD_OBJECT = discord.Object(id=int(GUILD_ID))

DAY1_PM = datetime(2026, 8, 13, 16, 0, 0, tzinfo=UTC)  # 2026-08-13 13:00 ADT (post due)
DAY1_AM = datetime(2026, 8, 13, 14, 59, 30, tzinfo=UTC)  # 2026-08-13 11:59:30 ADT
DAY2_NOON = datetime(2026, 8, 14, 15, 0, 0, tzinfo=UTC)  # 2026-08-14 12:00 ADT exactly

CUSTOM_API_BASE = "http://127.0.0.1:9"  # bogus local base, like VAL-OPS-008


def _settings(tmp_path: Path, *, discord_api_base: str = "https://discord.com/api/v10") -> Settings:
    """A valid Settings with NUMERIC guild/channel ids (the live client needs them)."""
    return Settings(
        discord_token="test-token",
        guild_id=GUILD_ID,
        channel_id=CHANNEL_ID,
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
        discord_api_base=discord_api_base,
    )


@dataclass
class _Stack:
    settings: Settings
    db: Database
    engine: GameEngine
    snippets: FakeSnippets


def _make_stack(tmp_path: Path, *, fail_snippets: bool = False) -> _Stack:
    settings = _settings(tmp_path)
    db = Database.open(settings.database_path)
    snippets = FakeSnippets(settings.snippet_cache_dir, fail=fail_snippets)
    engine = GameEngine(db, settings, snippets)
    # Mirror the live client's env bootstrap: the configured guild/channel
    # pair lands in guild_settings, which the scheduler iterates.
    engine.set_guild_channel(
        GUILD_ID, CHANNEL_ID, set_by="env", now=datetime(2026, 8, 1, tzinfo=UTC)
    )
    return _Stack(settings=settings, db=db, engine=engine, snippets=snippets)


class _FakeHealthHandle:
    def __init__(self) -> None:
        self.cleaned = False

    async def cleanup(self) -> None:
        self.cleaned = True


class _FakeHealthStarter:
    def __init__(self) -> None:
        self.calls: list[Settings] = []
        self.handle = _FakeHealthHandle()

    async def __call__(self, settings: Settings) -> _FakeHealthHandle:
        self.calls.append(settings)
        return self.handle


class _FakeCommandSyncer:
    def __init__(self) -> None:
        self.syncs: list[tuple[Any, int]] = []

    async def __call__(self, tree: Any, guild: discord.abc.Snowflake) -> None:
        self.syncs.append((tree, guild.id))


class _RecordingSender:
    """Post/reveal transport double: records calls in order, sends nothing."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    async def post(self, challenge: Challenge) -> None:
        self.events.append(("post", challenge))

    async def reveal(self, reveal: Reveal) -> None:
        self.events.append(("reveal", reveal))


@pytest.fixture
def stack(tmp_path: Path) -> Iterator[_Stack]:
    made = _make_stack(tmp_path)
    yield made
    made.db.close()  # sqlite3 close is idempotent (the client may have closed it)


@pytest.fixture
def sender() -> _RecordingSender:
    return _RecordingSender()


@pytest.fixture
def health() -> _FakeHealthStarter:
    return _FakeHealthStarter()


@pytest.fixture
def syncer() -> _FakeCommandSyncer:
    return _FakeCommandSyncer()


def _make_client(
    stack: _Stack,
    *,
    clock: Any = None,
    sender: _RecordingSender | None = None,
    health: _FakeHealthStarter | None = None,
    syncer: _FakeCommandSyncer | None = None,
) -> SongBotClient:
    return SongBotClient(
        stack.settings,
        stack.db,
        stack.engine,
        clock=clock if clock is not None else lambda: DAY1_PM,
        post_sender=(sender or _RecordingSender()).post,
        reveal_sender=(sender or _RecordingSender()).reveal,
        health_starter=health or _FakeHealthStarter(),
        command_syncer=syncer or _FakeCommandSyncer(),
    )


class TestConstruction:
    """The client object builds without any network, gateway, or side effects."""

    def test_minimal_guilds_only_intents(self, stack: _Stack) -> None:
        client = _make_client(stack)

        assert client.intents.guilds is True
        assert client.intents.message_content is False
        assert client.intents.members is False
        assert client.intents.presences is False

    def test_construction_has_no_side_effects(self, stack: _Stack) -> None:
        client = _make_client(stack)

        # Views, health server, and the scheduler only start in setup_hook.
        assert client._connection._view_store.persistent_views == []
        assert client._health_handle is None
        assert client._scheduler_task is None

    def test_from_settings_builds_the_real_stack(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)

        client = SongBotClient.from_settings(settings)

        assert isinstance(client._engine, GameEngine)
        assert client._db.path == settings.database_path
        # Database.open migrated the schema.
        row = client._db.query_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='challenges'"
        )
        assert row is not None

    def test_non_numeric_ids_are_a_clear_config_error(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        bad = Settings(
            **{**settings.__dict__, "guild_id": "guild-1"}  # type: ignore[arg-type]
        )
        stack = _make_stack(tmp_path)
        with pytest.raises(ConfigError, match="DISCORD_GUILD_ID"):
            SongBotClient(bad, stack.db, stack.engine)


class TestSetupHook:
    """setup_hook wiring: views, admin commands, health server, scheduler task."""

    async def test_registers_admin_commands_guild_scoped(
        self, stack: _Stack, syncer: _FakeCommandSyncer
    ) -> None:
        client = _make_client(stack, syncer=syncer)
        try:
            await client.setup_hook()

            guild_commands = client.tree.get_commands(guild=GUILD_OBJECT)
            assert {c.name for c in guild_commands} == {
                "songbot-setup",
                "songbot-post",
                "songbot-skip",
                "songbot-reload",
            }
            # Guild-scoped only: nothing lands on the global tree.
            assert client.tree.get_commands() == []
            # The (faked) sync ran exactly once, for the configured guild.
            assert syncer.syncs == [(client.tree, int(GUILD_ID))]
        finally:
            await client.close()

    async def test_setup_hook_seeds_the_env_bootstrap_guild(
        self, tmp_path: Path
    ) -> None:
        # A stack built WITHOUT the usual seeding simulates a first boot.
        settings = _settings(tmp_path)
        db = Database.open(settings.database_path)
        snippets = FakeSnippets(settings.snippet_cache_dir)
        engine = GameEngine(db, settings, snippets)
        client = _make_client(_Stack(settings, db, engine, snippets))
        try:
            assert engine.all_guild_settings() == []

            await client.setup_hook()

            row = engine.guild_settings(GUILD_ID)
            assert row is not None
            assert row.channel_id == CHANNEL_ID
            assert row.set_by == "env"
        finally:
            await client.close()

    async def test_setup_hook_without_bootstrap_pair_syncs_nothing(
        self, tmp_path: Path, syncer: _FakeCommandSyncer
    ) -> None:
        settings = _settings(tmp_path)
        multi_only = Settings(**{**settings.__dict__, "guild_id": None, "channel_id": None})  # type: ignore[arg-type]
        db = Database.open(settings.database_path)
        engine = GameEngine(db, multi_only, FakeSnippets(multi_only.snippet_cache_dir))
        client = SongBotClient(
            multi_only,
            db,
            engine,
            clock=lambda: DAY1_PM,
            post_sender=_RecordingSender().post,
            reveal_sender=_RecordingSender().reveal,
            health_starter=_FakeHealthStarter(),
            command_syncer=syncer,
        )
        try:
            await client.setup_hook()

            assert syncer.syncs == []
            assert client.tree.get_commands() == []
        finally:
            await client.close()

    async def test_registers_lazy_persistent_fallback_view(
        self, stack: _Stack
    ) -> None:
        _add_song(stack.db, "song-1")
        client = _make_client(stack)
        try:
            with mock.patch.object(client, "add_view", wraps=client.add_view) as spy:
                await client.setup_hook()

            assert spy.call_count == 1
            view = spy.call_args.args[0]
            assert isinstance(view, DailyChallengeView)
            assert view.is_persistent()
            # Multi-guild: the fallback is LAZY — no fixed challenge binding;
            # it resolves the clicking guild's latest challenge at press time.
            assert view._challenge_id is None
            assert view._guild_id is None
            custom_ids = {
                child.custom_id
                for child in view.children
                if isinstance(child, discord.ui.Button)
            }
            assert custom_ids == {
                "songbot:hear_more",
                "songbot:guess",
                "songbot:leaderboard",
            }
            # Really registered in the client's view store (restart dispatch).
            assert client._connection._view_store.persistent_views
        finally:
            await client.close()

    async def test_persistent_fallback_registered_even_with_no_challenges(
        self, stack: _Stack
    ) -> None:
        client = _make_client(stack)
        try:
            with mock.patch.object(client, "add_view", wraps=client.add_view) as spy:
                await client.setup_hook()

            # Lazy resolution makes registration safe on a fresh install.
            assert spy.call_count == 1
            assert client._connection._view_store.persistent_views
        finally:
            await client.close()

    async def test_on_guild_join_syncs_that_guild_once(
        self, stack: _Stack, syncer: _FakeCommandSyncer
    ) -> None:
        client = _make_client(stack, syncer=syncer)
        try:
            new_guild = discord.Object(id=424242)

            await client.on_guild_join(new_guild)  # type: ignore[arg-type]
            await client.on_guild_join(new_guild)  # type: ignore[arg-type]  # idempotent

            assert syncer.syncs == [(client.tree, 424242)]
            assert len(client.tree.get_commands(guild=new_guild)) == 4
        finally:
            await client.close()

    async def test_on_guild_remove_drops_configuration(
        self, stack: _Stack
    ) -> None:
        _add_song(stack.db, "song-1")
        challenge = stack.engine.ensure_today_challenge(GUILD_ID, CHANNEL_ID, DAY1_PM)
        client = _make_client(stack)
        try:
            assert stack.engine.guild_settings(GUILD_ID) is not None

            await client.on_guild_remove(discord.Object(id=int(GUILD_ID)))  # type: ignore[arg-type]

            assert stack.engine.guild_settings(GUILD_ID) is None
            # Game history is kept — only the post target is dropped.
            row = stack.db.query_one(
                "SELECT 1 FROM challenges WHERE id = ?", (challenge.id,)
            )
            assert row is not None
        finally:
            await client.close()

    async def test_starts_health_server_and_scheduler_task(
        self, stack: _Stack, health: _FakeHealthStarter
    ) -> None:
        client = _make_client(stack, health=health)
        try:
            await client.setup_hook()

            assert health.calls == [stack.settings]
            assert client._health_handle is health.handle
            task = client._scheduler_task
            assert isinstance(task, asyncio.Task)
            assert task.get_name() == "songbot-daily-scheduler"
            assert not task.done()
        finally:
            await client.close()

    async def test_close_cleans_up_scheduler_health_and_db(
        self, stack: _Stack, health: _FakeHealthStarter
    ) -> None:
        client = _make_client(stack, health=health)
        await client.setup_hook()
        task = client._scheduler_task
        assert task is not None

        await client.close()

        assert task.done()
        assert health.handle.cleaned is True
        with pytest.raises(Exception, match="closed"):  # sqlite3.ProgrammingError
            stack.db.query_one("SELECT 1")


class TestApiBaseOverride:
    """DISCORD_API_BASE -> discord.http.Route.BASE, applied BEFORE login."""

    @pytest.fixture(autouse=True)
    def _restore_route_base(self) -> Iterator[None]:
        original = discord.http.Route.BASE
        yield
        discord.http.Route.BASE = original

    def test_override_applied_when_custom_base_configured(self, tmp_path: Path) -> None:
        apply_api_base_override(_settings(tmp_path, discord_api_base=CUSTOM_API_BASE))

        assert discord.http.Route.BASE == CUSTOM_API_BASE

    def test_default_base_is_a_noop(self, tmp_path: Path) -> None:
        apply_api_base_override(_settings(tmp_path))

        assert discord.http.Route.BASE == "https://discord.com/api/v10"

    def test_run_bot_applies_override_before_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = _settings(tmp_path, discord_api_base=CUSTOM_API_BASE)
        stack = _make_stack(tmp_path)
        client = _make_client(stack)
        observed: dict[str, Any] = {}

        def fake_run(token: str, **kwargs: Any) -> None:
            # By the time run() (which performs login) executes, the override
            # must already be in effect.
            observed["base_at_login"] = discord.http.Route.BASE
            observed["token"] = token
            observed["log_level"] = kwargs.get("log_level")

        monkeypatch.setattr(client, "run", fake_run)

        run_bot(settings, client=client)

        assert observed == {
            "base_at_login": CUSTOM_API_BASE,
            "token": "test-token",
            "log_level": logging.INFO,
        }


class TestSchedulerTick:
    """One-tick-at-a-time scheduler verification against a real engine."""

    async def test_first_run_posts_immediately(
        self, stack: _Stack, sender: _RecordingSender
    ) -> None:
        _add_song(stack.db, "song-1")
        client = _make_client(stack, sender=sender)

        delay = await client._scheduler_tick(DAY1_PM)

        assert len(sender.events) == 1
        kind, challenge = sender.events[0]
        assert kind == "post"
        assert challenge.date == "2026-08-13"
        assert challenge.created is True
        # Next post is tomorrow 12:00 ADT, but the sleep is capped.
        assert delay == client_module.MAX_SLEEP_SEC

    async def test_not_due_after_todays_post(
        self, stack: _Stack, sender: _RecordingSender
    ) -> None:
        _add_song(stack.db, "song-1")
        client = _make_client(stack, sender=sender)
        await client._scheduler_tick(DAY1_PM)

        await client._scheduler_tick(DAY1_PM)

        assert len(sender.events) == 1  # no double post (restart-safe gating)

    async def test_next_day_reveals_previous_before_posting(
        self, stack: _Stack, sender: _RecordingSender
    ) -> None:
        _add_song(stack.db, "song-1")
        _add_song(stack.db, "song-2")
        client = _make_client(stack, sender=sender)
        await client._scheduler_tick(DAY1_PM)

        await client._scheduler_tick(DAY2_NOON)

        assert [kind for kind, _ in sender.events] == ["post", "reveal", "post"]
        _, reveal = sender.events[1]
        assert reveal.date == "2026-08-13"
        _, challenge2 = sender.events[2]
        assert challenge2.date == "2026-08-14"

    async def test_post_does_not_re_register_views(self, stack: _Stack) -> None:
        """The lazy persistent fallback needs no per-post re-binding."""
        _add_song(stack.db, "song-1")
        client = _make_client(stack)

        with mock.patch.object(client, "add_view", wraps=client.add_view) as spy:
            await client._scheduler_tick(DAY1_PM)

        assert spy.call_count == 0  # registration happens once, in setup_hook

    async def test_catalog_empty_is_logged_not_raised(
        self, stack: _Stack, sender: _RecordingSender, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _make_client(stack, sender=sender)  # no songs at all

        with caplog.at_level(logging.WARNING, logger="songbot.bot.client"):
            delay = await client._scheduler_tick(DAY1_PM)

        assert sender.events == []
        assert delay == client_module.MAX_SLEEP_SEC  # normal cadence, not retry spam
        assert any("catalog" in record.message.lower() for record in caplog.records)

    async def test_snippet_failure_backs_off_with_retry_delay(
        self,
        tmp_path: Path,
        sender: _RecordingSender,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        failing = _make_stack(tmp_path, fail_snippets=True)
        try:
            _add_song(failing.db, "song-1")
            client = _make_client(failing, sender=sender)

            with caplog.at_level(logging.ERROR, logger="songbot.bot.client"):
                delay = await client._scheduler_tick(DAY1_PM)

            assert sender.events == []
            assert delay == client_module.RETRY_DELAY_SEC
            assert any("daily post" in record.message.lower() for record in caplog.records)
        finally:
            failing.db.close()

    def test_seconds_until_next_check_uses_next_post_datetime(self, stack: _Stack) -> None:
        client = _make_client(stack)

        # 11:59:30 ADT -> 30s until the 12:00 post (below the cap, exact).
        assert client._seconds_until_next_check(DAY1_AM) == 30.0

    def test_seconds_until_next_check_is_capped(self, stack: _Stack) -> None:
        client = _make_client(stack)

        # 13:00 ADT -> 23h until the next post, capped at MAX_SLEEP_SEC.
        assert client._seconds_until_next_check(DAY1_PM) == client_module.MAX_SLEEP_SEC


class TestMultiGuildScheduler:
    """The scheduler iterates every configured guild, isolated per guild."""

    GUILD_B = "2222222222"
    CHANNEL_B = "999888777"

    def _add_guild_b(self, stack: _Stack) -> None:
        stack.engine.set_guild_channel(
            self.GUILD_B,
            self.CHANNEL_B,
            set_by="test",
            now=datetime(2026, 8, 1, tzinfo=UTC),
        )

    async def test_posts_to_every_configured_guild(
        self, stack: _Stack, sender: _RecordingSender
    ) -> None:
        _add_song(stack.db, "song-1")
        _add_song(stack.db, "song-2")
        self._add_guild_b(stack)
        client = _make_client(stack, sender=sender)

        await client._scheduler_tick(DAY1_PM)

        assert [kind for kind, _ in sender.events] == ["post", "post"]
        by_guild = {c.guild_id: c for _, c in sender.events}
        assert set(by_guild) == {GUILD_ID, self.GUILD_B}
        assert by_guild[GUILD_ID].channel_id == CHANNEL_ID
        assert by_guild[self.GUILD_B].channel_id == self.CHANNEL_B
        # Per-guild picks: each guild has its own challenge row for the date.
        rows = stack.db.query("SELECT guild_id, date FROM challenges")
        assert {(r["guild_id"], r["date"]) for r in rows} == {
            (GUILD_ID, "2026-08-13"),
            (self.GUILD_B, "2026-08-13"),
        }

    async def test_unconfigured_guilds_are_not_touched(
        self, stack: _Stack, sender: _RecordingSender
    ) -> None:
        """A guild with challenges but no guild_settings row gets no posts."""
        _add_song(stack.db, "song-1")
        # A challenge exists for some other guild (history), but only the
        # seeded guild is configured.
        stack.engine.ensure_today_challenge("ghost-guild", "ghost-channel", DAY1_PM)
        client = _make_client(stack, sender=sender)

        await client._scheduler_tick(DAY2_NOON)

        assert [c.guild_id for _, c in sender.events] == [GUILD_ID]

    async def test_one_guilds_failure_never_blocks_another(
        self, stack: _Stack, sender: _RecordingSender, caplog: pytest.LogCaptureFixture
    ) -> None:
        _add_song(stack.db, "song-1")
        _add_song(stack.db, "song-2")
        self._add_guild_b(stack)

        class _FlakySender(_RecordingSender):
            async def post(self, challenge: Challenge) -> None:
                if challenge.guild_id == GUILD_ID:
                    raise RuntimeError("discord 500")
                await super().post(challenge)

        flaky = _FlakySender()
        client = _make_client(stack, sender=flaky)

        with caplog.at_level(logging.ERROR, logger="songbot.bot.client"):
            delay = await client._scheduler_tick(DAY1_PM)

        # Guild B posted; guild A's failure is logged and drives the retry.
        assert [c.guild_id for _, c in flaky.events] == [self.GUILD_B]
        assert delay == client_module.RETRY_DELAY_SEC
        assert any(GUILD_ID in record.message for record in caplog.records)
        # Pinned #16: A's just-created challenge was rolled back.
        assert stack.db.query_one(
            "SELECT 1 FROM challenges WHERE guild_id = ?", (GUILD_ID,)
        ) is None

        # The retry posts A only (B is no longer due) on the same day.
        class _FixedSender(_RecordingSender):
            pass

        fixed = _FixedSender()
        client._post_sender = fixed.post
        await client._scheduler_tick(DAY1_PM)

        assert [c.guild_id for _, c in fixed.events] == [GUILD_ID]

    async def test_reveal_targets_the_channel_the_challenge_was_posted_in(
        self, stack: _Stack, sender: _RecordingSender
    ) -> None:
        """A channel change mid-game never moves the reveal of an old post."""
        _add_song(stack.db, "song-1")
        _add_song(stack.db, "song-2")
        client = _make_client(stack, sender=sender)
        await client._scheduler_tick(DAY1_PM)
        # Admin re-runs /songbot-setup pointing at a new channel.
        stack.engine.set_guild_channel(GUILD_ID, "333444555", set_by="admin", now=DAY2_NOON)

        await client._scheduler_tick(DAY2_NOON)

        kinds = [kind for kind, _ in sender.events]
        assert kinds == ["post", "reveal", "post"]
        reveal = sender.events[1][1]
        assert reveal.channel_id == CHANNEL_ID  # the OLD channel
        new_post = sender.events[2][1]
        assert new_post.channel_id == "333444555"


class TestEntrypoint:
    """`python -m songbot` wiring — verified without ever connecting."""

    def test_main_returns_1_on_config_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def fake_load_settings() -> Settings:
            raise ConfigError("bad config for test")

        run_bot_calls: list[Any] = []
        monkeypatch.setattr(client_module, "load_settings", fake_load_settings)
        monkeypatch.setattr(client_module, "run_bot", run_bot_calls.append)

        assert main() == 1
        assert "bad config for test" in capsys.readouterr().err
        assert run_bot_calls == []

    def test_main_runs_bot_with_loaded_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sentinel = object()
        run_bot_calls: list[Any] = []
        monkeypatch.setattr(client_module, "load_settings", lambda: sentinel)
        monkeypatch.setattr(client_module, "run_bot", run_bot_calls.append)

        assert main() == 0
        assert run_bot_calls == [sentinel]

    def test_python_m_songbot_invokes_main(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []
        monkeypatch.setattr(client_module, "main", lambda: calls.append(1) or 0)

        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("songbot", run_name="__main__")

        assert excinfo.value.code == 0
        assert calls == [1]

    def test_main_module_import_has_no_side_effects(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []
        monkeypatch.setattr(client_module, "main", lambda: calls.append(1) or 0)

        sys.modules.pop("songbot.__main__", None)
        importlib.import_module("songbot.__main__")

        assert calls == []

    def test_entrypoint_docstrings_mark_never_run(self) -> None:
        assert "NEVER" in (client_module.__doc__ or "")
        main_module = sys.modules.get("songbot.__main__") or importlib.import_module(
            "songbot.__main__"
        )
        assert "NEVER" in (main_module.__doc__ or "")
