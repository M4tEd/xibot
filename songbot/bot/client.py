"""Live Discord client wiring, gateway connect, view registration.

NEVER RUN ON THIS MACHINE during the mission: a Netskope agent blocks/flags
all Discord domains. This module is exercised at CONSTRUCTION level only
(pytest with faked transports — no ``client.run()``, no login, no gateway).
The live playtest is the user's deferred manual step, off-mission.

What ``python -m songbot`` does (when the user runs it off-machine):

1. `main` loads `.env` config (`load_settings`), then `run_bot`.
2. `run_bot` applies the ``DISCORD_API_BASE`` override FIRST —
   `apply_api_base_override` points ``discord.http.Route.BASE`` at the
   configured base BEFORE ``client.run`` performs login, so no request can
   slip out to the real API when an override is configured (documented for
   off-machine use, e.g. reaching Discord through a proxy; the default is a
   no-op). It then builds the client and blocks on the gateway.
3. `SongBotClient.setup_hook` (called by discord.py after login, before the
   websocket connects) registers the persistent `DailyChallengeView`,
   registers the /songbot-* admin commands guild-scoped and syncs them,
   starts the aiohttp ``/health`` server, and launches the daily scheduler
   task (`_daily_post_loop`).

The scheduler loop uses the PURE `songbot.scheduler` functions: it posts when
`is_post_due` says so (restart-safe: the challenges table is the record of
posts) and sleeps until `next_post_datetime`, capped so clock shifts and
out-of-band admin posts are noticed promptly. Each due tick first reveals the
previous challenge (song + winners) and then posts today's — the exact flow
the harness ``advance-day`` scenario validates headlessly.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import date, datetime
from typing import Any, Protocol, cast

import discord
from aiohttp import web
from discord import app_commands
from discord.http import Route

from songbot.bot.admin import AdminCommands, DailyPostSender, register_admin_commands
from songbot.bot.embeds import daily_challenge_embed, reveal_embed, snippet_attachment
from songbot.bot.health import build_health_app
from songbot.bot.modals import Clock, utc_now
from songbot.bot.views import DailyChallengeView
from songbot.config import ConfigError, Settings, load_settings
from songbot.db import Database
from songbot.engine import CatalogEmptyError, Challenge, GameEngine, Reveal
from songbot.scheduler import is_post_due, next_post_datetime
from songbot.snippets import SnippetGenerator

__all__ = [
    "MAX_SLEEP_SEC",
    "MIN_SLEEP_SEC",
    "RETRY_DELAY_SEC",
    "SCHEDULER_TASK_NAME",
    "SongBotClient",
    "apply_api_base_override",
    "main",
    "run_bot",
]

logger = logging.getLogger(__name__)

SCHEDULER_TASK_NAME = "songbot-daily-scheduler"
"""Name of the asyncio task running the daily post loop."""

MAX_SLEEP_SEC = 300.0
"""Cap on the scheduler sleep: re-check at least every 5 minutes so wall-clock
shifts, admin ``post-now`` posts, and skipped days are noticed promptly."""

MIN_SLEEP_SEC = 1.0
"""Floor on the scheduler sleep (avoids a busy spin exactly at post time)."""

RETRY_DELAY_SEC = 60.0
"""Backoff after a failed daily-post tick (e.g. a transient snippet error)."""


class HealthServerHandle(Protocol):
    """The slice of a started health server the client needs (cleanup)."""

    async def cleanup(self) -> None: ...


HealthStarter = Callable[[Settings], Awaitable[HealthServerHandle]]
"""Starts the /health server for the given settings; returns its handle."""

CommandSyncer = Callable[[app_commands.CommandTree[Any]], Awaitable[None]]
"""Syncs the command tree (the live default syncs guild-scoped)."""

RevealSender = Callable[[Reveal], Awaitable[None]]
"""Transport for the previous-challenge reveal (the live default sends it to
the configured channel; tests and the harness record instead)."""


class _MessageableChannel(Protocol):
    """The slice of a Discord channel the post/reveal transports need."""

    async def send(
        self,
        content: str | None = ...,
        *,
        embed: discord.Embed | None = ...,
        view: discord.ui.View | None = ...,
        file: discord.File | None = ...,
    ) -> object: ...


def _snowflake(raw: str, field: str) -> int:
    """Parse a Discord snowflake id from config, with a clear error."""
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(
            f"{field} '{raw}' is not a numeric Discord snowflake id"
        ) from None


async def _start_health_server(settings: Settings) -> HealthServerHandle:
    """The live health starter: aiohttp /health on 127.0.0.1:HEALTH_PORT.

    Loopback-only, mode ``"live"``, and performs NO Discord I/O — the same
    app the harness ``serve`` scenario exposes with mode ``"harness"``.
    """
    app = build_health_app(mode="live", guild_id=settings.guild_id)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=settings.health_port)
    await site.start()
    logger.info(
        "health endpoint listening on 127.0.0.1:%d (mode=live)", settings.health_port
    )
    return runner


class SongBotClient(discord.Client):
    """The live discord.py client: minimal intents, thin adapter wiring only.

    NEVER connected in-mission — construction and `setup_hook` are the tested
    surface. Connects with guilds-only intents (interactions need no
    privileged intents); owns no game rules (every decision is the engine's).

    Args:
        settings: validated configuration. ``guild_id``/``channel_id`` must be
            numeric snowflakes (checked eagerly — a clear `ConfigError` beats a
            late failure).
        db: the (migrated) database. The client TAKES OWNERSHIP and closes it
            in `close`.
        engine: the game engine (owns all rules).
        clock: time source for gameplay and the scheduler (default: the UTC
            wall clock — live mode posts on real time).
        post_sender: daily-post transport (default: send the real embed, a
            fresh `DailyChallengeView`, and the level-0
            ``songbot-snippet.mp3`` attachment to the configured channel).
        reveal_sender: reveal transport (default: send the reveal embed to the
            configured channel).
        health_starter: /health server starter (default: aiohttp on
            127.0.0.1:``health_port``).
        command_syncer: command-tree sync (default: guild-scoped
            ``tree.sync`` — instant availability in the configured guild).
    """

    def __init__(
        self,
        settings: Settings,
        db: Database,
        engine: GameEngine,
        *,
        clock: Clock | None = None,
        post_sender: DailyPostSender | None = None,
        reveal_sender: RevealSender | None = None,
        health_starter: HealthStarter | None = None,
        command_syncer: CommandSyncer | None = None,
    ) -> None:
        super().__init__(intents=discord.Intents(guilds=True))
        # Plain discord.Client has no command tree; constructing one here also
        # wires it into the connection state for interaction dispatch.
        self.tree: app_commands.CommandTree[SongBotClient] = app_commands.CommandTree(self)
        self._settings = settings
        self._db = db
        self._engine = engine
        self._clock: Clock = clock if clock is not None else utc_now
        self._guild = discord.Object(id=_snowflake(settings.guild_id, "DISCORD_GUILD_ID"))
        self._channel_id = _snowflake(settings.channel_id, "DISCORD_CHANNEL_ID")
        self._post_sender: DailyPostSender = (
            post_sender if post_sender is not None else self._send_daily_post
        )
        self._reveal_sender: RevealSender = (
            reveal_sender if reveal_sender is not None else self._send_reveal
        )
        self._health_starter: HealthStarter = (
            health_starter if health_starter is not None else _start_health_server
        )
        self._command_syncer: CommandSyncer = (
            command_syncer if command_syncer is not None else self._sync_commands
        )
        self._admin_commands = AdminCommands(
            engine, settings, post_sender=self._post_sender, clock=self._clock
        )
        self._health_handle: HealthServerHandle | None = None
        self._scheduler_task: asyncio.Task[None] | None = None

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        clock: Clock | None = None,
        post_sender: DailyPostSender | None = None,
        reveal_sender: RevealSender | None = None,
        health_starter: HealthStarter | None = None,
        command_syncer: CommandSyncer | None = None,
    ) -> SongBotClient:
        """Build the production stack (migrated db + engine + snippet generator).

        The keyword seams mirror the constructor (tests inject fakes; the live
        path uses the defaults).
        """
        db = Database.open(settings.database_path)
        engine = GameEngine(db, settings, SnippetGenerator(settings.snippet_cache_dir))
        return cls(
            settings,
            db,
            engine,
            clock=clock,
            post_sender=post_sender,
            reveal_sender=reveal_sender,
            health_starter=health_starter,
            command_syncer=command_syncer,
        )

    # -- startup/shutdown -------------------------------------------------

    async def setup_hook(self) -> None:
        """Wire everything after login, before the gateway connects.

        Called once by discord.py's ``login``. Registers the persistent view
        and the guild-scoped admin commands (then syncs them), starts the
        health server, and launches the daily scheduler task. The scheduler's
        ``wait_until_ready`` lives INSIDE the task — awaiting it here would
        deadlock (setup_hook runs before the websocket connects).
        """
        self._register_persistent_view()
        register_admin_commands(self.tree, self._admin_commands, guild=self._guild)
        await self._command_syncer(self.tree)
        self._health_handle = await self._health_starter(self._settings)
        self._scheduler_task = asyncio.create_task(
            self._daily_post_loop(), name=SCHEDULER_TASK_NAME
        )
        logger.info("setup complete: views + admin commands + health + scheduler")

    async def close(self) -> None:
        """Stop the scheduler and health server, close the db, then disconnect."""
        if self._scheduler_task is not None:
            self._scheduler_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._scheduler_task
            self._scheduler_task = None
        if self._health_handle is not None:
            await self._health_handle.cleanup()
            self._health_handle = None
        self._db.close()
        await super().close()

    # -- views and commands -------------------------------------------------

    def _build_view(self, challenge_id: int) -> DailyChallengeView:
        """A fresh persistent view bound to one challenge."""
        return DailyChallengeView(
            self._engine,
            challenge_id,
            guild_id=self._settings.guild_id,
            settings=self._settings,
            clock=self._clock,
        )

    def _register_persistent_view(self) -> None:
        """Register the persistent DailyChallengeView for restart-resilient buttons.

        Discord dispatches component interactions by custom_id, and the pinned
        custom_ids carry no challenge id, so the persistent fallback cannot
        know which day's message was clicked. It is therefore bound to the
        guild's LATEST challenge: the game only ever has one live challenge,
        so a click on any recent post acts on the current game (older,
        revealed posts answer with the closed-challenge notice). Messages sent
        this session dispatch through their own message-bound views first;
        this registration is the restart-recovery path (and is refreshed on
        every post). No challenges yet (fresh install): nothing to register —
        the first post attaches its own view.
        """
        challenge_id = self._latest_challenge_id()
        if challenge_id is None:
            logger.info("no challenges yet; the first post attaches the interactive view")
            return
        self.add_view(self._build_view(challenge_id))

    async def _sync_commands(self, tree: app_commands.CommandTree[Any]) -> None:
        """The live command sync: guild-scoped (instant in the configured guild)."""
        synced = await tree.sync(guild=self._guild)
        logger.info(
            "synced %d admin command(s) to guild %s", len(synced), self._settings.guild_id
        )

    # -- daily scheduler ----------------------------------------------------

    async def _daily_post_loop(self) -> None:
        """Reveal + post each day at the configured time, until the client closes."""
        await self.wait_until_ready()
        while not self.is_closed():
            delay = await self._scheduler_tick(self._clock())
            await asyncio.sleep(delay)

    async def _scheduler_tick(self, now: datetime) -> float:
        """One scheduler iteration: post if due; return seconds until the next check."""
        try:
            await self._post_if_due(now)
        except Exception:
            logger.exception(
                "daily post tick failed; retrying in %.0fs", RETRY_DELAY_SEC
            )
            return RETRY_DELAY_SEC
        return self._seconds_until_next_check(now)

    async def _post_if_due(self, now: datetime) -> None:
        """Reveal the previous challenge and post today's, when due.

        Restart-safe: ``is_post_due`` gates on the most recent challenge date
        recorded in the db, so a restart never double-posts and a missed day
        posts as soon as the bot comes back after the configured time. The
        reveal is sent BEFORE the new post (pinned #3 ordering) and is
        delivery-coupled (pinned #17): the previous challenge is marked
        revealed ONLY after its reveal announcement send succeeds — a failed
        send raises before any post is attempted, leaving the challenge
        active so the next tick retries the reveal first. An empty catalog is
        a persistent condition — logged once per check on the normal cadence,
        not a crash and not a fast retry.
        """
        if not is_post_due(
            self._last_post_date(),
            now,
            self._settings.tz,
            self._settings.daily_post_time,
        ):
            return
        reveal = self._engine.peek_reveal(self._settings.guild_id, now)
        if reveal is not None:
            await self._reveal_sender(reveal)
            # Pinned #17: mark revealed ONLY after the reveal send succeeded.
            self._engine.mark_revealed(self._settings.guild_id, now)
        try:
            challenge = self._engine.ensure_today_challenge(
                self._settings.guild_id, self._settings.channel_id, now
            )
        except CatalogEmptyError:
            logger.warning(
                "daily post skipped: catalog is empty — add songs and /songbot-reload"
            )
            return
        if challenge.created:
            try:
                await self._post_sender(challenge)
            except Exception:
                # Pinned #16: roll back the just-created challenge (never a
                # pre-existing row) so the day is not suppressed — the next
                # tick (60s backoff via `_scheduler_tick`) recreates the
                # identical challenge and retries the send.
                self._engine.delete_challenge(challenge.id)
                raise
            # Keep the persistent fallback binding pointed at the latest
            # challenge within this session too (same custom_ids overwrite the
            # restart-registration entry).
            self.add_view(self._build_view(challenge.id))

    def _seconds_until_next_check(self, now: datetime) -> float:
        """Sleep until the next scheduled post time, clamped to [MIN, MAX]."""
        next_at = next_post_datetime(
            now, self._settings.daily_post_time, self._settings.tz
        )
        delay = (next_at - now).total_seconds()
        return min(max(delay, MIN_SLEEP_SEC), MAX_SLEEP_SEC)

    # -- live transports (the ONLY methods that would touch Discord HTTP) -----

    async def _challenge_channel(self) -> _MessageableChannel:
        """Resolve the configured channel (gateway cache first, REST fallback)."""
        channel = self.get_channel(self._channel_id) or await self.fetch_channel(
            self._channel_id
        )
        return cast("_MessageableChannel", channel)

    async def _send_daily_post(self, challenge: Challenge) -> None:
        """Post the daily challenge: real embed + real view + level-0 snippet."""
        channel = await self._challenge_channel()
        await channel.send(
            embed=daily_challenge_embed(challenge, self._settings),
            view=self._build_view(challenge.id),
            file=snippet_attachment(challenge.snippet_paths[0]),
        )

    async def _send_reveal(self, reveal: Reveal) -> None:
        """Post the previous challenge's reveal (song + winners) to the channel."""
        channel = await self._challenge_channel()
        await channel.send(embed=reveal_embed(reveal))

    # -- db reads (same read-only pattern the harness uses) -------------------

    def _last_post_date(self) -> date | None:
        """The local date of the guild's most recent challenge (the post record)."""
        row = self._db.query_one(
            "SELECT MAX(date) AS latest FROM challenges WHERE guild_id = ?",
            (self._settings.guild_id,),
        )
        latest = row["latest"] if row is not None else None
        return date.fromisoformat(str(latest)) if latest is not None else None

    def _latest_challenge_id(self) -> int | None:
        """The id of the guild's most recent challenge (persistent-view binding)."""
        row = self._db.query_one(
            "SELECT id FROM challenges WHERE guild_id = ?"
            " ORDER BY date DESC, id DESC LIMIT 1",
            (self._settings.guild_id,),
        )
        return int(row["id"]) if row is not None else None


def apply_api_base_override(settings: Settings) -> None:
    """Point discord.py's REST client at ``settings.discord_api_base``.

    MUST run before any HTTP call — i.e. before ``client.run``/login:
    ``discord.http.Route.BASE`` is the module-level base every request URL is
    built from. Exists for off-machine use (e.g. networks where Discord is
    reached through a proxy or mock); NEVER exercised in-mission. A no-op
    when the configured base already matches the library default.
    """
    if settings.discord_api_base != Route.BASE:
        logger.warning("DISCORD_API_BASE override in effect: %s", settings.discord_api_base)
        Route.BASE = settings.discord_api_base


def run_bot(settings: Settings, *, client: SongBotClient | None = None) -> None:
    """Apply the API-base override, build the client, and BLOCK on the gateway.

    The override is applied BEFORE ``client.run`` (which performs the login),
    so no request can slip out to the real Discord API when an override is
    configured. ``client`` is injectable purely so tests can verify that
    ordering without a gateway connection.
    """
    apply_api_base_override(settings)
    bot = client if client is not None else SongBotClient.from_settings(settings)
    bot.run(settings.discord_token, log_level=_log_level(settings.log_level))


def _log_level(name: str) -> int:
    """Map the validated ``LOG_LEVEL`` name to a ``logging`` level constant."""
    level: int = getattr(logging, name, logging.INFO)
    return level


def main() -> int:
    """``python -m songbot`` entrypoint: load config, then run the live bot.

    NEVER run on this machine (Netskope flags all Discord traffic) — this
    exists for the user's off-machine deployment and manual playtest.
    """
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    run_bot(settings)
    return 0
