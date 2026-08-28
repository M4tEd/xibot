"""Live Discord client wiring, gateway connect, view registration.

NEVER RUN ON THIS MACHINE during the mission: a network security agent
blocks/flags all Discord domains. This module is exercised at CONSTRUCTION
level only (pytest with faked transports — no ``client.run()``, no login, no
gateway).
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
   websocket connects) seeds the optional env bootstrap guild into
   ``guild_settings``, registers the persistent `DailyChallengeView`
   fallback, registers + syncs the /songbot-* admin commands for every
   configured guild (``on_ready``/``on_guild_join`` cover the rest), starts
   the aiohttp ``/health`` server, and launches the daily scheduler task
   (`_daily_post_loop`).

Multi-guild: the bot serves every guild with a ``guild_settings`` row
(written by /songbot-setup or the env bootstrap). The scheduler loop
iterates that table and uses the PURE `songbot.scheduler` functions per
guild: it posts when `is_post_due` says so (restart-safe: the challenges
table is the record of posts) and sleeps until `next_post_datetime`, capped
so clock shifts and out-of-band admin posts are noticed promptly. Each due
tick first reveals the previous challenge (song + winners) and then posts
today's — the exact flow the harness ``advance-day`` scenario validates
headlessly. One guild's failure never blocks another's (per-guild isolation
with a 60s retry cadence for the failed guild only).
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

from songbot.bot.admin import (
    AdminCommands,
    DailyPostSender,
    register_admin_commands,
)
from songbot.bot.embeds import (
    daily_challenge_embed,
    ping_mention_content,
    reveal_embed,
    snippet_attachment,
)
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

CommandSyncer = Callable[
    [app_commands.CommandTree[Any], discord.abc.Snowflake], Awaitable[None]
]
"""Syncs the command tree for ONE guild (the live default syncs guild-scoped,
which is instant in that guild — no global-propagation delay)."""

RevealSender = Callable[[Reveal], Awaitable[None]]
"""Transport for one guild's previous-challenge reveal (the live default sends
it to the channel that challenge was posted in — ``reveal.channel_id``; tests
and the harness record instead)."""


class _MessageableChannel(Protocol):
    """The slice of a Discord channel the post/reveal transports need."""

    async def send(
        self,
        content: str | None = ...,
        *,
        embed: discord.Embed | None = ...,
        view: discord.ui.View | None = ...,
        file: discord.File | None = ...,
        allowed_mentions: discord.AllowedMentions | None = ...,
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
    app the harness ``serve`` scenario exposes with mode ``"harness"``. The
    reported guild is the env bootstrap guild, or ``"multi"`` when the bot is
    configured purely per-server via /songbot-setup.
    """
    app = build_health_app(mode="live", guild_id=settings.guild_id or "multi")
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
    surface. Connects with guilds+reactions intents (neither is privileged:
    interactions need no privileged intents, and the raw reaction events
    behind the /songbot-pingrole opt-in need ``reactions``); owns no game
    rules (every decision is the engine's).

    Args:
        settings: validated configuration. When the ``guild_id``/
            ``channel_id`` bootstrap pair is set, both must be numeric
            snowflakes (checked eagerly — a clear `ConfigError` beats a late
            failure); when absent, every guild configures itself via
            /songbot-setup.
        db: the (migrated) database. The client TAKES OWNERSHIP and closes it
            in `close`.
        engine: the game engine (owns all rules, including the
            ``guild_settings`` table the scheduler iterates).
        clock: time source for gameplay and the scheduler (default: the UTC
            wall clock — live mode posts on real time).
        post_sender: daily-post transport (default: send the real embed, a
            fresh `DailyChallengeView`, and the level-0
            ``songbot-snippet.mp3`` attachment to the challenge's channel).
        reveal_sender: reveal transport (default: send the reveal embed to
            the channel the revealed challenge was posted in).
        health_starter: /health server starter (default: aiohttp on
            127.0.0.1:``health_port``).
        command_syncer: per-guild command-tree sync (default: guild-scoped
            ``tree.sync`` — instant availability in each joined guild).
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
        # Intents: guilds (interactions/posts) + reactions (the raw reaction
        # events behind the /songbot-pingrole opt-in). Neither is privileged.
        super().__init__(intents=discord.Intents(guilds=True, reactions=True))
        # Plain discord.Client has no command tree; constructing one here also
        # wires it into the connection state for interaction dispatch.
        self.tree: app_commands.CommandTree[SongBotClient] = app_commands.CommandTree(self)
        self._settings = settings
        self._db = db
        self._engine = engine
        self._clock: Clock = clock if clock is not None else utc_now
        if settings.guild_id is not None:
            # The env bootstrap pair (validated as set-together at load).
            _snowflake(settings.guild_id, "DISCORD_GUILD_ID")
            _snowflake(settings.channel_id or "", "DISCORD_CHANNEL_ID")
        # Guilds whose command tree is already registered + synced this
        # session (setup_hook seeds the configured ones; on_ready and
        # on_guild_join cover every other joined guild exactly once).
        self._command_guilds: set[int] = set()
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
            engine,
            settings,
            post_sender=self._post_sender,
            announcement_poster=self._post_announcement,
            clock=self._clock,
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

        Called once by discord.py's ``login``. Seeds the optional env
        bootstrap guild into ``guild_settings``, registers the persistent
        fallback view, registers + syncs the admin commands for every
        CONFIGURED guild (the guilds cache is still empty here — every other
        joined guild is synced from ``on_ready``/``on_guild_join``), starts
        the health server, and launches the daily scheduler task. The
        scheduler's ``wait_until_ready`` lives INSIDE the task — awaiting it
        here would deadlock (setup_hook runs before the websocket connects).
        """
        self._seed_bootstrap_guild()
        self._register_persistent_view()
        for guild_settings in self._engine.all_guild_settings():
            try:
                guild_id = int(guild_settings.guild_id)
            except ValueError:
                logger.warning(
                    "skipping command sync for non-numeric guild id %r",
                    guild_settings.guild_id,
                )
                continue
            await self._ensure_guild_commands(discord.Object(id=guild_id))
        self._health_handle = await self._health_starter(self._settings)
        self._scheduler_task = asyncio.create_task(
            self._daily_post_loop(), name=SCHEDULER_TASK_NAME
        )
        logger.info("setup complete: views + admin commands + health + scheduler")

    def _seed_bootstrap_guild(self) -> None:
        """Upsert the optional DISCORD_GUILD_ID/DISCORD_CHANNEL_ID pair.

        Keeps env-managed single-server deployments working with zero
        commands: the pair is authoritative for that guild on every restart
        (a /songbot-setup change to the SAME guild is overwritten on reboot —
        edit the env instead). All other guilds are configured only via
        /songbot-setup.
        """
        settings = self._settings
        if settings.guild_id is None or settings.channel_id is None:
            return
        self._engine.set_guild_channel(
            settings.guild_id, settings.channel_id, set_by="env", now=self._clock()
        )

    async def on_ready(self) -> None:
        """Sync admin commands into every joined guild (post-connect).

        setup_hook runs before the guilds cache exists, so it can only cover
        configured guilds; READY is the first point every joined guild —
        including ones that never ran /songbot-setup — is visible, so they
        all get /songbot-setup itself. The per-guild guard makes reconnects
        (repeat READY events) no-ops.
        """
        for guild in self.guilds:
            await self._ensure_guild_commands(guild)
        logger.info("ready: %d guild(s) command-synced", len(self.guilds))

    async def on_guild_join(self, guild: discord.Guild) -> None:
        """Give a newly added guild the admin commands immediately."""
        await self._ensure_guild_commands(guild)
        logger.info("joined guild %s; admin commands synced", guild.id)

    async def on_guild_remove(self, guild: discord.Guild) -> None:
        """Stop scheduling a guild the bot was removed from.

        Drops the ``guild_settings`` row so the scheduler stops attempting
        posts that would 403/404 every tick. Game history (challenges,
        scores) is kept — re-adding the bot and re-running /songbot-setup
        resumes the guild's game.
        """
        self._command_guilds.discard(guild.id)
        self._engine.remove_guild_settings(str(guild.id))
        logger.info("removed from guild %s; configuration dropped", guild.id)

    # -- reaction-role opt-in (/songbot-pingrole) ------------------------------

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent) -> None:
        """Grant the ping role when a user reacts with the configured emoji.

        Raw events (not the cached-message variants) so reactions on an
        announcement posted before a restart keep working.
        """
        await self._sync_ping_role(payload, grant=True)

    async def on_raw_reaction_remove(
        self, payload: discord.RawReactionActionEvent
    ) -> None:
        """Revoke the ping role when the opt-in reaction is removed."""
        await self._sync_ping_role(payload, grant=False)

    async def _sync_ping_role(
        self, payload: discord.RawReactionActionEvent, *, grant: bool
    ) -> None:
        """Grant/revoke the ping role for one raw reaction event.

        Only reactions on a configured announcement message with the
        configured emoji act; everything else (other messages, other emoji,
        the bot's own seed reaction, other bots) is ignored. Role/member
        lookups degrade gracefully: a deleted role or a departed member is a
        no-op, and a missing Manage-Roles permission or role-hierarchy
        problem is logged, never raised (event handlers must not crash the
        dispatch loop).
        """
        config = self._engine.ping_role_for_message(str(payload.message_id))
        if config is None or str(payload.emoji) != config.emoji:
            return
        if payload.guild_id is None:  # pragma: no cover - announcements are guild posts
            return
        if self.user is not None and payload.user_id == self.user.id:
            return  # the bot's own seed reaction on the announcement
        guild = self.get_guild(payload.guild_id)
        if guild is None:
            return  # guild not in cache (bot removed; config cascades away anyway)
        role = guild.get_role(int(config.role_id))
        if role is None:
            logger.warning(
                "ping role %s not found in guild %s — was it deleted?",
                config.role_id,
                config.guild_id,
            )
            return
        # payload.member is only populated on ADD when the member happens to
        # be cached (no members intent here), so fetch over REST as needed;
        # REMOVE events never carry a member.
        member = payload.member
        if member is None:
            try:
                member = await guild.fetch_member(payload.user_id)
            except discord.NotFound:
                return  # the reacting user left the guild
        if member.bot:
            return
        action = "opt-in" if grant else "opt-out"
        try:
            if grant:
                await member.add_roles(role, reason="SongBot daily-song ping opt-in")
            else:
                await member.remove_roles(role, reason="SongBot daily-song ping opt-out")
        except discord.Forbidden:
            logger.warning(
                "ping %s failed in guild %s: missing Manage Roles or the role "
                "outranks the bot",
                action,
                config.guild_id,
            )
            return
        except discord.HTTPException:
            logger.exception("ping %s failed in guild %s", action, config.guild_id)
            return
        logger.info(
            "ping %s: user %s %s role %s in guild %s",
            action,
            payload.user_id,
            "granted" if grant else "revoked",
            config.role_id,
            config.guild_id,
        )

    async def _ensure_guild_commands(self, guild: discord.abc.Snowflake) -> None:
        """Register + sync the /songbot-* commands for one guild, once."""
        if guild.id in self._command_guilds:
            return
        register_admin_commands(self.tree, self._admin_commands, guild=guild)
        await self._command_syncer(self.tree, guild)
        self._command_guilds.add(guild.id)

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

    def _build_view(self, challenge: Challenge) -> DailyChallengeView:
        """A fresh message-bound view for one challenge's post."""
        return DailyChallengeView(
            self._engine,
            challenge.id,
            guild_id=challenge.guild_id,
            settings=self._settings,
            clock=self._clock,
        )

    def _register_persistent_view(self) -> None:
        """Register the persistent DailyChallengeView for restart-resilient buttons.

        Discord dispatches component interactions by custom_id, and the pinned
        custom_ids carry no challenge id, so the persistent fallback resolves
        its target LAZILY at click time: the clicking guild's latest challenge
        (see ``DailyChallengeView``). Lazy resolution keeps one global
        registration correct across MULTIPLE guilds — a statically bound
        fallback could act on the wrong guild's game — and makes registration
        unconditional (safe even with zero challenges anywhere: a stray click
        gets the graceful no-active-challenge notice). Messages sent this
        session dispatch through their own message-bound views first; this
        registration is purely the restart-recovery path.
        """
        self.add_view(
            DailyChallengeView(
                self._engine,
                None,
                guild_id=None,
                settings=self._settings,
                clock=self._clock,
            )
        )

    async def _sync_commands(
        self, tree: app_commands.CommandTree[Any], guild: discord.abc.Snowflake
    ) -> None:
        """The live command sync: guild-scoped (instant in that guild)."""
        synced = await tree.sync(guild=guild)
        logger.info("synced %d admin command(s) to guild %s", len(synced), guild.id)

    # -- daily scheduler ----------------------------------------------------

    async def _daily_post_loop(self) -> None:
        """Reveal + post each day at the configured time, until the client closes."""
        await self.wait_until_ready()
        while not self.is_closed():
            delay = await self._scheduler_tick(self._clock())
            await asyncio.sleep(delay)

    async def _scheduler_tick(self, now: datetime) -> float:
        """One scheduler iteration: post where due; return seconds until next check."""
        try:
            settled = await self._post_if_due(now)
        except Exception:
            logger.exception(
                "daily post tick failed; retrying in %.0fs", RETRY_DELAY_SEC
            )
            return RETRY_DELAY_SEC
        # A guild that failed (transient snippet/send error) retries on the
        # 60s backoff; already-posted guilds are gated by `is_post_due` and
        # simply no-op on that retry.
        return self._seconds_until_next_check(now) if settled else RETRY_DELAY_SEC

    async def _post_if_due(self, now: datetime) -> bool:
        """Run the daily flow for every configured guild; True iff all settled.

        Multi-guild isolation: one guild's failure is logged and the loop
        moves on — a broken guild (kicked bot, deleted channel, transient
        snippet error) never suppresses another guild's post. The bool drives
        the tick's retry cadence for the failed guild only.
        """
        all_settled = True
        for guild in self._engine.all_guild_settings():
            try:
                await self._post_if_due_for_guild(
                    guild.guild_id, guild.channel_id, now
                )
            except Exception:
                all_settled = False
                logger.exception(
                    "daily post failed for guild %s; other guilds unaffected",
                    guild.guild_id,
                )
        return all_settled

    async def _post_if_due_for_guild(
        self, guild_id: str, channel_id: str, now: datetime
    ) -> None:
        """Reveal the guild's previous challenge and post today's, when due.

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
            self._last_post_date(guild_id),
            now,
            self._settings.tz,
            self._settings.daily_post_time,
        ):
            return
        reveal = self._engine.peek_reveal(guild_id, now)
        if reveal is not None:
            await self._reveal_sender(reveal)
            # Pinned #17: mark revealed ONLY after the reveal send succeeded.
            self._engine.mark_revealed(guild_id, now)
        try:
            challenge = self._engine.ensure_today_challenge(guild_id, channel_id, now)
        except CatalogEmptyError:
            logger.warning(
                "daily post skipped for guild %s: catalog is empty — add songs"
                " and /songbot-reload",
                guild_id,
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

    def _seconds_until_next_check(self, now: datetime) -> float:
        """Sleep until the next scheduled post time, clamped to [MIN, MAX]."""
        next_at = next_post_datetime(
            now, self._settings.daily_post_time, self._settings.tz
        )
        delay = (next_at - now).total_seconds()
        return min(max(delay, MIN_SLEEP_SEC), MAX_SLEEP_SEC)

    # -- live transports (the ONLY methods that would touch Discord HTTP) -----

    async def _challenge_channel(self, channel_id: str) -> _MessageableChannel:
        """Resolve a guild's configured channel (gateway cache, REST fallback)."""
        snowflake = int(channel_id)
        channel = self.get_channel(snowflake) or await self.fetch_channel(snowflake)
        return cast("_MessageableChannel", channel)

    async def _send_daily_post(self, challenge: Challenge) -> None:
        """Post the daily challenge: real embed + real view + level-0 snippet.

        When the guild configured /songbot-pingrole, the message content
        mentions the opt-in role (pinging its members) — role mentions are
        explicitly allowed so a client-level allowed_mentions override can't
        silently drop the ping.
        """
        channel = await self._challenge_channel(challenge.channel_id)
        config = self._engine.ping_role_settings(challenge.guild_id)
        await channel.send(
            content=ping_mention_content(config.role_id) if config is not None else None,
            embed=daily_challenge_embed(challenge, self._settings),
            view=self._build_view(challenge),
            file=snippet_attachment(challenge.snippet_paths[0]),
            allowed_mentions=(
                discord.AllowedMentions(roles=True) if config is not None else None
            ),
        )

    async def _post_announcement(self, channel_id: str, content: str, emoji: str) -> str:
        """The live /songbot-pingrole transport: post + seed-reaction; return id.

        The bot adds its own ``emoji`` reaction so users can opt in with one
        tap. A seed-reaction failure (e.g. an invalid emoji string) deletes
        the just-posted announcement and raises — the caller persists
        nothing, so a retry starts clean.
        """
        channel = await self._challenge_channel(channel_id)
        message = cast("discord.Message", await channel.send(content=content))
        try:
            await message.add_reaction(emoji)
        except Exception:
            with suppress(discord.HTTPException):
                await message.delete()
            raise
        return str(message.id)

    async def _send_reveal(self, reveal: Reveal) -> None:
        """Post a previous challenge's reveal (song + winners) to its channel."""
        channel = await self._challenge_channel(reveal.channel_id)
        await channel.send(embed=reveal_embed(reveal))

    # -- db reads (same read-only pattern the harness uses) -------------------

    def _last_post_date(self, guild_id: str) -> date | None:
        """The local date of the guild's most recent challenge (the post record)."""
        row = self._db.query_one(
            "SELECT MAX(date) AS latest FROM challenges WHERE guild_id = ?",
            (guild_id,),
        )
        latest = row["latest"] if row is not None else None
        return date.fromisoformat(str(latest)) if latest is not None else None


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

    NEVER run on this machine (a network security agent flags all Discord
    traffic) — this exists for the user's off-machine deployment and manual
    playtest.
    """
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    run_bot(settings)
    return 0
