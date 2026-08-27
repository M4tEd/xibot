"""Admin slash commands: /songbot-setup, /songbot-post, /songbot-skip,
/songbot-reload, /songbot-fixsong.

All three are gated on the Manage-Guild permission and ack EPHEMERALLY. The
command bodies (`AdminCommands` methods) are plain coroutines that delegate
every game decision to the engine — the headless harness drives the SAME
bodies with its FakeInteraction (the ``--as-admin``/``--as-non-admin`` flag
maps onto the same `has_manage_guild` check the live commands use), while
the live client registers them as discord.py app_commands via
`register_admin_commands` (guild-scoped, with ``default_permissions`` so
Discord also hides them from non-admins client-side).

Drivability contract: the bodies touch only ``interaction.user`` (id and
``guild_permissions``), ``interaction.guild_id`` (the guild being acted on —
multi-guild: never the env bootstrap), and
``interaction.response.send_message``. The daily post itself goes through
the injected `DailyPostSender` — the live client sends it to the guild's
configured channel, the harness records a ``channel``-kind payload; both
build the message with the shared real builders (`daily_challenge_embed`,
`DailyChallengeView`, `snippet_attachment`).

Pinned decisions honored here: #4 (a same-day repeat post is idempotent and
never double-posts), #5 (skip is refused after a solve/reveal with zero
mutation; otherwise delete+recreate with no channel payload), #9 (no ack
ever names the song), #11 (empty catalog -> a clear ack, no challenge row),
#16 (a failed post send rolls back the just-created challenge so a retry
reposts it; a pre-existing row is never rolled back).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

import discord
from discord import app_commands

from songbot.bot.embeds import (
    ADMIN_CATALOG_EMPTY_MESSAGE,
    ADMIN_NOT_CONFIGURED_MESSAGE,
    ADMIN_POST_ALREADY_MESSAGE,
    ADMIN_POST_FAILED_MESSAGE,
    ADMIN_POST_SUCCESS_MESSAGE,
    ADMIN_SKIP_SUCCESS_MESSAGE,
    PERMISSION_DENIED_MESSAGE,
    fixsong_ack_content,
    fixsong_refusal_content,
    reload_ack_content,
    setup_ack_content,
    skip_refusal_content,
)
from songbot.bot.modals import Clock, utc_now
from songbot.catalog.refresh import RefreshResult
from songbot.config import Settings
from songbot.engine import (
    CatalogEmptyError,
    Challenge,
    FixSongRefusedError,
    FixSongRefusedReason,
    GameEngine,
    SkipRefusedError,
    SkipRefusedReason,
    SongFix,
)

__all__ = [
    "AdminCommands",
    "AdminResult",
    "DailyPostSender",
    "has_manage_guild",
    "register_admin_commands",
]

AdminOutcome = Literal[
    "posted",
    "already_posted",
    "catalog_empty",
    "denied",
    "skipped",
    "refused",
    "reloaded",
    "configured",
    "not_configured",
    "fixed",
    "error",
]
"""The machine-readable outcome of an admin command body.

``error`` is the pinned-#16 delivery failure: the daily-post send raised for
a challenge the call just created, which was rolled back (row deleted,
snippet cache purged) so a retry reposts the identical challenge.
``configured``/``not_configured`` are the /songbot-setup outcomes (and the
``not_configured`` refusal of the other commands in a guild that never ran
setup).
"""

DailyPostSender = Callable[[Challenge], Awaitable[None]]
"""Sends (or records) the daily challenge post for a freshly created challenge.

The message itself is always built from the shared real builders; only the
transport differs between the live client (channel send) and the harness
(recorded ``channel``-kind payload).
"""


@dataclass(frozen=True)
class AdminResult:
    """The outcome of an admin command body.

    discord.py ignores callback return values; the harness uses the outcome
    to shape its JSON output (e.g. the pinned-#4 ``already_posted`` compact
    form). ``reason`` is set for ``refused`` skips/fixes; ``refresh`` carries
    the per-source catalog summary for ``reloaded``; ``fix`` carries the
    old/new metadata record for ``fixed``; ``error`` carries the send
    failure's message for ``error`` (pinned #16).
    """

    outcome: AdminOutcome
    reason: SkipRefusedReason | FixSongRefusedReason | None = None
    refresh: RefreshResult | None = None
    fix: SongFix | None = None
    error: str | None = None


def has_manage_guild(interaction: discord.Interaction[Any]) -> bool:
    """The Manage-Guild check both discord.py and the harness honor.

    Reads ``interaction.user.guild_permissions.manage_guild`` — present on
    guild Members and on the harness's FakeUser; anything else (DM-shaped
    users, missing attributes) is denied.
    """
    permissions = getattr(interaction.user, "guild_permissions", None)
    return bool(getattr(permissions, "manage_guild", False))


class AdminCommands:
    """The /songbot-* admin command bodies: permission-gated, ephemeral acks.

    Args:
        engine: the game engine (owns all rules, including the per-guild
            channel configuration the setup command writes).
        settings: validated configuration (post time/timezone for the acks).
        clock: injected time source (defaults to the UTC wall clock; the
            harness passes the ``--now`` clock for determinism).
        post_sender: transport for the daily challenge post (see
            `DailyPostSender`); only invoked when a challenge is newly created.
    """

    def __init__(
        self,
        engine: GameEngine,
        settings: Settings,
        *,
        post_sender: DailyPostSender,
        clock: Clock | None = None,
    ) -> None:
        self._engine = engine
        self._settings = settings
        self._post_sender = post_sender
        self._clock: Clock = clock if clock is not None else utc_now

    async def _deny(self, interaction: discord.Interaction[Any]) -> AdminResult:
        """The generic ephemeral permission denial (no leaked detail)."""
        await interaction.response.send_message(PERMISSION_DENIED_MESSAGE, ephemeral=True)
        return AdminResult("denied")

    @staticmethod
    def _guild_id_of(interaction: discord.Interaction[Any]) -> str | None:
        """The invoking guild's id as a string (None in DMs — never configured)."""
        raw = interaction.guild_id
        return str(raw) if raw is not None else None

    async def setup_channel(
        self,
        interaction: discord.Interaction[Any],
        channel_id: str,
        channel_mention: str,
    ) -> AdminResult:
        """/songbot-setup: choose (or change) the guild's daily-post channel.

        Multi-guild configuration entry point: upserts the invoking guild's
        ``guild_settings`` row. The live command passes the picked channel's
        id and mention; the harness passes plain strings. Existing challenges
        keep the channel they were posted to — only future posts move.
        """
        if not has_manage_guild(interaction):
            return await self._deny(interaction)
        guild_id = self._guild_id_of(interaction)
        if guild_id is None:  # pragma: no cover - the command is guild-only
            await interaction.response.send_message(
                ADMIN_NOT_CONFIGURED_MESSAGE, ephemeral=True
            )
            return AdminResult("not_configured")
        self._engine.set_guild_channel(
            guild_id, channel_id, set_by=str(interaction.user.id), now=self._clock()
        )
        await interaction.response.send_message(
            setup_ack_content(channel_mention, self._settings), ephemeral=True
        )
        return AdminResult("configured")

    async def post_now(self, interaction: discord.Interaction[Any]) -> AdminResult:
        """/songbot-post: ensure today's challenge exists and post it.

        Idempotent (pinned #4): an existing challenge yields the
        already-posted ack and no second post. An empty catalog (pinned #11)
        yields the empty-catalog ack and no challenge row. A guild that never
        ran /songbot-setup gets the not-configured ack and no challenge row.
        If the channel send fails for the challenge THIS call created
        (pinned #16), the challenge is rolled back (row deleted, snippet
        cache purged) so a retry reposts the identical challenge — a
        transient send failure can never suppress the day — and the admin
        gets an ephemeral error ack.
        """
        if not has_manage_guild(interaction):
            return await self._deny(interaction)
        guild_id = self._guild_id_of(interaction)
        guild = self._engine.guild_settings(guild_id) if guild_id is not None else None
        if guild is None:
            await interaction.response.send_message(
                ADMIN_NOT_CONFIGURED_MESSAGE, ephemeral=True
            )
            return AdminResult("not_configured")
        try:
            challenge = await asyncio.to_thread(
                self._engine.ensure_today_challenge,
                guild.guild_id,
                guild.channel_id,
                self._clock(),
            )
        except CatalogEmptyError:
            await interaction.response.send_message(
                ADMIN_CATALOG_EMPTY_MESSAGE, ephemeral=True
            )
            return AdminResult("catalog_empty")
        if not challenge.created:
            await interaction.response.send_message(
                ADMIN_POST_ALREADY_MESSAGE, ephemeral=True
            )
            return AdminResult("already_posted")
        try:
            await self._post_sender(challenge)
        except Exception as exc:
            # Pinned #16: roll back the just-created challenge (never a
            # pre-existing row) so the retry recreates it identically.
            self._engine.delete_challenge(challenge.id)
            await interaction.response.send_message(
                ADMIN_POST_FAILED_MESSAGE, ephemeral=True
            )
            return AdminResult("error", error=str(exc))
        await interaction.response.send_message(ADMIN_POST_SUCCESS_MESSAGE, ephemeral=True)
        return AdminResult("posted")

    async def skip_song(self, interaction: discord.Interaction[Any]) -> AdminResult:
        """/songbot-skip: replace today's song (pinned #5).

        Refused with zero mutation when there is no challenge today, when it
        is already revealed, or when a user has solved it. On success the
        engine deletes + recreates the challenge row (cascading per-user
        state), purges and regenerates the snippet cache, and re-draws the
        song and offset from the skip-count seed. Skip emits NO channel
        payload — only the ephemeral ack.
        """
        if not has_manage_guild(interaction):
            return await self._deny(interaction)
        guild_id = self._guild_id_of(interaction)
        if guild_id is None:  # pragma: no cover - the command is guild-only
            await interaction.response.send_message(
                ADMIN_NOT_CONFIGURED_MESSAGE, ephemeral=True
            )
            return AdminResult("not_configured")
        try:
            await asyncio.to_thread(self._engine.skip_today_song, guild_id, self._clock())
        except SkipRefusedError as exc:
            await interaction.response.send_message(
                skip_refusal_content(exc.reason), ephemeral=True
            )
            return AdminResult("refused", reason=exc.reason)
        await interaction.response.send_message(ADMIN_SKIP_SUCCESS_MESSAGE, ephemeral=True)
        return AdminResult("skipped")

    async def fix_song(
        self,
        interaction: discord.Interaction[Any],
        *,
        title: str,
        artist: str | None = None,
        date: str | None = None,
    ) -> AdminResult:
        """/songbot-fixsong: correct the title/artist of a challenge's song.

        Targets the guild's most recent challenge's song, or the
        ``date``-selected one. The correction applies to new guesses
        immediately and is re-applied after every catalog reload (the
        ``song_overrides`` table); already-recorded guesses keep their
        original results. The ephemeral ack shows old -> new metadata — a
        scoped exception to the pinned-#9 secrecy rule (ephemeral and
        admin-gated; the command is unusable blind). Refusals
        (``no_challenge``/``invalid_date``/``blank_title``) mutate nothing.
        """
        if not has_manage_guild(interaction):
            return await self._deny(interaction)
        guild_id = self._guild_id_of(interaction)
        if guild_id is None:  # pragma: no cover - the command is guild-only
            await interaction.response.send_message(
                ADMIN_NOT_CONFIGURED_MESSAGE, ephemeral=True
            )
            return AdminResult("not_configured")
        try:
            fix = self._engine.fix_song_metadata(
                guild_id,
                title=title,
                artist=artist,
                date_str=date,
                set_by=str(interaction.user.id),
                now=self._clock(),
            )
        except FixSongRefusedError as exc:
            await interaction.response.send_message(
                fixsong_refusal_content(exc.reason), ephemeral=True
            )
            return AdminResult("refused", reason=exc.reason)
        await interaction.response.send_message(fixsong_ack_content(fix), ephemeral=True)
        return AdminResult("fixed", fix=fix)

    async def reload_catalog(self, interaction: discord.Interaction[Any]) -> AdminResult:
        """/songbot-reload: upsert the catalog from its sources.

        The ephemeral ack reports the per-source summary (added / updated /
        removed / retained, or the source's error — per-source failure
        isolation, pinned #12).
        """
        if not has_manage_guild(interaction):
            return await self._deny(interaction)
        result = self._engine.refresh_catalog()
        await interaction.response.send_message(reload_ack_content(result), ephemeral=True)
        return AdminResult("reloaded", refresh=result)


def register_admin_commands(
    tree: app_commands.CommandTree[Any],
    commands: AdminCommands,
    *,
    guild: discord.abc.Snowflake,
) -> None:
    """Register the /songbot-* commands on a command tree (guild-scoped).

    Called once per joined guild (multi-guild: guild-scoped sync gives
    instant availability in each server without the global-propagation
    delay); each call builds fresh command objects for that guild.
    ``default_permissions(manage_guild=True)`` hides them from non-admins in
    the Discord client; the bodies re-check `has_manage_guild` so the harness
    permission flag and any client-side permission drift are honored too.
    """

    @app_commands.command(
        name="songbot-setup",
        description="Choose the channel SongBot posts the daily challenge in.",
    )
    @app_commands.default_permissions(manage_guild=True)
    async def setup_command(
        interaction: discord.Interaction[Any], channel: discord.TextChannel
    ) -> None:
        await commands.setup_channel(interaction, str(channel.id), channel.mention)

    @app_commands.command(name="songbot-post", description="Post today's challenge now.")
    @app_commands.default_permissions(manage_guild=True)
    async def post_command(interaction: discord.Interaction[Any]) -> None:
        await commands.post_now(interaction)

    @app_commands.command(name="songbot-skip", description="Replace today's song with a new one.")
    @app_commands.default_permissions(manage_guild=True)
    async def skip_command(interaction: discord.Interaction[Any]) -> None:
        await commands.skip_song(interaction)

    @app_commands.command(
        name="songbot-reload", description="Reload the song catalog from its sources."
    )
    @app_commands.default_permissions(manage_guild=True)
    async def reload_command(interaction: discord.Interaction[Any]) -> None:
        await commands.reload_catalog(interaction)

    @app_commands.command(
        name="songbot-fixsong",
        description="Correct the title/artist of a challenge's song (bad parses).",
    )
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(
        title="The correct song title.",
        artist="The correct artist (omit to keep the current one).",
        date="Which challenge's song to fix, YYYY-MM-DD (default: the latest).",
    )
    async def fixsong_command(
        interaction: discord.Interaction[Any],
        title: str,
        artist: str | None = None,
        date: str | None = None,
    ) -> None:
        await commands.fix_song(interaction, title=title, artist=artist, date=date)

    registered: tuple[app_commands.Command[Any, Any, Any], ...] = (
        setup_command,
        post_command,
        skip_command,
        reload_command,
        fixsong_command,
    )
    for command in registered:
        tree.add_command(command, guild=guild)
