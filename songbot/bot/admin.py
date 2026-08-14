"""Admin slash commands: /songbot-post, /songbot-skip, /songbot-reload.

All three are gated on the Manage-Guild permission and ack EPHEMERALLY. The
command bodies (`AdminCommands` methods) are plain coroutines that delegate
every game decision to the engine — the headless harness drives the SAME
bodies with its FakeInteraction (the ``--as-admin``/``--as-non-admin`` flag
maps onto the same `has_manage_guild` check the live commands use), while
the live client registers them as discord.py app_commands via
`register_admin_commands` (guild-scoped, with ``default_permissions`` so
Discord also hides them from non-admins client-side).

Drivability contract: the bodies touch only ``interaction.user`` (id and
``guild_permissions``) and ``interaction.response.send_message``. The daily
post itself goes through the injected `DailyPostSender` — the live client
sends it to the configured channel, the harness records a ``channel``-kind
payload; both build the message with the shared real builders
(`daily_challenge_embed`, `DailyChallengeView`, `snippet_attachment`).

Pinned decisions honored here: #4 (a same-day repeat post is idempotent and
never double-posts), #5 (skip is refused after a solve/reveal with zero
mutation; otherwise delete+recreate with no channel payload), #9 (no ack
ever names the song), #11 (empty catalog -> a clear ack, no challenge row).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

import discord
from discord import app_commands

from songbot.bot.embeds import (
    ADMIN_CATALOG_EMPTY_MESSAGE,
    ADMIN_POST_ALREADY_MESSAGE,
    ADMIN_POST_SUCCESS_MESSAGE,
    ADMIN_SKIP_SUCCESS_MESSAGE,
    PERMISSION_DENIED_MESSAGE,
    reload_ack_content,
    skip_refusal_content,
)
from songbot.bot.modals import Clock, utc_now
from songbot.catalog.refresh import RefreshResult
from songbot.config import Settings
from songbot.engine import (
    CatalogEmptyError,
    Challenge,
    GameEngine,
    SkipRefusedError,
    SkipRefusedReason,
)

__all__ = [
    "AdminCommands",
    "AdminResult",
    "DailyPostSender",
    "has_manage_guild",
    "register_admin_commands",
]

AdminOutcome = Literal[
    "posted", "already_posted", "catalog_empty", "denied", "skipped", "refused", "reloaded"
]
"""The machine-readable outcome of an admin command body."""

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
    form). ``reason`` is set for ``refused`` skips; ``refresh`` carries the
    per-source catalog summary for ``reloaded``.
    """

    outcome: AdminOutcome
    reason: SkipRefusedReason | None = None
    refresh: RefreshResult | None = None


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
        engine: the game engine (owns all rules).
        settings: validated configuration (guild/channel ids, timezone).
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

    async def post_now(self, interaction: discord.Interaction[Any]) -> AdminResult:
        """/songbot-post: ensure today's challenge exists and post it.

        Idempotent (pinned #4): an existing challenge yields the
        already-posted ack and no second post. An empty catalog (pinned #11)
        yields the empty-catalog ack and no challenge row.
        """
        if not has_manage_guild(interaction):
            return await self._deny(interaction)
        try:
            challenge = self._engine.ensure_today_challenge(
                self._settings.guild_id, self._settings.channel_id, self._clock()
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
        await self._post_sender(challenge)
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
        try:
            self._engine.skip_today_song(self._settings.guild_id, self._clock())
        except SkipRefusedError as exc:
            await interaction.response.send_message(
                skip_refusal_content(exc.reason), ephemeral=True
            )
            return AdminResult("refused", reason=exc.reason)
        await interaction.response.send_message(ADMIN_SKIP_SUCCESS_MESSAGE, ephemeral=True)
        return AdminResult("skipped")

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

    ``default_permissions(manage_guild=True)`` hides them from non-admins in
    the Discord client; the bodies re-check `has_manage_guild` so the harness
    permission flag and any client-side permission drift are honored too.
    """

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

    registered: tuple[app_commands.Command[Any, Any, Any], ...] = (
        post_command,
        skip_command,
        reload_command,
    )
    for command in registered:
        tree.add_command(command, guild=guild)
