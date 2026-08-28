"""DailyChallengeView: Hear more / Guess / Leaderboard buttons.

A persistent (``timeout=None``) view with the three pinned custom_ids. The
view is a thin adapter: every game decision is delegated to the engine, and
engine results are formatted by the pure builders in `embeds.py`.

Drivability contract (the headless harness and the unit-test fakes drive the
REAL callbacks): callbacks only touch ``interaction.user``,
``interaction.response.send_message``/``send_modal``, and (indirectly, via
the modal) ``interaction.channel``. Buttons are pressed by custom_id:
``await button.callback(interaction)``.
"""

from __future__ import annotations

import logging
from typing import cast

import discord

from songbot.bot.embeds import (
    EMPTY_LEADERBOARD_MESSAGE,
    NO_ACTIVE_CHALLENGE_MESSAGE,
    hear_more_content,
    hear_more_refusal_content,
    leaderboard_embed,
    snippet_attachment,
)
from songbot.bot.modals import Clock, GuessModal, utc_now
from songbot.config import Settings
from songbot.engine import GameEngine, UnlockRefusedError

__all__ = ["DailyChallengeView"]

logger = logging.getLogger(__name__)


class DailyChallengeView(discord.ui.View):
    """The three-button view attached to every daily challenge post.

    Two binding modes:

    - **Message-bound** (``challenge_id`` AND ``guild_id`` set): the view
      attached to a post at send time; its buttons act on exactly that
      challenge — including after it is revealed, when gameplay maps to the
      closed notice.
    - **Persistent fallback** (both None): the restart-recovery view
      registered via ``client.add_view``. Discord dispatches component
      interactions by custom_id, and the pinned custom_ids carry no challenge
      id, so the fallback resolves the click LAZILY: the guild from
      ``interaction.guild_id``, the challenge from that guild's LATEST
      challenge at click time. The game only ever has one live challenge per
      guild, so a click on any recent post acts on the current game (older,
      revealed posts answer with the closed-challenge notice). Lazy
      resolution is what makes the fallback correct in MULTIPLE guilds: one
      global registration can never act on the wrong guild's challenge.

    Args:
        engine: the game engine (owns all rules).
        challenge_id: the challenge this message's buttons act on, or None
            for the persistent fallback mode.
        guild_id: the guild the challenge belongs to (leaderboard scope), or
            None for the persistent fallback mode.
        settings: display configuration (snippet ladder, points).
        clock: injected time source (defaults to the UTC wall clock; the
            harness passes a fixed clock for determinism).
    """

    def __init__(
        self,
        engine: GameEngine,
        challenge_id: int | None,
        *,
        guild_id: str | None,
        settings: Settings,
        clock: Clock | None = None,
    ) -> None:
        super().__init__(timeout=None)
        if (challenge_id is None) != (guild_id is None):
            raise ValueError(
                "challenge_id and guild_id must be set together (message-bound) "
                "or both None (persistent fallback)"
            )
        self._engine = engine
        self._challenge_id = challenge_id
        self._guild_id = guild_id
        self._settings = settings
        self._clock: Clock = clock if clock is not None else utc_now

    def _resolve_target(
        self, interaction: discord.Interaction[discord.Client]
    ) -> tuple[int, str] | None:
        """The (challenge_id, guild_id) a click acts on, or None.

        Message-bound views return their fixed binding. The persistent
        fallback resolves the clicking guild's latest challenge; None means
        the click came from a DM (impossible for a guild post) or the guild
        has no challenges at all — the caller answers with the graceful
        no-active-challenge notice.
        """
        if self._challenge_id is not None:
            # Both are set together (enforced in __init__).
            return self._challenge_id, cast(str, self._guild_id)
        raw_guild_id = interaction.guild_id
        if raw_guild_id is None:
            return None
        guild_id = str(raw_guild_id)
        challenge_id = self._engine.latest_challenge_id(guild_id)
        if challenge_id is None:
            return None
        return challenge_id, guild_id

    @discord.ui.button(
        label="Hear more",
        style=discord.ButtonStyle.primary,
        emoji="🎧",
        custom_id="songbot:hear_more",
    )
    async def hear_more(
        self,
        interaction: discord.Interaction[discord.Client],
        button: discord.ui.Button[DailyChallengeView],
    ) -> None:
        """Unlock the user's next-longer snippet (ephemeral + attachment).

        Refusals map to a single ephemeral notice with no attachment: the
        revealed-challenge lockout (VAL-GUESS-019), already-solved, and
        max-level (VAL-HEAR-008/009).
        """
        target = self._resolve_target(interaction)
        if target is None:
            await interaction.response.send_message(
                NO_ACTIVE_CHALLENGE_MESSAGE, ephemeral=True
            )
            return
        challenge_id, _ = target
        user_id = str(interaction.user.id)
        try:
            result = self._engine.unlock_snippet(challenge_id, user_id)
        except UnlockRefusedError as exc:
            await interaction.response.send_message(
                hear_more_refusal_content(exc.reason, self._settings), ephemeral=True
            )
            return
        await interaction.response.send_message(
            hear_more_content(result, self._settings),
            file=snippet_attachment(result.path),
            ephemeral=True,
        )

    @discord.ui.button(
        label="Guess",
        style=discord.ButtonStyle.success,
        emoji="💡",
        custom_id="songbot:guess",
    )
    async def guess(
        self,
        interaction: discord.Interaction[discord.Client],
        button: discord.ui.Button[DailyChallengeView],
    ) -> None:
        """Open the guess modal bound to the click's resolved challenge."""
        target = self._resolve_target(interaction)
        if target is None:
            await interaction.response.send_message(
                NO_ACTIVE_CHALLENGE_MESSAGE, ephemeral=True
            )
            return
        challenge_id, _ = target
        await interaction.response.send_modal(
            GuessModal(self._engine, challenge_id, settings=self._settings, clock=self._clock)
        )

    @discord.ui.button(
        label="Leaderboard",
        style=discord.ButtonStyle.secondary,
        emoji="🏆",
        custom_id="songbot:leaderboard",
    )
    async def leaderboard(
        self,
        interaction: discord.Interaction[discord.Client],
        button: discord.ui.Button[DailyChallengeView],
    ) -> None:
        """Show the guild's top-10 leaderboard (ephemeral, recipient-only)."""
        target = self._resolve_target(interaction)
        if target is None:
            # The leaderboard needs only a guild, but a fallback-mode click
            # with no resolvable guild gets the same graceful notice.
            await interaction.response.send_message(
                NO_ACTIVE_CHALLENGE_MESSAGE, ephemeral=True
            )
            return
        _, guild_id = target
        entries = self._engine.leaderboard(guild_id, self._clock())
        if not entries:
            await interaction.response.send_message(
                EMPTY_LEADERBOARD_MESSAGE, ephemeral=True
            )
            return
        await interaction.response.send_message(
            embed=leaderboard_embed(entries), ephemeral=True
        )
