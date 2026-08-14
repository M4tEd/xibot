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

import discord

from songbot.bot.embeds import (
    EMPTY_LEADERBOARD_MESSAGE,
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

    Args:
        engine: the game engine (owns all rules).
        challenge_id: the challenge this message's buttons act on — including
            after it is revealed, when gameplay maps to the closed notice.
        guild_id: the guild the challenge belongs to (leaderboard scope).
        settings: display configuration (snippet ladder, points).
        clock: injected time source (defaults to the UTC wall clock; the
            harness passes a fixed clock for determinism).
    """

    def __init__(
        self,
        engine: GameEngine,
        challenge_id: int,
        *,
        guild_id: str,
        settings: Settings,
        clock: Clock | None = None,
    ) -> None:
        super().__init__(timeout=None)
        self._engine = engine
        self._challenge_id = challenge_id
        self._guild_id = guild_id
        self._settings = settings
        self._clock: Clock = clock if clock is not None else utc_now

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
        user_id = str(interaction.user.id)
        try:
            result = self._engine.unlock_snippet(self._challenge_id, user_id)
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
        """Open the guess modal bound to this view's challenge."""
        await interaction.response.send_modal(
            GuessModal(self._engine, self._challenge_id, clock=self._clock)
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
        entries = self._engine.leaderboard(self._guild_id, self._clock())
        if not entries:
            await interaction.response.send_message(
                EMPTY_LEADERBOARD_MESSAGE, ephemeral=True
            )
            return
        await interaction.response.send_message(
            embed=leaderboard_embed(entries), ephemeral=True
        )
