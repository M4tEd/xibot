"""GuessModal (single text input).

The modal is a thin adapter: it reads the submitted text, delegates ALL game
decisions to ``engine.submit_guess``, and formats the result — an ephemeral
feedback message, plus the public first-solve announcement via
``interaction.channel.send`` when the engine flags ``announce``.

Drivability contract (the headless harness and the unit-test fakes drive the
REAL modal): ``on_submit`` only touches ``interaction.user``,
``interaction.channel``, and ``interaction.response.send_message``. The
harness injects the submitted text by setting the text input's value (as
discord.py itself does from component data) before calling ``on_submit``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol, cast

import discord

from songbot.bot.embeds import announcement_content, guess_feedback_content
from songbot.config import Settings
from songbot.engine import GameEngine

__all__ = ["Clock", "GuessModal", "utc_now"]

logger = logging.getLogger(__name__)

Clock = Callable[[], datetime]
"""Time source injected into views/modals so the engine never reads a clock."""


def utc_now() -> datetime:
    """The default (live-mode) clock: the current UTC time."""
    return datetime.now(UTC)


class _AnnouncementChannel(Protocol):
    """The slice of ``interaction.channel`` the announcement path needs.

    ``discord.Interaction.channel`` is a broad union (including channel types
    without ``send``); the bot only ever runs in text-capable guild channels,
    so the cast to this protocol is safe and keeps mypy strict happy.
    """

    async def send(self, content: str, /) -> object: ...


class GuessModal(discord.ui.Modal):
    """The Guess button's modal: exactly one required text input.

    The harness drives the real modal: ``modal.guess._value = text`` (mirroring
    how discord.py populates the component value), then ``await
    modal.on_submit(interaction)``.
    """

    guess: discord.ui.TextInput[GuessModal] = discord.ui.TextInput(
        label="Your guess",
        placeholder="Artist or title...",
        required=True,
        max_length=200,
        custom_id="songbot:guess_text",
    )

    def __init__(
        self,
        engine: GameEngine,
        challenge_id: int,
        *,
        settings: Settings,
        clock: Clock | None = None,
    ) -> None:
        super().__init__(title="Guess the song")
        self._engine = engine
        self._challenge_id = challenge_id
        self._settings = settings
        self._clock: Clock = clock if clock is not None else utc_now

    async def on_submit(self, interaction: discord.Interaction[discord.Client], /) -> None:
        """Submit the guess to the engine; reply ephemeral; announce a solve.

        A revealed challenge comes back as ``challenge_closed`` and maps to
        exactly one ephemeral "This challenge has closed." notice — no
        announcement, no error leak (VAL-GUESS-019 adapter half).
        """
        user_id = str(interaction.user.id)
        result = self._engine.submit_guess(
            self._challenge_id, user_id, self.guess.value, self._clock()
        )
        await interaction.response.send_message(
            guess_feedback_content(result), ephemeral=True
        )
        if result.announce:
            await self._announce(
                interaction,
                announcement_content(user_id, result, self._settings),
            )

    async def _announce(
        self, interaction: discord.Interaction[discord.Client], content: str
    ) -> None:
        """Post the public first-solve announcement to the challenge channel."""
        channel = interaction.channel
        if channel is None:  # pragma: no cover - guild interactions always have one
            logger.warning("interaction has no channel; solve announcement dropped")
            return
        await cast("_AnnouncementChannel", channel).send(content)
