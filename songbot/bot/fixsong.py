"""FixSongView + FixSongModal: the interactive /songbot-fixsong edit flow.

The command body (``AdminCommands.fix_song``) shows the song's current
metadata ephemerally with a single "Edit metadata" button; the button opens
this modal pre-filled with the current title/artist, and submitting it
applies the correction through ``engine.fix_song_metadata``. Nothing mutates
until the modal submit — dismissing the modal is a no-op.

Drivability contract (the headless harness and the unit-test fakes drive the
REAL view/modal): the callbacks touch only ``interaction.user`` (id and
``guild_permissions``) and ``interaction.response.send_message``/
``send_modal``. The button is pressed by custom_id (``await
button.callback(interaction)``); modal text is injected as
``modal.title_input._value = text`` / ``modal.artist_input._value = text``
before ``await modal.on_submit(interaction)`` — mirroring how discord.py populates
component values from the submit payload, where untouched fields arrive with
their pre-filled default (discord.py 2.7.1's ``TextInput.value`` has no
setter and never falls back to ``default``).
"""

from __future__ import annotations

from typing import Any

import discord

from songbot.bot.embeds import (
    PERMISSION_DENIED_MESSAGE,
    fixsong_ack_content,
    fixsong_refusal_content,
)
from songbot.bot.modals import Clock, utc_now
from songbot.bot.permissions import has_manage_guild
from songbot.engine import (
    FixSongRefusedError,
    FixSongRefusedReason,
    GameEngine,
    SongFix,
    SongFixTarget,
)

__all__ = ["FixSongModal", "FixSongView"]

_TEXT_MAX_LENGTH = 1000
"""Cap on the modal inputs; pre-filled defaults are truncated to fit."""


class FixSongModal(discord.ui.Modal):
    """The edit form: Title (required) and Artist (blank clears), pre-filled.

    The inputs are built per-instance in ``__init__`` (not as class
    attributes) because their ``default`` pre-fill depends on the displayed
    song. ``result`` carries the applied `SongFix` for the harness/tests
    (None when refused or never submitted); ``refused_reason`` carries the
    machine-readable refusal cause.
    """

    def __init__(
        self,
        engine: GameEngine,
        target: SongFixTarget,
        *,
        clock: Clock | None = None,
    ) -> None:
        super().__init__(title="Fix song metadata")
        self._engine = engine
        self._target = target
        self._clock: Clock = clock if clock is not None else utc_now
        self.result: SongFix | None = None
        self.refused_reason: FixSongRefusedReason | None = None
        # ``title_input``/``artist_input``: plain ``title`` would shadow
        # ``Modal.title`` (the form's own heading).
        self.title_input: discord.ui.TextInput[FixSongModal] = discord.ui.TextInput(
            label="Title",
            default=target.title[:_TEXT_MAX_LENGTH],
            required=True,
            max_length=_TEXT_MAX_LENGTH,
            custom_id="songbot:fixsong_title",
        )
        self.artist_input: discord.ui.TextInput[FixSongModal] = discord.ui.TextInput(
            label="Artist (leave blank to clear)",
            default=target.artist[:_TEXT_MAX_LENGTH] if target.artist else None,
            required=False,
            max_length=_TEXT_MAX_LENGTH,
            custom_id="songbot:fixsong_artist",
        )
        self.add_item(self.title_input)
        self.add_item(self.artist_input)

    async def on_submit(self, interaction: discord.Interaction[Any], /) -> None:
        """Apply the edit via the engine; ack ephemerally with old -> new.

        Re-checks Manage-Guild: the ephemeral flow restricts the button to
        the invoking admin, but permissions could have been revoked since.
        The raw artist value is always passed — never None — so the
        pre-filled form is WYSIWYG and a blanked field clears the artist via
        the engine's strip-to-None rule. Refusals (``blank_title``;
        ``no_challenge`` when the challenge vanished between show and
        submit) mutate nothing.
        """
        if not has_manage_guild(interaction):
            await interaction.response.send_message(PERMISSION_DENIED_MESSAGE, ephemeral=True)
            return
        target = self._target
        try:
            fix = self._engine.fix_song_metadata(
                target.guild_id,
                title=self.title_input.value,
                artist=self.artist_input.value,
                date_str=target.challenge_date,
                set_by=str(interaction.user.id),
                now=self._clock(),
            )
        except FixSongRefusedError as exc:
            self.refused_reason = exc.reason
            await interaction.response.send_message(
                fixsong_refusal_content(exc.reason), ephemeral=True
            )
            return
        self.result = fix
        await interaction.response.send_message(fixsong_ack_content(fix), ephemeral=True)


class FixSongView(discord.ui.View):
    """The ephemeral "Edit metadata" button attached to the show-first ack.

    Not persistent: the view lives only for the invoking admin's edit
    session (10-minute timeout); the modal submit is a fresh interaction
    that no longer needs the view.
    """

    def __init__(
        self,
        engine: GameEngine,
        target: SongFixTarget,
        *,
        clock: Clock | None = None,
    ) -> None:
        super().__init__(timeout=600)
        self._engine = engine
        self._target = target
        self._clock: Clock = clock if clock is not None else utc_now

    @discord.ui.button(
        label="Edit metadata",
        style=discord.ButtonStyle.primary,
        emoji="🛠️",
        custom_id="songbot:fixsong_edit",
    )
    async def edit(
        self,
        interaction: discord.Interaction[Any],
        button: discord.ui.Button[FixSongView],
    ) -> None:
        """Open the pre-filled edit modal (permission re-checked)."""
        if not has_manage_guild(interaction):
            await interaction.response.send_message(PERMISSION_DENIED_MESSAGE, ephemeral=True)
            return
        await interaction.response.send_modal(
            FixSongModal(self._engine, self._target, clock=self._clock)
        )
