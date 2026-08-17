"""Lightweight Discord interaction test doubles for driving views/modals.

These fakes mirror the documented harness ``FakeInteraction`` surface — the
ONLY attributes the adapter callbacks may touch:

- ``interaction.user`` (``id``/``name``)
- ``interaction.channel`` (``send``)
- ``interaction.response`` (``send_message``/``send_modal``)
- ``interaction.followup``
- ``interaction.message``
- ``interaction.guild_id`` (the guild being acted on — multi-guild admin/view
  resolution)

Every outgoing payload is recorded as a `RecordedPayload` so tests can assert
on kind (``ephemeral``/``channel``/``announcement``/``modal``), content,
embed, attachment metadata, and recipient — the same taxonomy the headless
harness records (pinned decision #3). Attachments are always expected to be
named ``songbot-snippet.mp3`` (pinned decision #9).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import discord


@dataclass
class RecordedAttachment:
    """Attachment metadata captured from a ``discord.File``."""

    filename: str
    path: str
    size: int


@dataclass
class RecordedPayload:
    """One outgoing message/modal captured by the fakes."""

    kind: str  # "ephemeral" | "channel" | "announcement" | "modal"
    content: str | None = None
    embed: discord.Embed | None = None
    attachment: RecordedAttachment | None = None
    modal: discord.ui.Modal[Any] | None = None
    recipient: str | None = None


@dataclass
class FakePermissions:
    """Stand-in for ``discord.Permissions`` (only ``manage_guild`` is used).

    The admin command bodies read ``interaction.user.guild_permissions
    .manage_guild`` — the same attribute a guild ``Member`` exposes — so the
    same check serves discord.py and the fakes (VAL-ADMIN-009).
    """

    manage_guild: bool = False


@dataclass
class FakeUser:
    """Stand-in for ``discord.User``/``Member`` (``id``/``name``/permissions)."""

    id: int
    name: str
    guild_permissions: FakePermissions = field(default_factory=FakePermissions)


@dataclass
class FakeResponse:
    """Stand-in for ``discord.InteractionResponse``."""

    interaction: FakeInteraction

    async def send_message(
        self,
        content: str | None = None,
        *,
        embed: discord.Embed | None = None,
        file: discord.File | None = None,
        ephemeral: bool = False,
        **kwargs: Any,
    ) -> None:
        attachment = _capture_attachment(file)
        self.interaction.payloads.append(
            RecordedPayload(
                kind="ephemeral" if ephemeral else "channel",
                content=content,
                embed=embed,
                attachment=attachment,
                recipient=str(self.interaction.user.id) if ephemeral else None,
            )
        )

    async def send_modal(self, modal: discord.ui.Modal[Any]) -> None:
        self.interaction.payloads.append(
            RecordedPayload(kind="modal", modal=modal, recipient=str(self.interaction.user.id))
        )


@dataclass
class FakeChannel:
    """Stand-in for the announcement channel (``interaction.channel``)."""

    interaction: FakeInteraction

    async def send(self, content: str | None = None, **kwargs: Any) -> None:
        self.interaction.payloads.append(RecordedPayload(kind="announcement", content=content))


@dataclass
class FakeFollowup:
    """Stand-in for ``interaction.followup`` (public follow-up messages)."""

    interaction: FakeInteraction

    async def send(self, content: str | None = None, **kwargs: Any) -> None:
        self.interaction.payloads.append(RecordedPayload(kind="channel", content=content))


@dataclass
class FakeInteraction:
    """Drives real view/modal callbacks without any discord.py gateway.

    ``guild_id`` mirrors ``discord.Interaction.guild_id`` (None in DMs); the
    admin bodies and the persistent-fallback view read it to resolve the
    guild being acted on. A string, matching the engine's TEXT guild ids.
    """

    user: FakeUser
    channel: FakeChannel
    response: FakeResponse
    followup: FakeFollowup
    message: None = None
    guild_id: str | None = None
    payloads: list[RecordedPayload] = field(default_factory=list)

    @classmethod
    def for_user(
        cls,
        user_id: int,
        name: str,
        *,
        manage_guild: bool = False,
        guild_id: str | None = "guild-1",
    ) -> FakeInteraction:
        interaction = cls.__new__(cls)
        interaction.user = FakeUser(
            user_id, name, FakePermissions(manage_guild=manage_guild)
        )
        interaction.payloads = []
        interaction.channel = FakeChannel(interaction)
        interaction.response = FakeResponse(interaction)
        interaction.followup = FakeFollowup(interaction)
        interaction.message = None
        interaction.guild_id = guild_id
        return interaction


def _capture_attachment(file: discord.File | None) -> RecordedAttachment | None:
    if file is None:
        return None
    attachment = RecordedAttachment(
        filename=file.filename or "",
        path=str(file.fp.name),
        size=os.fstat(file.fp.fileno()).st_size,
    )
    file.close()
    return attachment


def button_by_custom_id(view: discord.ui.View, custom_id: str) -> discord.ui.Button[Any]:
    """Find a view's button by custom_id (how the harness drives presses)."""
    for child in view.children:
        if isinstance(child, discord.ui.Button) and child.custom_id == custom_id:
            return child
    raise AssertionError(f"no button with custom_id {custom_id!r}")


async def press(view: discord.ui.View, custom_id: str, interaction: FakeInteraction) -> None:
    """Drive a real button callback with a fake interaction."""
    await button_by_custom_id(view, custom_id).callback(interaction)
