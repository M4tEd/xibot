"""Harness fakes: a Recorder plus the FakeInteraction surface for headless runs.

Built on the reference pattern of ``tests/unit/interaction_fakes.py`` (proven
against the real adapter) and kept structurally consistent with it. Two
harness-specific differences:

- every fake records into ONE shared `Recorder` (a whole scenario transcript,
  not per-interaction payload lists), and
- records are JSON-ready: embeds are stored as ``discord.Embed.to_dict()``
  dicts and view/modal components are described as plain dicts.

Recorded payload taxonomy (pinned decision #3): ``channel`` = the daily post,
``announcement`` = solve announcements and reveals, ``ephemeral`` = per-user
replies (always carrying ``recipient``). A fourth kind, ``modal``, records
modal handovers via ``response.send_modal`` (the GuessModal, the fixsong edit
form) so validators can inspect their inputs (VAL-GUESS-001). Messages
carrying a view keep the runtime view on the payload (never serialized) so
scenarios can press their buttons.

Drivability contract (see library/discord-adapter.md): buttons are pressed by
custom_id via ``await button.callback(interaction)``; modal text is injected
as ``modal.guess._value = text`` before ``await modal.on_submit(interaction)``
(discord.py 2.7.1's ``TextInput.value`` has no setter).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, cast

import discord

__all__ = [
    "FakeChannel",
    "FakeFollowup",
    "FakeInteraction",
    "FakePermissions",
    "FakeResponse",
    "FakeUser",
    "RecordedAttachment",
    "RecordedPayload",
    "Recorder",
    "find_button",
    "press_button",
    "submit_modal_text",
]


@dataclass(frozen=True)
class RecordedAttachment:
    """Attachment metadata captured from a ``discord.File``."""

    filename: str
    path: str
    size: int


@dataclass
class RecordedPayload:
    """One outgoing message/modal captured by the recorder.

    ``modal`` is the runtime handle for ``kind="modal"`` payloads (used by the
    scenario driver to submit the form); ``view`` is the runtime handle for
    messages carrying one (used to press their buttons). Neither is
    serialized.
    """

    kind: str  # "channel" | "ephemeral" | "announcement" | "modal"
    content: str | None = None
    embed: dict[str, Any] | None = None
    attachments: list[RecordedAttachment] = field(default_factory=list)
    components: list[dict[str, Any]] = field(default_factory=list)
    recipient: str | None = None
    modal: discord.ui.Modal | None = None
    view: discord.ui.View | None = None

    def to_dict(self) -> dict[str, Any]:
        """The JSON transcript shape: kind/content/embed/attachments/components/recipient."""
        return {
            "kind": self.kind,
            "content": self.content,
            "embed": self.embed,
            "attachments": [asdict(attachment) for attachment in self.attachments],
            "components": self.components,
            "recipient": self.recipient,
        }


class Recorder:
    """Captures every outgoing payload of a scenario run, in order."""

    def __init__(self) -> None:
        self.payloads: list[RecordedPayload] = []

    def record_message(
        self,
        *,
        kind: str,
        content: str | None = None,
        embed: discord.Embed | None = None,
        file: discord.File | None = None,
        view: discord.ui.View | None = None,
        recipient: str | None = None,
    ) -> RecordedPayload:
        """Record a message payload (channel post, ephemeral reply, announcement)."""
        payload = RecordedPayload(
            kind=kind,
            content=content,
            # Embed.to_dict() returns a TypedDict; at runtime it is a plain
            # JSON-ready dict, so cast across mypy's TypedDict/dict invariance.
            embed=cast("dict[str, Any]", embed.to_dict()) if embed is not None else None,
            attachments=_capture_file(file),
            components=view_components(view) if view is not None else [],
            recipient=recipient,
            view=view,
        )
        self.payloads.append(payload)
        return payload

    def record_modal(self, modal: discord.ui.Modal, *, recipient: str) -> RecordedPayload:
        """Record a modal handover (``response.send_modal``)."""
        payload = RecordedPayload(
            kind="modal",
            components=modal_components(modal),
            recipient=recipient,
            modal=modal,
        )
        self.payloads.append(payload)
        return payload

    def record_channel_post(
        self,
        *,
        content: str | None = None,
        embed: discord.Embed,
        view: discord.ui.View,
        file: discord.File,
    ) -> RecordedPayload:
        """Record the daily challenge post (the only ``channel``-kind payload).

        ``content`` carries the /songbot-pingrole mention when the guild
        configured one (the live send's message content).
        """
        return self.record_message(
            kind="channel", content=content, embed=embed, view=view, file=file
        )

    def to_list(self) -> list[dict[str, Any]]:
        """The JSON-serializable transcript of everything recorded so far."""
        return [payload.to_dict() for payload in self.payloads]


@dataclass(frozen=True)
class FakePermissions:
    """Stand-in for ``discord.Permissions`` (only ``manage_guild`` is used).

    The admin command bodies read ``interaction.user.guild_permissions
    .manage_guild`` — the same attribute a guild ``Member`` exposes — so the
    ``--as-admin``/``--as-non-admin`` flag flows through the identical check
    the live commands run (VAL-ADMIN-009).
    """

    manage_guild: bool = False


@dataclass(frozen=True)
class FakeUser:
    """Stand-in for ``discord.User``/``Member`` (``id``/``name``/permissions).

    The harness keeps ids as strings: a bare ``--user alice`` IS the stable
    deterministic id ``"alice"`` (pinned #2), so recorded recipients and
    sqlite3 ``user_id`` queries read naturally; ``--user 42:alice`` sets an
    explicit id with a display name.
    """

    id: str
    name: str
    guild_permissions: FakePermissions = field(default_factory=FakePermissions)


class FakeResponse:
    """Stand-in for ``discord.InteractionResponse``."""

    def __init__(self, recorder: Recorder, user: FakeUser) -> None:
        self._recorder = recorder
        self._user = user

    async def send_message(
        self,
        content: str | None = None,
        *,
        embed: discord.Embed | None = None,
        file: discord.File | None = None,
        view: discord.ui.View | None = None,
        ephemeral: bool = False,
        **kwargs: object,
    ) -> None:
        self._recorder.record_message(
            kind="ephemeral" if ephemeral else "channel",
            content=content,
            embed=embed,
            file=file,
            view=view,
            recipient=str(self._user.id) if ephemeral else None,
        )

    async def defer(self, *args: object, **kwargs: object) -> None:
        return None

    async def send_modal(self, modal: discord.ui.Modal) -> None:
        self._recorder.record_modal(modal, recipient=str(self._user.id))


class FakeChannel:
    """Stand-in for the announcement channel (``interaction.channel``).

    Everything sent here is a public non-post payload: solve announcements
    and day-advance reveals — both ``kind="announcement"`` (pinned #3).
    """

    def __init__(self, recorder: Recorder) -> None:
        self._recorder = recorder

    async def send(
        self,
        content: str | None = None,
        *,
        embed: discord.Embed | None = None,
        file: discord.File | None = None,
        **kwargs: object,
    ) -> None:
        self._recorder.record_message(
            kind="announcement", content=content, embed=embed, file=file
        )


class FakeFollowup:
    """Stand-in for ``interaction.followup`` (parity with the reference fakes)."""

    def __init__(self, recorder: Recorder, user: FakeUser) -> None:
        self._recorder = recorder
        self._user = user

    async def send(
        self,
        content: str | None = None,
        *,
        embed: discord.Embed | None = None,
        file: discord.File | None = None,
        ephemeral: bool = False,
        **kwargs: object,
    ) -> None:
        self._recorder.record_message(
            kind="ephemeral" if ephemeral else "channel",
            content=content,
            embed=embed,
            file=file,
            recipient=str(self._user.id) if ephemeral else None,
        )


class FakeInteraction:
    """Drives real view/modal callbacks without any discord.py gateway.

    Only the attributes the adapter callbacks may touch exist here:
    ``user``, ``channel``, ``response``, ``followup``, ``message``, and
    ``guild_id`` (the guild being acted on — multi-guild admin/view
    resolution; a string, matching the harness's string-id convention).
    """

    def __init__(
        self, recorder: Recorder, user: FakeUser, *, guild_id: str | None = None
    ) -> None:
        self.user = user
        self.channel = FakeChannel(recorder)
        self.response = FakeResponse(recorder, user)
        self.followup = FakeFollowup(recorder, user)
        self.message: None = None
        self.guild_id = guild_id


def find_button(view: discord.ui.View, custom_id: str) -> discord.ui.Button[Any]:
    """Find a view's button by custom_id (how the harness drives presses)."""
    for child in view.children:
        if isinstance(child, discord.ui.Button) and child.custom_id == custom_id:
            return child
    raise LookupError(f"no button with custom_id {custom_id!r}")


async def press_button(
    view: discord.ui.View, custom_id: str, interaction: FakeInteraction
) -> None:
    """Drive a real button callback with a fake interaction.

    discord.py wires decorator callbacks into ``_ItemCallback``, so calling
    ``button.callback(interaction)`` with just the interaction works; the cast
    satisfies mypy (the fake is duck-typed, not an Interaction subclass).
    """
    button = find_button(view, custom_id)
    await button.callback(cast("discord.Interaction[Any]", interaction))


async def submit_modal_text(
    modal: discord.ui.Modal,
    text_input: discord.ui.TextInput[Any],
    text: str,
    interaction: FakeInteraction,
) -> None:
    """Submit a real modal with injected text, the way discord.py would.

    discord.py populates component values from the interaction data before
    calling ``on_submit``; ``TextInput.value`` has no setter in 2.7.1, so the
    value is injected via ``_value`` exactly like the reference test fakes do.
    """
    text_input._value = text
    await modal.on_submit(cast("discord.Interaction[Any]", interaction))


def view_components(view: discord.ui.View) -> list[dict[str, Any]]:
    """Describe a view's buttons as plain dicts for the JSON transcript."""
    components: list[dict[str, Any]] = []
    for child in view.children:
        if isinstance(child, discord.ui.Button):
            components.append(
                {
                    "type": "button",
                    "custom_id": child.custom_id,
                    "label": child.label,
                    "style": child.style.name,
                    "emoji": str(child.emoji) if child.emoji is not None else None,
                    "disabled": child.disabled,
                }
            )
    return components


def modal_components(modal: discord.ui.Modal) -> list[dict[str, Any]]:
    """Describe a modal's text inputs as plain dicts for the JSON transcript."""
    components: list[dict[str, Any]] = []
    for child in modal.children:
        if isinstance(child, discord.ui.TextInput):
            # TextInput.label is deprecated in discord.py 2.6+ (warns on
            # access); the value lives on the underlying component payload.
            underlying = cast("Any", child)._underlying
            components.append(
                {
                    "type": "text_input",
                    "custom_id": child.custom_id,
                    "label": underlying.label,
                    "placeholder": child.placeholder,
                    "default": underlying.default,
                    "required": child.required,
                    "max_length": child.max_length,
                }
            )
    return components


def _capture_file(file: discord.File | None) -> list[RecordedAttachment]:
    """Capture attachment metadata from a ``discord.File``, then close it.

    ``discord.File`` opens the file EAGERLY; closing after capture avoids fd
    leaks across a scenario run (same pattern as the reference test fakes).
    """
    if file is None:
        return []
    fp = file.fp
    # The adapter only ever attaches path-backed files, which have a name.
    path = str(getattr(fp, "name", ""))
    attachment = RecordedAttachment(
        filename=file.filename or "",
        path=path,
        size=os.fstat(fp.fileno()).st_size,
    )
    file.close()
    return [attachment]
