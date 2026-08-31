"""Shared interaction permission predicates (admin gating).

Kept out of `admin.py` so the interactive admin UI (the fixsong view/modal)
runs the SAME Manage-Guild check the command bodies run — the harness
permission flag maps onto it identically — without a circular import.
"""

from __future__ import annotations

from typing import Any

import discord

__all__ = ["has_manage_guild"]


def has_manage_guild(interaction: discord.Interaction[Any]) -> bool:
    """The Manage-Guild check both discord.py and the harness honor.

    Reads ``interaction.user.guild_permissions.manage_guild`` — present on
    guild Members and on the harness's FakeUser; anything else (DM-shaped
    users, missing attributes) is denied.
    """
    permissions = getattr(interaction.user, "guild_permissions", None)
    return bool(getattr(permissions, "manage_guild", False))
