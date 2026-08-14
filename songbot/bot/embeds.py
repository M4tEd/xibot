"""Embed + message builders (daily challenge, reveal, leaderboard).

Every builder in this module is a PURE function: engine result in, Discord
payload out. No game rules, no I/O. Two pinned invariants live here:

- Pinned #9 (attachments): snippet files are ALWAYS attached under the
  filename ``songbot-snippet.mp3`` so the file name never leaks the song.
- Pinned #9 (secrecy): no pre-reveal copy (daily embed, guess/hear-more
  feedback, solve announcement) ever contains the song title or artist.
  Only `reveal_embed` names the song — it runs after the reveal.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import discord

from songbot.config import Settings
from songbot.engine import (
    Challenge,
    GuessResult,
    LeaderboardEntry,
    Reveal,
    UnlockRefusedReason,
    UnlockResult,
)

__all__ = [
    "ATTACHMENT_FILENAME",
    "CHALLENGE_CLOSED_MESSAGE",
    "EMPTY_GUESS_MESSAGE",
    "EMPTY_LEADERBOARD_MESSAGE",
    "HEAR_MORE_SOLVED_MESSAGE",
    "announcement_content",
    "daily_challenge_embed",
    "format_seconds",
    "guess_feedback_content",
    "hear_more_content",
    "hear_more_refusal_content",
    "leaderboard_embed",
    "reveal_embed",
    "snippet_attachment",
]

ATTACHMENT_FILENAME = "songbot-snippet.mp3"
"""Pinned #9: snippet attachments always use this filename (never the song)."""

CHALLENGE_CLOSED_MESSAGE = "This challenge has closed."
"""The single ephemeral notice for gameplay on a revealed challenge (VAL-GUESS-019)."""

EMPTY_GUESS_MESSAGE = "Please enter a guess — the text field can't be empty."
"""Pinned #15: empty-after-strip guesses are validation rejections, never counted."""

EMPTY_LEADERBOARD_MESSAGE = "No scores yet — be the first to solve a daily song! 🎵"
"""VAL-SCORE-012: friendly ephemeral when nobody has scored."""

HEAR_MORE_SOLVED_MESSAGE = "You've already solved this one — no more snippets needed. ✅"
"""Hear-more rejection for a user who already solved the challenge."""


def format_seconds(seconds: float) -> str:
    """Format a snippet length compactly: 1.0 -> "1s", 2.5 -> "2.5s"."""
    return f"{seconds:g}s"


def daily_challenge_embed(challenge: Challenge, settings: Settings) -> discord.Embed:
    """The daily post embed: title, how-to-play, and the level-0 advertisement.

    NEVER includes the song title/artist (secrecy invariant, pinned #9).
    """
    level0_seconds = format_seconds(settings.snippet_lengths[0])
    level0_points = settings.snippet_points[0]
    embed = discord.Embed(
        title=f"🎵 Daily Song — {challenge.date}",
        description=(
            "A new song just dropped — can you name it from a tiny snippet?\n\n"
            "**How to play**\n"
            "🎧 **Hear more** — unlock a longer snippet (worth fewer points)\n"
            "💡 **Guess** — name the artist or the title\n"
            "🏆 **Leaderboard** — see today's rankings\n\n"
            f"Right now you hear **{level0_seconds}** — a correct guess is worth "
            f"**{level0_points} points**. You have **{settings.max_guesses_per_day}** "
            "guesses today."
        ),
        color=discord.Color.blurple(),
    )
    embed.set_footer(
        text=(
            f"Snippet: {level0_seconds} • Worth {level0_points} points • "
            f"{settings.max_guesses_per_day} guesses per day"
        )
    )
    return embed


def reveal_embed(reveal: Reveal) -> discord.Embed:
    """The previous challenge's reveal: song identity + winners summary.

    This is the ONLY builder allowed to name the song — it runs post-reveal.
    Winners are listed in solve order with ``<@user_id>`` mentions.
    """
    artist = reveal.song.artist or "Unknown Artist"
    description = f"The song was **{reveal.song.title}** by **{artist}**."
    if reveal.winners:
        lines = [
            f"🎉 <@{winner.user_id}> — {winner.guesses_used} "
            f"{'guess' if winner.guesses_used == 1 else 'guesses'}, "
            f"{winner.points_awarded} points"
            for winner in reveal.winners
        ]
        description += "\n\n**Winners**\n" + "\n".join(lines)
    else:
        description += "\n\nNobody got it — better luck with today's song!"
    return discord.Embed(
        title=f"🎶 Yesterday's Song, Revealed — {reveal.date}",
        description=description,
        color=discord.Color.gold(),
    )


def leaderboard_embed(entries: Sequence[LeaderboardEntry]) -> discord.Embed:
    """The top-10 leaderboard embed: rank, ``<@user_id>``, points, wins, streak.

    Callers pass the engine's already-ordered entries (total_points DESC,
    wins DESC, user_id ASC — pinned #8); ranks are contiguous from 1.
    """
    lines = []
    for rank, entry in enumerate(entries, start=1):
        wins = f"{entry.wins} {'win' if entry.wins == 1 else 'wins'}"
        streak = (
            f"{entry.current_streak} {'day' if entry.current_streak == 1 else 'days'}"
        )
        lines.append(
            f"**{rank}.** <@{entry.user_id}> — **{entry.total_points}** pts • "
            f"{wins} • 🔥 {streak}"
        )
    return discord.Embed(
        title="🏆 SongBot Leaderboard",
        description="\n".join(lines),
        color=discord.Color.blurple(),
    )


def announcement_content(user_id: str, guesses_used: int, points_awarded: int) -> str:
    """The public first-solve announcement (VAL-GUESS-012).

    Uses the ``<@user_id>`` mention format and NEVER names the song (pinned #9).
    """
    guess_word = "guess" if guesses_used == 1 else "guesses"
    return (
        f"🎉 <@{user_id}> guessed today's song in {guesses_used} {guess_word} "
        f"for **{points_awarded} points**!"
    )


def guess_feedback_content(result: GuessResult) -> str:
    """The ephemeral reply to a guess submission (never names the song).

    Maps every `GuessOutcome` to user-facing copy; ``challenge_closed`` maps
    to the single closed-challenge notice (VAL-GUESS-019).
    """
    if result.outcome == "correct":
        if result.is_both:
            matched = "**both the title and the artist** — 1.5x bonus!"
        elif result.matched_title:
            matched = "the **title**"
        else:
            matched = "the **artist**"
        return (
            f"✅ Correct — you matched {matched} "
            f"**{result.points_awarded} points** banked."
        )
    if result.outcome == "wrong":
        if result.guesses_left > 0:
            return (
                f"❌ Not quite — **{result.guesses_left}** "
                f"{'guess' if result.guesses_left == 1 else 'guesses'} left today."
            )
        return "❌ Not quite — that was your **last guess** for today (0 left)."
    if result.outcome == "already_solved":
        return "✅ You've already solved today's song!"
    if result.outcome == "limit_reached":
        return (
            f"❌ No guesses left — you've used all {result.guesses_used} for today."
        )
    if result.outcome == "empty":
        return EMPTY_GUESS_MESSAGE
    return CHALLENGE_CLOSED_MESSAGE


def hear_more_content(result: UnlockResult, settings: Settings) -> str:
    """The ephemeral reply to a successful Hear-more press.

    Reports the new level, the new snippet length, and the new potential
    payout (VAL-HEAR-002/003); never names the song.
    """
    seconds = format_seconds(settings.snippet_lengths[result.level])
    return (
        f"🎧 Snippet level **{result.level}** unlocked — **{seconds}** of audio. "
        f"A correct guess is now worth **{result.potential_points} points**."
    )


def hear_more_refusal_content(reason: UnlockRefusedReason, settings: Settings) -> str:
    """The ephemeral reply when Hear-more is refused (no attachment)."""
    if reason == "closed":
        return CHALLENGE_CLOSED_MESSAGE
    if reason == "solved":
        return HEAR_MORE_SOLVED_MESSAGE
    max_seconds = format_seconds(max(settings.snippet_lengths))
    return (
        f"🎧 You're already at the longest snippet (**{max_seconds}**) — "
        "nothing more to unlock."
    )


def snippet_attachment(path: Path) -> discord.File:
    """Attach a snippet file under the pinned-#9 filename (never the song's)."""
    return discord.File(path, filename=ATTACHMENT_FILENAME)
