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

from songbot.catalog.refresh import RefreshResult
from songbot.config import Settings
from songbot.engine import (
    POST_LIMIT_SOLVE_POINTS,
    Challenge,
    FixSongRefusedReason,
    GuessResult,
    LeaderboardEntry,
    Reveal,
    SkipRefusedReason,
    SongFix,
    UnlockRefusedReason,
    UnlockResult,
)

__all__ = [
    "ADMIN_CATALOG_EMPTY_MESSAGE",
    "ADMIN_NOT_CONFIGURED_MESSAGE",
    "ADMIN_POST_ALREADY_MESSAGE",
    "ADMIN_POST_FAILED_MESSAGE",
    "ADMIN_POST_SUCCESS_MESSAGE",
    "ADMIN_SKIP_SUCCESS_MESSAGE",
    "ATTACHMENT_FILENAME",
    "CHALLENGE_CLOSED_MESSAGE",
    "EMPTY_GUESS_MESSAGE",
    "EMPTY_LEADERBOARD_MESSAGE",
    "HEAR_MORE_SOLVED_MESSAGE",
    "NO_ACTIVE_CHALLENGE_MESSAGE",
    "PERMISSION_DENIED_MESSAGE",
    "PING_ROLE_FAILED_MESSAGE",
    "announcement_content",
    "daily_challenge_embed",
    "fixsong_ack_content",
    "fixsong_refusal_content",
    "format_seconds",
    "guess_feedback_content",
    "hear_more_content",
    "hear_more_refusal_content",
    "leaderboard_embed",
    "ping_announcement_content",
    "ping_mention_content",
    "pingrole_ack_content",
    "reload_ack_content",
    "reveal_embed",
    "setup_ack_content",
    "skip_refusal_content",
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

NO_ACTIVE_CHALLENGE_MESSAGE = (
    "There's no active challenge right now — check back after the next daily post! 🎵"
)
"""The graceful ephemeral notice for gameplay before any challenge exists
(VAL-CROSS-017) — shared by the harness scenarios and the persistent fallback
view when a click resolves to a guild with no challenges at all."""

PERMISSION_DENIED_MESSAGE = (
    "🚫 You need the **Manage Server** permission to use SongBot admin commands."
)
"""VAL-ADMIN-009: the generic ephemeral denial — no leaked error detail."""

ADMIN_POST_SUCCESS_MESSAGE = "✅ Posted today's challenge."
"""Ephemeral ack for a successful /songbot-post."""

ADMIN_POST_ALREADY_MESSAGE = (
    "📋 Today's challenge is already posted — use /songbot-skip to replace the song."
)
"""Ephemeral ack for an idempotent /songbot-post repeat (pinned #4)."""

ADMIN_CATALOG_EMPTY_MESSAGE = (
    "📭 The catalog is empty — add songs to a catalog source, then /songbot-reload."
)
"""Ephemeral ack when /songbot-post finds no songs at all (pinned #11)."""

ADMIN_POST_FAILED_MESSAGE = (
    "⚠️ Couldn't deliver today's challenge post — the channel send failed. "
    "Nothing was saved; please try again."
)
"""Ephemeral ack when the /songbot-post channel send fails (pinned #16).

Generic on purpose: no transport internals and never the song identity. The
just-created challenge was rolled back, so a retry reposts the identical
challenge.
"""

ADMIN_SKIP_SUCCESS_MESSAGE = "⏭️ Skipped — today's song has been replaced with a new one."
"""Ephemeral ack for a successful /songbot-skip (never names either song)."""

ADMIN_NOT_CONFIGURED_MESSAGE = (
    "⚙️ SongBot isn't set up in this server yet — run /songbot-setup to pick "
    "the daily-challenge channel."
)
"""Ephemeral ack when an admin command needs a channel the guild never configured."""

PING_ROLE_FAILED_MESSAGE = (
    "⚠️ Couldn't post the opt-in announcement — the channel send or the emoji "
    "reaction failed (check the bot's permissions and that the emoji is valid). "
    "Nothing was saved; please try again."
)
"""Ephemeral ack when /songbot-pingrole's announcement post fails.

Generic on purpose: no transport internals. Nothing is persisted on failure,
so a retry posts a fresh announcement from scratch.
"""


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
            "full-value guesses today — after that, a correct guess still banks "
            f"**{POST_LIMIT_SOLVE_POINTS} points**."
        ),
        color=discord.Color.blurple(),
    )
    embed.set_footer(
        text=(
            f"Snippet: {level0_seconds} • Worth {level0_points} points • "
            f"{settings.max_guesses_per_day} full-value guesses per day, "
            f"then {POST_LIMIT_SOLVE_POINTS} pts per solve"
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


def announcement_content(user_id: str, result: GuessResult, settings: Settings) -> str:
    """The public first-solve announcement (VAL-GUESS-012).

    Uses the ``<@user_id>`` mention format and NEVER names the song (pinned #9).
    Reports the snippet length the solver was hearing at solve time
    (``result.snippet_level`` mapped through the configured ladder), formatted
    with `format_seconds`.
    """
    guess_word = "guess" if result.guesses_used == 1 else "guesses"
    seconds = format_seconds(settings.snippet_lengths[result.snippet_level])
    return (
        f"🎉 <@{user_id}> guessed today's song in {result.guesses_used} {guess_word} "
        f"for **{result.points_awarded} points** while hearing **{seconds}** of audio!"
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
    if result.outcome == "correct_after_limit":
        return (
            "✅ Correct — you're past today's full-value guesses, so this banks "
            f"**{result.points_awarded} points** (no win or streak)."
        )
    if result.outcome == "wrong":
        if result.guesses_left > 0:
            return (
                f"❌ Not quite — **{result.guesses_left}** "
                f"{'guess' if result.guesses_left == 1 else 'guesses'} left today."
            )
        return (
            "❌ Not quite — you're out of full-value guesses, but you can keep "
            f"trying: a correct guess still banks **{POST_LIMIT_SOLVE_POINTS} points**."
        )
    if result.outcome == "already_solved":
        return "✅ You've already solved today's song!"
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


def setup_ack_content(channel_mention: str, settings: Settings) -> str:
    """The ephemeral ack for /songbot-setup: where posts land and when."""
    return (
        f"✅ Daily challenges will be posted in {channel_mention}.\n"
        f"The first post lands at the next scheduled time "
        f"(**{settings.daily_post_time} {settings.timezone}**) — or post one "
        "right now with /songbot-post."
    )


def ping_announcement_content(emoji: str, role_mention: str) -> str:
    """The public opt-in announcement posted by /songbot-pingrole.

    Names the role and the emoji but NEVER any song (pinned #9 is trivially
    preserved — no song identity crosses this builder). Users react with the
    emoji to receive the role; removing the reaction removes the role.
    """
    return (
        "🎵 **Want a ping when the daily song drops?**\n"
        f"React with {emoji} to get the {role_mention} role — you'll be "
        "mentioned in every daily challenge post. Remove your reaction "
        "anytime to opt out."
    )


def pingrole_ack_content(role_mention: str, emoji: str) -> str:
    """The ephemeral ack for /songbot-pingrole: what was set up."""
    return (
        f"✅ Opt-in announcement posted. Reacting with {emoji} now grants the "
        f"{role_mention} role (removing the reaction revokes it), and daily "
        "challenge posts will mention it.\n"
        "Re-running /songbot-pingrole posts a fresh announcement and "
        "supersedes the old one."
    )


def ping_mention_content(role_id: str) -> str:
    """The daily post's message content when a ping role is configured.

    The ``<@&role_id>`` mention pings the role's members (role mentions are
    explicitly allowed on that send); the embed carries all the copy.
    """
    return f"<@&{role_id}>"


def skip_refusal_content(reason: SkipRefusedReason) -> str:
    """The ephemeral refusal for /songbot-skip (pinned #5: zero mutation)."""
    if reason == "no_challenge":
        return "There's no challenge to skip today — post one first."
    if reason == "revealed":
        return "Today's challenge has already been revealed — it can't be skipped."
    return "Today's challenge already has a solver — it can't be skipped."


def fixsong_ack_content(fix: SongFix) -> str:
    """The ephemeral ack for /songbot-fixsong: exactly what changed, old -> new.

    Names the song — a deliberate, scoped exception to the pinned-#9 secrecy
    rule: the ack is ephemeral and admin-gated (Manage Server), and the
    command is unusable blind. Public payloads still never name songs.
    """
    old_artist = fix.old_artist or "—"
    new_artist = fix.new_artist or "—"
    return (
        f"🛠️ Corrected the song from the {fix.challenge_date} challenge:\n"
        f"Title: **{fix.old_title}** → **{fix.new_title}**\n"
        f"Artist: **{old_artist}** → **{new_artist}**\n"
        "Applies to new guesses immediately and survives catalog reloads; "
        "already-recorded guesses keep their original results."
    )


def fixsong_refusal_content(reason: FixSongRefusedReason) -> str:
    """The ephemeral refusal for /songbot-fixsong (zero mutation)."""
    if reason == "invalid_date":
        return "⚠️ Invalid date — use YYYY-MM-DD (the challenge's local date)."
    if reason == "blank_title":
        return "⚠️ The corrected title can't be empty."
    return (
        "There's no challenge to fix — post one first, "
        "or pass the date of an earlier challenge."
    )


def reload_ack_content(result: RefreshResult) -> str:
    """The ephemeral ack for /songbot-reload: the per-source summary.

    One line per enabled source with its added/updated/removed/retained
    counts, or its error when the source failed (per-source failure isolation,
    pinned #12). Never contains song identity (counts only).
    """
    if not result.sources:
        return "🔄 Catalog reload: no catalog sources are configured."
    lines = ["🔄 Catalog reload complete."]
    for source in result.sources:
        if source.error is not None:
            lines.append(f"**{source.source}**: ⚠️ failed — {source.error}")
        else:
            lines.append(
                f"**{source.source}**: {source.added} added • {source.updated} updated "
                f"• {source.removed} removed • {source.retained} retained"
            )
    return "\n".join(lines)
