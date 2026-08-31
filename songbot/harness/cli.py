"""``python -m songbot.harness <scenario>`` — the headless validation harness.

Each scenario is a one-shot process against ``DATABASE_PATH`` (default
``data/songbot.db``) that drives the REAL ``DailyChallengeView``/``GuessModal``
callbacks via `FakeInteraction` — never engine shortcuts — and prints a JSON
transcript to stdout::

    {"scenario": ..., "payloads": [...], "state": {...}}

Every payload is ``{kind, content, embed, attachments, components,
recipient}`` (pinned #3 taxonomy: channel / announcement / ephemeral, plus
``modal`` for the guess-modal handover). Two pinned exact-shape outputs:
a same-day repeat ``post`` prints ``{"already_posted": true, "messages": []}``
(pinned #4) and an empty catalog at post time prints
``{"error": "catalog_empty"}`` (pinned #11); domain errors exit 1. A failed
daily-post send rolls the just-created challenge back and prints
``{"error": "post_failed", "message": ...}`` (pinned #16, exit 1) — the retry
reposts the identical challenge.

Clock (pinned #1): every scenario accepts ``--now "ISO-8601"``; the engine
only ever sees the injected time. Users (pinned #2): ``--user alice`` uses
the bare name AS the stable deterministic user id (so recorded ``recipient``
fields and sqlite3 ``user_id`` queries read naturally); ``--user 42:alice``
sets an explicit id with a display name.

Challenge targeting for gameplay scenarios (hear-more / guess / leaderboard):
with ``--now`` the harness drives the view bound to THAT date's challenge —
a revealed challenge therefore yields the real closed-challenge notice
(VAL-GUESS-019). Without ``--now`` it presses the buttons on the CURRENT post
(the guild's most recent ACTIVE challenge), which keeps bare multi-day
``advance-day`` flows pointed at the live challenge.

``advance-day`` is state-anchored: it posts the day AFTER the guild's latest
challenge (at the configured post time), revealing stale challenges first
(reveal announcement recorded BEFORE the new daily post, pinned #3). The
reveal is delivery-coupled (pinned #17): the previous challenge is marked
revealed ONLY after the reveal send succeeds — a failed reveal send mutates
nothing and prints ``{"error": "reveal_failed", "message": ...}`` (exit 1),
and the retry delivers the reveal before posting. From an empty state it
posts the local date of ``--now`` (or the real clock) with no reveal
(VAL-CROSS-018).

Admin scenarios — ``admin-setup`` / ``admin-post`` / ``admin-skip`` /
``admin-reload`` / ``admin-fixsong`` / ``admin-pingrole`` — drive the REAL
``AdminCommands`` bodies (songbot/bot/admin.py) with the invoking user's
Manage-Guild permission simulated by ``--as-admin``/``--as-non-admin``
(exactly one required). A denied invocation records exactly one ephemeral
permission-denied payload and mutates nothing (VAL-ADMIN-009). A same-day
repeat ``admin-post`` prints the pinned-#4 compact form
``{"already_posted": true, "messages": []}`` just like ``post``
(VAL-CROSS-015); an empty catalog prints ``{"error": "catalog_empty"}``
(pinned #11). ``admin-setup`` upserts the guild's post channel
(multi-guild configuration, ``--channel`` overrides the default harness
channel). ``admin-fixsong`` shows a challenge song's current metadata, then presses
the REAL Edit button and submits the REAL modal (``--title``/``--artist``
are the texts typed into the pre-filled fields — omit either to keep the
current value; ``--date`` selects the challenge); its ephemeral ack and
``state.fix`` show the old -> new metadata (admin-only secrecy
exception — the harness's ``status`` surface exposes song identity too).
``admin-pingrole`` posts the reaction-role opt-in announcement (recorded as
an ``announcement`` payload) and upserts ``ping_role_settings``
(``--role``/``--emoji`` override the defaults); later ``post``/
``advance-day`` runs then carry the role mention in the daily post's
``content``.

Multi-guild: the harness drives ONE guild per run — the
DISCORD_GUILD_ID/DISCORD_CHANNEL_ID bootstrap pair when set, else the
deterministic ``harness-guild``/``harness-channel`` defaults — and seeds its
``guild_settings`` row (if absent) at startup, mirroring the live client's
env bootstrap.

``serve`` runs ONLY the health endpoint (mode="harness", HEALTH_PORT) and
makes no Discord requests. The harness never constructs a Discord client, so
``DISCORD_BOT_TOKEN`` is never used; a placeholder is substituted when the
environment provides none.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, cast

import discord

from songbot.bot.admin import AdminCommands
from songbot.bot.embeds import (
    NO_ACTIVE_CHALLENGE_MESSAGE,
    daily_challenge_embed,
    ping_mention_content,
    reveal_embed,
    snippet_attachment,
)
from songbot.bot.fixsong import FixSongModal, FixSongView
from songbot.bot.health import serve_health
from songbot.bot.modals import GuessModal
from songbot.bot.views import DailyChallengeView
from songbot.config import ConfigError, Settings, load_settings
from songbot.db import ChallengeRow, Database
from songbot.engine import (
    CatalogEmptyError,
    Challenge,
    EngineError,
    GameEngine,
    Reveal,
)
from songbot.harness.fakes import (
    FakeInteraction,
    FakePermissions,
    FakeUser,
    Recorder,
    press_button,
    submit_modal_text,
)
from songbot.snippets import SnippetError, SnippetGenerator

__all__ = [
    "NO_ACTIVE_CHALLENGE_MESSAGE",
    "HarnessContext",
    "RecordPost",
    "RecordReveal",
    "UsageError",
    "build_parser",
    "main",
    "parse_now",
    "parse_user",
    "scenario_admin_fixsong",
    "scenario_admin_pingrole",
    "scenario_admin_post",
    "scenario_admin_reload",
    "scenario_admin_setup",
    "scenario_admin_skip",
    "scenario_advance_day",
    "scenario_guess",
    "scenario_hear_more",
    "scenario_leaderboard",
    "scenario_post",
    "scenario_reset",
    "scenario_serve",
    "scenario_status",
]

DEFAULT_HARNESS_GUILD_ID = "harness-guild"
"""The guild scenarios act on when no DISCORD_GUILD_ID bootstrap is set."""

DEFAULT_HARNESS_CHANNEL_ID = "harness-channel"
"""The channel scenarios post to when no DISCORD_CHANNEL_ID bootstrap is set."""


class UsageError(Exception):
    """Invalid harness invocation (e.g. a malformed --now timestamp)."""


@dataclass
class HarnessContext:
    """The real stack a scenario runs against: settings + db + engine.

    ``guild_id``/``channel_id`` are the ONE guild the harness drives: the env
    bootstrap pair when set, else the deterministic harness defaults (pass
    explicit values to override; the empty string means "resolve"). The
    guild's ``guild_settings`` row is seeded (if absent) whenever the context
    is built, so the admin command bodies' configured-channel lookup works
    exactly like live — and an earlier ``admin-setup`` invocation's choice
    survives across harness runs.
    """

    settings: Settings
    db: Database
    engine: GameEngine
    guild_id: str = ""
    channel_id: str = ""

    def __post_init__(self) -> None:
        if not self.guild_id:
            self.guild_id = self.settings.guild_id or DEFAULT_HARNESS_GUILD_ID
        if not self.channel_id:
            self.channel_id = self.settings.channel_id or DEFAULT_HARNESS_CHANNEL_ID
        if self.engine.guild_settings(self.guild_id) is None:
            self.engine.set_guild_channel(
                self.guild_id, self.channel_id, set_by="harness", now=datetime.now(UTC)
            )

    @classmethod
    def from_settings(cls, settings: Settings) -> HarnessContext:
        """Build the production stack: real DB (migrated) + real SnippetGenerator."""
        db = Database.open(settings.database_path)
        engine = GameEngine(db, settings, SnippetGenerator(settings.snippet_cache_dir))
        return cls(settings=settings, db=db, engine=engine)

    def close(self) -> None:
        """Close the database connection (one-shot process hygiene)."""
        self.db.close()


RecordPost = Callable[[Recorder, HarnessContext, Challenge, datetime], None]
"""The harness's daily-post "send": record the channel payload.

Default ``_record_daily_post`` (the real builders, recorded transport). Tests
inject a failing seam to exercise the pinned-#16 delivery-failure rollback;
the CLI itself always uses the default.
"""

RecordReveal = Callable[[Recorder, HarnessContext, Reveal], None]
"""The harness's reveal "send": record the announcement payload.

Default ``_record_reveal`` (the real embed builder, recorded transport).
Tests inject a failing seam to exercise the pinned-#17 delivery-coupled
reveal retry; the CLI itself always uses the default.
"""


# -- argument parsing helpers ---------------------------------------------------


def parse_user(spec: str) -> FakeUser:
    """Parse ``--user``: a bare name is its own stable id; ``id:name`` is explicit."""
    user_id, sep, name = spec.partition(":")
    if not sep:
        return FakeUser(id=spec, name=spec)
    if not user_id:
        raise UsageError(f"invalid --user {spec!r}: expected <name> or <id>:<name>")
    return FakeUser(id=user_id, name=name or user_id)


def parse_now(raw: str | None) -> datetime:
    """Parse ``--now`` as ISO-8601; naive timestamps are read as UTC."""
    if raw is None:
        return datetime.now(UTC)
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise UsageError(f"invalid --now {raw!r}: expected an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"value must be >= 1, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """The harness CLI: one subcommand per scenario, ``--now`` everywhere."""
    parser = argparse.ArgumentParser(
        prog="songbot.harness",
        description=(
            "Headless SongBot validation harness: drives the real views/modals/engine "
            "and prints a JSON transcript of every recorded payload."
        ),
    )
    sub = parser.add_subparsers(dest="scenario", required=True)

    def add_now(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--now",
            default=None,
            help="ISO-8601 timestamp injected as the current time (pinned #1).",
        )

    def add_user(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--user",
            required=True,
            help="<name> (the name is the stable user id) or <id>:<name> (pinned #2).",
        )

    post = sub.add_parser("post", help="Ensure today's challenge exists and post it.")
    add_now(post)

    hear_more = sub.add_parser(
        "hear-more", help="Press the Hear more button (--times N presses)."
    )
    add_user(hear_more)
    hear_more.add_argument("--times", type=_positive_int, default=1)
    add_now(hear_more)

    guess = sub.add_parser("guess", help="Open the Guess modal and submit --text.")
    add_user(guess)
    guess.add_argument("--text", required=True, help="The guess text (may be empty).")
    add_now(guess)

    leaderboard = sub.add_parser("leaderboard", help="Press the Leaderboard button.")
    add_user(leaderboard)
    add_now(leaderboard)

    advance = sub.add_parser(
        "advance-day", help="Reveal the previous challenge and post the next day."
    )
    add_now(advance)

    def add_admin_flags(subparser: argparse.ArgumentParser) -> None:
        """--as-admin/--as-non-admin (exactly one) + the optional invoker."""
        group = subparser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "--as-admin",
            action="store_true",
            help="Simulate an invoker WITH the Manage-Guild permission.",
        )
        group.add_argument(
            "--as-non-admin",
            action="store_true",
            help="Simulate an invoker WITHOUT the Manage-Guild permission.",
        )
        subparser.add_argument(
            "--user",
            default="admin",
            help="The invoking user, <name> or <id>:<name> (default: 'admin').",
        )
        add_now(subparser)

    admin_post = sub.add_parser(
        "admin-post", help="Run /songbot-post (ensure + post today's challenge)."
    )
    add_admin_flags(admin_post)

    admin_skip = sub.add_parser(
        "admin-skip", help="Run /songbot-skip (replace today's song, pinned #5)."
    )
    add_admin_flags(admin_skip)

    admin_reload = sub.add_parser(
        "admin-reload", help="Run /songbot-reload (refresh the catalog)."
    )
    add_admin_flags(admin_reload)

    admin_fixsong = sub.add_parser(
        "admin-fixsong",
        help="Run /songbot-fixsong (show a challenge song's metadata, then edit it).",
    )
    add_admin_flags(admin_fixsong)
    admin_fixsong.add_argument(
        "--title",
        default=None,
        help="Text to type into the modal's Title field (omit to keep the current one).",
    )
    admin_fixsong.add_argument(
        "--artist",
        default=None,
        help="Text to type into the modal's Artist field (omit to keep the current "
        "one; an empty string clears it).",
    )
    admin_fixsong.add_argument(
        "--date",
        default=None,
        help="Which challenge's song to fix, YYYY-MM-DD (default: the latest).",
    )

    admin_setup = sub.add_parser(
        "admin-setup", help="Run /songbot-setup (set the guild's post channel)."
    )
    add_admin_flags(admin_setup)
    admin_setup.add_argument(
        "--channel",
        default=None,
        help="The channel id to configure (default: the harness channel).",
    )

    admin_pingrole = sub.add_parser(
        "admin-pingrole",
        help="Run /songbot-pingrole (post the reaction-role opt-in announcement).",
    )
    add_admin_flags(admin_pingrole)
    admin_pingrole.add_argument(
        "--role",
        default="ping-role",
        help="The role id to grant on reaction (default: 'ping-role').",
    )
    admin_pingrole.add_argument(
        "--emoji",
        default="🎵",
        help="The opt-in reaction emoji (default: 🎵).",
    )

    reset = sub.add_parser("reset", help="Wipe all DB tables and the snippet cache.")
    add_now(reset)

    status = sub.add_parser(
        "status", help="Print today's date/challenge/song/counts/leaderboard as JSON."
    )
    add_now(status)

    serve = sub.add_parser(
        "serve", help="Run ONLY the health endpoint (mode=harness) until killed."
    )
    add_now(serve)
    return parser


# -- shared scenario helpers -----------------------------------------------------


def _local_date(settings: Settings, now: datetime) -> date:
    """The calendar date of ``now`` in the configured timezone."""
    return now.astimezone(settings.tz).date()


def _transcript(
    scenario: str, recorder: Recorder, state: dict[str, Any]
) -> dict[str, Any]:
    """The standard harness output shape: scenario + payloads + state."""
    return {
        "scenario": scenario,
        "payloads": recorder.to_list(),
        "state": state,
    }


def _challenge_state(challenge: Challenge) -> dict[str, Any]:
    """Post/advance-day state. NEVER includes song title/artist (secrecy, pinned #9)."""
    return {
        "id": challenge.id,
        "date": challenge.date,
        "status": challenge.status,
        "song_id": challenge.song.id,
        "snippet_offset_sec": challenge.snippet_offset_sec,
        "created": challenge.created,
    }


def _user_state(ctx: HarnessContext, challenge_id: int, user_id: str) -> dict[str, Any] | None:
    """The per-user challenge row as JSON, or None when no interaction happened."""
    row = ctx.db.query_one(
        "SELECT snippet_level, guesses_used, solved, points_awarded"
        " FROM challenge_users WHERE challenge_id = ? AND user_id = ?",
        (challenge_id, user_id),
    )
    if row is None:
        return None
    return {
        "user_id": user_id,
        "snippet_level": int(row["snippet_level"]),
        "guesses_used": int(row["guesses_used"]),
        "solved": bool(row["solved"]),
        "points_awarded": int(row["points_awarded"]),
    }


def _counts(ctx: HarnessContext) -> dict[str, int]:
    """Row counts for the status/reset state blocks."""
    def count(table: str) -> int:
        row = ctx.db.query_one(f"SELECT COUNT(*) AS c FROM {table}")
        return int(row["c"]) if row is not None else 0

    return {
        "songs": count("songs"),
        "challenges": count("challenges"),
        "challenge_users": count("challenge_users"),
        "guesses": count("guesses"),
        "users": count("user_stats"),
    }


def _view_for(ctx: HarnessContext, challenge_id: int, now: datetime) -> DailyChallengeView:
    """The REAL message-bound view for a challenge, with the scenario clock."""
    return DailyChallengeView(
        ctx.engine,
        challenge_id,
        guild_id=ctx.guild_id,
        settings=ctx.settings,
        clock=lambda: now,
    )


def _resolve_challenge(
    ctx: HarnessContext, now: datetime, *, now_pinned: bool
) -> ChallengeRow | None:
    """The challenge whose view a gameplay scenario drives.

    With ``--now``: the challenge of that exact local date, whatever its
    status (a revealed one drives the real closed-challenge notice). Without
    ``--now``: the guild's most recent ACTIVE challenge (the "current post").
    """
    guild_id = ctx.guild_id
    if now_pinned:
        row = ctx.db.query_one(
            "SELECT * FROM challenges WHERE guild_id = ? AND date = ?",
            (guild_id, _local_date(ctx.settings, now).isoformat()),
        )
        return ChallengeRow.from_row(row) if row is not None else None
    row = ctx.db.query_one(
        "SELECT * FROM challenges WHERE guild_id = ? AND status = 'active'"
        " ORDER BY date DESC, id DESC LIMIT 1",
        (guild_id,),
    )
    return ChallengeRow.from_row(row) if row is not None else None


async def _send_no_active_challenge(recorder: Recorder, user: FakeUser) -> None:
    """The graceful ephemeral notice when there is no challenge to act on."""
    interaction = FakeInteraction(recorder, user)
    await interaction.response.send_message(NO_ACTIVE_CHALLENGE_MESSAGE, ephemeral=True)


def _record_daily_post(
    recorder: Recorder, ctx: HarnessContext, challenge: Challenge, now: datetime
) -> None:
    """Record the daily post exactly as the live client would send it.

    Built with the REAL embed builder, the REAL persistent view, and the REAL
    attachment helper (pinned #9 filename); only the transport is recorded.
    When the guild configured /songbot-pingrole, the payload's ``content``
    carries the role mention exactly like the live send.
    """
    config = ctx.engine.ping_role_settings(challenge.guild_id)
    recorder.record_channel_post(
        content=ping_mention_content(config.role_id) if config is not None else None,
        embed=daily_challenge_embed(challenge, ctx.settings),
        view=_view_for(ctx, challenge.id, now),
        file=snippet_attachment(challenge.snippet_paths[0]),
    )


def _record_reveal(recorder: Recorder, ctx: HarnessContext, reveal: Reveal) -> None:
    """Record the reveal announcement exactly as the live client would send it.

    Built with the REAL ``reveal_embed`` builder (the same embed the live
    ``_send_reveal`` posts); only the transport is recorded — an
    ``announcement``-kind payload (pinned #3). ``ctx`` is unused but kept for
    symmetry with `RecordPost`.
    """
    recorder.record_message(kind="announcement", embed=reveal_embed(reveal))


HARNESS_PING_MESSAGE_ID = "harness-ping-announcement"
"""The deterministic announcement message id the harness's poster returns."""


# -- scenarios ---------------------------------------------------------------------


async def scenario_post(
    ctx: HarnessContext, now: datetime, *, record_post: RecordPost | None = None
) -> dict[str, Any]:
    """Ensure today's challenge via the engine and record the daily post.

    Idempotent (pinned #4): a same-day repeat prints exactly
    ``{"already_posted": true, "messages": []}`` — no second channel payload,
    no state change (the engine still re-heals the snippet cache, pinned #14).
    An empty catalog even after the auto-bootstrap yields the pinned-#11
    ``{"error": "catalog_empty"}`` with no challenge row and no payload. If
    recording the channel payload fails for the challenge THIS call created
    (pinned #16), the challenge is rolled back (row deleted, snippet cache
    purged) and the scenario prints ``{"error": "post_failed", ...}`` (exit 1)
    — a retry recreates the identical challenge and delivers it.
    """
    record = record_post if record_post is not None else _record_daily_post
    try:
        challenge = ctx.engine.ensure_today_challenge(ctx.guild_id, ctx.channel_id, now)
    except CatalogEmptyError:
        return {"error": "catalog_empty"}
    if not challenge.created:
        return {"already_posted": True, "messages": []}
    recorder = Recorder()
    try:
        record(recorder, ctx, challenge, now)
    except Exception as exc:
        # Pinned #16: roll back the just-created challenge (never a
        # pre-existing row) so a transient send failure never suppresses the day.
        ctx.engine.delete_challenge(challenge.id)
        return {"error": "post_failed", "message": str(exc)}
    return _transcript("post", recorder, {"challenge": _challenge_state(challenge)})


async def scenario_hear_more(
    ctx: HarnessContext, user: FakeUser, times: int, now: datetime, *, now_pinned: bool
) -> dict[str, Any]:
    """Press the REAL Hear-more button ``times`` times (fresh interaction each)."""
    recorder = Recorder()
    row = _resolve_challenge(ctx, now, now_pinned=now_pinned)
    if row is None:
        await _send_no_active_challenge(recorder, user)
        return _transcript("hear-more", recorder, {"challenge": None, "user": None})
    view = _view_for(ctx, row.id, now)
    for _ in range(times):
        await press_button(view, "songbot:hear_more", FakeInteraction(recorder, user))
    return _transcript(
        "hear-more",
        recorder,
        {
            "challenge": {"id": row.id, "date": row.date},
            "user": _user_state(ctx, row.id, user.id),
        },
    )


async def scenario_guess(
    ctx: HarnessContext, user: FakeUser, text: str, now: datetime, *, now_pinned: bool
) -> dict[str, Any]:
    """Press the REAL Guess button, then submit the REAL modal with ``text``."""
    recorder = Recorder()
    row = _resolve_challenge(ctx, now, now_pinned=now_pinned)
    if row is None:
        await _send_no_active_challenge(recorder, user)
        return _transcript("guess", recorder, {"challenge": None, "user": None})
    view = _view_for(ctx, row.id, now)
    interaction = FakeInteraction(recorder, user)
    await press_button(view, "songbot:guess", interaction)
    modal = recorder.payloads[-1].modal
    if not isinstance(modal, GuessModal):  # defensive: the button always sends one
        raise RuntimeError("guess button did not open a GuessModal")
    await submit_modal_text(modal, modal.guess, text, interaction)
    return _transcript(
        "guess",
        recorder,
        {
            "challenge": {"id": row.id, "date": row.date},
            "user": _user_state(ctx, row.id, user.id),
        },
    )


async def scenario_leaderboard(
    ctx: HarnessContext, user: FakeUser, now: datetime, *, now_pinned: bool
) -> dict[str, Any]:
    """Press the REAL Leaderboard button (works even with no challenge yet)."""
    recorder = Recorder()
    row = _resolve_challenge(ctx, now, now_pinned=now_pinned)
    # The leaderboard callback never touches the challenge id, so a placeholder
    # keeps the real callback drivable before any challenge exists (VAL-CROSS-017).
    view = _view_for(ctx, row.id if row is not None else 0, now)
    await press_button(view, "songbot:leaderboard", FakeInteraction(recorder, user))
    entries = ctx.engine.leaderboard(ctx.guild_id, now)
    return _transcript(
        "leaderboard", recorder, {"entries": [asdict(entry) for entry in entries]}
    )


async def scenario_advance_day(
    ctx: HarnessContext,
    now: datetime,
    *,
    record_post: RecordPost | None = None,
    record_reveal: RecordReveal | None = None,
) -> dict[str, Any]:
    """Reveal the previous challenge, then post the next day's challenge.

    State-anchored (pinned #1 sugar): the target is the day AFTER the guild's
    latest challenge at the configured post time; from an empty state it is
    the local date of ``now``. The reveal is delivery-coupled (pinned #17):
    peek (read-only) -> send the reveal announcement -> mark revealed -> then
    post, with the reveal payload recorded BEFORE the new daily post payload
    (pinned #3). A failed reveal send mutates NOTHING (the previous challenge
    stays active, no new post is delivered) and prints
    ``{"error": "reveal_failed", ...}`` (exit 1); the retry re-attempts the
    reveal before posting. A failed daily-post send rolls back the
    just-created challenge (pinned #16) and prints
    ``{"error": "post_failed", ...}`` (exit 1); the already-sent reveal is
    idempotent, so the retry records exactly one new daily post.
    """
    record = record_post if record_post is not None else _record_daily_post
    send_reveal = record_reveal if record_reveal is not None else _record_reveal
    recorder = Recorder()
    target = _advance_target(ctx, now)
    reveal = ctx.engine.peek_reveal(ctx.guild_id, target)
    reveal_state: dict[str, Any] | None = None
    if reveal is not None:
        try:
            send_reveal(recorder, ctx, reveal)
        except Exception as exc:
            # Pinned #17: zero mutation — the challenge stays active and the
            # next advance-day retries the reveal BEFORE posting.
            return {"error": "reveal_failed", "message": str(exc)}
        # Mark revealed ONLY after the reveal send succeeded (pinned #17).
        ctx.engine.mark_revealed(ctx.guild_id, target)
        reveal_state = {
            "challenge_id": reveal.challenge_id,
            "date": reveal.date,
            "winners": len(reveal.winners),
        }
    try:
        challenge = ctx.engine.ensure_today_challenge(ctx.guild_id, ctx.channel_id, target)
    except CatalogEmptyError:
        return {"error": "catalog_empty"}
    if challenge.created:
        try:
            record(recorder, ctx, challenge, target)
        except Exception as exc:
            # Pinned #16: roll back the just-created challenge (never a
            # pre-existing row) so the retry reposts the identical challenge.
            ctx.engine.delete_challenge(challenge.id)
            return {"error": "post_failed", "message": str(exc)}
    return _transcript(
        "advance-day",
        recorder,
        {"challenge": _challenge_state(challenge), "reveal": reveal_state},
    )


def _advance_target(ctx: HarnessContext, now: datetime) -> datetime:
    """The next day's post datetime: latest challenge date + 1 day at post time."""
    settings = ctx.settings
    row = ctx.db.query_one(
        "SELECT MAX(date) AS latest FROM challenges WHERE guild_id = ?",
        (ctx.guild_id,),
    )
    latest_raw = row["latest"] if row is not None else None
    if latest_raw is None:
        target_date = _local_date(settings, now)
    else:
        target_date = date.fromisoformat(str(latest_raw)) + timedelta(days=1)
    hour, minute = settings.daily_post_time.split(":")
    return datetime.combine(target_date, time(int(hour), int(minute)), tzinfo=settings.tz)


# -- admin scenarios ---------------------------------------------------------------
#
# Each drives the REAL AdminCommands body (songbot/bot/admin.py) with a
# FakeInteraction whose user carries the simulated Manage-Guild permission —
# the bodies run the identical `has_manage_guild` check the live discord.py
# commands use (VAL-ADMIN-009).


def _admin_user(spec: str, *, as_admin: bool) -> FakeUser:
    """The invoking admin user with the simulated Manage-Guild permission."""
    base = parse_user(spec)
    return FakeUser(
        id=base.id,
        name=base.name,
        guild_permissions=FakePermissions(manage_guild=as_admin),
    )


def _admin_commands(
    ctx: HarnessContext,
    recorder: Recorder,
    now: datetime,
    *,
    record_post: RecordPost | None = None,
) -> AdminCommands:
    """The REAL admin command bodies with the harness post sender injected.

    The sender records the daily post exactly like ``post`` does — built with
    the REAL embed/view/attachment builders, only the transport is recorded.
    ``record_post`` is the pinned-#16 failure-injection seam for tests.
    """
    record = record_post if record_post is not None else _record_daily_post

    async def send_daily_post(challenge: Challenge) -> None:
        record(recorder, ctx, challenge, now)

    async def post_announcement(channel_id: str, content: str, emoji: str) -> str:
        """The harness's /songbot-pingrole transport: record, return a stable id."""
        recorder.record_message(kind="announcement", content=content)
        return HARNESS_PING_MESSAGE_ID

    return AdminCommands(
        ctx.engine,
        ctx.settings,
        post_sender=send_daily_post,
        announcement_poster=post_announcement,
        clock=lambda: now,
    )


def _today_challenge_state(ctx: HarnessContext, now: datetime) -> dict[str, Any] | None:
    """The challenge row for the local date of ``now`` as JSON (or None).

    NEVER includes song title/artist (secrecy, pinned #9) — ``status`` is the
    only harness surface that exposes song identity (pinned #2).
    """
    row = ctx.db.query_one(
        "SELECT * FROM challenges WHERE guild_id = ? AND date = ?",
        (ctx.guild_id, _local_date(ctx.settings, now).isoformat()),
    )
    if row is None:
        return None
    challenge = ChallengeRow.from_row(row)
    return {
        "id": challenge.id,
        "date": challenge.date,
        "status": challenge.status,
        "song_id": challenge.song_id,
        "snippet_offset_sec": challenge.snippet_offset_sec,
        "skip_count": challenge.skip_count,
    }


async def scenario_admin_post(
    ctx: HarnessContext,
    user: FakeUser,
    now: datetime,
    *,
    record_post: RecordPost | None = None,
) -> dict[str, Any]:
    """Drive the REAL /songbot-post body (VAL-ADMIN-001/002/009).

    Success: one ``channel`` payload (the daily post, same shape as the
    scheduled post) plus one ephemeral ack to the invoker. A same-day repeat
    prints the pinned-#4 compact form — the body still sends its ephemeral
    already-posted ack (the live client needs it), but the CLI output mirrors
    ``post``: ``{"already_posted": true, "messages": []}`` (VAL-CROSS-015).
    An empty catalog prints the pinned-#11 error. A failed send of the
    just-created challenge (pinned #16) rolls it back and prints
    ``{"error": "post_failed", ...}`` (exit 1). A non-admin invocation
    records exactly one ephemeral denial and mutates nothing.
    """
    recorder = Recorder()
    commands = _admin_commands(ctx, recorder, now, record_post=record_post)
    # The fake is duck-typed, not an Interaction subclass — same cast the
    # button/modal drivers use (see fakes.press_button).
    interaction = cast(
        "discord.Interaction[Any]",
        FakeInteraction(recorder, user, guild_id=ctx.guild_id),
    )
    result = await commands.post_now(interaction)
    if result.outcome == "already_posted":
        return {"already_posted": True, "messages": []}
    if result.outcome == "catalog_empty":
        return {"error": "catalog_empty"}
    if result.outcome == "error":
        return {"error": "post_failed", "message": result.error}
    return _transcript(
        "admin-post",
        recorder,
        {"outcome": result.outcome, "challenge": _today_challenge_state(ctx, now)},
    )


async def scenario_admin_setup(
    ctx: HarnessContext, user: FakeUser, channel_id: str, now: datetime
) -> dict[str, Any]:
    """Drive the REAL /songbot-setup body (multi-guild configuration).

    Success: exactly one ephemeral ack naming the channel mention, and the
    guild's ``guild_settings`` row is upserted (``state.guild_settings``
    carries the stored row). A non-admin invocation records exactly one
    ephemeral denial and mutates nothing (VAL-ADMIN-009).
    """
    recorder = Recorder()
    commands = _admin_commands(ctx, recorder, now)
    interaction = cast(
        "discord.Interaction[Any]",
        FakeInteraction(recorder, user, guild_id=ctx.guild_id),
    )
    result = await commands.setup_channel(
        interaction, channel_id, channel_mention=f"#{channel_id}"
    )
    row = ctx.engine.guild_settings(ctx.guild_id)
    return _transcript(
        "admin-setup",
        recorder,
        {
            "outcome": result.outcome,
            "guild_settings": (
                {
                    "guild_id": row.guild_id,
                    "channel_id": row.channel_id,
                    "set_by": row.set_by,
                }
                if row is not None
                else None
            ),
        },
    )


async def scenario_admin_skip(
    ctx: HarnessContext, user: FakeUser, now: datetime
) -> dict[str, Any]:
    """Drive the REAL /songbot-skip body (pinned #5, VAL-ADMIN-003..006/009).

    Success: exactly one ephemeral ack (never a channel/announcement payload)
    after the engine deleted + recreated today's challenge. Refusals
    (``no_challenge``/``revealed``/``solved``) record one ephemeral refusal
    with zero mutation; ``state.reason`` carries the machine-readable cause.
    """
    recorder = Recorder()
    commands = _admin_commands(ctx, recorder, now)
    interaction = cast(
        "discord.Interaction[Any]",
        FakeInteraction(recorder, user, guild_id=ctx.guild_id),
    )
    result = await commands.skip_song(interaction)
    return _transcript(
        "admin-skip",
        recorder,
        {
            "outcome": result.outcome,
            "reason": result.reason,
            "challenge": _today_challenge_state(ctx, now),
        },
    )


async def scenario_admin_reload(
    ctx: HarnessContext, user: FakeUser, now: datetime
) -> dict[str, Any]:
    """Drive the REAL /songbot-reload body (VAL-ADMIN-007/008/009).

    The ephemeral ack and ``state.sources`` both report the per-source
    summary (added/updated/removed/retained, or the source's error).
    """
    recorder = Recorder()
    commands = _admin_commands(ctx, recorder, now)
    interaction = cast(
        "discord.Interaction[Any]",
        FakeInteraction(recorder, user, guild_id=ctx.guild_id),
    )
    result = await commands.reload_catalog(interaction)
    sources = [
        {
            "source": source.source,
            "added": source.added,
            "updated": source.updated,
            "removed": source.removed,
            "retained": source.retained,
            "error": source.error,
        }
        for source in (result.refresh.sources if result.refresh is not None else ())
    ]
    return _transcript(
        "admin-reload",
        recorder,
        {"outcome": result.outcome, "sources": sources, "counts": _counts(ctx)},
    )


async def scenario_admin_fixsong(
    ctx: HarnessContext,
    user: FakeUser,
    title: str | None,
    artist: str | None,
    date: str | None,
    now: datetime,
) -> dict[str, Any]:
    """Drive the REAL /songbot-fixsong flow: show, press Edit, submit the modal.

    The command body answers with exactly one ephemeral payload carrying the
    current metadata and the REAL FixSongView — zero mutation, reported as
    ``state.target`` — then the scenario presses ``songbot:fixsong_edit``
    and submits the REAL FixSongModal, injecting ``title``/``artist`` where
    given and mirroring Discord's submit payload (the pre-filled default)
    where a flag is omitted. Success: a second ephemeral payload with the
    old -> new ack (the admin-only, ephemeral exception to the pinned-#9
    secrecy rule) and ``state.fix`` carries the same record. Refusals record
    exactly one ephemeral refusal with zero mutation —
    ``no_challenge``/``invalid_date`` at the command step, ``blank_title``
    at the modal submit; ``state.reason`` carries the machine-readable cause.
    """
    recorder = Recorder()
    commands = _admin_commands(ctx, recorder, now)
    fake = FakeInteraction(recorder, user, guild_id=ctx.guild_id)
    # The fake is duck-typed, not an Interaction subclass — same cast the
    # button/modal drivers use (see fakes.press_button).
    interaction = cast("discord.Interaction[Any]", fake)
    result = await commands.fix_song(interaction, date=date)
    target = result.target
    state: dict[str, Any] = {
        "outcome": result.outcome,
        "reason": result.reason,
        "target": (
            {
                "song_id": target.song_id,
                "challenge_id": target.challenge_id,
                "challenge_date": target.challenge_date,
                "title": target.title,
                "artist": target.artist,
            }
            if target is not None
            else None
        ),
        "fix": None,
    }
    if result.outcome != "shown":
        return _transcript("admin-fixsong", recorder, state)
    view = recorder.payloads[-1].view
    if not isinstance(view, FixSongView):  # defensive: the body always sends one
        raise RuntimeError("fix_song did not answer with a FixSongView")
    await press_button(view, "songbot:fixsong_edit", fake)
    modal = recorder.payloads[-1].modal
    if not isinstance(modal, FixSongModal):  # defensive: the button always opens one
        raise RuntimeError("fixsong edit button did not open a FixSongModal")
    # Discord's submit payload carries every field: the edited text where the
    # admin typed, the pre-filled default where the field was left untouched.
    modal.title_input._value = (
        title if title is not None else (modal.title_input.default or "")
    )
    modal.artist_input._value = (
        artist if artist is not None else (modal.artist_input.default or "")
    )
    await modal.on_submit(interaction)
    fix = modal.result
    state["outcome"] = "fixed" if fix is not None else "refused"
    state["reason"] = modal.refused_reason
    state["fix"] = (
        {
            "song_id": fix.song_id,
            "challenge_id": fix.challenge_id,
            "challenge_date": fix.challenge_date,
            "old_title": fix.old_title,
            "old_artist": fix.old_artist,
            "new_title": fix.new_title,
            "new_artist": fix.new_artist,
        }
        if fix is not None
        else None
    )
    return _transcript("admin-fixsong", recorder, state)


async def scenario_admin_pingrole(
    ctx: HarnessContext,
    user: FakeUser,
    role_id: str,
    emoji: str,
    now: datetime,
) -> dict[str, Any]:
    """Drive the REAL /songbot-pingrole body (reaction-role opt-in).

    Success: one ``announcement`` payload (the opt-in message the live client
    posts to the configured channel) plus one ephemeral ack, and the guild's
    ``ping_role_settings`` row is upserted (``state.ping_role`` carries the
    stored row; the harness poster returns the deterministic
    ``harness-ping-announcement`` message id). A non-admin invocation records
    exactly one ephemeral denial and mutates nothing (VAL-ADMIN-009).
    """
    recorder = Recorder()
    commands = _admin_commands(ctx, recorder, now)
    interaction = cast(
        "discord.Interaction[Any]",
        FakeInteraction(recorder, user, guild_id=ctx.guild_id),
    )
    result = await commands.setup_ping_role(
        interaction,
        role_id=role_id,
        role_mention=f"<@&{role_id}>",
        emoji=emoji,
    )
    row = ctx.engine.ping_role_settings(ctx.guild_id)
    return _transcript(
        "admin-pingrole",
        recorder,
        {
            "outcome": result.outcome,
            "error": result.error,
            "ping_role": (
                {
                    "guild_id": row.guild_id,
                    "channel_id": row.channel_id,
                    "message_id": row.message_id,
                    "role_id": row.role_id,
                    "emoji": row.emoji,
                }
                if row is not None
                else None
            ),
        },
    )


def scenario_reset(ctx: HarnessContext) -> dict[str, Any]:
    """Wipe every data table and the snippet cache (schema/migrations survive).

    ``guild_settings`` is wiped too (full clean slate); the next harness run
    re-seeds the harness guild's row at context build.
    """
    with ctx.db.transaction():
        for table in (
            "guesses",
            "challenge_users",
            "challenges",
            "user_stats",
            "songs",
            "song_overrides",
            "ping_role_settings",
            "guild_settings",
        ):
            ctx.db.execute(f"DELETE FROM {table}")
    cache_dir = ctx.settings.snippet_cache_dir
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return _transcript("reset", Recorder(), {"reset": True, "counts": _counts(ctx)})


def scenario_status(ctx: HarnessContext, now: datetime) -> dict[str, Any]:
    """Today's date, challenge status, song identity (test-only), counts, leaderboard."""
    guild_id = ctx.guild_id
    today = _local_date(ctx.settings, now).isoformat()
    challenge_state: dict[str, Any] | None = None
    row = ctx.db.query_one(
        "SELECT * FROM challenges WHERE guild_id = ? AND date = ?", (guild_id, today)
    )
    if row is not None:
        challenge = ChallengeRow.from_row(row)
        song = ctx.db.query_one(
            "SELECT title, artist FROM songs WHERE id = ?", (challenge.song_id,)
        )
        challenge_state = {
            "id": challenge.id,
            "date": challenge.date,
            "status": challenge.status,
            "song_id": challenge.song_id,
            "snippet_offset_sec": challenge.snippet_offset_sec,
            # status is the pinned-#2 test-only surface that exposes the song.
            "song": (
                {"title": song["title"], "artist": song["artist"]}
                if song is not None
                else None
            ),
        }
    entries = ctx.engine.leaderboard(guild_id, now)
    return _transcript(
        "status",
        Recorder(),
        {
            "date": today,
            "challenge": challenge_state,
            "counts": _counts(ctx),
            "leaderboard": [asdict(entry) for entry in entries],
        },
    )


async def scenario_serve(ctx: HarnessContext) -> int:
    """Run ONLY the health endpoint (mode="harness") until SIGINT/SIGTERM."""
    return await serve_health(
        host="127.0.0.1",
        port=ctx.settings.health_port,
        mode="harness",
        guild_id=ctx.guild_id,
    )


# -- entrypoint ---------------------------------------------------------------------


def _harness_environ() -> dict[str, str]:
    """The environment for `load_settings`, with a placeholder Discord token.

    The harness never constructs a Discord client, so the token is never
    used; substituting a placeholder when the environment provides none keeps
    a real token out of the Settings object entirely.
    """
    env = dict(os.environ)
    env.setdefault("DISCORD_BOT_TOKEN", "harness-unused-token")
    return env


async def _dispatch(
    args: argparse.Namespace, ctx: HarnessContext, now: datetime
) -> dict[str, Any]:
    now_pinned: bool = args.now is not None
    if args.scenario == "post":
        return await scenario_post(ctx, now)
    if args.scenario == "hear-more":
        return await scenario_hear_more(
            ctx, parse_user(args.user), args.times, now, now_pinned=now_pinned
        )
    if args.scenario == "guess":
        return await scenario_guess(
            ctx, parse_user(args.user), args.text, now, now_pinned=now_pinned
        )
    if args.scenario == "leaderboard":
        return await scenario_leaderboard(ctx, parse_user(args.user), now, now_pinned=now_pinned)
    if args.scenario == "advance-day":
        return await scenario_advance_day(ctx, now)
    if args.scenario == "admin-post":
        return await scenario_admin_post(
            ctx, _admin_user(args.user, as_admin=args.as_admin), now
        )
    if args.scenario == "admin-skip":
        return await scenario_admin_skip(
            ctx, _admin_user(args.user, as_admin=args.as_admin), now
        )
    if args.scenario == "admin-reload":
        return await scenario_admin_reload(
            ctx, _admin_user(args.user, as_admin=args.as_admin), now
        )
    if args.scenario == "admin-fixsong":
        return await scenario_admin_fixsong(
            ctx,
            _admin_user(args.user, as_admin=args.as_admin),
            args.title,
            args.artist,
            args.date,
            now,
        )
    if args.scenario == "admin-setup":
        return await scenario_admin_setup(
            ctx,
            _admin_user(args.user, as_admin=args.as_admin),
            args.channel if args.channel is not None else ctx.channel_id,
            now,
        )
    if args.scenario == "admin-pingrole":
        return await scenario_admin_pingrole(
            ctx,
            _admin_user(args.user, as_admin=args.as_admin),
            args.role,
            args.emoji,
            now,
        )
    if args.scenario == "reset":
        return scenario_reset(ctx)
    if args.scenario == "status":
        return scenario_status(ctx, now)
    raise UsageError(f"unknown scenario {args.scenario!r}")  # argparse prevents this


def _emit(out: dict[str, Any]) -> int:
    """Print the scenario output and return the process exit code.

    The pinned exact-shape outputs (``already_posted``, errors) print compact;
    transcripts pretty-print for human inspection. All JSON is UTF-8
    (``ensure_ascii=False``) so embed emoji survive raw greps too.
    """
    if "error" in out:
        print(json.dumps(out, ensure_ascii=False))
        return 1
    if out.get("already_posted"):
        print(json.dumps(out, ensure_ascii=False))
        return 0
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint: parse args, build the real stack, run one scenario."""
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(environ=_harness_environ())
    except ConfigError as exc:
        return _emit({"error": "config_invalid", "message": str(exc)})
    try:
        now = parse_now(args.now)
    except UsageError as exc:
        print(json.dumps({"error": "usage", "message": str(exc)}, ensure_ascii=False))
        return 2
    ctx = HarnessContext.from_settings(settings)
    try:
        if args.scenario == "serve":
            return asyncio.run(scenario_serve(ctx))
        try:
            out = asyncio.run(_dispatch(args, ctx, now))
        except CatalogEmptyError:
            out = {"error": "catalog_empty"}
        except SnippetError as exc:
            out = {"error": "snippet_error", "message": str(exc)}
        except EngineError as exc:
            out = {"error": "engine_error", "message": str(exc)}
    finally:
        ctx.close()
    return _emit(out)
