"""Unit tests for delivery-coupled challenge creation (pinned #16, VAL-DAILY-013).

Every daily-post path — the scheduler tick (songbot/bot/client.py), the admin
/songbot-post body (songbot/bot/admin.py), and the harness ``post`` /
``advance-day`` / ``admin-post`` scenarios (songbot/harness/cli.py) — creates
the challenge row and THEN sends the channel payload. If the send fails for a
challenge THAT call created (``Challenge.created``), the path rolls back:
the row is deleted (`GameEngine.delete_challenge`) and its snippet cache is
purged, then the error surfaces (scheduler: the existing 60s retry backoff;
admin: an ephemeral error ack; harness: ``{"error": "post_failed"}`` + exit 1).
A retry recreates the IDENTICAL challenge (deterministic date+guild+skip_count
seed — same song/offset, skip_count unchanged) and delivers exactly one post.
A pre-existing challenge row is NEVER rolled back.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from songbot.bot import client as client_module
from songbot.bot.admin import AdminCommands
from songbot.bot.client import SongBotClient
from songbot.bot.embeds import ADMIN_POST_FAILED_MESSAGE, ADMIN_POST_SUCCESS_MESSAGE
from songbot.db import Database
from songbot.engine import Challenge, GameEngine, Reveal
from songbot.harness.cli import (
    HarnessContext,
    _emit,
    _record_daily_post,
    scenario_admin_post,
    scenario_advance_day,
    scenario_post,
)
from songbot.harness.fakes import FakePermissions, FakeUser, Recorder
from tests.unit.interaction_fakes import FakeInteraction
from tests.unit.test_client import DAY1_PM, _make_stack, _Stack
from tests.unit.test_engine_daily import _challenge_count, _make_engine, _settings
from tests.unit.test_engine_gameplay import _add_song

DAY1 = datetime(2026, 8, 13, 16, 0, 0, tzinfo=UTC)  # 2026-08-13 13:00 ADT
DAY2 = datetime(2026, 8, 14, 16, 0, 0, tzinfo=UTC)  # 2026-08-14 13:00 ADT
ADMIN_ID = 9001

TITLE = "Neon Skyline"
ARTIST = "Midnight Circuit"

ADMIN = FakeUser(id="admin", name="admin", guild_permissions=FakePermissions(manage_guild=True))


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    database = Database.open(tmp_path / "songbot.db")
    yield database
    database.close()


def _interaction(*, manage_guild: bool = True) -> FakeInteraction:
    return FakeInteraction.for_user(ADMIN_ID, "admin", manage_guild=manage_guild)


def _seed_guild(engine: GameEngine) -> None:
    """Configure guild-1's post channel (what /songbot-setup or the env
    bootstrap does live); the admin bodies look it up per interaction."""
    engine.set_guild_channel("guild-1", "channel-1", set_by="test", now=DAY1)


def _kinds(out: dict[str, Any]) -> list[str]:
    return [p["kind"] for p in out["payloads"]]


class _FlakyPoster:
    """Admin DailyPostSender double: fails until switched off, records attempts."""

    def __init__(self) -> None:
        self.fail = True
        self.attempts: list[Challenge] = []

    async def __call__(self, challenge: Challenge) -> None:
        self.attempts.append(challenge)
        if self.fail:
            raise RuntimeError("simulated send failure")


class _FlakySender:
    """Scheduler post/reveal transport double: fails posts until switched off."""

    def __init__(self) -> None:
        self.fail_post = True
        self.post_attempts: list[Challenge] = []
        self.events: list[tuple[str, Any]] = []

    async def post(self, challenge: Challenge) -> None:
        self.post_attempts.append(challenge)
        if self.fail_post:
            raise RuntimeError("simulated send failure")
        self.events.append(("post", challenge))

    async def reveal(self, reveal: Reveal) -> None:
        self.events.append(("reveal", reveal))


class _FlakyRecordPost:
    """Harness record_post seam: fails until switched off, records attempts."""

    def __init__(self) -> None:
        self.fail = True
        self.attempts: list[Challenge] = []

    def __call__(
        self, recorder: Recorder, ctx: HarnessContext, challenge: Challenge, now: datetime
    ) -> None:
        self.attempts.append(challenge)
        if self.fail:
            raise RuntimeError("simulated send failure")
        _record_daily_post(recorder, ctx, challenge, now)


def _client(stack: _Stack, sender: _FlakySender) -> SongBotClient:
    """A client wired for one-tick-at-a-time scheduler drives (no setup_hook)."""
    return SongBotClient(
        stack.settings,
        stack.db,
        stack.engine,
        clock=lambda: DAY1_PM,
        post_sender=sender.post,
        reveal_sender=sender.reveal,
    )


def _assert_identical_recreation(retry: Challenge, failed: Challenge) -> None:
    """The retry recreated the SAME challenge: deterministic date+guild+skip seed."""
    assert retry.song.id == failed.song.id
    assert retry.snippet_offset_sec == failed.snippet_offset_sec
    assert retry.skip_count == 0 == failed.skip_count
    assert retry.date == failed.date
    assert retry.created is True


class TestDeleteChallenge:
    """The engine rollback helper: delete the row + purge the snippet cache."""

    def test_deletes_row_and_purges_snippet_cache(
        self, db: Database, tmp_path: Path
    ) -> None:
        _add_song(db)
        engine, fake = _make_engine(tmp_path, db)
        challenge = engine.ensure_today_challenge("guild-1", "channel-1", DAY1)
        cache_dir = tmp_path / "snippets" / str(challenge.id)
        assert cache_dir.is_dir()

        engine.delete_challenge(challenge.id)

        assert _challenge_count(db) == 0
        assert fake.purged == [challenge.id]
        assert not cache_dir.exists()

    def test_ensure_after_delete_recreates_identical_challenge(
        self, db: Database, tmp_path: Path
    ) -> None:
        _add_song(db)
        engine, _ = _make_engine(tmp_path, db)
        first = engine.ensure_today_challenge("guild-1", "channel-1", DAY1)

        engine.delete_challenge(first.id)
        second = engine.ensure_today_challenge("guild-1", "channel-1", DAY1)

        _assert_identical_recreation(second, first)
        assert _challenge_count(db) == 1


class TestSchedulerPath:
    """client._scheduler_tick with an injected failing post sender."""

    async def test_failed_send_rolls_back_and_schedules_retry(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        stack = _make_stack(tmp_path)
        try:
            _add_song(stack.db, "song-1")
            sender = _FlakySender()
            client = _client(stack, sender)

            with caplog.at_level(logging.ERROR, logger="songbot.bot.client"):
                delay = await client._scheduler_tick(DAY1_PM)

            assert delay == client_module.RETRY_DELAY_SEC  # existing 60s backoff
            assert len(sender.post_attempts) == 1  # the send was attempted
            assert sender.events == []  # nothing was delivered
            # The just-created challenge was rolled back: row + cache gone.
            assert _challenge_count(stack.db) == 0
            attempted = sender.post_attempts[0]
            assert attempted.id in stack.snippets.purged
            assert not (tmp_path / "snippets" / str(attempted.id)).exists()
            assert any("daily post" in r.message.lower() for r in caplog.records)
        finally:
            stack.db.close()

    async def test_retry_after_failure_delivers_identical_challenge_once(
        self, tmp_path: Path
    ) -> None:
        stack = _make_stack(tmp_path)
        try:
            _add_song(stack.db, "song-1")
            sender = _FlakySender()
            client = _client(stack, sender)

            await client._scheduler_tick(DAY1_PM)
            assert _challenge_count(stack.db) == 0

            sender.fail_post = False
            delay = await client._scheduler_tick(DAY1_PM)  # the retry tick

            # The day was NOT suppressed: the retry posted on the normal cadence.
            assert delay == client_module.MAX_SLEEP_SEC
            assert [kind for kind, _ in sender.events] == ["post"]  # exactly one post
            assert len(sender.post_attempts) == 2
            _assert_identical_recreation(sender.post_attempts[1], sender.post_attempts[0])
            assert _challenge_count(stack.db) == 1
        finally:
            stack.db.close()

    async def test_preexisting_challenge_is_never_rolled_back(
        self, tmp_path: Path
    ) -> None:
        stack = _make_stack(tmp_path)
        try:
            _add_song(stack.db, "song-1")
            sender = _FlakySender()
            sender.fail_post = False
            client = _client(stack, sender)
            await client._scheduler_tick(DAY1_PM)
            assert _challenge_count(stack.db) == 1

            # A later same-day tick with a now-failing sender: the delivered
            # challenge gates the post off — no send attempt, no rollback.
            sender.fail_post = True
            await client._scheduler_tick(DAY1_PM)

            assert len(sender.post_attempts) == 1
            assert _challenge_count(stack.db) == 1
            assert stack.snippets.purged == []
        finally:
            stack.db.close()


class TestAdminPath:
    """AdminCommands.post_now with an injected failing post sender."""

    async def test_failed_send_rolls_back_and_acks_error(
        self, db: Database, tmp_path: Path
    ) -> None:
        _add_song(db)
        engine, fake = _make_engine(tmp_path, db)
        _seed_guild(engine)
        poster = _FlakyPoster()
        commands = AdminCommands(
            engine, _settings(tmp_path), clock=lambda: DAY1, post_sender=poster
        )

        interaction = _interaction()
        result = await commands.post_now(interaction)

        assert result.outcome == "error"
        assert result.error is not None
        assert "simulated send failure" in result.error
        assert len(poster.attempts) == 1
        # Rolled back: the challenge row and its snippet cache are gone.
        assert _challenge_count(db) == 0
        assert poster.attempts[0].id in fake.purged
        # Exactly one ephemeral error ack to the invoking admin.
        assert [p.kind for p in interaction.payloads] == ["ephemeral"]
        ack = interaction.payloads[0]
        assert ack.recipient == str(ADMIN_ID)
        assert ack.content == ADMIN_POST_FAILED_MESSAGE

    async def test_retry_after_failure_posts_identical_challenge(
        self, db: Database, tmp_path: Path
    ) -> None:
        _add_song(db)
        engine, _ = _make_engine(tmp_path, db)
        _seed_guild(engine)
        poster = _FlakyPoster()
        commands = AdminCommands(
            engine, _settings(tmp_path), clock=lambda: DAY1, post_sender=poster
        )

        failed = await commands.post_now(_interaction())
        assert failed.outcome == "error"
        assert _challenge_count(db) == 0

        poster.fail = False
        interaction = _interaction()
        result = await commands.post_now(interaction)

        assert result.outcome == "posted"
        assert len(poster.attempts) == 2  # exactly one retry send
        _assert_identical_recreation(poster.attempts[1], poster.attempts[0])
        assert _challenge_count(db) == 1
        assert [p.kind for p in interaction.payloads] == ["ephemeral"]
        assert interaction.payloads[0].content == ADMIN_POST_SUCCESS_MESSAGE

    async def test_preexisting_challenge_with_failing_sender_is_untouched(
        self, db: Database, tmp_path: Path
    ) -> None:
        _add_song(db)
        engine, fake = _make_engine(tmp_path, db)
        _seed_guild(engine)
        poster = _FlakyPoster()
        poster.fail = False
        commands = AdminCommands(
            engine, _settings(tmp_path), clock=lambda: DAY1, post_sender=poster
        )
        await commands.post_now(_interaction())
        assert _challenge_count(db) == 1

        # Repeat post after a delivered one: the sender is never invoked and
        # the pre-existing row is never rolled back (pinned #16).
        poster.fail = True
        interaction = _interaction()
        result = await commands.post_now(interaction)

        assert result.outcome == "already_posted"
        assert len(poster.attempts) == 1
        assert _challenge_count(db) == 1
        assert fake.purged == []
        assert [p.kind for p in interaction.payloads] == ["ephemeral"]

    async def test_error_ack_never_names_the_song_or_leaks_internals(
        self, db: Database, tmp_path: Path
    ) -> None:
        _add_song(db)  # Neon Skyline / Midnight Circuit
        engine, _ = _make_engine(tmp_path, db)
        _seed_guild(engine)
        poster = _FlakyPoster()
        commands = AdminCommands(
            engine, _settings(tmp_path), clock=lambda: DAY1, post_sender=poster
        )

        interaction = _interaction()
        await commands.post_now(interaction)

        content = interaction.payloads[0].content or ""
        assert TITLE.lower() not in content.lower()
        assert ARTIST.lower() not in content.lower()
        assert "simulated send failure" not in content


class TestHarnessPostPath:
    """Harness post/advance-day/admin-post with an injected failing record seam."""

    def _ctx(self, db: Database, engine: GameEngine, tmp_path: Path) -> HarnessContext:
        return HarnessContext(settings=_settings(tmp_path), db=db, engine=engine)

    async def test_failed_send_rolls_back_and_emits_clean_error_json(
        self, db: Database, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _add_song(db)
        engine, fake = _make_engine(tmp_path, db)
        ctx = self._ctx(db, engine, tmp_path)
        seam = _FlakyRecordPost()

        out = await scenario_post(ctx, DAY1, record_post=seam)

        # Clean error JSON + non-zero exit (pinned harness error shape).
        assert json.loads(json.dumps(out, ensure_ascii=False)) == out
        assert out["error"] == "post_failed"
        assert "simulated send failure" in out["message"]
        assert _emit(out) == 1
        capsys.readouterr()
        # Rolled back: row + snippet cache gone.
        assert _challenge_count(db) == 0
        assert len(seam.attempts) == 1
        assert seam.attempts[0].id in fake.purged

    async def test_retry_after_failure_delivers_identical_post(
        self, db: Database, tmp_path: Path
    ) -> None:
        _add_song(db)
        engine, _ = _make_engine(tmp_path, db)
        ctx = self._ctx(db, engine, tmp_path)
        seam = _FlakyRecordPost()

        failed = await scenario_post(ctx, DAY1, record_post=seam)
        assert failed["error"] == "post_failed"

        seam.fail = False
        out = await scenario_post(ctx, DAY1, record_post=seam)

        assert out["scenario"] == "post"
        assert _kinds(out) == ["channel"]  # exactly one delivered post
        assert len(seam.attempts) == 2
        _assert_identical_recreation(seam.attempts[1], seam.attempts[0])
        row = db.query_one("SELECT skip_count FROM challenges")
        assert row is not None
        assert row["skip_count"] == 0  # rollback is not a skip: seed unchanged
        assert _challenge_count(db) == 1

    async def test_repeat_post_with_failing_seam_never_rolls_back(
        self, db: Database, tmp_path: Path
    ) -> None:
        _add_song(db)
        engine, fake = _make_engine(tmp_path, db)
        ctx = self._ctx(db, engine, tmp_path)
        seam = _FlakyRecordPost()
        seam.fail = False
        await scenario_post(ctx, DAY1, record_post=seam)
        assert _challenge_count(db) == 1

        seam.fail = True
        out = await scenario_post(ctx, DAY1, record_post=seam)

        # Pinned #4 repeat: the seam is never invoked, the row is untouched.
        assert out == {"already_posted": True, "messages": []}
        assert len(seam.attempts) == 1
        assert _challenge_count(db) == 1
        assert fake.purged == []

    async def test_advance_day_failed_send_rolls_back_and_retries(
        self, db: Database, tmp_path: Path
    ) -> None:
        _add_song(db)
        _add_song(db, "song-2", title="Digital Horizon", artist="Quantum Drift",
                  raw_title="Quantum Drift - Digital Horizon")
        engine, fake = _make_engine(tmp_path, db)
        ctx = self._ctx(db, engine, tmp_path)
        await scenario_post(ctx, DAY1)  # day 1 delivered normally

        seam = _FlakyRecordPost()
        failed = await scenario_advance_day(ctx, DAY2, record_post=seam)

        assert failed["error"] == "post_failed"
        # Day-2 challenge rolled back; day-1 row survives (already revealed).
        rows = db.query("SELECT date, status FROM challenges ORDER BY date")
        assert [(row["date"], row["status"]) for row in rows] == [
            ("2026-08-13", "revealed")
        ]
        assert seam.attempts[0].id in fake.purged

        seam.fail = False
        out = await scenario_advance_day(ctx, DAY2, record_post=seam)

        # The reveal already went out on the failed attempt: the retry records
        # exactly ONE new payload — the day-2 daily post (no double reveal).
        assert _kinds(out) == ["channel"]
        assert out["state"]["challenge"]["date"] == "2026-08-14"
        assert len(seam.attempts) == 2
        _assert_identical_recreation(seam.attempts[1], seam.attempts[0])
        assert _challenge_count(db) == 2

    async def test_admin_post_failed_send_rolls_back_and_emits_error_json(
        self, db: Database, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _add_song(db)
        engine, fake = _make_engine(tmp_path, db)
        ctx = self._ctx(db, engine, tmp_path)
        seam = _FlakyRecordPost()

        out = await scenario_admin_post(ctx, ADMIN, DAY1, record_post=seam)

        assert out["error"] == "post_failed"
        assert "simulated send failure" in out["message"]
        assert _emit(out) == 1
        capsys.readouterr()
        assert _challenge_count(db) == 0
        assert seam.attempts[0].id in fake.purged

        seam.fail = False
        retried = await scenario_admin_post(ctx, ADMIN, DAY1, record_post=seam)

        assert retried["scenario"] == "admin-post"
        assert retried["state"]["outcome"] == "posted"
        assert _kinds(retried) == ["channel", "ephemeral"]
        assert len(seam.attempts) == 2
        _assert_identical_recreation(seam.attempts[1], seam.attempts[0])
        assert _challenge_count(db) == 1
