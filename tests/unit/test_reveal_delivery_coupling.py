"""Unit tests for the delivery-coupled reveal (pinned #17, VAL-DAILY-014).

The reveal-side twin of pinned #16 (test_post_delivery_rollback.py). The old
``get_reveal`` marked the previous challenge revealed BEFORE the reveal
announcement was sent, so a failed reveal send permanently dropped the
previous song's reveal. The engine now splits it into a read-only
`peek_reveal` (compute the stale challenge's `Reveal` — song + winners —
with zero mutation) and a `mark_revealed` mutation the caller applies ONLY
after the reveal send succeeds. Both reveal flows — the scheduler tick
(songbot/bot/client.py) and the harness ``advance-day``
(songbot/harness/cli.py) — run peek -> send reveal -> mark revealed -> post.

VAL-DAILY-014: (1) a failed reveal send leaves the previous challenge ACTIVE
(zero mutation), surfaces the error, and does not deliver the new post;
(2) the next tick/advance-day retries and delivers the reveal announcement
BEFORE the new post, exactly once, and only then marks the challenge
revealed; (3) a delivered reveal is never re-sent on later ticks.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from songbot.bot import client as client_module
from songbot.bot.client import SongBotClient
from songbot.db import Database
from songbot.engine import Challenge, GameEngine, Reveal
from songbot.harness.cli import (
    HarnessContext,
    _emit,
    _record_reveal,
    scenario_advance_day,
    scenario_guess,
    scenario_post,
)
from songbot.harness.fakes import FakeUser, Recorder
from tests.unit.test_client import DAY1_PM, DAY2_NOON, _make_stack, _Stack
from tests.unit.test_engine_daily import (
    _challenge_count,
    _db_snapshot,
    _make_engine,
    _settings,
)
from tests.unit.test_engine_gameplay import _add_song

DAY1 = datetime(2026, 8, 13, 16, 0, 0, tzinfo=UTC)  # 2026-08-13 13:00 ADT (post due)
DAY2 = datetime(2026, 8, 14, 16, 0, 0, tzinfo=UTC)  # 2026-08-14 13:00 ADT
DAY3_NOON = datetime(2026, 8, 15, 15, 0, 0, tzinfo=UTC)  # 2026-08-15 12:00 ADT exactly

DAY1_DATE = "2026-08-13"
DAY2_DATE = "2026-08-14"
DAY3_DATE = "2026-08-15"

ALICE = FakeUser(id="alice", name="alice")


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    database = Database.open(tmp_path / "songbot.db")
    yield database
    database.close()


def _kinds(out: dict[str, Any]) -> list[str]:
    return [p["kind"] for p in out["payloads"]]


def _challenge_status(db: Database, date_str: str) -> tuple[str, str | None]:
    row = db.query_one(
        "SELECT status, revealed_at FROM challenges WHERE date = ?", (date_str,)
    )
    assert row is not None
    return str(row["status"]), row["revealed_at"]


def _song_for_date(db: Database, date_str: str) -> tuple[str, str]:
    row = db.query_one(
        "SELECT s.title, s.artist FROM challenges c"
        " JOIN songs s ON s.id = c.song_id WHERE c.date = ?",
        (date_str,),
    )
    assert row is not None
    return str(row["title"]), str(row["artist"])


class _FlakyRevealSender:
    """Scheduler post/reveal transport double: fails reveals until switched off."""

    def __init__(self) -> None:
        self.fail_reveal = True
        self.reveal_attempts: list[Reveal] = []
        self.post_attempts: list[Challenge] = []
        self.events: list[tuple[str, Any]] = []
        # Optional hook run inside post() — proves the reveal was marked
        # BEFORE the new post send (e.g. reads the previous row's status).
        self.post_probe: Callable[[], str] | None = None
        self.probes: list[str] = []

    async def post(self, challenge: Challenge) -> None:
        self.post_attempts.append(challenge)
        if self.post_probe is not None:
            self.probes.append(self.post_probe())
        self.events.append(("post", challenge))

    async def reveal(self, reveal: Reveal) -> None:
        self.reveal_attempts.append(reveal)
        if self.fail_reveal:
            raise RuntimeError("simulated reveal send failure")
        self.events.append(("reveal", reveal))


class _FlakyRecordReveal:
    """Harness record_reveal seam: fails until switched off, records attempts."""

    def __init__(self) -> None:
        self.fail = True
        self.attempts: list[Reveal] = []

    def __call__(self, recorder: Recorder, ctx: HarnessContext, reveal: Reveal) -> None:
        self.attempts.append(reveal)
        if self.fail:
            raise RuntimeError("simulated reveal send failure")
        _record_reveal(recorder, ctx, reveal)


def _client(stack: _Stack, sender: _FlakyRevealSender) -> SongBotClient:
    """A client wired for one-tick-at-a-time scheduler drives (no setup_hook)."""
    return SongBotClient(
        stack.settings,
        stack.db,
        stack.engine,
        clock=lambda: DAY1_PM,
        post_sender=sender.post,
        reveal_sender=sender.reveal,
    )


def _ctx(db: Database, engine: GameEngine, tmp_path: Path) -> HarnessContext:
    return HarnessContext(settings=_settings(tmp_path), db=db, engine=engine)


class TestEngineRetrySemantics:
    """The engine heart of VAL-DAILY-014: a failed send changes nothing, so
    the retry peeks the identical reveal; only a delivered reveal is marked."""

    def test_failed_send_then_retry_peeks_identical_reveal_then_marks(
        self, db: Database, tmp_path: Path
    ) -> None:
        _add_song(db)
        _add_song(db, "song-2")
        engine, _ = _make_engine(tmp_path, db)
        day1 = engine.ensure_today_challenge("g1", "c1", DAY1)
        before = _db_snapshot(db)

        first = engine.peek_reveal("g1", DAY2)  # send "fails": no mark_revealed
        assert first is not None
        assert _db_snapshot(db) == before  # zero mutation from the peek

        retry = engine.peek_reveal("g1", DAY2)  # the retry re-peeks
        assert retry == first  # identical challenge, song, winners, revealed_at
        assert _challenge_status(db, DAY1_DATE) == ("active", None)

        engine.mark_revealed("g1", DAY2)  # only after the (retried) send succeeds
        assert _challenge_status(db, DAY1_DATE)[0] == "revealed"
        assert engine.peek_reveal("g1", DAY2) is None  # never computed again
        assert day1.id == first.challenge_id


class TestSchedulerRevealPath:
    """client._scheduler_tick with an injected failing reveal sender."""

    async def test_failed_reveal_send_leaves_challenge_active_and_skips_post(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        stack = _make_stack(tmp_path)
        try:
            _add_song(stack.db, "song-1")
            _add_song(stack.db, "song-2")
            sender = _FlakyRevealSender()
            sender.fail_reveal = False  # day 1 has nothing to reveal anyway
            client = _client(stack, sender)
            await client._scheduler_tick(DAY1_PM)  # day-1 post delivered
            assert [kind for kind, _ in sender.events] == ["post"]

            sender.fail_reveal = True
            before = _db_snapshot(stack.db)
            with caplog.at_level(logging.ERROR, logger="songbot.bot.client"):
                delay = await client._scheduler_tick(DAY2_NOON)

            # The error surfaces via the existing 60s retry backoff + log.
            assert delay == client_module.RETRY_DELAY_SEC
            assert any("daily post" in r.message.lower() for r in caplog.records)
            # The reveal send was attempted; NOTHING was delivered (no reveal,
            # and the new day's post was never even attempted).
            assert len(sender.reveal_attempts) == 1
            assert [kind for kind, _ in sender.events] == ["post"]
            assert len(sender.post_attempts) == 1
            # Zero mutation: day-1 NOT marked revealed, day-2 never created.
            assert _db_snapshot(stack.db) == before
            assert _challenge_status(stack.db, DAY1_DATE) == ("active", None)
            assert _challenge_count(stack.db) == 1
        finally:
            stack.db.close()

    async def test_retry_delivers_reveal_before_post_and_marks_exactly_once(
        self, tmp_path: Path
    ) -> None:
        stack = _make_stack(tmp_path)
        try:
            _add_song(stack.db, "song-1")
            _add_song(stack.db, "song-2")
            sender = _FlakyRevealSender()
            sender.fail_reveal = False
            client = _client(stack, sender)
            sender.post_probe = lambda: _challenge_status(stack.db, DAY1_DATE)[0]
            await client._scheduler_tick(DAY1_PM)

            sender.fail_reveal = True
            await client._scheduler_tick(DAY2_NOON)  # reveal send fails
            assert _challenge_status(stack.db, DAY1_DATE) == ("active", None)

            sender.fail_reveal = False
            delay = await client._scheduler_tick(DAY2)  # the retry tick (13:00 ADT)

            # The day was NOT suppressed: retry posted on the normal cadence.
            assert delay == client_module.MAX_SLEEP_SEC
            # Reveal delivered BEFORE the new post, exactly once.
            assert [kind for kind, _ in sender.events] == ["post", "reveal", "post"]
            _, reveal = sender.events[1]
            assert reveal.date == DAY1_DATE
            assert len(sender.reveal_attempts) == 2  # one failed + one delivered
            # ...and only THEN marked revealed: the day-2 post send already
            # observed day-1 as revealed (mark happens between the two sends).
            assert sender.probes == ["active", "revealed"]
            status, revealed_at = _challenge_status(stack.db, DAY1_DATE)
            assert status == "revealed"
            assert revealed_at is not None
            assert _challenge_status(stack.db, DAY2_DATE)[0] == "active"
            assert _challenge_count(stack.db) == 2
        finally:
            stack.db.close()

    async def test_delivered_reveal_is_never_resent_on_later_ticks(
        self, tmp_path: Path
    ) -> None:
        stack = _make_stack(tmp_path)
        try:
            for i in range(1, 4):
                _add_song(stack.db, f"song-{i}")
            sender = _FlakyRevealSender()
            sender.fail_reveal = False
            client = _client(stack, sender)
            await client._scheduler_tick(DAY1_PM)
            await client._scheduler_tick(DAY2_NOON)  # reveals day 1, posts day 2

            await client._scheduler_tick(DAY2_NOON)  # same day: gated, nothing
            assert [kind for kind, _ in sender.events] == ["post", "reveal", "post"]

            await client._scheduler_tick(DAY3_NOON)  # reveals day 2, posts day 3

            assert [kind for kind, _ in sender.events] == [
                "post",
                "reveal",
                "post",
                "reveal",
                "post",
            ]
            reveals = [r for kind, r in sender.events if kind == "reveal"]
            # Each challenge's reveal was delivered exactly once — day 1's
            # delivered reveal was never re-sent.
            assert [r.date for r in reveals] == [DAY1_DATE, DAY2_DATE]
            assert _challenge_status(stack.db, DAY1_DATE)[0] == "revealed"
            assert _challenge_status(stack.db, DAY2_DATE)[0] == "revealed"
            assert _challenge_status(stack.db, DAY3_DATE)[0] == "active"
        finally:
            stack.db.close()


class TestHarnessAdvanceDayRevealPath:
    """Harness advance-day with an injected failing record_reveal seam."""

    async def test_failed_reveal_send_emits_error_json_with_zero_mutation(
        self, db: Database, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _add_song(db)
        _add_song(db, "song-2", title="Digital Horizon", artist="Quantum Drift",
                  raw_title="Quantum Drift - Digital Horizon")
        engine, fake = _make_engine(tmp_path, db)
        ctx = _ctx(db, engine, tmp_path)
        await scenario_post(ctx, DAY1)  # day 1 delivered normally
        before = _db_snapshot(db)
        ensure_calls_before = len(fake.ensure_calls)
        seam = _FlakyRecordReveal()

        out = await scenario_advance_day(ctx, DAY2, record_reveal=seam)

        # Clean error JSON + non-zero exit (the pinned harness error shape).
        assert json.loads(json.dumps(out, ensure_ascii=False)) == out
        assert out["error"] == "reveal_failed"
        assert "simulated reveal send failure" in out["message"]
        assert _emit(out) == 1
        capsys.readouterr()
        # The reveal send was attempted exactly once.
        assert len(seam.attempts) == 1
        assert seam.attempts[0].date == DAY1_DATE
        # Zero mutation: day-1 NOT marked revealed; the new post was never
        # delivered and day-2's challenge was never even created.
        assert _db_snapshot(db) == before
        assert len(fake.ensure_calls) == ensure_calls_before
        assert _challenge_status(db, DAY1_DATE) == ("active", None)
        assert _challenge_count(db) == 1

    async def test_retry_delivers_reveal_before_post_exactly_once_then_marks(
        self, db: Database, tmp_path: Path
    ) -> None:
        _add_song(db)
        _add_song(db, "song-2", title="Digital Horizon", artist="Quantum Drift",
                  raw_title="Quantum Drift - Digital Horizon")
        engine, _ = _make_engine(tmp_path, db)
        ctx = _ctx(db, engine, tmp_path)
        posted = await scenario_post(ctx, DAY1)
        day1_id = posted["state"]["challenge"]["id"]
        # alice solves day 1 through the REAL view/modal flow, so the retried
        # reveal must carry her winners entry.
        title, artist = _song_for_date(db, DAY1_DATE)
        solved = await scenario_guess(ctx, ALICE, title, DAY1, now_pinned=True)
        assert solved["state"]["user"]["solved"] is True

        seam = _FlakyRecordReveal()
        failed = await scenario_advance_day(ctx, DAY2, record_reveal=seam)
        assert failed["error"] == "reveal_failed"
        assert _challenge_status(db, DAY1_DATE) == ("active", None)

        seam.fail = False
        out = await scenario_advance_day(ctx, DAY2, record_reveal=seam)

        # The reveal announcement is recorded BEFORE the new daily post,
        # exactly once (one announcement payload in the whole transcript).
        assert _kinds(out) == ["announcement", "channel"]
        assert out["state"]["reveal"] == {
            "challenge_id": day1_id,
            "date": DAY1_DATE,
            "winners": 1,
        }
        assert out["state"]["challenge"]["date"] == DAY2_DATE
        description = out["payloads"][0]["embed"]["description"]
        assert title in description
        assert artist in description
        assert "<@alice>" in description  # the winners summary survived the retry
        assert len(seam.attempts) == 2  # one failed + one delivered
        # ...and only then marked revealed.
        status, revealed_at = _challenge_status(db, DAY1_DATE)
        assert status == "revealed"
        assert revealed_at is not None
        assert _challenge_status(db, DAY2_DATE) == ("active", None)
        assert _challenge_count(db) == 2

    async def test_delivered_reveal_is_never_resent_on_later_advance_days(
        self, db: Database, tmp_path: Path
    ) -> None:
        _add_song(db)
        _add_song(db, "song-2", title="Digital Horizon", artist="Quantum Drift",
                  raw_title="Quantum Drift - Digital Horizon")
        _add_song(db, "song-3", title="Velvet Static", artist="Analog Hearts",
                  raw_title="Analog Hearts - Velvet Static")
        engine, _ = _make_engine(tmp_path, db)
        ctx = _ctx(db, engine, tmp_path)
        await scenario_post(ctx, DAY1)
        seam = _FlakyRecordReveal()
        seam.fail = False
        day2 = await scenario_advance_day(ctx, DAY2, record_reveal=seam)
        day2_id = day2["state"]["challenge"]["id"]
        assert _kinds(day2) == ["announcement", "channel"]  # day-1 reveal + post

        day3 = await scenario_advance_day(ctx, DAY3_NOON, record_reveal=seam)

        # Day 3 reveals DAY 2 — day 1's delivered reveal is not re-sent.
        assert _kinds(day3) == ["announcement", "channel"]
        assert day3["state"]["reveal"]["challenge_id"] == day2_id
        assert day3["state"]["reveal"]["date"] == DAY2_DATE
        assert day3["state"]["challenge"]["date"] == DAY3_DATE
        day1_title, _ = _song_for_date(db, DAY1_DATE)
        day2_title, day2_artist = _song_for_date(db, DAY2_DATE)
        description = day3["payloads"][0]["embed"]["description"]
        assert day2_title in description
        assert day2_artist in description
        assert day1_title not in description
        # Two advance-days, one reveal announcement each, distinct challenges.
        assert len(seam.attempts) == 2
        assert [r.date for r in seam.attempts] == [DAY1_DATE, DAY2_DATE]
        assert _challenge_status(db, DAY1_DATE)[0] == "revealed"
        assert _challenge_status(db, DAY2_DATE)[0] == "revealed"
        assert _challenge_status(db, DAY3_DATE)[0] == "active"
