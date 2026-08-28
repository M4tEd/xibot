"""Unit tests for the headless harness: fakes, scenario drivers, JSON output.

The harness scenarios drive the REAL DailyChallengeView/GuessModal callbacks
via the harness FakeInteraction against a REAL GameEngine on a tmp SQLite DB
with the fake snippet service — no discord.py gateway, no network, no ffmpeg.

Covers the recorded-payload taxonomy (pinned #3), the repeat-post pinned
output (pinned #4), --now determinism (pinned #1), user parsing (pinned #2),
the empty-catalog error (pinned #11), the no-challenge graceful notices
(VAL-CROSS-017), the secrecy invariant (VAL-CROSS-010), and the advance-day
reveal ordering (VAL-DAILY-008/009, VAL-CROSS-018).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from songbot.catalog.refresh import RefreshResult, SourceRefresh
from songbot.db import Database
from songbot.engine import GameEngine
from songbot.harness.cli import (
    HarnessContext,
    parse_now,
    parse_user,
    scenario_admin_fixsong,
    scenario_admin_post,
    scenario_admin_reload,
    scenario_admin_setup,
    scenario_admin_skip,
    scenario_advance_day,
    scenario_guess,
    scenario_hear_more,
    scenario_leaderboard,
    scenario_post,
    scenario_reset,
    scenario_status,
)
from songbot.harness.fakes import FakePermissions, FakeUser
from tests.unit.test_engine_daily import _make_engine, _settings
from tests.unit.test_engine_gameplay import _add_song

TITLE = "Neon Skyline"
ARTIST = "Midnight Circuit"
WRONG = "zxqv unrelated noise"

DAY1 = datetime(2026, 8, 13, 16, 0, 0, tzinfo=UTC)  # 2026-08-13 13:00 ADT
DAY2 = datetime(2026, 8, 14, 16, 0, 0, tzinfo=UTC)  # 2026-08-14 13:00 ADT
HALIFAX = timezone(timedelta(hours=-3))  # ADT (August)

ALICE = FakeUser(id="alice", name="alice")
BOB = FakeUser(id="bob", name="bob")
ADMIN = FakeUser(id="admin", name="admin", guild_permissions=FakePermissions(manage_guild=True))
NON_ADMIN = FakeUser(id="admin", name="admin")


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    database = Database.open(tmp_path / "songbot.db")
    yield database
    database.close()


@pytest.fixture
def engine(db: Database, tmp_path: Path) -> GameEngine:
    engine, _ = _make_engine(tmp_path, db)
    return engine


@pytest.fixture
def ctx(db: Database, engine: GameEngine, tmp_path: Path) -> HarnessContext:
    return HarnessContext(settings=_settings(tmp_path), db=db, engine=engine)


@pytest.fixture
def posted(ctx: HarnessContext, db: Database) -> dict[str, Any]:
    _add_song(db)
    return json_roundtrip(scenario_post_sync(ctx, DAY1))


def scenario_post_sync(ctx: HarnessContext, now: datetime) -> dict[str, Any]:
    """Sync wrapper so non-async fixtures/tests can post."""
    return asyncio.run(scenario_post(ctx, now))


def json_roundtrip(out: dict[str, Any]) -> dict[str, Any]:
    """Every scenario output must be JSON-serializable; return the parsed form."""
    return json.loads(json.dumps(out, ensure_ascii=False))  # type: ignore[no-any-return]


def kinds(out: dict[str, Any]) -> list[str]:
    return [p["kind"] for p in out["payloads"]]


class TestUserAndNowParsing:
    def test_bare_name_is_its_own_stable_id(self) -> None:
        user = parse_user("alice")
        assert user.id == "alice"
        assert user.name == "alice"

    def test_explicit_id_name_pair(self) -> None:
        user = parse_user("123456789:alice")
        assert user.id == "123456789"
        assert user.name == "alice"

    def test_parse_now_iso_with_offset(self) -> None:
        now = parse_now("2026-08-10T15:00:00-03:00")
        assert now.tzinfo is not None
        assert now.utcoffset() == timedelta(hours=-3)

    def test_parse_now_naive_means_utc(self) -> None:
        now = parse_now("2026-08-10T15:00:00")
        assert now.tzinfo is not None
        assert now.utcoffset() == timedelta(0)


class TestPost:
    async def test_first_post_records_one_channel_payload(
        self, ctx: HarnessContext, db: Database
    ) -> None:
        _add_song(db)
        out = json_roundtrip(await scenario_post(ctx, DAY1))

        assert len(out["payloads"]) == 1
        payload = out["payloads"][0]
        assert payload["kind"] == "channel"
        assert payload["recipient"] is None
        embed = payload["embed"]
        assert embed["title"].startswith("🎵 Daily Song — ")
        assert "2026-08-13" in embed["title"]
        assert "how to play" in embed["description"].lower()
        custom_ids = {c["custom_id"] for c in payload["components"]}
        assert custom_ids == {"songbot:hear_more", "songbot:guess", "songbot:leaderboard"}
        assert len(payload["attachments"]) == 1
        attachment = payload["attachments"][0]
        assert attachment["filename"] == "songbot-snippet.mp3"
        assert attachment["path"].endswith("0.mp3")
        assert attachment["size"] >= 0

        row = db.query_one("SELECT * FROM challenges WHERE guild_id = 'guild-1'")
        assert row is not None
        assert row["status"] == "active"
        assert row["date"] == "2026-08-13"
        assert out["state"]["challenge"]["created"] is True
        assert out["state"]["challenge"]["date"] == "2026-08-13"

    async def test_repeat_post_same_day_is_already_posted(
        self, ctx: HarnessContext, db: Database
    ) -> None:
        _add_song(db)
        await scenario_post(ctx, DAY1)
        out = await scenario_post(ctx, DAY1)

        # Pinned #4: exactly this shape, no new payloads, no state change.
        assert out == {"already_posted": True, "messages": []}
        row = db.query_one("SELECT COUNT(*) AS c FROM challenges")
        assert row is not None
        assert row["c"] == 1

    async def test_post_with_empty_catalog_returns_clean_error(
        self, ctx: HarnessContext, db: Database
    ) -> None:
        # _settings has both providers disabled and the songs table is empty.
        out = await scenario_post(ctx, DAY1)
        assert out == {"error": "catalog_empty"}
        row = db.query_one("SELECT COUNT(*) AS c FROM challenges")
        assert row is not None
        assert row["c"] == 0

    async def test_post_now_drives_the_challenge_date(
        self, ctx: HarnessContext, db: Database
    ) -> None:
        _add_song(db)
        out = json_roundtrip(await scenario_post(ctx, datetime(2026, 8, 10, 15, 0, tzinfo=HALIFAX)))
        assert out["state"]["challenge"]["date"] == "2026-08-10"
        row = db.query_one("SELECT date FROM challenges")
        assert row is not None
        assert row["date"] == "2026-08-10"


class TestHearMore:
    async def test_ladder_escalation_payloads(
        self, ctx: HarnessContext, posted: dict[str, Any], db: Database
    ) -> None:
        challenge_id = posted["state"]["challenge"]["id"]
        out = json_roundtrip(
            await scenario_hear_more(ctx, ALICE, 4, DAY1, now_pinned=True)
        )

        assert kinds(out) == ["ephemeral"] * 4
        for payload, points, level in zip(
            out["payloads"], ("75", "50", "30", "15"), (1, 2, 3, 4), strict=True
        ):
            assert payload["recipient"] == "alice"
            assert points in payload["content"]
            assert len(payload["attachments"]) == 1
            assert payload["attachments"][0]["filename"] == "songbot-snippet.mp3"
            assert payload["attachments"][0]["path"].endswith(f"{level}.mp3")
        row = db.query_one(
            "SELECT snippet_level FROM challenge_users"
            " WHERE challenge_id = ? AND user_id = 'alice'",
            (challenge_id,),
        )
        assert row is not None
        assert row["snippet_level"] == 4
        assert out["state"]["user"]["snippet_level"] == 4

    async def test_multi_press_stops_at_max_level(
        self, ctx: HarnessContext, posted: dict[str, Any]
    ) -> None:
        out = json_roundtrip(
            await scenario_hear_more(ctx, ALICE, 6, DAY1, now_pinned=True)
        )
        assert len(out["payloads"]) == 6
        attachments = [p for p in out["payloads"] if p["attachments"]]
        assert len(attachments) == 4
        for notice in out["payloads"][4:]:
            assert notice["kind"] == "ephemeral"
            assert notice["attachments"] == []
        assert out["state"]["user"]["snippet_level"] == 4

    async def test_levels_are_independent_per_user(
        self, ctx: HarnessContext, posted: dict[str, Any], db: Database
    ) -> None:
        challenge_id = posted["state"]["challenge"]["id"]
        await scenario_hear_more(ctx, ALICE, 2, DAY1, now_pinned=True)
        out = json_roundtrip(await scenario_hear_more(ctx, BOB, 1, DAY1, now_pinned=True))

        assert len(out["payloads"]) == 1
        payload = out["payloads"][0]
        assert payload["recipient"] == "bob"
        assert "75" in payload["content"]  # bob starts from his own level 0
        assert payload["attachments"][0]["path"].endswith("1.mp3")
        alice = db.query_one(
            "SELECT snippet_level FROM challenge_users"
            " WHERE challenge_id = ? AND user_id = 'alice'",
            (challenge_id,),
        )
        assert alice is not None
        assert alice["snippet_level"] == 2  # untouched by bob's press

    async def test_no_active_challenge_yields_graceful_ephemeral(
        self, ctx: HarnessContext, db: Database
    ) -> None:
        out = json_roundtrip(await scenario_hear_more(ctx, ALICE, 1, DAY1, now_pinned=True))
        assert kinds(out) == ["ephemeral"]
        assert out["payloads"][0]["recipient"] == "alice"
        assert "no active challenge" in out["payloads"][0]["content"].lower()
        for table in ("challenge_users", "guesses", "user_stats"):
            row = db.query_one(f"SELECT COUNT(*) AS c FROM {table}")
            assert row is not None
            assert row["c"] == 0

    async def test_hear_more_creates_zero_valued_user_stats_row(
        self, ctx: HarnessContext, posted: dict[str, Any], db: Database
    ) -> None:
        """VAL-SCORE-005 through the real view: a hear-more-only user gets a
        zero-valued user_stats row, yet the leaderboard stays friendly-empty
        (VAL-SCORE-012) until somebody scores."""
        await scenario_hear_more(ctx, ALICE, 1, DAY1, now_pinned=True)

        row = db.query_one(
            "SELECT total_points, wins, current_streak, best_streak, last_win_date"
            " FROM user_stats WHERE guild_id = 'guild-1' AND user_id = 'alice'"
        )
        assert row is not None
        assert (row["total_points"], row["wins"]) == (0, 0)
        assert (row["current_streak"], row["best_streak"]) == (0, 0)
        assert row["last_win_date"] is None

        out = json_roundtrip(await scenario_leaderboard(ctx, BOB, DAY1, now_pinned=True))
        assert kinds(out) == ["ephemeral"]
        assert "no scores" in out["payloads"][0]["content"].lower()
        assert out["state"]["entries"] == []


class TestGuess:
    async def test_correct_guess_flow_modal_ephemeral_announcement(
        self, ctx: HarnessContext, posted: dict[str, Any], db: Database
    ) -> None:
        challenge_id = posted["state"]["challenge"]["id"]
        out = json_roundtrip(await scenario_guess(ctx, ALICE, TITLE, DAY1, now_pinned=True))

        assert kinds(out) == ["modal", "ephemeral", "announcement"]
        modal = out["payloads"][0]
        assert modal["recipient"] == "alice"
        assert len(modal["components"]) == 1
        text_input = modal["components"][0]
        assert text_input["type"] == "text_input"
        assert text_input["placeholder"] == "Artist or title..."
        assert text_input["required"] is True

        feedback = out["payloads"][1]
        assert feedback["recipient"] == "alice"
        assert "✅" in feedback["content"]
        assert "100" in feedback["content"]
        assert "title" in feedback["content"].lower()
        # Issue #7: the solve feedback carries the full song, pinned-#9 filename.
        assert len(feedback["attachments"]) == 1
        assert feedback["attachments"][0]["filename"] == "songbot-full.mp3"
        assert feedback["attachments"][0]["path"].endswith("full.mp3")

        announcement = out["payloads"][2]
        assert announcement["recipient"] is None
        assert announcement["attachments"] == []  # nothing public changes
        assert "<@alice>" in announcement["content"]
        assert "1 guess" in announcement["content"]
        assert "100" in announcement["content"]

        row = db.query_one(
            "SELECT solved, points_awarded, guesses_used FROM challenge_users"
            " WHERE challenge_id = ? AND user_id = 'alice'",
            (challenge_id,),
        )
        assert row is not None
        assert row["solved"] == 1
        assert row["points_awarded"] == 100
        assert row["guesses_used"] == 1

    async def test_wrong_guess_reports_remaining_count(
        self, ctx: HarnessContext, posted: dict[str, Any], db: Database
    ) -> None:
        out = json_roundtrip(await scenario_guess(ctx, ALICE, WRONG, DAY1, now_pinned=True))
        assert kinds(out) == ["modal", "ephemeral"]
        feedback = out["payloads"][1]
        assert "❌" in feedback["content"]
        assert "5" in feedback["content"]
        assert feedback["attachments"] == []  # the full song is a SOLVER reward
        assert out["state"]["user"]["guesses_used"] == 1
        assert out["state"]["user"]["solved"] == 0

    async def test_empty_guess_is_a_notice_not_a_guess(
        self, ctx: HarnessContext, posted: dict[str, Any], db: Database
    ) -> None:
        out = json_roundtrip(await scenario_guess(ctx, ALICE, "   ", DAY1, now_pinned=True))
        assert kinds(out) == ["modal", "ephemeral"]
        assert "enter a guess" in out["payloads"][1]["content"].lower()
        row = db.query_one("SELECT COUNT(*) AS c FROM guesses")
        assert row is not None
        assert row["c"] == 0
        assert out["state"]["user"] is None or out["state"]["user"]["guesses_used"] == 0

    async def test_guess_after_solve_is_rejected_without_reannounce(
        self, ctx: HarnessContext, posted: dict[str, Any]
    ) -> None:
        await scenario_guess(ctx, ALICE, TITLE, DAY1, now_pinned=True)
        out = json_roundtrip(await scenario_guess(ctx, ALICE, WRONG, DAY1, now_pinned=True))
        assert kinds(out) == ["modal", "ephemeral"]
        assert "already solved" in out["payloads"][1]["content"].lower()
        assert out["state"]["user"]["guesses_used"] == 1  # unchanged

    async def test_no_active_challenge_yields_graceful_ephemeral(
        self, ctx: HarnessContext
    ) -> None:
        out = json_roundtrip(await scenario_guess(ctx, ALICE, TITLE, DAY1, now_pinned=True))
        assert kinds(out) == ["ephemeral"]
        assert "no active challenge" in out["payloads"][0]["content"].lower()

    async def test_pinned_now_on_revealed_challenge_drives_closed_notice(
        self, ctx: HarnessContext, posted: dict[str, Any], db: Database
    ) -> None:
        # With --now pinned to a revealed challenge's date, the harness drives
        # the REAL view bound to that challenge -> the closed-challenge notice.
        await scenario_advance_day(ctx, DAY1)
        out = json_roundtrip(await scenario_guess(ctx, ALICE, TITLE, DAY1, now_pinned=True))
        assert kinds(out) == ["modal", "ephemeral"]
        assert out["payloads"][1]["content"] == "This challenge has closed."
        row = db.query_one("SELECT COUNT(*) AS c FROM guesses")
        assert row is not None
        assert row["c"] == 0

    async def test_unpinned_gameplay_targets_the_latest_active_challenge(
        self, ctx: HarnessContext, posted: dict[str, Any], db: Database
    ) -> None:
        # Bare (no --now) interactions press the CURRENT post's buttons: after
        # advance-day the day-1 challenge is revealed and day-2 is active.
        await scenario_advance_day(ctx, DAY1)
        out = json_roundtrip(await scenario_hear_more(ctx, ALICE, 1, DAY1, now_pinned=False))
        assert kinds(out) == ["ephemeral"]
        assert "75" in out["payloads"][0]["content"]  # level 1 on the NEW challenge
        day2 = db.query_one("SELECT id FROM challenges WHERE date = '2026-08-14'")
        assert day2 is not None
        row = db.query_one(
            "SELECT snippet_level FROM challenge_users"
            " WHERE challenge_id = ? AND user_id = 'alice'",
            (day2["id"],),
        )
        assert row is not None
        assert row["snippet_level"] == 1


class TestLeaderboard:
    async def test_empty_leaderboard_is_a_friendly_ephemeral(
        self, ctx: HarnessContext, posted: dict[str, Any]
    ) -> None:
        out = json_roundtrip(await scenario_leaderboard(ctx, ALICE, DAY1, now_pinned=True))
        assert kinds(out) == ["ephemeral"]
        payload = out["payloads"][0]
        assert payload["recipient"] == "alice"
        assert "no scores" in payload["content"].lower()
        assert out["state"]["entries"] == []

    async def test_leaderboard_entries_payload(
        self, ctx: HarnessContext, posted: dict[str, Any]
    ) -> None:
        await scenario_guess(ctx, ALICE, TITLE, DAY1, now_pinned=True)
        out = json_roundtrip(await scenario_leaderboard(ctx, BOB, DAY1, now_pinned=True))
        assert kinds(out) == ["ephemeral"]
        payload = out["payloads"][0]
        assert payload["recipient"] == "bob"
        assert "<@alice>" in payload["embed"]["description"]
        assert "100" in payload["embed"]["description"]
        assert out["state"]["entries"] == [
            {"user_id": "alice", "total_points": 100, "wins": 1, "current_streak": 1}
        ]

    async def test_leaderboard_without_any_challenge_still_works(
        self, ctx: HarnessContext
    ) -> None:
        out = json_roundtrip(await scenario_leaderboard(ctx, ALICE, DAY1, now_pinned=True))
        assert kinds(out) == ["ephemeral"]
        assert "no scores" in out["payloads"][0]["content"].lower()


class TestAdvanceDay:
    async def test_reveal_then_new_post_with_winners(
        self, ctx: HarnessContext, posted: dict[str, Any], db: Database
    ) -> None:
        await scenario_guess(ctx, ALICE, TITLE, DAY1, now_pinned=True)
        out = json_roundtrip(await scenario_advance_day(ctx, DAY1))

        assert kinds(out) == ["announcement", "channel"]
        reveal = out["payloads"][0]
        assert TITLE in reveal["embed"]["description"]
        assert ARTIST in reveal["embed"]["description"]
        assert "<@alice>" in reveal["embed"]["description"]
        new_post = out["payloads"][1]
        assert "2026-08-14" in new_post["embed"]["title"]
        assert len(new_post["attachments"]) == 1

        day1 = db.query_one("SELECT status, revealed_at FROM challenges WHERE date = '2026-08-13'")
        assert day1 is not None
        assert day1["status"] == "revealed"
        assert day1["revealed_at"] is not None
        day2 = db.query_one("SELECT status FROM challenges WHERE date = '2026-08-14'")
        assert day2 is not None
        assert day2["status"] == "active"
        assert out["state"]["challenge"]["date"] == "2026-08-14"
        assert out["state"]["reveal"]["date"] == "2026-08-13"

    async def test_reveal_with_no_winners_says_nobody_got_it(
        self, ctx: HarnessContext, posted: dict[str, Any]
    ) -> None:
        out = json_roundtrip(await scenario_advance_day(ctx, DAY1))
        assert kinds(out) == ["announcement", "channel"]
        description = out["payloads"][0]["embed"]["description"]
        assert TITLE in description
        assert "nobody got it" in description.lower()

    async def test_advance_day_from_empty_state_posts_without_reveal(
        self, ctx: HarnessContext, db: Database
    ) -> None:
        _add_song(db)
        out = json_roundtrip(await scenario_advance_day(ctx, DAY1))
        assert kinds(out) == ["channel"]
        assert out["state"]["reveal"] is None
        assert out["state"]["challenge"]["date"] == "2026-08-13"
        row = db.query_one("SELECT COUNT(*) AS c FROM challenges")
        assert row is not None
        assert row["c"] == 1

    async def test_advance_day_is_state_anchored_sequential(
        self, ctx: HarnessContext, db: Database
    ) -> None:
        _add_song(db)
        await scenario_post(ctx, DAY1)
        await scenario_advance_day(ctx, DAY1)  # -> 2026-08-14
        out = json_roundtrip(await scenario_advance_day(ctx, DAY1))  # -> 2026-08-15
        assert out["state"]["challenge"]["date"] == "2026-08-15"
        dates = [r["date"] for r in db.query("SELECT date FROM challenges ORDER BY date")]
        assert dates == ["2026-08-13", "2026-08-14", "2026-08-15"]


class TestReset:
    async def test_reset_wipes_tables_and_snippet_cache(
        self, ctx: HarnessContext, posted: dict[str, Any], tmp_path: Path
    ) -> None:
        await scenario_guess(ctx, ALICE, WRONG, DAY1, now_pinned=True)
        cache_file = tmp_path / "snippets" / str(posted["state"]["challenge"]["id"]) / "0.mp3"
        assert cache_file.exists()  # FakeSnippets touches level files

        out = json_roundtrip(scenario_reset(ctx))

        assert out["payloads"] == []
        for table in (
            "songs",
            "challenges",
            "challenge_users",
            "guesses",
            "user_stats",
            "guild_settings",
            "song_overrides",
        ):
            row = ctx.db.query_one(f"SELECT COUNT(*) AS c FROM {table}")
            assert row is not None
            assert row["c"] == 0
        snippet_dir = tmp_path / "snippets"
        assert list(snippet_dir.rglob("*.mp3")) == []
        # Migrations survive a reset (the schema itself is not wiped).
        assert ctx.db.schema_version() == 4

    async def test_post_after_reset_works(self, ctx: HarnessContext, db: Database) -> None:
        _add_song(db)
        await scenario_post(ctx, DAY1)
        scenario_reset(ctx)
        _add_song(db)
        out = json_roundtrip(await scenario_post(ctx, DAY1))
        assert kinds(out) == ["channel"]


class TestStatus:
    def test_status_on_empty_state(self, ctx: HarnessContext) -> None:
        out = json_roundtrip(scenario_status(ctx, DAY1))
        state = out["state"]
        assert state["date"] == "2026-08-13"
        assert state["challenge"] is None
        assert state["counts"]["challenges"] == 0
        assert state["counts"]["guesses"] == 0
        assert state["leaderboard"] == []

    async def test_status_reflects_post_and_guess(
        self, ctx: HarnessContext, posted: dict[str, Any]
    ) -> None:
        await scenario_guess(ctx, BOB, WRONG, DAY1, now_pinned=True)
        out = json_roundtrip(scenario_status(ctx, DAY1))
        state = out["state"]
        assert state["date"] == "2026-08-13"
        assert state["challenge"]["status"] == "active"
        # status is the test-only surface that exposes song identity (pinned #2).
        assert state["challenge"]["song"] == {"title": TITLE, "artist": ARTIST}
        assert state["counts"]["challenges"] == 1
        assert state["counts"]["guesses"] == 1
        assert state["counts"]["songs"] == 1

    async def test_status_now_drives_the_reported_date(
        self, ctx: HarnessContext, posted: dict[str, Any]
    ) -> None:
        out = json_roundtrip(scenario_status(ctx, DAY2))
        assert out["state"]["date"] == "2026-08-14"
        assert out["state"]["challenge"] is None


class TestAdminSetup:
    """The admin-setup scenario drives the REAL /songbot-setup body."""

    async def test_admin_setup_configures_channel_and_acks(
        self, ctx: HarnessContext, db: Database
    ) -> None:
        out = json_roundtrip(await scenario_admin_setup(ctx, ADMIN, "999888", DAY1))

        assert kinds(out) == ["ephemeral"]
        ack = out["payloads"][0]
        assert ack["recipient"] == "admin"
        assert "#999888" in ack["content"]
        assert out["state"]["outcome"] == "configured"
        assert out["state"]["guild_settings"] == {
            "guild_id": "guild-1",
            "channel_id": "999888",
            "set_by": "admin",
        }
        row = db.query_one("SELECT channel_id FROM guild_settings WHERE guild_id = 'guild-1'")
        assert row is not None
        assert row["channel_id"] == "999888"

    async def test_admin_setup_non_admin_denied_without_changing_the_row(
        self, ctx: HarnessContext, db: Database
    ) -> None:
        # The context seeded guild-1 -> channel-1 at build time.
        out = json_roundtrip(await scenario_admin_setup(ctx, NON_ADMIN, "999888", DAY1))

        assert kinds(out) == ["ephemeral"]
        assert "manage server" in out["payloads"][0]["content"].lower()
        assert out["state"]["outcome"] == "denied"
        row = db.query_one("SELECT channel_id FROM guild_settings WHERE guild_id = 'guild-1'")
        assert row is not None
        assert row["channel_id"] == "channel-1"  # unchanged

    async def test_admin_setup_then_admin_post_uses_the_new_channel(
        self, ctx: HarnessContext, db: Database
    ) -> None:
        _add_song(db)
        await scenario_admin_setup(ctx, ADMIN, "999888", DAY1)

        out = json_roundtrip(await scenario_admin_post(ctx, ADMIN, DAY1))

        assert out["state"]["outcome"] == "posted"
        row = db.query_one("SELECT channel_id FROM challenges")
        assert row is not None
        assert row["channel_id"] == "999888"


class TestAdminPost:
    """The admin-post scenario drives the REAL /songbot-post body."""

    async def test_admin_post_records_channel_post_and_ephemeral_ack(
        self, ctx: HarnessContext, db: Database
    ) -> None:
        _add_song(db)
        out = json_roundtrip(await scenario_admin_post(ctx, ADMIN, DAY1))

        assert kinds(out) == ["channel", "ephemeral"]
        post = out["payloads"][0]
        assert post["recipient"] is None
        assert post["embed"]["title"].startswith("🎵 Daily Song — 2026-08-13")
        custom_ids = {c["custom_id"] for c in post["components"]}
        assert custom_ids == {"songbot:hear_more", "songbot:guess", "songbot:leaderboard"}
        assert len(post["attachments"]) == 1
        assert post["attachments"][0]["filename"] == "songbot-snippet.mp3"
        ack = out["payloads"][1]
        assert ack["recipient"] == "admin"
        assert out["state"]["outcome"] == "posted"
        assert out["state"]["challenge"]["date"] == "2026-08-13"
        assert out["state"]["challenge"]["status"] == "active"
        row = db.query_one("SELECT COUNT(*) AS c FROM challenges")
        assert row is not None
        assert row["c"] == 1

    async def test_admin_post_repeat_prints_compact_already_posted(
        self, ctx: HarnessContext, db: Database
    ) -> None:
        _add_song(db)
        await scenario_admin_post(ctx, ADMIN, DAY1)

        out = await scenario_admin_post(ctx, ADMIN, DAY1)

        # Pinned #4 / VAL-CROSS-015: the exact compact shape, no second post.
        assert out == {"already_posted": True, "messages": []}
        row = db.query_one("SELECT COUNT(*) AS c FROM challenges")
        assert row is not None
        assert row["c"] == 1

    async def test_admin_post_non_admin_denied_with_zero_mutation(
        self, ctx: HarnessContext, db: Database
    ) -> None:
        out = json_roundtrip(await scenario_admin_post(ctx, NON_ADMIN, DAY1))

        assert kinds(out) == ["ephemeral"]
        denial = out["payloads"][0]
        assert denial["recipient"] == "admin"
        assert "manage server" in denial["content"].lower()
        assert out["state"]["outcome"] == "denied"
        for table in ("songs", "challenges"):
            row = db.query_one(f"SELECT COUNT(*) AS c FROM {table}")
            assert row is not None
            assert row["c"] == 0

    async def test_admin_post_with_empty_catalog_returns_clean_error(
        self, ctx: HarnessContext, db: Database
    ) -> None:
        out = await scenario_admin_post(ctx, ADMIN, DAY1)
        assert out == {"error": "catalog_empty"}
        row = db.query_one("SELECT COUNT(*) AS c FROM challenges")
        assert row is not None
        assert row["c"] == 0


class TestAdminSkip:
    """The admin-skip scenario drives the REAL /songbot-skip body (pinned #5)."""

    async def test_admin_skip_replaces_song_and_resets_users(
        self, ctx: HarnessContext, db: Database, tmp_path: Path
    ) -> None:
        _add_song(db)
        _add_song(db, "song-2", title="Digital Horizon", artist="Quantum Drift")
        await scenario_post(ctx, DAY1)
        before = db.query_one(
            "SELECT id, song_id, snippet_offset_sec FROM challenges"
        )
        assert before is not None
        # todd builds in-progress state against the pre-skip song.
        await scenario_hear_more(ctx, FakeUser(id="todd", name="todd"), 2, DAY1, now_pinned=True)
        await scenario_guess(ctx, FakeUser(id="todd", name="todd"), WRONG, DAY1, now_pinned=True)

        out = json_roundtrip(await scenario_admin_skip(ctx, ADMIN, DAY1))

        assert kinds(out) == ["ephemeral"], "skip emits no channel/announcement payload"
        assert out["payloads"][0]["recipient"] == "admin"
        assert out["state"]["outcome"] == "skipped"
        assert out["state"]["reason"] is None
        after = db.query_one("SELECT * FROM challenges WHERE date = '2026-08-13'")
        assert after is not None
        assert after["song_id"] != before["song_id"]
        assert after["snippet_offset_sec"] != before["snippet_offset_sec"]
        assert after["status"] == "active"
        assert after["skip_count"] == 1
        assert out["state"]["challenge"]["song_id"] == after["song_id"]
        # Per-user state fully reset (cascade delete + recreate).
        for table in ("challenge_users", "guesses"):
            row = db.query_one(f"SELECT COUNT(*) AS c FROM {table}")
            assert row is not None
            assert row["c"] == 0
        # The snippet cache was purged and regenerated for the replacement.
        cache_dir = tmp_path / "snippets" / str(after["id"])
        assert sorted(p.name for p in cache_dir.glob("*.mp3")) == [
            "0.mp3", "1.mp3", "2.mp3", "3.mp3", "4.mp3",
        ]
        # Secrecy: the ack names neither song.
        ack = out["payloads"][0]["content"].lower()
        for secret in (TITLE, ARTIST, "digital horizon", "quantum drift"):
            assert secret not in ack

    async def test_admin_skip_refused_after_solve_with_zero_mutation(
        self, ctx: HarnessContext, db: Database
    ) -> None:
        _add_song(db)
        _add_song(db, "song-2", title="Digital Horizon", artist="Quantum Drift")
        await scenario_post(ctx, DAY1)
        song = db.query_one(
            "SELECT s.title AS title FROM challenges c JOIN songs s ON s.id = c.song_id"
        )
        assert song is not None
        await scenario_guess(ctx, ALICE, song["title"], DAY1, now_pinned=True)

        out = json_roundtrip(await scenario_admin_skip(ctx, ADMIN, DAY1))

        assert kinds(out) == ["ephemeral"]
        assert out["state"]["outcome"] == "refused"
        assert out["state"]["reason"] == "solved"
        challenge = db.query_one("SELECT song_id, status, skip_count FROM challenges")
        assert challenge is not None
        assert challenge["status"] == "active"
        assert challenge["skip_count"] == 0
        user = db.query_one(
            "SELECT solved, points_awarded FROM challenge_users WHERE user_id = 'alice'"
        )
        assert user is not None
        assert (user["solved"], user["points_awarded"]) == (1, 100)
        stats = db.query_one("SELECT total_points, wins FROM user_stats WHERE user_id = 'alice'")
        assert stats is not None
        assert (stats["total_points"], stats["wins"]) == (100, 1)

    async def test_admin_skip_refused_when_revealed(
        self, ctx: HarnessContext, db: Database
    ) -> None:
        _add_song(db)
        await scenario_post(ctx, DAY1)
        db.execute(
            "UPDATE challenges SET status = 'revealed',"
            " revealed_at = '2026-08-13T16:30:00+00:00'"
        )

        out = json_roundtrip(await scenario_admin_skip(ctx, ADMIN, DAY1))

        assert kinds(out) == ["ephemeral"]
        assert out["state"]["outcome"] == "refused"
        assert out["state"]["reason"] == "revealed"
        row = db.query_one("SELECT status, skip_count FROM challenges")
        assert row is not None
        assert row["status"] == "revealed"
        assert row["skip_count"] == 0

    async def test_admin_skip_non_admin_denied_with_zero_mutation(
        self, ctx: HarnessContext, db: Database
    ) -> None:
        _add_song(db)
        _add_song(db, "song-2", title="Digital Horizon", artist="Quantum Drift")
        await scenario_post(ctx, DAY1)
        before = db.query_one("SELECT song_id, snippet_offset_sec FROM challenges")
        assert before is not None

        out = json_roundtrip(await scenario_admin_skip(ctx, NON_ADMIN, DAY1))

        assert kinds(out) == ["ephemeral"]
        assert "manage server" in out["payloads"][0]["content"].lower()
        assert out["state"]["outcome"] == "denied"
        after = db.query_one("SELECT song_id, snippet_offset_sec, skip_count FROM challenges")
        assert after is not None
        assert after["song_id"] == before["song_id"]
        assert after["snippet_offset_sec"] == before["snippet_offset_sec"]
        assert after["skip_count"] == 0


class TestAdminFixsong:
    """The admin-fixsong scenario drives the REAL /songbot-fixsong body."""

    async def test_admin_fixsong_corrects_metadata_and_acks_old_new(
        self, ctx: HarnessContext, db: Database
    ) -> None:
        _add_song(db)
        await scenario_post(ctx, DAY1)

        out = json_roundtrip(
            await scenario_admin_fixsong(ctx, ADMIN, "Fixed Title", "Fixed Artist", None, DAY1)
        )

        assert kinds(out) == ["ephemeral"]
        assert out["payloads"][0]["recipient"] == "admin"
        assert out["state"]["outcome"] == "fixed"
        assert out["state"]["reason"] is None
        fix = out["state"]["fix"]
        assert fix["challenge_date"] == "2026-08-13"
        assert (fix["old_title"], fix["old_artist"]) == (TITLE, ARTIST)
        assert (fix["new_title"], fix["new_artist"]) == ("Fixed Title", "Fixed Artist")
        # The ack shows old -> new (the admin-only ephemeral secrecy exception).
        ack = out["payloads"][0]["content"]
        for text in (TITLE, ARTIST, "Fixed Title", "Fixed Artist"):
            assert text in ack
        # The songs row and the durable override row both carry the fix.
        song = db.query_one("SELECT title, artist FROM songs")
        assert song is not None
        assert (song["title"], song["artist"]) == ("Fixed Title", "Fixed Artist")
        override = db.query_one("SELECT title, artist FROM song_overrides")
        assert override is not None
        assert (override["title"], override["artist"]) == ("Fixed Title", "Fixed Artist")

    async def test_admin_fixsong_then_guess_uses_corrected_metadata(
        self, ctx: HarnessContext, db: Database
    ) -> None:
        _add_song(db)
        await scenario_post(ctx, DAY1)
        await scenario_admin_fixsong(ctx, ADMIN, "Fixed Title", "Fixed Artist", None, DAY1)

        out = json_roundtrip(
            await scenario_guess(ctx, ALICE, "Fixed Title Fixed Artist", DAY1, now_pinned=True)
        )

        assert out["state"]["user"]["solved"] is True
        assert out["state"]["user"]["points_awarded"] == 150  # 100 x 1.5 both-bonus

    async def test_admin_fixsong_non_admin_denied_with_zero_mutation(
        self, ctx: HarnessContext, db: Database
    ) -> None:
        _add_song(db)
        await scenario_post(ctx, DAY1)

        out = json_roundtrip(
            await scenario_admin_fixsong(ctx, NON_ADMIN, "Fixed Title", None, None, DAY1)
        )

        assert kinds(out) == ["ephemeral"]
        assert "manage server" in out["payloads"][0]["content"].lower()
        assert out["state"]["outcome"] == "denied"
        assert out["state"]["fix"] is None
        song = db.query_one("SELECT title, artist FROM songs")
        assert song is not None
        assert (song["title"], song["artist"]) == (TITLE, ARTIST)
        row = db.query_one("SELECT COUNT(*) AS c FROM song_overrides")
        assert row is not None
        assert row["c"] == 0

    async def test_admin_fixsong_refused_without_a_challenge(
        self, ctx: HarnessContext, db: Database
    ) -> None:
        _add_song(db)

        out = json_roundtrip(
            await scenario_admin_fixsong(ctx, ADMIN, "Fixed Title", None, None, DAY1)
        )

        assert kinds(out) == ["ephemeral"]
        assert out["state"]["outcome"] == "refused"
        assert out["state"]["reason"] == "no_challenge"
        assert out["state"]["fix"] is None


class TestAdminReload:
    """The admin-reload scenario drives the REAL /songbot-reload body."""

    async def test_admin_reload_reports_per_source_summary(
        self, db: Database, tmp_path: Path
    ) -> None:
        refresh = RefreshResult(
            sources=(
                SourceRefresh(source="local", added=8),
                SourceRefresh(source="youtube", error="YouTubeCatalogError: boom"),
            )
        )
        engine, _ = _make_engine(tmp_path, db, catalog_refresher=lambda: refresh)
        ctx = HarnessContext(settings=_settings(tmp_path), db=db, engine=engine)

        out = json_roundtrip(await scenario_admin_reload(ctx, ADMIN, DAY1))

        assert kinds(out) == ["ephemeral"]
        ack = out["payloads"][0]
        assert ack["recipient"] == "admin"
        content = ack["content"]
        assert "local" in content
        assert "8 added" in content
        assert "youtube" in content
        assert "YouTubeCatalogError" in content
        assert out["state"]["outcome"] == "reloaded"
        assert out["state"]["sources"] == [
            {
                "source": "local",
                "added": 8,
                "updated": 0,
                "removed": 0,
                "retained": 0,
                "error": None,
            },
            {
                "source": "youtube",
                "added": 0,
                "updated": 0,
                "removed": 0,
                "retained": 0,
                "error": "YouTubeCatalogError: boom",
            },
        ]

    async def test_admin_reload_with_no_providers_says_so(
        self, ctx: HarnessContext
    ) -> None:
        # _settings disables both providers -> an empty RefreshResult.
        out = json_roundtrip(await scenario_admin_reload(ctx, ADMIN, DAY1))
        assert kinds(out) == ["ephemeral"]
        assert "no catalog sources" in out["payloads"][0]["content"].lower()
        assert out["state"]["sources"] == []

    async def test_admin_reload_non_admin_denied_without_refreshing(
        self, db: Database, tmp_path: Path
    ) -> None:
        calls = 0

        def _spy() -> RefreshResult:
            nonlocal calls
            calls += 1
            return RefreshResult(sources=())

        engine, _ = _make_engine(tmp_path, db, catalog_refresher=_spy)
        ctx = HarnessContext(settings=_settings(tmp_path), db=db, engine=engine)

        out = json_roundtrip(await scenario_admin_reload(ctx, NON_ADMIN, DAY1))

        assert kinds(out) == ["ephemeral"]
        assert "manage server" in out["payloads"][0]["content"].lower()
        assert out["state"]["outcome"] == "denied"
        assert calls == 0, "a denied reload must not touch the catalog"
        row = db.query_one("SELECT COUNT(*) AS c FROM songs")
        assert row is not None
        assert row["c"] == 0


class TestSecrecy:
    async def test_no_pre_reveal_payload_leaks_song_identity(
        self, ctx: HarnessContext, posted: dict[str, Any]
    ) -> None:
        transcript: list[dict[str, Any]] = []
        transcript += posted["payloads"]
        for out in (
            await scenario_hear_more(ctx, ALICE, 4, DAY1, now_pinned=True),
            await scenario_guess(ctx, ALICE, WRONG, DAY1, now_pinned=True),
            await scenario_guess(ctx, ALICE, TITLE, DAY1, now_pinned=True),
            await scenario_guess(ctx, BOB, TITLE, DAY1, now_pinned=True),
            await scenario_leaderboard(ctx, BOB, DAY1, now_pinned=True),
        ):
            transcript += json_roundtrip(out)["payloads"]

        for payload in transcript:
            haystack = json.dumps(payload, ensure_ascii=False).lower()
            assert TITLE.lower() not in haystack
            assert ARTIST.lower() not in haystack

        # ...and the reveal DOES name the song (post-reveal).
        out = json_roundtrip(await scenario_advance_day(ctx, DAY1))
        reveal = out["payloads"][0]
        assert TITLE in reveal["embed"]["description"]
        assert ARTIST in reveal["embed"]["description"]
