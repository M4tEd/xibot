"""Unit tests for DailyChallengeView: the three persistent buttons.

The REAL view callbacks are driven by lightweight fakes
(`tests/unit/interaction_fakes.py`) against a REAL GameEngine on a tmp
SQLite DB with the fake snippet service — no discord.py gateway, no network.
Covers the exact custom_ids, the hear-more ladder payloads (VAL-HEAR-002/003
adapter half), max-level/already-solved rejections (VAL-HEAR-008/009), the
guess-button modal handoff (VAL-GUESS-001), the ephemeral leaderboard
(VAL-SCORE-011/012), and the revealed-challenge lockout mapping
(VAL-GUESS-019 adapter half: exactly one ephemeral "This challenge has
closed." and nothing else).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from songbot.bot.modals import GuessModal
from songbot.bot.views import DailyChallengeView
from songbot.db import Database
from songbot.engine import Challenge, GameEngine
from tests.unit.interaction_fakes import FakeInteraction, press
from tests.unit.test_engine_daily import NOW, _make_engine, _reveal_previous, _settings
from tests.unit.test_engine_gameplay import _add_song

NEXT_DAY = datetime(2026, 8, 14, 16, 0, 0, tzinfo=UTC)

ALICE = FakeInteraction.for_user(1001, "alice")
TITLE = "Neon Skyline"
ARTIST = "Midnight Circuit"


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
def challenge(engine: GameEngine, db: Database) -> Challenge:
    _add_song(db)
    return engine.ensure_today_challenge("g1", "c1", NOW)


@pytest.fixture
def view(engine: GameEngine, challenge: Challenge, tmp_path: Path) -> DailyChallengeView:
    return DailyChallengeView(
        engine,
        challenge.id,
        guild_id=challenge.guild_id,
        settings=_settings(tmp_path),
        clock=lambda: NOW,
    )


def _interaction(user_id: int = 1001, name: str = "alice") -> FakeInteraction:
    return FakeInteraction.for_user(user_id, name)


class TestViewShape:
    def test_three_buttons_with_exact_custom_ids(self, view: DailyChallengeView) -> None:
        custom_ids = {
            getattr(child, "custom_id", None) for child in view.children
        }
        assert custom_ids == {"songbot:hear_more", "songbot:guess", "songbot:leaderboard"}

    def test_persistent_view_has_no_timeout(self, view: DailyChallengeView) -> None:
        assert view.timeout is None


class TestHearMore:
    async def test_first_press_escalates_and_replies_ephemeral(
        self, view: DailyChallengeView, challenge: Challenge, db: Database
    ) -> None:
        interaction = _interaction()
        await press(view, "songbot:hear_more", interaction)

        assert len(interaction.payloads) == 1
        payload = interaction.payloads[0]
        assert payload.kind == "ephemeral"
        assert payload.recipient == "1001"
        assert payload.content is not None
        assert "75" in payload.content  # new potential points (level 1)
        assert "2s" in payload.content  # new snippet length
        assert payload.attachment is not None
        assert payload.attachment.filename == "songbot-snippet.mp3"  # pinned #9
        assert payload.attachment.path == str(challenge.snippet_paths[1])
        assert payload.attachment.size >= 0
        row = db.query_one(
            "SELECT snippet_level FROM challenge_users WHERE challenge_id = ? AND user_id = '1001'",
            (challenge.id,),
        )
        assert row is not None
        assert row["snippet_level"] == 1

    async def test_full_ladder_reports_descending_points(
        self, view: DailyChallengeView, challenge: Challenge
    ) -> None:
        interaction = _interaction()
        for _ in range(4):
            await press(view, "songbot:hear_more", interaction)

        assert [p.kind for p in interaction.payloads] == ["ephemeral"] * 4
        points_seen = []
        for expected, payload in zip(("75", "50", "30", "15"), interaction.payloads, strict=True):
            assert payload.content is not None
            assert expected in payload.content
            assert payload.attachment is not None
            points_seen.append(expected)
        assert points_seen == ["75", "50", "30", "15"]
        levels = [p.attachment.path for p in interaction.payloads if p.attachment is not None]
        assert levels == [str(challenge.snippet_paths[i]) for i in (1, 2, 3, 4)]

    async def test_max_level_press_returns_notice_without_attachment(
        self, view: DailyChallengeView, challenge: Challenge, db: Database
    ) -> None:
        interaction = _interaction()
        for _ in range(4):
            await press(view, "songbot:hear_more", interaction)
        interaction.payloads.clear()

        await press(view, "songbot:hear_more", interaction)

        assert len(interaction.payloads) == 1
        payload = interaction.payloads[0]
        assert payload.kind == "ephemeral"
        assert payload.attachment is None
        assert payload.content is not None
        assert "16s" in payload.content
        row = db.query_one(
            "SELECT snippet_level FROM challenge_users WHERE challenge_id = ? AND user_id = '1001'",
            (challenge.id,),
        )
        assert row is not None
        assert row["snippet_level"] == 4

    async def test_rejected_after_solve(
        self, view: DailyChallengeView, challenge: Challenge, db: Database, engine: GameEngine
    ) -> None:
        engine.submit_guess(challenge.id, "1001", TITLE, NOW)
        interaction = _interaction()
        await press(view, "songbot:hear_more", interaction)

        assert len(interaction.payloads) == 1
        payload = interaction.payloads[0]
        assert payload.kind == "ephemeral"
        assert payload.attachment is None
        assert payload.content is not None
        assert "already" in payload.content.lower()
        row = db.query_one(
            "SELECT snippet_level, solved FROM challenge_users"
            " WHERE challenge_id = ? AND user_id = '1001'",
            (challenge.id,),
        )
        assert row is not None
        assert row["snippet_level"] == 0
        assert row["solved"] == 1

    async def test_revealed_challenge_yields_single_closed_notice(
        self, view: DailyChallengeView, challenge: Challenge, engine: GameEngine
    ) -> None:
        # VAL-GUESS-019 adapter half: one ephemeral closed notice, nothing else.
        _reveal_previous(engine, "g1", NEXT_DAY)
        interaction = _interaction()
        await press(view, "songbot:hear_more", interaction)

        assert len(interaction.payloads) == 1
        payload = interaction.payloads[0]
        assert payload.kind == "ephemeral"
        assert payload.recipient == "1001"
        assert payload.content == "This challenge has closed."
        assert payload.attachment is None
        assert payload.embed is None


class TestGuessButton:
    async def test_guess_button_sends_guess_modal(
        self, view: DailyChallengeView, challenge: Challenge
    ) -> None:
        interaction = _interaction()
        await press(view, "songbot:guess", interaction)

        assert len(interaction.payloads) == 1
        payload = interaction.payloads[0]
        assert payload.kind == "modal"
        assert isinstance(payload.modal, GuessModal)

    async def test_modal_from_button_is_bound_to_the_view_challenge(
        self, view: DailyChallengeView, challenge: Challenge, db: Database
    ) -> None:
        interaction = _interaction()
        await press(view, "songbot:guess", interaction)
        modal = interaction.payloads[0].modal
        assert isinstance(modal, GuessModal)

        modal.guess._value = TITLE  # harness drives the real modal like discord would
        await modal.on_submit(interaction)

        row = db.query_one(
            "SELECT solved FROM challenge_users WHERE challenge_id = ? AND user_id = '1001'",
            (challenge.id,),
        )
        assert row is not None
        assert row["solved"] == 1


class TestLeaderboardButton:
    async def test_empty_leaderboard_sends_friendly_ephemeral(
        self, view: DailyChallengeView
    ) -> None:
        interaction = _interaction()
        await press(view, "songbot:leaderboard", interaction)

        assert len(interaction.payloads) == 1
        payload = interaction.payloads[0]
        assert payload.kind == "ephemeral"
        assert payload.recipient == "1001"
        assert payload.embed is None
        assert payload.content is not None
        assert "no scores" in payload.content.lower()

    async def test_leaderboard_embed_lists_entries_with_mentions(
        self, view: DailyChallengeView, challenge: Challenge, engine: GameEngine
    ) -> None:
        engine.submit_guess(challenge.id, "1001", TITLE, NOW)  # 100 pts
        engine.submit_guess(challenge.id, "1002", ARTIST, NOW)  # 100 pts, tie -> user_id ASC
        interaction = _interaction(1003, "rita")
        await press(view, "songbot:leaderboard", interaction)

        assert len(interaction.payloads) == 1
        payload = interaction.payloads[0]
        assert payload.kind == "ephemeral"
        assert payload.recipient == "1003"
        assert payload.embed is not None
        description = payload.embed.description or ""
        assert "<@1001>" in description
        assert "<@1002>" in description
        assert description.index("<@1001>") < description.index("<@1002>")  # tiebreak
        assert "100" in description
        # No public payload emitted by the leaderboard press.
        assert all(p.kind == "ephemeral" for p in interaction.payloads)
