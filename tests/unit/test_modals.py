"""Unit tests for GuessModal: single text input -> engine.submit_guess.

The REAL modal is constructed and submitted via lightweight fakes against a
REAL GameEngine (tmp SQLite DB, fake snippet service) — no gateway, no
network. Covers the modal shape (VAL-GUESS-001), correct/wrong/empty/
already-solved/post-limit feedback (VAL-GUESS-002/004/006/009/011/018
adapter halves), the first-solve public announcement with <@user_id> mention
format (VAL-GUESS-012), the secrecy invariant (VAL-GUESS-013: no title/artist
in any payload), and the revealed-challenge lockout (VAL-GUESS-019 adapter
half: exactly one ephemeral "This challenge has closed.").
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import discord
import pytest

from songbot.bot.modals import GuessModal
from songbot.db import Database
from songbot.engine import Challenge, GameEngine
from tests.unit.interaction_fakes import FakeInteraction
from tests.unit.test_engine_daily import NOW, _make_engine, _reveal_previous
from tests.unit.test_engine_gameplay import _add_song

NEXT_DAY = datetime(2026, 8, 14, 16, 0, 0, tzinfo=UTC)

TITLE = "Neon Skyline"
ARTIST = "Midnight Circuit"
BOTH = "Midnight Circuit - Neon Skyline"
WRONG = "zxqv unrelated noise"


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


def _modal(engine: GameEngine, challenge: Challenge) -> GuessModal:
    return GuessModal(engine, challenge.id, clock=lambda: NOW)


async def _submit(
    modal: GuessModal, text: str, user_id: int = 1001, name: str = "alice"
) -> FakeInteraction:
    interaction = FakeInteraction.for_user(user_id, name)
    modal.guess._value = text  # the harness injects the submitted value like discord does
    await modal.on_submit(interaction)
    return interaction


class TestModalShape:
    def test_exactly_one_required_text_input(
        self, engine: GameEngine, challenge: Challenge
    ) -> None:
        modal = _modal(engine, challenge)
        text_inputs = [c for c in modal.children if isinstance(c, discord.ui.TextInput)]
        assert len(text_inputs) == 1
        (text_input,) = text_inputs
        assert text_input.required is True
        assert text_input.placeholder == "Artist or title..."

    def test_modal_has_a_title(self, engine: GameEngine, challenge: Challenge) -> None:
        assert _modal(engine, challenge).title


class TestSubmitGuess:
    async def test_correct_title_banks_points_and_announces(
        self, engine: GameEngine, challenge: Challenge, db: Database
    ) -> None:
        interaction = await _submit(_modal(engine, challenge), TITLE)

        kinds = [p.kind for p in interaction.payloads]
        assert kinds == ["ephemeral", "announcement"]

        reply = interaction.payloads[0]
        assert reply.recipient == "1001"
        assert reply.content is not None
        assert "✅" in reply.content
        assert "100" in reply.content
        assert "title" in reply.content.lower()

        announcement = interaction.payloads[1]
        assert announcement.content is not None
        assert "<@1001>" in announcement.content  # mention format
        assert "1 guess" in announcement.content
        assert "100" in announcement.content

        row = db.query_one(
            "SELECT solved, points_awarded, guesses_used FROM challenge_users"
            " WHERE challenge_id = ? AND user_id = '1001'",
            (challenge.id,),
        )
        assert dict(row) == {"solved": 1, "points_awarded": 100, "guesses_used": 1}

    async def test_artist_only_guess_solves(
        self, engine: GameEngine, challenge: Challenge
    ) -> None:
        interaction = await _submit(_modal(engine, challenge), ARTIST, user_id=1002, name="bob")
        reply = interaction.payloads[0]
        assert reply.kind == "ephemeral"
        assert reply.content is not None
        assert "✅" in reply.content
        assert "artist" in reply.content.lower()

    async def test_both_match_reports_bonus(
        self, engine: GameEngine, challenge: Challenge, db: Database
    ) -> None:
        interaction = await _submit(_modal(engine, challenge), BOTH, user_id=1003, name="carol")
        reply = interaction.payloads[0]
        assert reply.content is not None
        assert "bonus" in reply.content.lower()
        assert "150" in reply.content
        row = db.query_one(
            "SELECT points_awarded FROM challenge_users"
            " WHERE challenge_id = ? AND user_id = '1003'",
            (challenge.id,),
        )
        assert row is not None
        assert row["points_awarded"] == 150

    async def test_wrong_guess_reports_remaining_without_announcement(
        self, engine: GameEngine, challenge: Challenge
    ) -> None:
        interaction = await _submit(_modal(engine, challenge), WRONG)
        assert [p.kind for p in interaction.payloads] == ["ephemeral"]
        reply = interaction.payloads[0]
        assert reply.content is not None
        assert "❌" in reply.content
        assert "5" in reply.content  # guesses left

    async def test_empty_guess_is_rejected_without_counting(
        self, engine: GameEngine, challenge: Challenge, db: Database
    ) -> None:
        interaction = await _submit(_modal(engine, challenge), "   ")
        assert [p.kind for p in interaction.payloads] == ["ephemeral"]
        reply = interaction.payloads[0]
        assert reply.content is not None
        assert "enter a guess" in reply.content.lower()
        assert db.query_one(
            "SELECT guesses_used FROM challenge_users WHERE challenge_id = ? AND user_id = '1001'",
            (challenge.id,),
        ) is None
        assert (
            db.query_one(
                "SELECT COUNT(*) AS c FROM guesses WHERE challenge_id = ?", (challenge.id,)
            )["c"]
            == 0
        )

    async def test_guess_after_solve_rejected_without_second_announcement(
        self, engine: GameEngine, challenge: Challenge
    ) -> None:
        engine.submit_guess(challenge.id, "1001", TITLE, NOW)
        interaction = await _submit(_modal(engine, challenge), "anything else")
        assert [p.kind for p in interaction.payloads] == ["ephemeral"]
        reply = interaction.payloads[0]
        assert reply.content is not None
        assert "already" in reply.content.lower()

    async def test_guess_past_the_limit_solves_for_10_points(
        self, engine: GameEngine, challenge: Challenge
    ) -> None:
        for i in range(6):
            engine.submit_guess(challenge.id, "1001", f"{WRONG} {i}", NOW)
        interaction = await _submit(_modal(engine, challenge), TITLE)
        assert [p.kind for p in interaction.payloads] == ["ephemeral", "announcement"]
        reply = interaction.payloads[0]
        assert reply.content is not None
        assert "✅" in reply.content
        assert "10" in reply.content
        announcement = interaction.payloads[1]
        assert announcement.content is not None
        assert "<@1001>" in announcement.content
        assert "10" in announcement.content

    async def test_wrong_guess_past_the_limit_keeps_playing(
        self, engine: GameEngine, challenge: Challenge
    ) -> None:
        for i in range(6):
            engine.submit_guess(challenge.id, "1001", f"{WRONG} {i}", NOW)
        interaction = await _submit(_modal(engine, challenge), WRONG)
        assert [p.kind for p in interaction.payloads] == ["ephemeral"]
        reply = interaction.payloads[0]
        assert reply.content is not None
        assert "❌" in reply.content
        assert "10" in reply.content  # the keep-playing 10-point offer

    async def test_revealed_challenge_yields_single_closed_notice(
        self, engine: GameEngine, challenge: Challenge, db: Database
    ) -> None:
        # VAL-GUESS-019 adapter half: one ephemeral closed notice, nothing else.
        _reveal_previous(engine, "g1", NEXT_DAY)
        interaction = await _submit(_modal(engine, challenge), TITLE)

        assert len(interaction.payloads) == 1
        payload = interaction.payloads[0]
        assert payload.kind == "ephemeral"
        assert payload.recipient == "1001"
        assert payload.content == "This challenge has closed."
        assert payload.embed is None
        assert payload.attachment is None
        # Zero mutation: no guesses logged, no challenge_users row.
        assert (
            db.query_one(
                "SELECT COUNT(*) AS c FROM guesses WHERE challenge_id = ?", (challenge.id,)
            )["c"]
            == 0
        )
        assert (
            db.query_one(
                "SELECT * FROM challenge_users WHERE challenge_id = ?", (challenge.id,)
            )
            is None
        )


class TestSecrecy:
    async def test_no_payload_contains_song_identity(
        self, engine: GameEngine, challenge: Challenge
    ) -> None:
        """VAL-GUESS-013/VAL-CROSS-010: title/artist never appear pre-reveal."""
        interaction = await _submit(_modal(engine, challenge), TITLE)
        for payload in interaction.payloads:
            text = payload.content or ""
            assert TITLE.lower() not in text.lower()
            assert ARTIST.lower() not in text.lower()

    async def test_announcement_uses_mention_not_plain_name(
        self, engine: GameEngine, challenge: Challenge
    ) -> None:
        interaction = await _submit(_modal(engine, challenge), TITLE)
        announcement = interaction.payloads[1]
        assert announcement.content is not None
        assert "<@1001>" in announcement.content
        assert "alice" not in announcement.content
