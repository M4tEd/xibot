"""Unit tests for the pure embed/message builders in songbot.bot.embeds.

Deterministic fixtures only — no engine, no I/O. Covers the daily challenge
embed (VAL-DAILY-001 shape + VAL-HEAR-001 level-0 advertisement), the reveal
embed (VAL-DAILY-008/009), the leaderboard embed (VAL-SCORE-009/011 field
shape), the solve announcement copy (VAL-GUESS-012 + mention format), the
guess/hear-more feedback copy, the pinned-#9 attachment filename, and the
pinned-#9 secrecy invariant for all pre-reveal copy.
"""

from __future__ import annotations

from pathlib import Path

import discord
import pytest

from songbot.bot.embeds import (
    ATTACHMENT_FILENAME,
    CHALLENGE_CLOSED_MESSAGE,
    announcement_content,
    daily_challenge_embed,
    format_seconds,
    guess_feedback_content,
    hear_more_content,
    hear_more_refusal_content,
    leaderboard_embed,
    reveal_embed,
    snippet_attachment,
)
from songbot.db import SongRow
from songbot.engine import (
    Challenge,
    GuessResult,
    LeaderboardEntry,
    Reveal,
    UnlockResult,
    Winner,
)
from tests.unit.test_engine_daily import _settings

SONG = SongRow(
    id=7,
    source="local",
    source_id="song-7",
    title="Neon Skyline",
    artist="Midnight Circuit",
    duration_sec=30.0,
    audio_ref="/music/song-7.mp3",
    raw_title="Midnight Circuit - Neon Skyline",
    created_at="2026-01-01T00:00:00+00:00",
)

CHALLENGE = Challenge(
    id=3,
    guild_id="g1",
    channel_id="c1",
    date="2026-08-13",
    song=SONG,
    snippet_offset_sec=4.0,
    status="active",
    skip_count=0,
    created_at="2026-08-13T16:00:00+00:00",
    revealed_at=None,
    snippet_paths={0: Path("/tmp/snippets/3/0.mp3")},
    created=True,
)


def _embed_text(embed: discord.Embed) -> str:
    """Flatten every user-visible string of an embed for content assertions."""
    parts: list[str] = [embed.title or "", embed.description or ""]
    if embed.footer and embed.footer.text:
        parts.append(embed.footer.text)
    for field_ in embed.fields:
        parts.append(field_.name)
        parts.append(str(field_.value))
    return "\n".join(parts)


class TestDailyChallengeEmbed:
    def test_title_is_daily_song_with_date(self, tmp_path: Path) -> None:
        embed = daily_challenge_embed(CHALLENGE, _settings(tmp_path))
        assert embed.title == "🎵 Daily Song — 2026-08-13"

    def test_how_to_play_instructions_present(self, tmp_path: Path) -> None:
        text = _embed_text(daily_challenge_embed(CHALLENGE, _settings(tmp_path)))
        assert "How to play" in text
        assert "Hear more" in text
        assert "Guess" in text
        assert "Leaderboard" in text

    def test_advertises_level_zero_state(self, tmp_path: Path) -> None:
        # VAL-HEAR-001: the post advertises the 1s snippet and 100 points.
        text = _embed_text(daily_challenge_embed(CHALLENGE, _settings(tmp_path)))
        assert "1s" in text
        assert "100" in text
        assert "6" in text  # max guesses per day

    def test_never_leaks_song_identity(self, tmp_path: Path) -> None:
        # Pinned #9 secrecy invariant: no title/artist/raw_title pre-reveal.
        text = _embed_text(daily_challenge_embed(CHALLENGE, _settings(tmp_path))).lower()
        for secret in (SONG.title, SONG.artist or "", SONG.raw_title):
            assert secret
            assert secret.lower() not in text


class TestRevealEmbed:
    def test_names_song_and_winners(self) -> None:
        reveal = Reveal(
            challenge_id=3,
            guild_id="guild-1",
            channel_id="channel-1",
            date="2026-08-13",
            song=SONG,
            winners=(
                Winner(
                    user_id="1001",
                    guesses_used=1,
                    points_awarded=100,
                    solved_at="2026-08-13T16:10:00+00:00",
                ),
                Winner(
                    user_id="1002",
                    guesses_used=3,
                    points_awarded=75,
                    solved_at="2026-08-13T17:00:00+00:00",
                ),
            ),
            revealed_at="2026-08-14T15:00:00+00:00",
        )
        text = _embed_text(reveal_embed(reveal))
        assert "Neon Skyline" in text
        assert "Midnight Circuit" in text
        assert "<@1001>" in text
        assert "<@1002>" in text
        assert "100" in text
        assert "75" in text
        # Winners appear in solve order.
        assert text.index("<@1001>") < text.index("<@1002>")

    def test_without_winners_says_nobody_got_it(self) -> None:
        reveal = Reveal(
            challenge_id=3,
            guild_id="guild-1",
            channel_id="channel-1",
            date="2026-08-13",
            song=SONG,
            winners=(),
            revealed_at="2026-08-14T15:00:00+00:00",
        )
        text = _embed_text(reveal_embed(reveal)).lower()
        assert "neon skyline" in text
        assert "midnight circuit" in text
        assert "nobody" in text
        assert "<@" not in text

    def test_reveal_title_is_not_a_daily_post_title(self) -> None:
        reveal = Reveal(
            challenge_id=3,
            guild_id="guild-1",
            channel_id="channel-1",
            date="2026-08-13",
            song=SONG,
            winners=(),
            revealed_at="2026-08-14T15:00:00+00:00",
        )
        embed = reveal_embed(reveal)
        assert embed.title is not None
        assert not embed.title.startswith("🎵 Daily Song — ")


class TestLeaderboardEmbed:
    def test_renders_rank_user_points_wins_streak(self) -> None:
        entries = [
            LeaderboardEntry(user_id="1001", total_points=225, wins=3, current_streak=3),
            LeaderboardEntry(user_id="1002", total_points=100, wins=1, current_streak=1),
        ]
        embed = leaderboard_embed(entries)
        text = _embed_text(embed)
        assert embed.title is not None
        assert "Leaderboard" in embed.title
        assert "<@1001>" in text
        assert "<@1002>" in text
        assert "225" in text
        assert "100" in text
        assert "3" in text  # wins/streak digits present
        assert "1" in text
        # Order preserved: rank 1 before rank 2.
        assert text.index("<@1001>") < text.index("<@1002>")

    def test_ranks_are_contiguous_from_one(self) -> None:
        entries = [
            LeaderboardEntry(user_id=str(2000 + i), total_points=100 - i, wins=1, current_streak=1)
            for i in range(4)
        ]
        text = _embed_text(leaderboard_embed(entries))
        for rank in (1, 2, 3, 4):
            assert f"{rank}." in text


def _solve_result(**overrides: object) -> GuessResult:
    """A correct-guess `GuessResult` for announcement tests (level 0 by default)."""
    base: dict[str, object] = {
        "outcome": "correct",
        "matched_title": True,
        "matched_artist": False,
        "is_both": False,
        "guesses_used": 1,
        "guesses_left": 5,
        "points_awarded": 100,
        "snippet_level": 0,
        "announce": True,
    }
    base.update(overrides)
    return GuessResult(**base)  # type: ignore[arg-type]


class TestAnnouncementContent:
    def test_mention_format_and_singular_guess(self, tmp_path: Path) -> None:
        content = announcement_content("1001", _solve_result(), _settings(tmp_path))
        assert "<@1001>" in content
        assert "1 guess" in content
        assert "100" in content
        assert content.startswith("🎉")

    def test_plural_guesses(self, tmp_path: Path) -> None:
        content = announcement_content(
            "1002", _solve_result(guesses_used=3, points_awarded=75), _settings(tmp_path)
        )
        assert "3 guesses" in content
        assert "75" in content

    def test_includes_snippet_length_heard_at_solve(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        content = announcement_content("1001", _solve_result(snippet_level=2), settings)
        # Level 2 of the configured ladder is 4s, formatted with format_seconds.
        assert format_seconds(settings.snippet_lengths[2]) in content
        assert "**4s** of audio" in content

    def test_snippet_length_tracks_the_solve_level(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        level_zero = announcement_content("1001", _solve_result(snippet_level=0), settings)
        level_four = announcement_content("1001", _solve_result(snippet_level=4), settings)
        assert "1s" in level_zero
        assert "16s" in level_four

    def test_never_contains_song_identity(self, tmp_path: Path) -> None:
        content = announcement_content(
            "1001", _solve_result(guesses_used=2), _settings(tmp_path)
        ).lower()
        assert "neon skyline" not in content
        assert "midnight circuit" not in content


class TestGuessFeedbackContent:
    def _result(self, **overrides: object) -> GuessResult:
        base: dict[str, object] = {
            "outcome": "wrong",
            "matched_title": False,
            "matched_artist": False,
            "is_both": False,
            "guesses_used": 1,
            "guesses_left": 5,
            "points_awarded": 0,
            "snippet_level": 0,
            "announce": False,
        }
        base.update(overrides)
        return GuessResult(**base)  # type: ignore[arg-type]

    def test_correct_title(self) -> None:
        content = guess_feedback_content(
            self._result(outcome="correct", matched_title=True, points_awarded=100, announce=True)
        )
        assert "✅" in content
        assert "title" in content.lower()
        assert "100" in content
        assert "Neon Skyline" not in content  # secrecy: never name the song

    def test_correct_artist(self) -> None:
        content = guess_feedback_content(
            self._result(outcome="correct", matched_artist=True, points_awarded=100, announce=True)
        )
        assert "artist" in content.lower()
        assert "Midnight Circuit" not in content

    def test_correct_both_shows_bonus(self) -> None:
        content = guess_feedback_content(
            self._result(
                outcome="correct",
                matched_title=True,
                matched_artist=True,
                is_both=True,
                points_awarded=150,
                announce=True,
            )
        )
        assert "bonus" in content.lower()
        assert "150" in content

    def test_wrong_reports_guesses_left(self) -> None:
        content = guess_feedback_content(
            self._result(outcome="wrong", guesses_used=1, guesses_left=5)
        )
        assert "❌" in content
        assert "5" in content

    def test_wrong_last_guess_says_limit_reached(self) -> None:
        content = guess_feedback_content(
            self._result(outcome="wrong", guesses_used=6, guesses_left=0)
        )
        assert "❌" in content
        assert "0" in content or "last" in content.lower() or "no guesses" in content.lower()

    def test_already_solved(self) -> None:
        content = guess_feedback_content(self._result(outcome="already_solved"))
        assert "already" in content.lower()

    def test_limit_reached(self) -> None:
        content = guess_feedback_content(
            self._result(outcome="limit_reached", guesses_used=6, guesses_left=0)
        )
        assert "6" in content or "limit" in content.lower()

    def test_empty_guess_asks_for_input(self) -> None:
        content = guess_feedback_content(self._result(outcome="empty", guesses_used=0))
        assert "enter a guess" in content.lower()

    def test_challenge_closed(self) -> None:
        assert (
            guess_feedback_content(self._result(outcome="challenge_closed"))
            == CHALLENGE_CLOSED_MESSAGE
        )
        assert CHALLENGE_CLOSED_MESSAGE == "This challenge has closed."


class TestHearMoreContent:
    def test_reports_level_seconds_and_points(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        content = hear_more_content(
            UnlockResult(level=1, path=Path("/tmp/1.mp3"), potential_points=75), settings
        )
        assert "1" in content  # level
        assert "2s" in content  # seconds for level 1
        assert "75" in content  # new potential points

    def test_ladder_levels(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        expectations = {1: ("2s", "75"), 2: ("4s", "50"), 3: ("8s", "30"), 4: ("16s", "15")}
        for level, (seconds, points) in expectations.items():
            content = hear_more_content(
                UnlockResult(
                    level=level,
                    path=Path(f"/tmp/{level}.mp3"),
                    potential_points=settings.snippet_points[level],
                ),
                settings,
            )
            assert seconds in content
            assert points in content

    def test_refusal_max_level_mentions_longest_snippet(self, tmp_path: Path) -> None:
        content = hear_more_refusal_content("max_level", _settings(tmp_path))
        assert "16s" in content

    def test_refusal_solved(self, tmp_path: Path) -> None:
        assert "already" in hear_more_refusal_content("solved", _settings(tmp_path)).lower()

    def test_refusal_closed_is_the_closed_message(self, tmp_path: Path) -> None:
        assert hear_more_refusal_content("closed", _settings(tmp_path)) == CHALLENGE_CLOSED_MESSAGE


class TestSnippetAttachment:
    def test_filename_is_always_songbot_snippet(self, tmp_path: Path) -> None:
        assert ATTACHMENT_FILENAME == "songbot-snippet.mp3"
        source = tmp_path / "3.mp3"
        source.write_bytes(b"fake-mp3")
        attachment = snippet_attachment(source)
        try:
            assert attachment.filename == "songbot-snippet.mp3"
            assert "Neon" not in attachment.filename
        finally:
            attachment.close()


@pytest.mark.parametrize("secret", ["Neon Skyline", "Midnight Circuit"])
def test_no_pre_reveal_copy_contains_song_identity(secret: str, tmp_path: Path) -> None:
    """Pinned #9: none of the pre-reveal builder output may name the song."""
    settings = _settings(tmp_path)
    pre_reveal_texts = [
        _embed_text(daily_challenge_embed(CHALLENGE, settings)),
        announcement_content("1001", _solve_result(), settings),
        guess_feedback_content(
            GuessResult(
                outcome="correct",
                matched_title=True,
                matched_artist=True,
                is_both=True,
                guesses_used=1,
                guesses_left=5,
                points_awarded=150,
                snippet_level=0,
                announce=True,
            )
        ),
        hear_more_content(
            UnlockResult(level=4, path=Path("/tmp/4.mp3"), potential_points=15), settings
        ),
    ]
    for text in pre_reveal_texts:
        assert secret.lower() not in text.lower()
