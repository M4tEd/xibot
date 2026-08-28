"""End-to-end tests for the harness CLI as a one-shot OS process.

Drives the REAL stack — real SnippetGenerator + ffmpeg, the real fixture
music library, a real SQLite file — through subprocess invocations of
``python -m songbot.harness`` with a per-test tmp DATABASE_PATH /
SNIPPET_CACHE_DIR. YouTube is disabled (local-only catalog) and every run
uses a bogus Discord token plus the unroutable DISCORD_API_BASE
(``http://127.0.0.1:9``): all scenarios must exit 0 with zero Discord traffic
(VAL-OPS-008).

Covers the harness-level daily/snippet assertions (VAL-DAILY-001/002/003/004,
VAL-SNIP-001/002/008), the hear-more ladder (VAL-HEAR-003/004/005/008), the
guess flow incl. secrecy (VAL-GUESS-001/002/012, VAL-CROSS-010), reset
(VAL-OPS-007), status (VAL-OPS-006), --now determinism (VAL-OPS-009),
graceful pre-challenge interactions (VAL-CROSS-017), and the health endpoint
(VAL-OPS-001).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import mutagen
import pytest

REPO = Path(__file__).resolve().parents[2]
FIXTURE_MUSIC = REPO / "data" / "fixture-music"
HEALTH_PORT = 3108

DAY1_NOW = "2026-08-10T15:00:00-03:00"  # Halifax-local -> challenge date 2026-08-10
DAY2_NOW = "2026-08-11T13:00:00-03:00"  # -> 2026-08-11
DAY3_NOW = "2026-08-12T13:00:00-03:00"  # -> 2026-08-12
DAY4_NOW = "2026-08-13T13:00:00-03:00"  # -> 2026-08-13

PLAYLIST_URL = "https://youtube.com/playlist?list=PLDzqiyJzN_jBRIJhUB_vmD5jPIHNkflc1"
_HALIFAX_DST = timezone(timedelta(hours=-3))  # ADT (August)


def day_now(offset_days: int) -> str:
    """The ISO --now for ``2026-08-10 + offset_days`` at 15:00 Halifax-local."""
    base = datetime(2026, 8, 10, 15, 0, tzinfo=_HALIFAX_DST)
    return (base + timedelta(days=offset_days)).isoformat()

pytestmark = pytest.mark.skipif(
    not FIXTURE_MUSIC.is_dir() or len(list(FIXTURE_MUSIC.iterdir())) < 8,
    reason="fixture music library not present",
)


def run_harness(
    tmp_path: Path, *args: str, env_overrides: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run one harness scenario as a one-shot process with isolated state."""
    env = os.environ.copy()
    env.update(
        {
            "DATABASE_PATH": str(tmp_path / "songbot.db"),
            "SNIPPET_CACHE_DIR": str(tmp_path / "snippets"),
            "LOCAL_MUSIC_DIR": str(FIXTURE_MUSIC),
            "YOUTUBE_PLAYLIST_URL": "",  # local-only catalog (pinned #10)
            "DISCORD_BOT_TOKEN": "invalid-token",  # never used (VAL-OPS-008)
            "DISCORD_API_BASE": "http://127.0.0.1:9",  # unroutable discard address
            "DISCORD_GUILD_ID": "harness-guild",
            "DISCORD_CHANNEL_ID": "harness-channel",
        }
    )
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "songbot.harness", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,  # no .env here: every setting comes from the environment
        timeout=120,
    )


def run_json(
    tmp_path: Path, *args: str, env_overrides: dict[str, str] | None = None
) -> dict[str, Any]:
    """Run a scenario, assert exit 0, and return its parsed stdout JSON."""
    result = run_harness(tmp_path, *args, env_overrides=env_overrides)
    assert result.returncode == 0, f"{args} failed: {result.stderr[-2000:]}"
    return json.loads(result.stdout)


def db_rows(tmp_path: Path, sql: str) -> list[tuple[Any, ...]]:
    conn = sqlite3.connect(tmp_path / "songbot.db")
    try:
        return list(conn.execute(sql))
    finally:
        conn.close()


def db_exec(tmp_path: Path, sql: str, params: tuple[Any, ...] = ()) -> None:
    """Execute a write against the test DB (e.g. arranging a revealed state)."""
    conn = sqlite3.connect(tmp_path / "songbot.db")
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 0, out.stderr
    return float(out.stdout.strip())


def kinds(out: dict[str, Any]) -> list[str]:
    return [p["kind"] for p in out["payloads"]]


class TestDailyPostFlow:
    def test_post_records_channel_payload_and_full_snippet_set(
        self, tmp_path: Path
    ) -> None:
        run_json(tmp_path, "reset")
        out = run_json(tmp_path, "post", "--now", DAY1_NOW)

        # VAL-DAILY-001: exactly one channel payload with embed/buttons/attachment.
        assert kinds(out) == ["channel"]
        payload = out["payloads"][0]
        assert payload["recipient"] is None
        assert payload["embed"]["title"].startswith("🎵 Daily Song — 2026-08-10")
        assert "how to play" in payload["embed"]["description"].lower()
        custom_ids = {c["custom_id"] for c in payload["components"]}
        assert custom_ids == {"songbot:hear_more", "songbot:guess", "songbot:leaderboard"}

        # VAL-DAILY-002: the attachment is the level-0 (1s) snippet on disk.
        assert len(payload["attachments"]) == 1
        attachment = payload["attachments"][0]
        assert attachment["filename"] == "songbot-snippet.mp3"
        challenge_id = out["state"]["challenge"]["id"]
        assert attachment["path"] == str(tmp_path / "snippets" / str(challenge_id) / "0.mp3")
        snippet = Path(attachment["path"])
        assert snippet.is_file()
        assert attachment["size"] == snippet.stat().st_size
        assert abs(ffprobe_duration(snippet) - 1.0) <= 0.05

        # VAL-DAILY-003 / VAL-SNIP-001/002: all five levels exist on disk with
        # ladder durations within 50 ms.
        for level, target in enumerate((1.0, 2.0, 4.0, 8.0, 16.0)):
            level_file = tmp_path / "snippets" / str(challenge_id) / f"{level}.mp3"
            assert level_file.is_file()
            assert level_file.stat().st_size > 0
            assert abs(ffprobe_duration(level_file) - target) <= 0.05

        # The challenge row matches (active, offset within the song).
        rows = db_rows(
            tmp_path,
            "SELECT c.status, c.date, c.snippet_offset_sec, s.duration_sec"
            " FROM challenges c JOIN songs s ON s.id = c.song_id",
        )
        assert len(rows) == 1
        status, date_str, offset, duration = rows[0]
        assert (status, date_str) == ("active", "2026-08-10")
        assert 0 <= offset <= duration - 16.0 + 1e-6

    def test_repeat_post_is_already_posted_and_reheals_cache(
        self, tmp_path: Path
    ) -> None:
        run_json(tmp_path, "reset")
        first = run_json(tmp_path, "post", "--now", DAY1_NOW)
        challenge_id = first["state"]["challenge"]["id"]

        # VAL-DAILY-004 / pinned #4: exact compact JSON, no second post.
        second = run_harness(tmp_path, "post", "--now", DAY1_NOW)
        assert second.returncode == 0
        assert second.stdout.strip() == '{"already_posted": true, "messages": []}'
        assert db_rows(tmp_path, "SELECT COUNT(*) FROM challenges") == [(1,)]

        # VAL-SNIP-008 (engine re-heal): delete the cache dir; the repeat post
        # regenerates all five levels while still reporting already_posted.
        shutil.rmtree(tmp_path / "snippets" / str(challenge_id))
        third = run_harness(tmp_path, "post", "--now", DAY1_NOW)
        assert third.returncode == 0
        assert third.stdout.strip() == '{"already_posted": true, "messages": []}'
        for level, target in enumerate((1.0, 2.0, 4.0, 8.0, 16.0)):
            level_file = tmp_path / "snippets" / str(challenge_id) / f"{level}.mp3"
            assert level_file.is_file()
            assert abs(ffprobe_duration(level_file) - target) <= 0.05
        assert db_rows(tmp_path, "SELECT COUNT(*) FROM challenges") == [(1,)]

    def test_now_drives_challenge_dates(self, tmp_path: Path) -> None:
        run_json(tmp_path, "reset")
        run_json(tmp_path, "post", "--now", DAY1_NOW)
        status = run_json(tmp_path, "status", "--now", "2026-08-10T16:00:00-03:00")
        assert status["state"]["date"] == "2026-08-10"
        assert status["state"]["challenge"]["date"] == "2026-08-10"

        run_json(tmp_path, "post", "--now", DAY2_NOW)
        dates = [r[0] for r in db_rows(tmp_path, "SELECT date FROM challenges ORDER BY date")]
        assert dates == ["2026-08-10", "2026-08-11"]


class TestGameplayFlow:
    def test_full_day_cycle_with_secrecy(self, tmp_path: Path) -> None:
        run_json(tmp_path, "reset")
        post = run_json(tmp_path, "post", "--now", DAY1_NOW)
        challenge_id = post["state"]["challenge"]["id"]
        status = run_json(tmp_path, "status", "--now", DAY1_NOW)
        title = status["state"]["challenge"]["song"]["title"]
        artist = status["state"]["challenge"]["song"]["artist"]

        transcript: list[dict[str, Any]] = list(post["payloads"])

        # VAL-HEAR-003/004/005: four escalations, points 75/50/30/15, correct
        # per-level attachments with real durations.
        hear = run_json(tmp_path, "hear-more", "--user", "alice", "--times", "4",
                        "--now", DAY1_NOW)
        assert kinds(hear) == ["ephemeral"] * 4
        for payload, points, level in zip(
            hear["payloads"], ("75", "50", "30", "15"), (1, 2, 3, 4), strict=True
        ):
            assert payload["recipient"] == "alice"
            assert points in payload["content"]
            assert len(payload["attachments"]) == 1
            path = Path(payload["attachments"][0]["path"])
            assert path == tmp_path / "snippets" / str(challenge_id) / f"{level}.mp3"
            assert path.is_file()
            assert payload["attachments"][0]["size"] > 0
        transcript += hear["payloads"]
        assert hear["state"]["user"]["snippet_level"] == 4

        # VAL-HEAR-008: a 5th press is an ephemeral max-level notice, no attachment.
        extra = run_json(tmp_path, "hear-more", "--user", "alice", "--now", DAY1_NOW)
        assert kinds(extra) == ["ephemeral"]
        assert extra["payloads"][0]["attachments"] == []
        transcript += extra["payloads"]

        # VAL-GUESS-001/002/012: modal shape, ephemeral correct feedback, one
        # public announcement mentioning the user, guess count, and points.
        # alice is at level 4, so a title-only solve banks 15 points.
        guess = run_json(tmp_path, "guess", "--user", "alice", "--text", title,
                         "--now", DAY1_NOW)
        assert kinds(guess) == ["modal", "ephemeral", "announcement"]
        modal = guess["payloads"][0]
        assert len(modal["components"]) == 1
        assert modal["components"][0]["placeholder"] == "Artist or title..."
        feedback = guess["payloads"][1]
        assert feedback["recipient"] == "alice"
        assert "✅" in feedback["content"]
        assert "15" in feedback["content"]
        announcement = guess["payloads"][2]
        assert "<@alice>" in announcement["content"]
        assert "1 guess" in announcement["content"]
        assert "15" in announcement["content"]
        transcript += guess["payloads"]

        # bob: one wrong guess (5 left), then a correct artist guess (100 pts).
        wrong = run_json(tmp_path, "guess", "--user", "bob", "--text",
                         "zxqv unrelated noise", "--now", DAY1_NOW)
        assert kinds(wrong) == ["modal", "ephemeral"]
        assert "❌" in wrong["payloads"][1]["content"]
        assert "5" in wrong["payloads"][1]["content"]
        solved = run_json(tmp_path, "guess", "--user", "bob", "--text", artist,
                          "--now", DAY1_NOW)
        assert kinds(solved) == ["modal", "ephemeral", "announcement"]
        assert "100" in solved["payloads"][1]["content"]
        transcript += wrong["payloads"] + solved["payloads"]

        # Leaderboard: ephemeral to the requester, alice+bob listed.
        board = run_json(tmp_path, "leaderboard", "--user", "carol", "--now", DAY1_NOW)
        assert kinds(board) == ["ephemeral"]
        assert board["payloads"][0]["recipient"] == "carol"
        description = board["payloads"][0]["embed"]["description"]
        assert "<@bob>" in description
        assert "<@alice>" in description
        transcript += board["payloads"]

        # VAL-CROSS-010: nothing pre-reveal leaks the song identity.
        for payload in transcript:
            haystack = json.dumps(payload, ensure_ascii=False).lower()
            assert title.lower() not in haystack
            assert artist.lower() not in haystack

        # VAL-DAILY-008: advance-day reveals (song + winners) BEFORE the new post.
        advanced = run_json(tmp_path, "advance-day", "--now", DAY1_NOW)
        assert kinds(advanced) == ["announcement", "channel"]
        reveal_text = advanced["payloads"][0]["embed"]["description"]
        assert title in reveal_text
        assert artist in reveal_text
        assert "<@alice>" in reveal_text
        assert "<@bob>" in reveal_text
        assert "2026-08-11" in advanced["payloads"][1]["embed"]["title"]
        rows = db_rows(
            tmp_path, "SELECT status FROM challenges WHERE date = '2026-08-10'"
        )
        assert rows == [("revealed",)]

    def test_interactions_before_any_challenge_are_graceful(
        self, tmp_path: Path
    ) -> None:
        run_json(tmp_path, "reset")
        for args in (
            ("hear-more", "--user", "alice"),
            ("guess", "--user", "alice", "--text", "anything"),
            ("leaderboard", "--user", "alice"),
        ):
            out = run_json(tmp_path, *args, "--now", DAY1_NOW)
            assert kinds(out) == ["ephemeral"]
            assert out["payloads"][0]["recipient"] == "alice"
        for table in ("challenge_users", "guesses", "user_stats"):
            assert db_rows(tmp_path, f"SELECT COUNT(*) FROM {table}") == [(0,)]

    def test_reset_wipes_state_and_post_recovers(self, tmp_path: Path) -> None:
        run_json(tmp_path, "reset")
        run_json(tmp_path, "post", "--now", DAY1_NOW)
        run_json(tmp_path, "hear-more", "--user", "alice", "--now", DAY1_NOW)
        run_json(tmp_path, "guess", "--user", "alice", "--text", "nope", "--now", DAY1_NOW)

        run_json(tmp_path, "reset")
        for table in ("challenges", "challenge_users", "guesses", "user_stats"):
            assert db_rows(tmp_path, f"SELECT COUNT(*) FROM {table}") == [(0,)]
        assert list((tmp_path / "snippets").rglob("*.mp3")) == []

        out = run_json(tmp_path, "post", "--now", DAY1_NOW)
        assert kinds(out) == ["channel"]


class TestAdminFlow:
    """Admin scenarios driven as one-shot CLI processes (real stack + ffmpeg)."""

    def test_admin_post_happy_path_and_idempotent_repeat(self, tmp_path: Path) -> None:
        """VAL-ADMIN-001/002: post + ack, then a compact already_posted repeat."""
        run_json(tmp_path, "reset")
        out = run_json(tmp_path, "admin-post", "--as-admin", "--now", DAY1_NOW)

        # Exactly one channel post (same shape as the scheduled post) + one ack.
        assert kinds(out) == ["channel", "ephemeral"]
        post = out["payloads"][0]
        assert post["recipient"] is None
        assert post["embed"]["title"].startswith("🎵 Daily Song — 2026-08-10")
        custom_ids = {c["custom_id"] for c in post["components"]}
        assert custom_ids == {"songbot:hear_more", "songbot:guess", "songbot:leaderboard"}
        assert len(post["attachments"]) == 1
        attachment = post["attachments"][0]
        assert attachment["filename"] == "songbot-snippet.mp3"
        challenge_id = out["state"]["challenge"]["id"]
        assert attachment["path"] == str(tmp_path / "snippets" / str(challenge_id) / "0.mp3")
        assert abs(ffprobe_duration(Path(attachment["path"])) - 1.0) <= 0.05
        ack = out["payloads"][1]
        assert ack["recipient"] == "admin"
        assert out["state"]["outcome"] == "posted"

        rows = db_rows(
            tmp_path,
            "SELECT c.status, c.date, c.snippet_offset_sec, s.duration_sec"
            " FROM challenges c JOIN songs s ON s.id = c.song_id",
        )
        assert len(rows) == 1
        status, date_str, offset, duration = rows[0]
        assert (status, date_str) == ("active", "2026-08-10")
        assert 0 <= offset <= duration - 16.0 + 1e-6
        # All five snippet levels exist for the challenge.
        for level in range(5):
            assert (tmp_path / "snippets" / str(challenge_id) / f"{level}.mp3").is_file()

        # Repeat: pinned-#4 compact JSON, still one row, no new snippet files.
        second = run_harness(tmp_path, "admin-post", "--as-admin", "--now", DAY1_NOW)
        assert second.returncode == 0
        assert second.stdout.strip() == '{"already_posted": true, "messages": []}'
        assert db_rows(tmp_path, "SELECT COUNT(*) FROM challenges") == [(1,)]

    def test_admin_post_non_admin_denied_with_zero_mutation(self, tmp_path: Path) -> None:
        """VAL-ADMIN-009 (post): one ephemeral denial, no state change."""
        run_json(tmp_path, "reset")
        out = run_json(tmp_path, "admin-post", "--as-non-admin", "--now", DAY1_NOW)

        assert kinds(out) == ["ephemeral"]
        denial = out["payloads"][0]
        assert denial["recipient"] == "admin"
        assert "manage server" in denial["content"].lower()
        assert out["state"]["outcome"] == "denied"
        assert db_rows(tmp_path, "SELECT COUNT(*) FROM challenges") == [(0,)]
        assert db_rows(tmp_path, "SELECT COUNT(*) FROM songs") == [(0,)]
        assert list((tmp_path / "snippets").rglob("*.mp3")) == []

    def test_admin_pingrole_then_daily_posts_mention_the_role(self, tmp_path: Path) -> None:
        """Reaction-role opt-in end to end: announce, persist, ping on posts."""
        run_json(tmp_path, "reset")
        out = run_json(
            tmp_path, "admin-pingrole", "--as-admin", "--role", "role-9", "--now", DAY1_NOW
        )

        assert kinds(out) == ["announcement", "ephemeral"]
        announcement = out["payloads"][0]
        assert "🎵" in announcement["content"]
        assert "<@&role-9>" in announcement["content"]
        assert out["state"]["outcome"] == "ping_configured"
        assert db_rows(
            tmp_path,
            "SELECT role_id, emoji FROM ping_role_settings"
            " WHERE guild_id = 'harness-guild'",
        ) == [("role-9", "🎵")]

        # The next daily post (scheduled path AND admin path) pings the role.
        post = run_json(tmp_path, "post", "--now", DAY1_NOW)
        assert kinds(post) == ["channel"]
        assert post["payloads"][0]["content"] == "<@&role-9>"
        admin_post = run_json(tmp_path, "admin-post", "--as-admin", "--now", DAY2_NOW)
        assert admin_post["payloads"][0]["content"] == "<@&role-9>"

    def test_admin_pingrole_non_admin_denied_with_zero_mutation(self, tmp_path: Path) -> None:
        """VAL-ADMIN-009 (pingrole): one ephemeral denial, nothing persisted."""
        run_json(tmp_path, "reset")
        out = run_json(tmp_path, "admin-pingrole", "--as-non-admin", "--now", DAY1_NOW)

        assert kinds(out) == ["ephemeral"]
        assert "manage server" in out["payloads"][0]["content"].lower()
        assert out["state"]["outcome"] == "denied"
        assert db_rows(tmp_path, "SELECT COUNT(*) FROM ping_role_settings") == [(0,)]

    def test_admin_skip_replaces_song_and_resets_players(self, tmp_path: Path) -> None:
        """VAL-ADMIN-003/004 + VAL-CROSS-006: new song/offset, state reset."""
        run_json(tmp_path, "reset")
        post = run_json(tmp_path, "post", "--now", DAY1_NOW)
        old_challenge_id = post["state"]["challenge"]["id"]
        status = run_json(tmp_path, "status", "--now", DAY1_NOW)
        old_song_id = status["state"]["challenge"]["song_id"]
        old_title = status["state"]["challenge"]["song"]["title"]
        old_offset = status["state"]["challenge"]["snippet_offset_sec"]

        # todd builds in-progress state: level 2 + one wrong guess.
        run_json(tmp_path, "hear-more", "--user", "todd", "--times", "2", "--now", DAY1_NOW)
        run_json(tmp_path, "guess", "--user", "todd", "--text", "zxqv noise", "--now", DAY1_NOW)
        assert db_rows(
            tmp_path,
            "SELECT snippet_level, guesses_used FROM challenge_users WHERE user_id = 'todd'",
        ) == [(2, 1)]

        out = run_json(tmp_path, "admin-skip", "--as-admin", "--now", DAY1_NOW)

        # Pinned #5: exactly one ephemeral ack, no channel/announcement payload.
        assert kinds(out) == ["ephemeral"]
        assert out["payloads"][0]["recipient"] == "admin"
        assert out["state"]["outcome"] == "skipped"
        assert out["state"]["reason"] is None

        rows = db_rows(
            tmp_path,
            "SELECT song_id, snippet_offset_sec, status, skip_count FROM challenges",
        )
        assert len(rows) == 1
        new_song_id, new_offset, status, skip_count = rows[0]
        assert new_song_id != old_song_id
        assert new_offset != old_offset
        assert status == "active"
        assert skip_count == 1
        # Per-user state fully reset; old guesses gone.
        assert db_rows(tmp_path, "SELECT COUNT(*) FROM challenge_users") == [(0,)]
        assert db_rows(tmp_path, "SELECT COUNT(*) FROM guesses") == [(0,)]
        # The snippet cache was purged and regenerated (5 levels, correct durations).
        new_challenge_id = out["state"]["challenge"]["id"]
        for level, target in enumerate((1.0, 2.0, 4.0, 8.0, 16.0)):
            level_file = tmp_path / "snippets" / str(new_challenge_id) / f"{level}.mp3"
            assert level_file.is_file()
            assert abs(ffprobe_duration(level_file) - target) <= 0.05

        # todd restarts at level 0 -> 1 (75 pts) on the NEW song...
        hear = run_json(tmp_path, "hear-more", "--user", "todd", "--now", DAY1_NOW)
        assert "75" in hear["payloads"][0]["content"]
        assert hear["state"]["user"]["snippet_level"] == 1
        # ...the OLD title is now wrong...
        new_status = run_json(tmp_path, "status", "--now", DAY1_NOW)
        new_title = new_status["state"]["challenge"]["song"]["title"]
        wrong = run_json(
            tmp_path, "guess", "--user", "todd", "--text", old_title, "--now", DAY1_NOW
        )
        assert "❌" in wrong["payloads"][1]["content"]
        # ...and the NEW title solves (level 1 -> 75 points banked).
        solved = run_json(
            tmp_path, "guess", "--user", "todd", "--text", new_title, "--now", DAY1_NOW
        )
        assert "✅" in solved["payloads"][1]["content"]
        assert "75" in solved["payloads"][1]["content"]
        assert db_rows(
            tmp_path,
            "SELECT solved, points_awarded FROM challenge_users WHERE user_id = 'todd'",
        ) == [(1, 75)]
        # Secrecy: the skip ack named neither song.
        ack = out["payloads"][0]["content"].lower()
        assert old_title.lower() not in ack
        assert new_title.lower() not in ack
        # The old challenge id's cache dir is gone or repurposed for the new set
        # (SQLite rowid reuse means the ids usually coincide — either way the
        # files on disk belong to the replacement challenge).
        assert old_challenge_id == new_challenge_id or not (
            tmp_path / "snippets" / str(old_challenge_id)
        ).exists()

    def test_admin_skip_refused_after_solve_with_zero_mutation(self, tmp_path: Path) -> None:
        """VAL-ADMIN-006: a solved challenge refuses skip; solver state intact."""
        run_json(tmp_path, "reset")
        run_json(tmp_path, "post", "--now", DAY1_NOW)
        status = run_json(tmp_path, "status", "--now", DAY1_NOW)
        title = status["state"]["challenge"]["song"]["title"]
        song_id = status["state"]["challenge"]["song_id"]
        run_json(tmp_path, "guess", "--user", "uma", "--text", title, "--now", DAY1_NOW)

        out = run_json(tmp_path, "admin-skip", "--as-admin", "--now", DAY1_NOW)

        assert kinds(out) == ["ephemeral"]
        assert out["payloads"][0]["recipient"] == "admin"
        assert "solv" in out["payloads"][0]["content"].lower()
        assert out["state"]["outcome"] == "refused"
        assert out["state"]["reason"] == "solved"
        rows = db_rows(tmp_path, "SELECT song_id, status, skip_count FROM challenges")
        assert rows == [(song_id, "active", 0)]
        assert db_rows(
            tmp_path,
            "SELECT solved, points_awarded FROM challenge_users WHERE user_id = 'uma'",
        ) == [(1, 100)]
        assert db_rows(
            tmp_path,
            "SELECT total_points, wins, current_streak FROM user_stats WHERE user_id = 'uma'",
        ) == [(100, 1, 1)]

    def test_admin_skip_refused_when_revealed(self, tmp_path: Path) -> None:
        """VAL-ADMIN-005: a revealed challenge refuses skip with zero mutation."""
        run_json(tmp_path, "reset")
        run_json(tmp_path, "post", "--now", DAY1_NOW)
        db_exec(
            tmp_path,
            "UPDATE challenges SET status = 'revealed', revealed_at = ?"
            " WHERE date = '2026-08-10'",
            ("2026-08-10T16:30:00+00:00",),
        )

        out = run_json(tmp_path, "admin-skip", "--as-admin", "--now", DAY1_NOW)

        assert kinds(out) == ["ephemeral"]
        content = out["payloads"][0]["content"].lower()
        assert "revealed" in content or "no longer active" in content
        assert out["state"]["outcome"] == "refused"
        assert out["state"]["reason"] == "revealed"
        assert db_rows(tmp_path, "SELECT status, skip_count FROM challenges") == [
            ("revealed", 0)
        ]

    def test_admin_skip_non_admin_denied_with_zero_mutation(self, tmp_path: Path) -> None:
        """VAL-ADMIN-009 (skip): one ephemeral denial, song/offset untouched."""
        run_json(tmp_path, "reset")
        run_json(tmp_path, "post", "--now", DAY1_NOW)
        before = db_rows(tmp_path, "SELECT song_id, snippet_offset_sec FROM challenges")

        out = run_json(tmp_path, "admin-skip", "--as-non-admin", "--now", DAY1_NOW)

        assert kinds(out) == ["ephemeral"]
        assert "manage server" in out["payloads"][0]["content"].lower()
        assert out["state"]["outcome"] == "denied"
        assert db_rows(tmp_path, "SELECT song_id, snippet_offset_sec FROM challenges") == before
        assert db_rows(tmp_path, "SELECT COUNT(*) FROM challenges") == [(1,)]

    def test_admin_reload_local_catalog_and_idempotency(self, tmp_path: Path) -> None:
        """VAL-CATALOG-011 (local part) / VAL-CATALOG-012: upsert, no dupes."""
        run_json(tmp_path, "reset")
        out = run_json(tmp_path, "admin-reload", "--as-admin", "--now", DAY1_NOW)

        assert kinds(out) == ["ephemeral"]
        assert out["payloads"][0]["recipient"] == "admin"
        content = out["payloads"][0]["content"]
        assert "local" in content
        assert "8 added" in content
        sources = {s["source"]: s for s in out["state"]["sources"]}
        assert sources["local"]["added"] == 8
        assert sources["local"]["error"] is None
        assert db_rows(
            tmp_path, "SELECT source, COUNT(*) FROM songs GROUP BY source"
        ) == [("local", 8)]
        # Every row is well-formed; the untagged fixture fell back to filename.
        assert db_rows(
            tmp_path,
            "SELECT COUNT(*) FROM songs WHERE title IS NULL OR duration_sec <= 0"
            " OR audio_ref = ''",
        ) == [(0,)]
        assert db_rows(
            tmp_path,
            "SELECT title, artist FROM songs WHERE audio_ref LIKE '%Retro Waves%'",
        ) == [("Sunset Drive", "Retro Waves")]
        assert db_rows(
            tmp_path,
            "SELECT source, source_id, COUNT(*) c FROM songs GROUP BY 1, 2 HAVING c > 1",
        ) == []

        # Second reload: no new rows, no duplicates, ids stable (updates in place).
        max_id_before = db_rows(tmp_path, "SELECT MAX(id) FROM songs")[0][0]
        second = run_json(tmp_path, "admin-reload", "--as-admin", "--now", DAY1_NOW)
        sources2 = {s["source"]: s for s in second["state"]["sources"]}
        assert sources2["local"]["added"] == 0
        assert sources2["local"]["updated"] == 8
        assert db_rows(tmp_path, "SELECT COUNT(*), MAX(id) FROM songs") == [(8, max_id_before)]

    def test_admin_reload_adds_drops_and_retains_songs(self, tmp_path: Path) -> None:
        """VAL-ADMIN-007/008 + VAL-CROSS-008: reload reflects the music dir."""
        music = tmp_path / "music"
        shutil.copytree(FIXTURE_MUSIC, music)
        env = {"LOCAL_MUSIC_DIR": str(music)}
        run_json(tmp_path, "reset", env_overrides=env)
        run_json(tmp_path, "admin-reload", "--as-admin", "--now", DAY1_NOW, env_overrides=env)
        assert db_rows(tmp_path, "SELECT COUNT(*) FROM songs") == [(8,)]

        # VAL-ADMIN-007: a newly added tagged file is picked up.
        added_file = music / "Reload Test Artist - Reload Test Title.mp3"
        shutil.copy(FIXTURE_MUSIC / "Midnight Circuit - Neon Skyline.mp3", added_file)
        audio = mutagen.File(added_file, easy=True)
        assert audio is not None
        audio["title"] = ["Reload Test Title"]
        audio["artist"] = ["Reload Test Artist"]
        audio.save()
        out = run_json(
            tmp_path, "admin-reload", "--as-admin", "--now", DAY1_NOW, env_overrides=env
        )
        sources = {s["source"]: s for s in out["state"]["sources"]}
        assert sources["local"]["added"] == 1
        assert db_rows(
            tmp_path,
            "SELECT title, artist FROM songs"
            " WHERE source_id = 'Reload Test Artist - Reload Test Title.mp3'",
        ) == [("Reload Test Title", "Reload Test Artist")]
        assert db_rows(tmp_path, "SELECT COUNT(*) FROM songs") == [(9,)]
        # A second reload does not duplicate the row.
        run_json(tmp_path, "admin-reload", "--as-admin", "--now", DAY1_NOW, env_overrides=env)
        assert db_rows(tmp_path, "SELECT COUNT(*) FROM songs") == [(9,)]

        # VAL-ADMIN-008a: removing a never-used song deletes its row.
        added_file.unlink()
        out = run_json(
            tmp_path, "admin-reload", "--as-admin", "--now", DAY1_NOW, env_overrides=env
        )
        sources = {s["source"]: s for s in out["state"]["sources"]}
        assert sources["local"]["removed"] == 1
        assert db_rows(tmp_path, "SELECT COUNT(*) FROM songs") == [(8,)]

        # VAL-ADMIN-008b: removing a REFERENCED song retains its row.
        run_json(tmp_path, "post", "--now", DAY1_NOW, env_overrides=env)
        ref = db_rows(
            tmp_path,
            "SELECT s.audio_ref FROM challenges c JOIN songs s ON s.id = c.song_id",
        )[0][0]
        referenced = Path(ref)
        moved = referenced.parent / (referenced.name + ".moved")
        referenced.rename(moved)
        try:
            out = run_json(
                tmp_path, "admin-reload", "--as-admin", "--now", DAY1_NOW, env_overrides=env
            )
            sources = {s["source"]: s for s in out["state"]["sources"]}
            assert sources["local"]["removed"] == 0
            assert sources["local"]["retained"] == 1
            assert db_rows(tmp_path, "SELECT COUNT(*) FROM songs") == [(8,)]
            # The challenge still joins its song (history intact).
            assert db_rows(
                tmp_path,
                "SELECT COUNT(*) FROM challenges c JOIN songs s ON s.id = c.song_id",
            ) == [(1,)]
        finally:
            moved.rename(referenced)

        # VAL-CROSS-008: the next day posts fine and references a song that
        # is still in the catalog (never a deleted one).
        advanced = run_json(
            tmp_path, "advance-day", "--now", DAY1_NOW, env_overrides=env
        )
        assert kinds(advanced) == ["announcement", "channel"]
        day2_song = int(advanced["state"]["challenge"]["song_id"])
        assert db_rows(
            tmp_path, f"SELECT COUNT(*) FROM songs WHERE id = {day2_song}"
        ) == [(1,)]

    def test_admin_reload_non_admin_denied_with_zero_mutation(self, tmp_path: Path) -> None:
        """VAL-ADMIN-009 (reload): one ephemeral denial, songs table untouched."""
        run_json(tmp_path, "reset")
        out = run_json(tmp_path, "admin-reload", "--as-non-admin", "--now", DAY1_NOW)

        assert kinds(out) == ["ephemeral"]
        assert "manage server" in out["payloads"][0]["content"].lower()
        assert out["state"]["outcome"] == "denied"
        assert out["state"]["sources"] == []
        assert db_rows(tmp_path, "SELECT COUNT(*) FROM songs") == [(0,)]

    def test_admin_reload_with_real_youtube_playlist(self, tmp_path: Path) -> None:
        """VAL-CATALOG-011 (youtube part): the real playlist through admin-reload.

        Network: YouTube only (allowed). Tolerates playlist drift (>= 200).
        """
        env = {"YOUTUBE_PLAYLIST_URL": PLAYLIST_URL}
        run_json(tmp_path, "reset", env_overrides=env)
        out = run_json(
            tmp_path, "admin-reload", "--as-admin", "--now", DAY1_NOW, env_overrides=env
        )

        assert kinds(out) == ["ephemeral"]
        sources = {s["source"]: s for s in out["state"]["sources"]}
        assert sources["local"]["added"] == 8
        assert sources["youtube"]["error"] is None
        assert sources["youtube"]["added"] >= 200
        counts = dict(
            db_rows(tmp_path, "SELECT source, COUNT(*) FROM songs GROUP BY source")
        )
        assert counts["local"] == 8
        assert counts["youtube"] >= 200
        assert db_rows(
            tmp_path,
            "SELECT source, source_id, COUNT(*) c FROM songs GROUP BY 1, 2 HAVING c > 1",
        ) == []
        # Every youtube row has a watch URL; every local row an absolute path.
        assert db_rows(
            tmp_path,
            "SELECT COUNT(*) FROM songs WHERE source = 'youtube'"
            " AND audio_ref NOT LIKE 'https://%watch%'",
        ) == [(0,)]
        assert db_rows(
            tmp_path,
            "SELECT COUNT(*) FROM songs WHERE source = 'local' AND audio_ref NOT LIKE '/%'",
        ) == [(0,)]

    def test_same_day_repost_idempotent_across_scheduler_and_admin(
        self, tmp_path: Path
    ) -> None:
        """VAL-CROSS-015: post + post + admin-post -> exactly one channel post."""
        run_json(tmp_path, "reset")
        first = run_json(tmp_path, "post", "--now", DAY1_NOW)
        assert kinds(first) == ["channel"]
        challenge_id = first["state"]["challenge"]["id"]
        snippet_dir = tmp_path / "snippets" / str(challenge_id)
        mtimes = {p.name: p.stat().st_mtime_ns for p in snippet_dir.glob("*.mp3")}

        second = run_harness(tmp_path, "post", "--now", DAY1_NOW)
        assert second.returncode == 0
        assert second.stdout.strip() == '{"already_posted": true, "messages": []}'
        third = run_harness(tmp_path, "admin-post", "--as-admin", "--now", DAY1_NOW)
        assert third.returncode == 0
        assert third.stdout.strip() == '{"already_posted": true, "messages": []}'

        assert db_rows(tmp_path, "SELECT COUNT(*) FROM challenges") == [(1,)]
        # Snippet cache reused, not regenerated.
        assert {p.name: p.stat().st_mtime_ns for p in snippet_dir.glob("*.mp3")} == mtimes


class TestCrossDayFlows:
    """Multi-day harness chains (advance-day + admin) from the contract."""

    def test_no_repeat_across_eight_days_then_history_reset(self, tmp_path: Path) -> None:
        """VAL-CROSS-007: 8 distinct picks over 8 days; day 9 resets history."""
        run_json(tmp_path, "reset")
        picked: list[int] = []
        for day in range(9):
            out = run_json(tmp_path, "post", "--now", day_now(day))
            picked.append(out["state"]["challenge"]["song_id"])

        catalog = {row[0] for row in db_rows(tmp_path, "SELECT id FROM songs")}
        assert len(catalog) == 8
        first_cycle = picked[:8]
        assert len(set(first_cycle)) == 8, f"repeat within 8 days: {first_cycle}"
        assert set(first_cycle) == catalog
        assert picked[8] in catalog, "day 9 must succeed via the history reset"
        assert db_rows(tmp_path, "SELECT COUNT(*) FROM challenges") == [(9,)]

    def test_guess_limit_resets_with_next_day(self, tmp_path: Path) -> None:
        """VAL-CROSS-011: full-value guesses run out (post-limit 10-point mode
        kicks in), then the count resets on the new day."""
        run_json(tmp_path, "reset")
        run_json(tmp_path, "post", "--now", DAY1_NOW)
        for n in range(1, 7):
            out = run_json(
                tmp_path, "guess", "--user", "erin", "--text", f"wrong {n}", "--now", DAY1_NOW
            )
            assert "❌" in out["payloads"][1]["content"]
        seventh = run_json(
            tmp_path, "guess", "--user", "erin", "--text", "wrong 7", "--now", DAY1_NOW
        )
        # Past the limit the player is NOT locked out: the guess is processed
        # and logged, and the feedback offers the 10-point mode.
        assert "10" in seventh["payloads"][1]["content"]
        assert db_rows(
            tmp_path, "SELECT COUNT(*) FROM guesses WHERE user_id = 'erin'"
        ) == [(7,)]

        run_json(tmp_path, "advance-day", "--now", DAY1_NOW)
        status = run_json(tmp_path, "status", "--now", DAY2_NOW)
        title = status["state"]["challenge"]["song"]["title"]
        solved = run_json(
            tmp_path, "guess", "--user", "erin", "--text", title, "--now", DAY2_NOW
        )
        assert "✅" in solved["payloads"][1]["content"]
        assert solved["state"]["user"]["guesses_used"] == 1

    def test_post_solve_lockout_is_per_user(self, tmp_path: Path) -> None:
        """VAL-CROSS-012: frank's solve locks HIM out; grace is unaffected."""
        run_json(tmp_path, "reset")
        run_json(tmp_path, "post", "--now", DAY1_NOW)
        status = run_json(tmp_path, "status", "--now", DAY1_NOW)
        title = status["state"]["challenge"]["song"]["title"]

        run_json(tmp_path, "guess", "--user", "frank", "--text", title, "--now", DAY1_NOW)
        locked_hear = run_json(tmp_path, "hear-more", "--user", "frank", "--now", DAY1_NOW)
        assert kinds(locked_hear) == ["ephemeral"]
        assert locked_hear["payloads"][0]["attachments"] == []
        locked_guess = run_json(
            tmp_path, "guess", "--user", "frank", "--text", "anything", "--now", DAY1_NOW
        )
        assert "already solved" in locked_guess["payloads"][1]["content"].lower()

        hear = run_json(tmp_path, "hear-more", "--user", "grace", "--now", DAY1_NOW)
        assert "75" in hear["payloads"][0]["content"]
        solved = run_json(
            tmp_path, "guess", "--user", "grace", "--text", title, "--now", DAY1_NOW
        )
        assert "✅" in solved["payloads"][1]["content"]

        board = run_json(tmp_path, "leaderboard", "--user", "frank", "--now", DAY1_NOW)
        assert kinds(board) == ["ephemeral"]
        assert "<@frank>" in board["payloads"][0]["embed"]["description"]

    def test_points_ladder_and_bonus_compose_across_days(self, tmp_path: Path) -> None:
        """VAL-CROSS-013: 100 + 15 + 75 = 190 over three days; 16s ffprobe."""
        run_json(tmp_path, "reset")

        # Day 1: title-only solve at level 0 -> 100.
        run_json(tmp_path, "post", "--now", DAY1_NOW)
        day1 = run_json(tmp_path, "status", "--now", DAY1_NOW)["state"]["challenge"]
        run_json(
            tmp_path, "guess", "--user", "henry", "--text", day1["song"]["title"],
            "--now", DAY1_NOW,
        )
        run_json(tmp_path, "advance-day", "--now", DAY1_NOW)

        # Day 2: four hear-more presses (75/50/30/15) then a level-4 solve -> 15.
        day2 = run_json(tmp_path, "status", "--now", DAY2_NOW)["state"]["challenge"]
        hear = run_json(
            tmp_path, "hear-more", "--user", "henry", "--times", "4", "--now", DAY2_NOW
        )
        for payload, points in zip(hear["payloads"], ("75", "50", "30", "15"), strict=True):
            assert points in payload["content"]
        level4 = Path(hear["payloads"][3]["attachments"][0]["path"])
        assert abs(ffprobe_duration(level4) - 16.0) <= 0.05
        run_json(
            tmp_path, "guess", "--user", "henry", "--text", day2["song"]["title"],
            "--now", DAY2_NOW,
        )
        run_json(tmp_path, "advance-day", "--now", DAY2_NOW)

        # Day 3: level 2 (50 base) + a both-fields guess -> 75.
        day3 = run_json(tmp_path, "status", "--now", DAY3_NOW)["state"]["challenge"]
        run_json(tmp_path, "hear-more", "--user", "henry", "--times", "2", "--now", DAY3_NOW)
        both = f"{day3['song']['artist']} - {day3['song']['title']}"
        guess = run_json(
            tmp_path, "guess", "--user", "henry", "--text", both, "--now", DAY3_NOW
        )
        assert "75" in guess["payloads"][1]["content"]
        assert "bonus" in guess["payloads"][1]["content"].lower()

        assert db_rows(
            tmp_path,
            "SELECT points_awarded FROM challenge_users WHERE user_id = 'henry'"
            " ORDER BY solved_at",
        ) == [(100,), (15,), (75,)]
        assert db_rows(
            tmp_path,
            "SELECT total_points, wins, current_streak FROM user_stats WHERE user_id = 'henry'",
        ) == [(190, 3, 3)]

    def test_leaderboard_reorders_across_days(self, tmp_path: Path) -> None:
        """VAL-CROSS-014: ordering by total_points on both days; ephemeral."""
        run_json(tmp_path, "reset")
        run_json(tmp_path, "post", "--now", DAY1_NOW)
        day1 = run_json(tmp_path, "status", "--now", DAY1_NOW)["state"]["challenge"]
        run_json(
            tmp_path, "guess", "--user", "alice", "--text", day1["song"]["title"],
            "--now", DAY1_NOW,
        )
        run_json(tmp_path, "hear-more", "--user", "bob", "--times", "2", "--now", DAY1_NOW)
        run_json(
            tmp_path, "guess", "--user", "bob", "--text", day1["song"]["title"],
            "--now", DAY1_NOW,
        )
        board1 = run_json(tmp_path, "leaderboard", "--user", "alice", "--now", DAY1_NOW)
        description1 = board1["payloads"][0]["embed"]["description"]
        assert description1.index("<@alice>") < description1.index("<@bob>")
        assert "<@carol>" not in description1

        run_json(tmp_path, "advance-day", "--now", DAY1_NOW)
        day2 = run_json(tmp_path, "status", "--now", DAY2_NOW)["state"]["challenge"]
        run_json(
            tmp_path, "guess", "--user", "bob", "--text", day2["song"]["title"],
            "--now", DAY2_NOW,
        )
        board2 = run_json(tmp_path, "leaderboard", "--user", "bob", "--now", DAY2_NOW)
        assert kinds(board2) == ["ephemeral"]
        assert board2["payloads"][0]["recipient"] == "bob"
        description2 = board2["payloads"][0]["embed"]["description"]
        assert description2.index("<@bob>") < description2.index("<@alice>")
        assert "150" in description2  # bob: 50 + 100
        assert "100" in description2  # alice: 100
        assert "<@carol>" not in description2

    def test_reveal_content_matches_db_truth(self, tmp_path: Path) -> None:
        """VAL-CROSS-016: winners case (solve order) and nobody-got-it case."""
        # Case (a): two winners in solve order.
        run_json(tmp_path, "reset")
        run_json(tmp_path, "post", "--now", DAY1_NOW)
        day1 = run_json(tmp_path, "status", "--now", DAY1_NOW)["state"]["challenge"]
        title, artist = day1["song"]["title"], day1["song"]["artist"]
        run_json(tmp_path, "guess", "--user", "alice", "--text", title, "--now", DAY1_NOW)
        for n in (1, 2):
            run_json(
                tmp_path, "guess", "--user", "bob", "--text", f"wrong {n}", "--now", DAY1_NOW
            )
        run_json(tmp_path, "guess", "--user", "bob", "--text", title, "--now", DAY1_NOW)

        advanced = run_json(tmp_path, "advance-day", "--now", DAY1_NOW)
        assert kinds(advanced) == ["announcement", "channel"]
        reveal = advanced["payloads"][0]["embed"]["description"]
        assert title in reveal
        assert artist in reveal
        # Winners listed in solve order with DB-truth guess counts and points.
        winners = db_rows(
            tmp_path,
            "SELECT user_id, guesses_used, points_awarded FROM challenge_users"
            " WHERE solved = 1 ORDER BY solved_at",
        )
        assert winners == [("alice", 1, 100), ("bob", 3, 100)]
        assert reveal.index("<@alice>") < reveal.index("<@bob>")
        assert "1 guess" in reveal
        assert "3 guesses" in reveal

        # Case (b): nobody got it.
        run_json(tmp_path, "reset")
        run_json(tmp_path, "post", "--now", DAY1_NOW)
        day1b = run_json(tmp_path, "status", "--now", DAY1_NOW)["state"]["challenge"]
        run_json(tmp_path, "guess", "--user", "carol", "--text", "wrong", "--now", DAY1_NOW)
        advanced_b = run_json(tmp_path, "advance-day", "--now", DAY1_NOW)
        assert kinds(advanced_b) == ["announcement", "channel"]
        reveal_b = advanced_b["payloads"][0]["embed"]["description"]
        assert day1b["song"]["title"] in reveal_b
        assert day1b["song"]["artist"] in reveal_b
        assert "nobody got it" in reveal_b.lower()
        assert "<@carol>" not in reveal_b
        # VAL-SCORE-005: carol's wrong guess registered a zero-valued stats
        # row — no points, no win, no streak — so "nobody got it" stays true.
        assert db_rows(
            tmp_path,
            "SELECT total_points, wins, current_streak, best_streak, last_win_date"
            " FROM user_stats WHERE user_id = 'carol'",
        ) == [(0, 0, 0, 0, None)]


class TestServe:
    def test_health_endpoint_on_3108(self, tmp_path: Path) -> None:
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", HEALTH_PORT)) == 0:
                pytest.skip("port 3108 already in use")
        env = os.environ.copy()
        env.update(
            {
                "DATABASE_PATH": str(tmp_path / "songbot.db"),
                "SNIPPET_CACHE_DIR": str(tmp_path / "snippets"),
                "DISCORD_BOT_TOKEN": "invalid-token",
                "DISCORD_API_BASE": "http://127.0.0.1:9",
                "DISCORD_GUILD_ID": "harness-guild",
                "DISCORD_CHANNEL_ID": "harness-channel",
            }
        )
        process = subprocess.Popen(
            [sys.executable, "-m", "songbot.harness", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=tmp_path,
        )
        try:
            deadline_body: dict[str, Any] | None = None
            for _ in range(100):
                try:
                    with urllib.request.urlopen(
                        f"http://localhost:{HEALTH_PORT}/health", timeout=1
                    ) as response:
                        assert response.status == 200
                        deadline_body = json.loads(response.read())
                    break
                except OSError:
                    if process.poll() is not None:
                        raise AssertionError(
                            f"serve exited early: {process.stderr.read() if process.stderr else ''}"
                        ) from None
                    import time

                    time.sleep(0.1)
            assert deadline_body == {
                "status": "ok",
                "mode": "harness",
                "guild": "harness-guild",
            }
        finally:
            process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=10) == 0
