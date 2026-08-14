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
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
FIXTURE_MUSIC = REPO / "data" / "fixture-music"
HEALTH_PORT = 3108

DAY1_NOW = "2026-08-10T15:00:00-03:00"  # Halifax-local -> challenge date 2026-08-10
DAY2_NOW = "2026-08-11T13:00:00-03:00"  # -> 2026-08-11

pytestmark = pytest.mark.skipif(
    not FIXTURE_MUSIC.is_dir() or len(list(FIXTURE_MUSIC.iterdir())) < 8,
    reason="fixture music library not present",
)


def run_harness(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
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
    return subprocess.run(
        [sys.executable, "-m", "songbot.harness", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,  # no .env here: every setting comes from the environment
        timeout=120,
    )


def run_json(tmp_path: Path, *args: str) -> dict[str, Any]:
    """Run a scenario, assert exit 0, and return its parsed stdout JSON."""
    result = run_harness(tmp_path, *args)
    assert result.returncode == 0, f"{args} failed: {result.stderr[-2000:]}"
    return json.loads(result.stdout)


def db_rows(tmp_path: Path, sql: str) -> list[tuple[Any, ...]]:
    conn = sqlite3.connect(tmp_path / "songbot.db")
    try:
        return list(conn.execute(sql))
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
