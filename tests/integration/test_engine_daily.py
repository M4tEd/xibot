"""Integration tests for the GameEngine daily lifecycle with REAL components.

Uses the real fixture music library (read-only), the real
LocalDirectoryProvider + refresh_catalog to populate ``songs``, and a real
SnippetGenerator writing to a tmp cache dir (real ffmpeg runs, no network).
Verifies the shapes unit tests with fakes cannot: real snippet files on disk,
offset constraints against real durations, cache re-heal (pinned #14) and
skip-song cache purging (pinned #5) end to end.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from songbot.catalog.refresh import refresh_catalog
from songbot.config import Settings
from songbot.db import Database
from songbot.engine import GameEngine
from songbot.snippets import SnippetGenerator

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "data" / "fixture-music"

pytestmark = pytest.mark.skipif(
    not FIXTURE_DIR.is_dir(), reason="fixture music library not present at data/fixture-music"
)

NOW = datetime(2026, 8, 13, 16, 0, 0, tzinfo=UTC)  # 2026-08-13 13:00 ADT


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        discord_token="test-token",
        guild_id="guild-1",
        channel_id="channel-1",
        youtube_playlist_url=None,
        local_music_dir=FIXTURE_DIR,
        daily_post_time="12:00",
        timezone="America/Halifax",
        max_guesses_per_day=6,
        snippet_lengths=(1.0, 2.0, 4.0, 8.0, 16.0),
        snippet_points=(100, 75, 50, 30, 15),
        both_correct_multiplier=1.5,
        guess_match_mode="either",
        database_path=tmp_path / "songbot.db",
        snippet_cache_dir=tmp_path / "snippets",
        health_port=3108,
        log_level="INFO",
        discord_api_base="https://discord.com/api/v10",
    )


def _duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return float(out.stdout.strip())


@pytest.fixture
def engine_stack(tmp_path: Path) -> Iterator[tuple[GameEngine, Database, Settings, Path]]:
    """Engine + real catalog (8 fixture songs) + real snippet generator."""
    settings = _settings(tmp_path)
    db = Database.open(settings.database_path)
    try:
        refresh_catalog(db, settings)
        snippets = SnippetGenerator(settings.snippet_cache_dir)
        engine = GameEngine(db, settings, snippets)
        yield engine, db, settings, settings.snippet_cache_dir
    finally:
        db.close()


class TestEnsureTodayChallengeReal:
    def test_full_snippet_set_on_disk_with_valid_offset(
        self, engine_stack: tuple[GameEngine, Database, Settings, Path]
    ) -> None:
        engine, db, _settings_obj, cache_dir = engine_stack
        challenge = engine.ensure_today_challenge("g1", "c1", NOW)

        assert challenge.created is True
        assert len(challenge.snippet_paths) == 5
        expected = {0: 1.0, 1: 2.0, 2: 4.0, 3: 8.0, 4: 16.0}
        for level, target in expected.items():
            path = cache_dir / str(challenge.id) / f"{level}.mp3"
            assert challenge.snippet_paths[level] == path
            assert path.is_file()
            assert path.stat().st_size > 0
            assert abs(_duration(path) - target) <= 0.05
        assert challenge.snippet_offset_sec >= 0
        assert challenge.snippet_offset_sec + 16 <= challenge.song.duration_sec + 1e-6

        row = db.query_one("SELECT * FROM challenges")
        assert row is not None
        assert row["song_id"] == challenge.song.id

    def test_repost_reuses_cache_and_reheals_deleted_dir(
        self, engine_stack: tuple[GameEngine, Database, Settings, Path]
    ) -> None:
        """Pinned #14 + VAL-SNIP-007/008 engine level: reuse is byte-stable and
        a deleted cache dir is regenerated for the SAME challenge row."""
        engine, _db, _s, cache_dir = engine_stack
        first = engine.ensure_today_challenge("g1", "c1", NOW)
        level0 = cache_dir / str(first.id) / "0.mp3"
        mtime_before = level0.stat().st_mtime_ns

        second = engine.ensure_today_challenge("g1", "c1", NOW)
        assert second.created is False
        assert second.id == first.id
        assert level0.stat().st_mtime_ns == mtime_before  # skipped, not re-encoded

        shutil.rmtree(cache_dir / str(first.id))
        third = engine.ensure_today_challenge("g1", "c1", NOW)
        assert third.created is False
        assert third.id == first.id
        for level in range(5):
            path = cache_dir / str(first.id) / f"{level}.mp3"
            assert path.is_file()
            assert path.stat().st_size > 0
        assert abs(_duration(cache_dir / str(first.id) / "4.mp3") - 16.0) <= 0.05


class TestSkipTodaySongReal:
    def test_skip_regenerates_cache_for_new_song(
        self, engine_stack: tuple[GameEngine, Database, Settings, Path]
    ) -> None:
        engine, db, _s, cache_dir = engine_stack
        old = engine.ensure_today_challenge("g1", "c1", NOW)
        assert (cache_dir / str(old.id)).is_dir()
        # Pad with a later-rowid challenge in another guild so the skipped row
        # is not the max rowid: the replacement gets a fresh id and the old
        # cache dir's purge is observable (SQLite reuses freed max rowids).
        other = engine.ensure_today_challenge("g2", "c1", NOW)

        new = engine.skip_today_song("g1", NOW)

        assert new.id != old.id
        assert new.song.id != old.song.id
        assert new.snippet_offset_sec != old.snippet_offset_sec
        assert not (cache_dir / str(old.id)).exists()  # purged, incl. intermediates
        assert (cache_dir / str(other.id)).is_dir()  # other guild untouched
        for level in range(5):
            path = cache_dir / str(new.id) / f"{level}.mp3"
            assert path.is_file()
            assert path.stat().st_size > 0
        assert abs(_duration(cache_dir / str(new.id) / "0.mp3") - 1.0) <= 0.05
        rows = db.query("SELECT guild_id, song_id, skip_count FROM challenges ORDER BY id")
        assert len(rows) == 2  # g1's replacement + g2's untouched challenge
        g1_row = next(r for r in rows if r["guild_id"] == "g1")
        assert g1_row["song_id"] == new.song.id
        assert g1_row["skip_count"] == 1


class TestNoRepeatRealCatalog:
    def test_eight_distinct_days_then_reset(
        self, engine_stack: tuple[GameEngine, Database, Settings, Path]
    ) -> None:
        """VAL-DAILY-006/007 engine level against the real 8-song fixture catalog."""
        engine, db, _s, _c = engine_stack
        picked: list[int] = []
        for day in range(1, 10):
            now = datetime(2026, 8, day, 16, 0, 0, tzinfo=UTC)
            picked.append(engine.ensure_today_challenge("g1", "c1", now).song.id)

        all_ids = {int(r["id"]) for r in db.query("SELECT id FROM songs")}
        assert len(all_ids) == 8
        assert set(picked[:8]) == all_ids  # 8 distinct, full coverage
        assert picked[8] in all_ids  # day 9 reset succeeds
