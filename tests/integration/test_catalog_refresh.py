"""Integration tests for refresh_catalog with the real provider classes.

The local side uses copies of the fixture library in tmp dirs (`data/` is
never touched). The YouTube side uses YouTubePlaylistProvider with a stubbed
``entry_loader``, except the bogus-URL test, which performs a real (allowed)
YouTube request against an invalid playlist.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import mutagen
import pytest

from songbot.catalog.local import LocalDirectoryProvider
from songbot.catalog.refresh import RefreshResult, SourceRefresh, refresh_catalog
from songbot.catalog.youtube import YouTubePlaylistProvider
from songbot.config import Settings
from songbot.db import Database, SongRow

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "data" / "fixture-music"
BOGUS_PLAYLIST_URL = "https://youtube.com/playlist?list=PL_INVALID_SONGBOT_TEST"

pytestmark = pytest.mark.skipif(
    not FIXTURE_DIR.is_dir(), reason="fixture music library not present at data/fixture-music"
)


def _settings(
    tmp_path: Path,
    *,
    local_music_dir: Path | None = None,
    youtube_playlist_url: str | None = None,
) -> Settings:
    """A valid Settings with both catalog providers disabled unless specified."""
    return Settings(
        discord_token="test-token",
        guild_id="guild-1",
        channel_id="channel-1",
        youtube_playlist_url=youtube_playlist_url,
        local_music_dir=local_music_dir,
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


@pytest.fixture
def db() -> Iterator[Database]:
    database = Database.open(":memory:")
    yield database
    database.close()


def _songs_table(db: Database) -> list[SongRow]:
    rows = db.query("SELECT * FROM songs ORDER BY source, source_id")
    return [SongRow.from_row(row) for row in rows]


def _insert_challenge(db: Database, song_id: int, date: str = "2026-08-13") -> None:
    db.execute(
        "INSERT INTO challenges"
        " (guild_id, channel_id, song_id, date, snippet_offset_sec, status, created_at)"
        " VALUES (?, ?, ?, ?, ?, 'active', ?)",
        ("guild-1", "channel-1", song_id, date, 0.0, datetime.now(UTC).isoformat()),
    )


def _copy_fixture(music_dir: Path, name: str) -> Path:
    music_dir.mkdir(parents=True, exist_ok=True)
    target = music_dir / name
    shutil.copy(FIXTURE_DIR / name, target)
    return target


class TestRealLocalProvider:
    """refresh_catalog drives the real LocalDirectoryProvider from settings."""

    def test_refresh_populates_songs_from_fixture_copies(
        self, db: Database, tmp_path: Path
    ) -> None:
        music = tmp_path / "music"
        _copy_fixture(music, "Midnight Circuit - Neon Skyline.mp3")
        _copy_fixture(music, "Retro Waves - Sunset Drive.mp3")  # untagged -> filename parse

        result = refresh_catalog(db, _settings(tmp_path, local_music_dir=music))

        assert result == RefreshResult(sources=(SourceRefresh(source="local", added=2),))
        rows = _songs_table(db)
        by_id = {row.source_id: row for row in rows}
        for row in rows:
            print(f"row: {row}")
            assert row.source == "local"
            assert Path(row.audio_ref).is_absolute()
            assert Path(row.audio_ref).is_file()
            assert abs(row.duration_sec - 30.0) <= 0.5
        neon = by_id["Midnight Circuit - Neon Skyline.mp3"]
        assert (neon.title, neon.artist) == ("Neon Skyline", "Midnight Circuit")
        retro = by_id["Retro Waves - Sunset Drive.mp3"]
        assert (retro.title, retro.artist) == ("Sunset Drive", "Retro Waves")

    def test_retagged_file_updates_title_preserving_row_id(
        self, db: Database, tmp_path: Path
    ) -> None:
        """VAL-CATALOG-013: same filename + new tags -> update in place, stable id."""
        music = tmp_path / "music"
        target = _copy_fixture(music, "Midnight Circuit - Neon Skyline.mp3")
        settings = _settings(tmp_path, local_music_dir=music)

        refresh_catalog(db, settings)
        before = _songs_table(db)[0]
        assert before.title == "Neon Skyline"

        audio = mutagen.File(target, easy=True)
        assert audio is not None
        audio["title"] = ["Neon Skyline (Remaster)"]
        audio.save()

        second = refresh_catalog(db, settings)

        after = _songs_table(db)
        assert len(after) == 1, "no duplicate row may be inserted"
        row = after[0]
        print(f"id {before.id} -> {row.id}; title {before.title!r} -> {row.title!r}")
        assert row.id == before.id, "row id must be preserved (update, not delete+insert)"
        assert row.created_at == before.created_at
        assert row.title == "Neon Skyline (Remaster)"
        assert row.artist == "Midnight Circuit"
        assert second.by_source("local") == SourceRefresh(source="local", updated=1)

    def test_removed_file_deleted_on_next_refresh(
        self, db: Database, tmp_path: Path
    ) -> None:
        music = tmp_path / "music"
        _copy_fixture(music, "Midnight Circuit - Neon Skyline.mp3")
        victim = _copy_fixture(music, "Quantum Drift - Digital Horizon.mp3")
        settings = _settings(tmp_path, local_music_dir=music)
        refresh_catalog(db, settings)
        assert len(_songs_table(db)) == 2

        victim.unlink()
        result = refresh_catalog(db, settings)

        assert result.by_source("local") == SourceRefresh(source="local", updated=1, removed=1)
        assert [r.source_id for r in _songs_table(db)] == [
            "Midnight Circuit - Neon Skyline.mp3"
        ]

    def test_removed_but_referenced_file_retained(
        self, db: Database, tmp_path: Path
    ) -> None:
        music = tmp_path / "music"
        victim = _copy_fixture(music, "Quantum Drift - Digital Horizon.mp3")
        settings = _settings(tmp_path, local_music_dir=music)
        refresh_catalog(db, settings)
        song_id = _songs_table(db)[0].id
        _insert_challenge(db, song_id)

        victim.unlink()
        result = refresh_catalog(db, settings)

        src = result.by_source("local")
        print(f"referenced-removal result: {src}")
        assert src.removed == 0
        assert src.retained == 1
        rows = _songs_table(db)
        assert len(rows) == 1, "a challenge-referenced song must be retained for history"
        assert rows[0].id == song_id


class TestCombinedProviders:
    """Real LocalDirectoryProvider + real YouTubePlaylistProvider (stubbed dump)."""

    def test_local_and_stubbed_youtube_combined(
        self, db: Database, tmp_path: Path
    ) -> None:
        music = tmp_path / "music"
        _copy_fixture(music, "The Cartographers - Paper Moons.m4a")
        entries = [
            {
                "id": "vid1",
                "title": "XI - Akasha",
                "duration": 200,
                "uploader": "Raudi",
                "url": "https://www.youtube.com/watch?v=vid1",
            },
            {
                "id": "vid2",
                "title": "Agartha",
                "duration": 300,
                "uploader": "xi - Topic",
                "url": "https://www.youtube.com/watch?v=vid2",
            },
        ]
        youtube = YouTubePlaylistProvider(
            "https://youtube.com/playlist?list=PL_TEST",
            entry_loader=lambda url, timeout: entries,
        )
        providers = {"local": LocalDirectoryProvider(music), "youtube": youtube}

        result = refresh_catalog(db, _settings(tmp_path), providers=providers)

        print(f"result: {result}")
        assert result.ok
        assert result.by_source("local") == SourceRefresh(source="local", added=1)
        assert result.by_source("youtube") == SourceRefresh(source="youtube", added=2)
        by_pair = {(r.source, r.source_id): r for r in _songs_table(db)}
        assert set(by_pair) == {
            ("local", "The Cartographers - Paper Moons.m4a"),
            ("youtube", "vid1"),
            ("youtube", "vid2"),
        }
        assert (by_pair[("youtube", "vid1")].artist, by_pair[("youtube", "vid1")].title) == (
            "XI",
            "Akasha",
        )
        # Bare title: uploader fallback with ' - Topic' stripped.
        assert (by_pair[("youtube", "vid2")].artist, by_pair[("youtube", "vid2")].title) == (
            "xi",
            "Agartha",
        )
        assert by_pair[("youtube", "vid1")].audio_ref == (
            "https://www.youtube.com/watch?v=vid1"
        )


class TestBogusPlaylist:
    """VAL-CATALOG-014 at the refresh level (real YouTube network, invalid target)."""

    def test_bogus_playlist_error_isolated_and_lossless(
        self, db: Database, tmp_path: Path
    ) -> None:
        music = tmp_path / "music"
        _copy_fixture(music, "Midnight Circuit - Neon Skyline.mp3")
        # Pre-seed a youtube row as a previous successful refresh would have.
        db.execute(
            "INSERT INTO songs"
            " (source, source_id, title, artist, duration_sec, audio_ref, raw_title,"
            " created_at)"
            " VALUES ('youtube', 'preexisting', 'Old', 'Someone', 200.0,"
            " 'https://www.youtube.com/watch?v=preexisting', 'Old', ?)",
            (datetime.now(UTC).isoformat(),),
        )
        settings = _settings(
            tmp_path, local_music_dir=music, youtube_playlist_url=BOGUS_PLAYLIST_URL
        )

        start = time.monotonic()
        result = refresh_catalog(db, settings)  # real providers built from settings
        elapsed = time.monotonic() - start
        print(f"refresh with bogus playlist took {elapsed:.1f}s -> {result}")

        assert elapsed <= 60.0, "the fetch must be bounded by a timeout (<= 60s)"
        assert not result.ok
        yt = result.by_source("youtube")
        assert yt.error is not None, "a failed source must report a named error"
        assert "YouTubeCatalogError" in yt.error
        assert "PL_INVALID_SONGBOT_TEST" in yt.error

        assert result.by_source("local") == SourceRefresh(source="local", added=1)
        pairs = {(r.source, r.source_id) for r in _songs_table(db)}
        assert pairs == {
            ("local", "Midnight Circuit - Neon Skyline.mp3"),
            ("youtube", "preexisting"),  # failed source's rows neither deleted nor corrupted
        }
