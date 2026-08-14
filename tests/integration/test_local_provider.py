"""Integration tests for LocalDirectoryProvider (real fixture library + tmp dirs)."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import pytest

from songbot.catalog import CatalogProvider, Song
from songbot.catalog.local import LocalDirectoryProvider

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "data" / "fixture-music"

pytestmark = pytest.mark.skipif(
    not FIXTURE_DIR.is_dir(), reason="fixture music library not present at data/fixture-music"
)


def _print_songs(songs: list[Song]) -> None:
    """Evidence helper: list every returned Song record."""
    for song in songs:
        print(
            f"source_id={song.source_id!r} title={song.title!r} artist={song.artist!r} "
            f"duration_sec={song.duration_sec:.3f} audio_ref={song.audio_ref!r} "
            f"raw_title={song.raw_title!r}"
        )


class TestFixtureLibrary:
    """VAL-CATALOG-001: one well-formed Song per supported fixture file."""

    def test_returns_8_wellformed_songs(self) -> None:
        provider = LocalDirectoryProvider(FIXTURE_DIR)
        assert isinstance(provider, CatalogProvider)
        songs = provider.fetch()
        _print_songs(songs)

        assert len(songs) == 8
        source_ids = [song.source_id for song in songs]
        assert len(set(source_ids)) == 8, "source_id values must be unique"
        for song in songs:
            assert song.source == "local"
            assert song.source_id
            assert not Path(song.source_id).is_absolute()
            assert song.title
            assert song.artist
            assert abs(song.duration_sec - 30.0) <= 0.5
            audio_ref = Path(song.audio_ref)
            assert audio_ref.is_absolute()
            assert audio_ref.is_file()
            assert audio_ref.resolve().is_relative_to(FIXTURE_DIR.resolve())
            assert song.raw_title

    def test_untagged_file_falls_back_to_filename(self) -> None:
        """VAL-CATALOG-003: untagged mp3 parses artist/title from the filename."""
        target = FIXTURE_DIR / "Retro Waves - Sunset Drive.mp3"
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags",
             "-of", "json", str(target)],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
        print(f"ffprobe format_tags for {target.name}: {probe}")
        assert '"title"' not in probe.lower()
        assert '"artist"' not in probe.lower()

        songs = LocalDirectoryProvider(FIXTURE_DIR).fetch()
        retro = next(s for s in songs if s.audio_ref.endswith("Retro Waves - Sunset Drive.mp3"))
        print(f"fallback song: {retro!r}")
        assert retro.title == "Sunset Drive"
        assert retro.artist == "Retro Waves"
        assert abs(retro.duration_sec - 30.0) <= 0.5
        assert retro.raw_title == "Retro Waves - Sunset Drive"


class TestTagPrecedence:
    """VAL-CATALOG-002: embedded tags beat a misleading filename."""

    def test_mp3_tags_beat_filename(self, tmp_path: Path) -> None:
        shutil.copy(
            FIXTURE_DIR / "Midnight Circuit - Neon Skyline.mp3",
            tmp_path / "Wrong Artist - Wrong Title.mp3",
        )
        songs = LocalDirectoryProvider(tmp_path).fetch()
        assert len(songs) == 1
        song = songs[0]
        print(
            f"filename='Wrong Artist - Wrong Title.mp3' -> "
            f"title={song.title!r} artist={song.artist!r}"
        )
        assert song.title == "Neon Skyline"
        assert song.artist == "Midnight Circuit"
        assert song.source_id == "Wrong Artist - Wrong Title.mp3"
        assert song.raw_title == "Wrong Artist - Wrong Title"

    def test_m4a_tags_beat_filename(self, tmp_path: Path) -> None:
        shutil.copy(
            FIXTURE_DIR / "The Cartographers - Paper Moons.m4a",
            tmp_path / "Bogus Act - Bogus Name.m4a",
        )
        songs = LocalDirectoryProvider(tmp_path).fetch()
        assert len(songs) == 1
        song = songs[0]
        print(
            f"filename='Bogus Act - Bogus Name.m4a' -> "
            f"title={song.title!r} artist={song.artist!r}"
        )
        assert song.title == "Paper Moons"
        assert song.artist == "The Cartographers"


class TestDecoys:
    """VAL-CATALOG-004: unsupported/corrupt files are skipped, scan never crashes."""

    def test_unsupported_and_corrupt_files_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        shutil.copy(
            FIXTURE_DIR / "Midnight Circuit - Neon Skyline.mp3",
            tmp_path / "Real - Song.mp3",
        )
        (tmp_path / "notes.txt").write_text("track listing, not audio")
        (tmp_path / "Fake - Wave.wav").write_bytes(b"RIFF\x24\x00\x00\x00WAVE")
        (tmp_path / "archive.zip").write_bytes(b"PK\x03\x04" + b"\x00" * 32)
        (tmp_path / "Ghost - Empty.mp3").write_bytes(b"")

        with caplog.at_level(logging.WARNING, logger="songbot.catalog.local"):
            songs = LocalDirectoryProvider(tmp_path).fetch()

        print(f"tmp dir contents: {sorted(p.name for p in tmp_path.iterdir())}")
        _print_songs(songs)
        assert len(songs) == 1
        assert songs[0].title == "Neon Skyline"
        assert songs[0].artist == "Midnight Circuit"
        warnings = [record.getMessage() for record in caplog.records]
        assert any("Ghost - Empty.mp3" in message for message in warnings), (
            f"expected a warning naming the corrupt file, got: {warnings}"
        )


class TestDirectories:
    """VAL-CATALOG-005 + layout rules."""

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        assert LocalDirectoryProvider(tmp_path).fetch() == []

    def test_nonexistent_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            LocalDirectoryProvider(tmp_path / "nope").fetch()

    def test_nested_file_uses_relative_source_id(self, tmp_path: Path) -> None:
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        shutil.copy(
            FIXTURE_DIR / "Midnight Circuit - Neon Skyline.mp3",
            subdir / "Nested - Tune.mp3",
        )
        songs = LocalDirectoryProvider(tmp_path).fetch()
        _print_songs(songs)
        assert [song.source_id for song in songs] == ["subdir/Nested - Tune.mp3"]
        assert Path(songs[0].audio_ref).is_absolute()
