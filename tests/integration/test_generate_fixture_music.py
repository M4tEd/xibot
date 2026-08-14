"""Integration tests for scripts/generate_fixture_music.py (real ffmpeg + mutagen).

The fixture library at data/fixture-music/ is gitignored, so a fresh clone must
regenerate it; this script is the reproduction path. These tests run the script
as a subprocess into a tmp dir and assert the same contract the catalog tests
assert against the committed-then-regenerated library: exactly 8 files, ~30s
durations, tags matching the "Artist - Title" stems, and one deliberately
UNTAGGED file (Retro Waves - Sunset Drive.mp3) for the filename fallback.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from songbot.catalog.local import LocalDirectoryProvider

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "generate_fixture_music.py"

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)

# filename -> (artist, title, expected audio codec). Tags deliberately match the
# "Artist - Title" filename stems (mirrors the original fixture library).
EXPECTED_FILES: dict[str, tuple[str, str, str]] = {
    "Aurora Fields - Solar Bloom.mp3": ("Aurora Fields", "Solar Bloom", "mp3"),
    "Echo Cartography - Glass Rivers.mp3": ("Echo Cartography", "Glass Rivers", "mp3"),
    "Iron Meadow - Midnight Freight.mp3": ("Iron Meadow", "Midnight Freight", "mp3"),
    "Midnight Circuit - Neon Skyline.mp3": ("Midnight Circuit", "Neon Skyline", "mp3"),
    "Quantum Drift - Digital Horizon.mp3": ("Quantum Drift", "Digital Horizon", "mp3"),
    "Retro Waves - Sunset Drive.mp3": ("Retro Waves", "Sunset Drive", "mp3"),
    "The Analog Ghosts - Velvet Static.mp3": ("The Analog Ghosts", "Velvet Static", "mp3"),
    "The Cartographers - Paper Moons.m4a": ("The Cartographers", "Paper Moons", "aac"),
}
UNTAGGED_FILE = "Retro Waves - Sunset Drive.mp3"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("generate_fixture_music", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations via sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probe(path: Path) -> dict:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:format_tags:stream=codec_name,sample_rate,channels",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, timeout=30, check=True,
    ).stdout
    return json.loads(out)


@pytest.fixture(scope="module")
def generated_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the script once into a fresh tmp dir (simulates a clean checkout)."""
    dest = tmp_path_factory.mktemp("fixture-music")
    result = _run_script("--output-dir", str(dest))
    print(result.stdout)
    assert result.returncode == 0, f"script failed:\n{result.stdout}\n{result.stderr}"
    return dest


class TestCleanCheckoutReproduction:
    """Running the script on a clean checkout reproduces all 8 fixture files."""

    def test_all_8_files_created(self, generated_dir: Path) -> None:
        produced = {p.name for p in generated_dir.iterdir() if p.is_file()}
        print(f"produced: {sorted(produced)}")
        assert produced == set(EXPECTED_FILES)

    def test_durations_codecs_and_channel_layout(self, generated_dir: Path) -> None:
        for filename, (_, _, codec) in sorted(EXPECTED_FILES.items()):
            info = _probe(generated_dir / filename)
            duration = float(info["format"]["duration"])
            stream = info["streams"][0]
            print(
                f"{filename}: codec={stream['codec_name']} duration={duration:.3f} "
                f"sample_rate={stream['sample_rate']} channels={stream['channels']}"
            )
            assert stream["codec_name"] == codec
            assert abs(duration - 30.0) <= 0.5
            assert int(stream["sample_rate"]) == 44100
            assert stream["channels"] == 1

    def test_tags_match_filename_stems(self, generated_dir: Path) -> None:
        import mutagen

        for filename, (artist, title, _) in sorted(EXPECTED_FILES.items()):
            if filename == UNTAGGED_FILE:
                continue
            audio = mutagen.File(generated_dir / filename, easy=True)
            assert audio is not None
            print(f"{filename}: tags={dict(audio.tags)}")
            assert audio["title"] == [title]
            assert audio["artist"] == [artist]

    def test_fallback_file_is_untagged(self, generated_dir: Path) -> None:
        target = generated_dir / UNTAGGED_FILE
        tags = _probe(target)["format"].get("tags", {})
        print(f"{UNTAGGED_FILE} ffprobe format tags: {tags}")
        assert "title" not in {k.lower() for k in tags}
        assert "artist" not in {k.lower() for k in tags}

    def test_melodies_are_distinct(self, generated_dir: Path) -> None:
        hashes = {_sha256(generated_dir / name) for name in EXPECTED_FILES}
        assert len(hashes) == len(EXPECTED_FILES), "fixture songs must have distinct audio"

    def test_regenerated_library_feeds_local_provider(self, generated_dir: Path) -> None:
        """The regenerated library satisfies the VAL-CATALOG-001/003 contract shape."""
        songs = LocalDirectoryProvider(generated_dir).fetch()
        for song in songs:
            print(
                f"source_id={song.source_id!r} title={song.title!r} artist={song.artist!r} "
                f"duration_sec={song.duration_sec:.3f}"
            )
        assert len(songs) == 8
        for song in songs:
            assert song.source == "local"
            assert song.title
            assert song.artist
            assert abs(song.duration_sec - 30.0) <= 0.5
            assert Path(song.audio_ref).is_file()
            assert song.raw_title
        retro = next(s for s in songs if s.audio_ref.endswith(UNTAGGED_FILE))
        assert retro.title == "Sunset Drive"
        assert retro.artist == "Retro Waves"


class TestIdempotency:
    """Re-running the script is a no-op; --force rebuilds deterministically."""

    def test_second_run_skips_without_touching_files(self, generated_dir: Path) -> None:
        before = {p.name: (_sha256(p), p.stat().st_mtime_ns) for p in generated_dir.iterdir()}
        result = _run_script("--output-dir", str(generated_dir))
        print(result.stdout)
        assert result.returncode == 0, result.stderr
        assert result.stdout.lower().count("skip") == len(EXPECTED_FILES)
        after = {p.name: (_sha256(p), p.stat().st_mtime_ns) for p in generated_dir.iterdir()}
        assert before == after

    def test_force_regenerates_valid_library(self, generated_dir: Path) -> None:
        result = _run_script("--output-dir", str(generated_dir), "--force")
        print(result.stdout)
        assert result.returncode == 0, result.stderr
        assert "skip" not in result.stdout.lower()
        produced = {p.name for p in generated_dir.iterdir() if p.is_file()}
        assert produced == set(EXPECTED_FILES)
        songs = LocalDirectoryProvider(generated_dir).fetch()
        assert len(songs) == 8
        retro = next(s for s in songs if s.audio_ref.endswith(UNTAGGED_FILE))
        assert (retro.artist, retro.title) == ("Retro Waves", "Sunset Drive")


class TestSongTable:
    """Pure-data checks on the script's song table (no ffmpeg needed)."""

    def test_song_table_matches_expected_library(self) -> None:
        module = _load_script_module()
        songs = module.SONGS
        assert len(songs) == 8
        filenames = [song.filename for song in songs]
        assert len(set(filenames)) == 8
        assert set(filenames) == set(EXPECTED_FILES)
        untagged = [song for song in songs if not song.tagged]
        assert [song.filename for song in untagged] == [UNTAGGED_FILE]
        for song in songs:
            if song.tagged:
                assert song.filename.startswith(f"{song.artist} - {song.title}.")

    def test_melody_expression_is_valid_and_escaped(self) -> None:
        module = _load_script_module()
        for song in module.SONGS:
            expr = module.melody_expression(song)
            print(f"{song.filename}: {expr[:120]}...")
            # Every note frequency must appear; commas must be filtergraph-escaped.
            for note in song.notes:
                assert f"{note:.3f}" in expr
            assert "," not in expr.replace("\\,", ""), f"unescaped comma in {expr!r}"
