"""Integration tests for SnippetGenerator: real ffmpeg/ffprobe on tmp cache dirs.

Covers the feature's expected behavior plus the generator-level shapes of
VAL-SNIP-002/003/004/008/009/010 and all of VAL-SNIP-011: exact durations,
offset fidelity (reference-cut comparison), cross-level prefix consistency,
idempotent caching, m4a -> mp3 re-encode, the YouTube section-download flow,
and clean failure semantics (named errors, no partial cache artifacts).

Network: only the ``TestRealYouTube`` class touches the network (YouTube is
allowed by mission boundaries); everything else is fully local.
"""

from __future__ import annotations

import hashlib
import subprocess
import time
from array import array
from pathlib import Path

import pytest
import yt_dlp

from songbot import snippets as snippets_module
from songbot.catalog import Song
from songbot.catalog.youtube import YouTubePlaylistProvider
from songbot.snippets import (
    SnippetError,
    SnippetGenerationError,
    SnippetGenerator,
    SnippetSourceError,
    download_section_with_timeout,
)

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "data" / "fixture-music"
MP3_FIXTURE = FIXTURE_DIR / "Midnight Circuit - Neon Skyline.mp3"
M4A_FIXTURE = FIXTURE_DIR / "The Cartographers - Paper Moons.m4a"
PLAYLIST_URL = "https://youtube.com/playlist?list=PLDzqiyJzN_jBRIJhUB_vmD5jPIHNkflc1"
REAL_VIDEO_ID = "jO4fTcziVRM"  # raw title "XI - Akasha", duration in [105, 561]s

pytestmark = pytest.mark.skipif(
    not FIXTURE_DIR.is_dir(), reason="fixture music library not present at data/fixture-music"
)

LENGTHS = (1.0, 2.0, 4.0, 8.0, 16.0)
OFFSET = 7.5
TOLERANCE_SEC = 0.05
SAMPLE_RATE = 44100


def _local_song(path: Path = MP3_FIXTURE) -> Song:
    return Song(
        source="local",
        source_id=path.name,
        title="Test Title",
        artist="Test Artist",
        duration_sec=30.0,
        audio_ref=str(path),
        raw_title=path.stem,
    )


def _youtube_song(video_id: str = "stubbedvideo1", duration_sec: float = 300.0) -> Song:
    return Song(
        source="youtube",
        source_id=video_id,
        title="Stubbed",
        artist="Stubber",
        duration_sec=duration_sec,
        audio_ref=f"https://www.youtube.com/watch?v={video_id}",
        raw_title="Stubber - Stubbed",
    )


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )


def _probe_duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout
    return float(out.strip())


def _probe_codec(path: Path) -> str:
    return subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
            "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout.strip()


def _decode_pcm(
    path: Path, ss: float | None = None, t: float | None = None, rate: int = SAMPLE_RATE
) -> array:
    """Decode to mono s16le samples (the VAL-SNIP-003/004 comparison shape)."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if ss is not None:
        cmd += ["-ss", repr(ss)]
    if t is not None:
        cmd += ["-t", repr(t)]
    cmd += ["-i", str(path), "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True, timeout=60, check=True).stdout
    samples = array("h")
    samples.frombytes(raw)
    return samples


def _mean_abs_diff(a: array, b: array) -> float:
    """Mean absolute normalized sample difference over the shared prefix."""
    n = min(len(a), len(b))
    assert n > 0, "cannot compare empty sample streams"
    return sum(abs(x - y) for x, y in zip(a[:n], b[:n], strict=True)) / n / 32768.0


def _file_state(path: Path) -> tuple[str, int]:
    """(sha256, mtime_ns) fingerprint for idempotency assertions."""
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns


def _leftover_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file()) if root.exists() else []


class StubSectionDownloader:
    """Test double for the yt-dlp section download: cuts the section locally.

    Mimics the real downloader contract: writes ``<dest_base>.wav`` holding the
    audio of ``source`` from ``start`` to ``end`` seconds, returns its path.
    A LOSSLESS intermediate is used on purpose: it isolates the generator's
    cut math (offset relative to the section start) from lossy double-encode
    noise, which alone pushes a re-encoded mp3 intermediate past the 0.01
    comparison threshold (verified: min MAD 0.0113 at exact alignment).
    """

    def __init__(self, source: Path) -> None:
        self.source = source
        self.calls: list[tuple[str, float, float]] = []

    def __call__(self, url: str, start: float, end: float, dest_base: Path, timeout: float) -> Path:
        self.calls.append((url, start, end))
        dest = dest_base.parent / f"{dest_base.name}.wav"
        _ffmpeg(
            "-ss", repr(start), "-t", repr(end - start),
            "-i", str(self.source), "-c:a", "pcm_s16le", str(dest),
        )
        return dest


class StubFullDownloader:
    """Test double for the yt-dlp full-audio download: re-encodes locally.

    Mimics the real downloader contract: writes the WHOLE ``source`` as
    ``<dest_base>.mp3`` (the production downloader's FFmpegExtractAudio
    postprocessor shape) and returns its path.
    """

    def __init__(self, source: Path) -> None:
        self.source = source
        self.calls: list[str] = []

    def __call__(self, url: str, dest_base: Path, timeout: float) -> Path:
        self.calls.append(url)
        dest = dest_base.parent / f"{dest_base.name}.mp3"
        _ffmpeg("-i", str(self.source), "-c:a", "libmp3lame", str(dest))
        return dest


class TestLocalGeneration:
    """Expected behavior: full ladder at the documented cache path, exact durations."""

    def test_all_levels_generated_with_exact_durations(self, tmp_path: Path) -> None:
        gen = SnippetGenerator(tmp_path)
        result = gen.ensure_snippets(_local_song(), challenge_id=1, offset=OFFSET, lengths=LENGTHS)

        assert set(result) == {0, 1, 2, 3, 4}
        for level, target in enumerate(LENGTHS):
            path = result[level]
            assert path == tmp_path / "1" / f"{level}.mp3"
            assert path.is_file()
            assert path.stat().st_size > 0
            measured = _probe_duration(path)
            delta_ms = abs(measured - target) * 1000
            print(
                f"level {level}: target={target}s measured={measured:.4f}s "
                f"delta={delta_ms:.1f}ms"
            )
            assert delta_ms <= TOLERANCE_SEC * 1000

    def test_snippet_content_starts_at_requested_offset(self, tmp_path: Path) -> None:
        """VAL-SNIP-003 shape: cached 4.mp3 matches an independent reference cut."""
        gen = SnippetGenerator(tmp_path / "cache")
        result = gen.ensure_snippets(_local_song(), challenge_id=7, offset=OFFSET, lengths=LENGTHS)

        ref = tmp_path / "ref.mp3"
        _ffmpeg("-ss", repr(OFFSET), "-t", "16", "-i", str(MP3_FIXTURE),
                "-c:a", "libmp3lame", str(ref))
        wrong_ref = tmp_path / "wrong-offset.mp3"
        _ffmpeg("-ss", "0", "-t", "16", "-i", str(MP3_FIXTURE),
                "-c:a", "libmp3lame", str(wrong_ref))

        edge = int(0.05 * SAMPLE_RATE)  # ignore first/last 50ms (encoder edge effects)
        snippet = _decode_pcm(result[4])

        ref_pcm = _decode_pcm(ref)
        n = min(len(snippet), len(ref_pcm)) - edge
        mad = _mean_abs_diff(snippet[edge:n], ref_pcm[edge:n])
        print(f"offset-fidelity mean abs diff vs reference cut at {OFFSET}s: {mad:.6f}")
        assert mad < 0.01

        wrong_pcm = _decode_pcm(wrong_ref)
        n_wrong = min(len(snippet), len(wrong_pcm)) - edge
        mad_wrong = _mean_abs_diff(snippet[edge:n_wrong], wrong_pcm[edge:n_wrong])
        print(f"control mean abs diff vs offset-0 cut: {mad_wrong:.6f}")
        assert mad_wrong >= 0.01, "comparison has no teeth: offset-0 audio should differ"

    def test_all_levels_share_one_start_point(self, tmp_path: Path) -> None:
        """VAL-SNIP-004 shape: shorter snippets are prefixes of the longest one."""
        gen = SnippetGenerator(tmp_path)
        result = gen.ensure_snippets(_local_song(), challenge_id=2, offset=OFFSET, lengths=LENGTHS)

        longest = _decode_pcm(result[4])
        for level, window_sec in ((0, 0.95), (1, 1.95)):
            shorter = _decode_pcm(result[level])
            count = int(window_sec * SAMPLE_RATE)
            mad = _mean_abs_diff(shorter[:count], longest[:count])
            print(f"level {level} vs level 4 over first {window_sec}s: mean abs diff {mad:.6f}")
            assert mad < 0.01

    def test_m4a_source_reencodes_to_mp3(self, tmp_path: Path) -> None:
        """VAL-SNIP-009 shape: output codec is mp3 regardless of source container."""
        gen = SnippetGenerator(tmp_path)
        result = gen.ensure_snippets(
            _local_song(M4A_FIXTURE), challenge_id=3, offset=OFFSET, lengths=LENGTHS
        )
        for level, target in enumerate(LENGTHS):
            codec = _probe_codec(result[level])
            measured = _probe_duration(result[level])
            print(f"level {level}: codec={codec} duration={measured:.4f}s")
            assert codec == "mp3"
            assert abs(measured - target) <= TOLERANCE_SEC


class TestIdempotency:
    """Expected behavior: existing files skipped; single deleted level regenerated."""

    def test_rerun_skips_existing_files(self, tmp_path: Path) -> None:
        gen = SnippetGenerator(tmp_path)
        song = _local_song()
        first = gen.ensure_snippets(song, challenge_id=4, offset=OFFSET, lengths=LENGTHS)
        before = {level: _file_state(path) for level, path in first.items()}

        second = gen.ensure_snippets(song, challenge_id=4, offset=OFFSET, lengths=LENGTHS)

        assert second == first
        for level, path in second.items():
            assert _file_state(path) == before[level], f"level {level} was regenerated"

    def test_deleted_level_regenerates_exactly_that_file(self, tmp_path: Path) -> None:
        """VAL-SNIP-008 part 1: delete one level, rerun, only that file comes back."""
        gen = SnippetGenerator(tmp_path)
        song = _local_song()
        first = gen.ensure_snippets(song, challenge_id=5, offset=OFFSET, lengths=LENGTHS)
        kept = {level: _file_state(path) for level, path in first.items() if level != 2}

        first[2].unlink()
        second = gen.ensure_snippets(song, challenge_id=5, offset=OFFSET, lengths=LENGTHS)

        assert set(second) == {0, 1, 2, 3, 4}
        for level, state in kept.items():
            assert _file_state(second[level]) == state, f"level {level} was needlessly regenerated"
        assert abs(_probe_duration(second[2]) - LENGTHS[2]) <= TOLERANCE_SEC

    def test_zero_byte_level_file_is_regenerated(self, tmp_path: Path) -> None:
        """A 0-byte file is a partial artifact: treated as missing, not skipped."""
        gen = SnippetGenerator(tmp_path)
        song = _local_song()
        first = gen.ensure_snippets(song, challenge_id=6, offset=OFFSET, lengths=LENGTHS)

        first[1].write_bytes(b"")
        second = gen.ensure_snippets(song, challenge_id=6, offset=OFFSET, lengths=LENGTHS)

        assert second[1].stat().st_size > 0
        assert abs(_probe_duration(second[1]) - LENGTHS[1]) <= TOLERANCE_SEC


class TestFailures:
    """VAL-SNIP-011: missing/corrupt source -> clear named error, no partial cache."""

    def test_missing_source_raises_named_error_without_partials(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.mp3"
        gen = SnippetGenerator(tmp_path / "cache")

        with pytest.raises(SnippetSourceError, match=r"nope\.mp3") as excinfo:
            gen.ensure_snippets(
                _local_song(missing), challenge_id=8, offset=OFFSET, lengths=LENGTHS
            )

        print(f"error: {excinfo.value}")
        assert isinstance(excinfo.value, SnippetError)
        leftovers = _leftover_files(tmp_path / "cache")
        print(f"leftover cache files: {leftovers}")
        assert leftovers == []

    def test_corrupt_sources_raise_named_error_without_partials(self, tmp_path: Path) -> None:
        for name, payload in (("empty.mp3", b""), ("text.mp3", b"this is not audio")):
            corrupt = tmp_path / name
            corrupt.write_bytes(payload)
            gen = SnippetGenerator(tmp_path / "cache")

            with pytest.raises(SnippetSourceError, match=name) as excinfo:
                gen.ensure_snippets(
                    _local_song(corrupt), challenge_id=9, offset=OFFSET, lengths=LENGTHS
                )

            print(f"{name} error: {excinfo.value}")
            assert isinstance(excinfo.value, SnippetError)
            leftovers = _leftover_files(tmp_path / "cache")
            print(f"leftover cache files after {name}: {leftovers}")
            assert leftovers == []

    def test_failure_preserves_preexisting_levels_without_partials(self, tmp_path: Path) -> None:
        """A complete pre-existing set may remain, but the failure adds nothing."""
        gen = SnippetGenerator(tmp_path / "cache")
        good = gen.ensure_snippets(_local_song(), challenge_id=10, offset=OFFSET, lengths=LENGTHS)
        good[3].unlink()
        kept = {level: _file_state(path) for level, path in good.items() if level != 3}

        corrupt = tmp_path / "broken.mp3"
        corrupt.write_bytes(b"garbage audio")
        with pytest.raises(SnippetSourceError, match=r"broken\.mp3"):
            gen.ensure_snippets(
                _local_song(corrupt), challenge_id=10, offset=OFFSET, lengths=LENGTHS
            )

        for level, state in kept.items():
            assert _file_state(good[level]) == state, f"pre-existing level {level} was touched"
        assert not good[3].exists(), "failed run left a partial level file behind"

    def test_too_short_source_raises_named_error(self, tmp_path: Path) -> None:
        """offset + max length beyond the source duration fails before any cut."""
        gen = SnippetGenerator(tmp_path / "cache")
        with pytest.raises(SnippetSourceError, match="too short"):
            gen.ensure_snippets(_local_song(), challenge_id=16, offset=20.0, lengths=LENGTHS)
        assert _leftover_files(tmp_path / "cache") == []

    def test_source_failure_removes_stale_incomplete_levels(self, tmp_path: Path) -> None:
        """VAL-SNIP-011: a failed regeneration removes the 0-byte partials it targeted.

        Regression: source-preparation failure used to leave pre-existing
        0-byte level files behind even though the call intended to regenerate
        them — a partial artifact survived in the cache.
        """
        gen = SnippetGenerator(tmp_path / "cache")
        good = gen.ensure_snippets(_local_song(), challenge_id=18, offset=OFFSET, lengths=LENGTHS)
        # Levels 1 and 3 become stale partials (0-byte); the rest stay complete.
        good[1].write_bytes(b"")
        good[3].write_bytes(b"")
        kept = {level: _file_state(path) for level, path in good.items() if level not in (1, 3)}

        missing = tmp_path / "gone.mp3"
        with pytest.raises(SnippetSourceError, match=r"gone\.mp3"):
            gen.ensure_snippets(
                _local_song(missing), challenge_id=18, offset=OFFSET, lengths=LENGTHS
            )

        assert not good[1].exists(), "stale 0-byte level 1 survived the failed regeneration"
        assert not good[3].exists(), "stale 0-byte level 3 survived the failed regeneration"
        for level, state in kept.items():
            assert _file_state(good[level]) == state, (
                f"pre-existing complete level {level} was touched"
            )
        partials = [p for p in _leftover_files(tmp_path / "cache") if p.stat().st_size == 0]
        assert partials == [], f"zero-byte partials left behind: {partials}"

    def test_source_failure_with_only_stale_partials_removes_challenge_dir(
        self, tmp_path: Path
    ) -> None:
        """VAL-SNIP-011: a challenge dir holding only 0-byte partials is removed entirely."""
        gen = SnippetGenerator(tmp_path / "cache")
        challenge_dir = tmp_path / "cache" / "20"
        challenge_dir.mkdir(parents=True)
        (challenge_dir / "0.mp3").write_bytes(b"")
        (challenge_dir / "1.mp3").write_bytes(b"")

        with pytest.raises(SnippetSourceError, match=r"gone\.mp3"):
            gen.ensure_snippets(
                _local_song(tmp_path / "gone.mp3"), challenge_id=20, offset=OFFSET, lengths=LENGTHS
            )

        assert not challenge_dir.exists(), "empty challenge dir survived the failed run"
        assert _leftover_files(tmp_path / "cache") == []

    def test_download_failure_removes_stale_incomplete_levels(self, tmp_path: Path) -> None:
        """VAL-SNIP-011, YouTube path: a failed re-download also clears stale partials."""
        stub = StubSectionDownloader(MP3_FIXTURE)
        gen = SnippetGenerator(tmp_path, section_downloader=stub)
        song = _youtube_song()
        good = gen.ensure_snippets(song, challenge_id=19, offset=OFFSET, lengths=LENGTHS)
        good[2].write_bytes(b"")
        kept = {level: _file_state(path) for level, path in good.items() if level != 2}
        # Drop the cached intermediate so the next run MUST re-download (and fail).
        for section in (tmp_path / "sections").glob("19.*"):
            section.unlink()

        def failing_downloader(
            url: str, start: float, end: float, dest_base: Path, timeout: float
        ) -> Path:
            raise RuntimeError("network boom")

        gen2 = SnippetGenerator(tmp_path, section_downloader=failing_downloader)
        with pytest.raises(SnippetGenerationError, match="network boom"):
            gen2.ensure_snippets(song, challenge_id=19, offset=OFFSET, lengths=LENGTHS)

        assert not good[2].exists(), "stale 0-byte level 2 survived the failed regeneration"
        for level, state in kept.items():
            assert _file_state(good[level]) == state, (
                f"pre-existing complete level {level} was touched"
            )
        partials = [p for p in _leftover_files(tmp_path) if p.stat().st_size == 0]
        assert partials == [], f"zero-byte partials left behind: {partials}"


class TestYouTubeFlow:
    """Expected behavior: one section download, local cuts, cached intermediate."""

    def test_single_section_download_then_local_cuts(self, tmp_path: Path) -> None:
        stub = StubSectionDownloader(MP3_FIXTURE)
        gen = SnippetGenerator(tmp_path, section_downloader=stub)
        song = _youtube_song()

        result = gen.ensure_snippets(song, challenge_id=11, offset=OFFSET, lengths=LENGTHS)

        assert len(stub.calls) == 1, "expected exactly one section download"
        url, start, end = stub.calls[0]
        print(f"section download: url={url} start={start}s end={end}s")
        assert url == song.audio_ref
        assert start == pytest.approx(OFFSET - 2.0)
        assert end == pytest.approx(OFFSET + max(LENGTHS) + 2.0)

        for level, target in enumerate(LENGTHS):
            assert _probe_codec(result[level]) == "mp3"
            assert abs(_probe_duration(result[level]) - target) <= TOLERANCE_SEC

        # Content alignment: the stubbed section starts at OFFSET-2s of the
        # fixture, so snippets must match a direct reference cut at OFFSET.
        ref = tmp_path / "ref.mp3"
        _ffmpeg("-ss", repr(OFFSET), "-t", "16", "-i", str(MP3_FIXTURE),
                "-c:a", "libmp3lame", str(ref))
        edge = int(0.05 * SAMPLE_RATE)
        snippet, ref_pcm = _decode_pcm(result[4]), _decode_pcm(ref)
        n = min(len(snippet), len(ref_pcm)) - edge
        mad = _mean_abs_diff(snippet[edge:n], ref_pcm[edge:n])
        print(f"youtube-path offset fidelity vs reference cut: {mad:.6f}")
        assert mad < 0.01

        # Intermediate cached OUTSIDE the challenge dir; challenge dir holds exactly 0..4.mp3.
        sections = list((tmp_path / "sections").glob("11.*"))
        assert len(sections) == 1
        print(f"cached intermediate: {sections[0]}")
        challenge_files = sorted(p.name for p in (tmp_path / "11").iterdir())
        assert challenge_files == ["0.mp3", "1.mp3", "2.mp3", "3.mp3", "4.mp3"]

    def test_offset_below_padding_clamps_section_start(self, tmp_path: Path) -> None:
        stub = StubSectionDownloader(MP3_FIXTURE)
        gen = SnippetGenerator(tmp_path, section_downloader=stub)

        result = gen.ensure_snippets(_youtube_song(), challenge_id=12, offset=1.0, lengths=LENGTHS)

        _, start, end = stub.calls[0]
        print(f"clamped section: start={start}s end={end}s")
        assert start == 0.0
        assert end == pytest.approx(1.0 + max(LENGTHS) + 2.0)
        # Content still starts at the requested offset (1.0s into the source).
        ref = tmp_path / "ref.mp3"
        _ffmpeg("-ss", "1.0", "-t", "16", "-i", str(MP3_FIXTURE), "-c:a", "libmp3lame", str(ref))
        edge = int(0.05 * SAMPLE_RATE)
        snippet, ref_pcm = _decode_pcm(result[4]), _decode_pcm(ref)
        n = min(len(snippet), len(ref_pcm)) - edge
        assert _mean_abs_diff(snippet[edge:n], ref_pcm[edge:n]) < 0.01

    def test_second_run_and_partial_regen_skip_download(self, tmp_path: Path) -> None:
        stub = StubSectionDownloader(MP3_FIXTURE)
        gen = SnippetGenerator(tmp_path, section_downloader=stub)
        song = _youtube_song()

        first = gen.ensure_snippets(song, challenge_id=13, offset=OFFSET, lengths=LENGTHS)
        assert len(stub.calls) == 1

        again = gen.ensure_snippets(song, challenge_id=13, offset=OFFSET, lengths=LENGTHS)
        assert len(stub.calls) == 1, "fully cached run must not re-download"
        assert again == first

        first[0].unlink()
        healed = gen.ensure_snippets(song, challenge_id=13, offset=OFFSET, lengths=LENGTHS)
        assert len(stub.calls) == 1, "partial regeneration must reuse the cached intermediate"
        assert abs(_probe_duration(healed[0]) - LENGTHS[0]) <= TOLERANCE_SEC

    def test_download_failure_leaves_no_partials(self, tmp_path: Path) -> None:
        def failing_downloader(
            url: str, start: float, end: float, dest_base: Path, timeout: float
        ) -> Path:
            raise RuntimeError("network boom")

        gen = SnippetGenerator(tmp_path, section_downloader=failing_downloader)
        with pytest.raises(SnippetGenerationError, match="stubbedvideo1") as excinfo:
            gen.ensure_snippets(_youtube_song(), challenge_id=14, offset=OFFSET, lengths=LENGTHS)

        print(f"error: {excinfo.value}")
        leftovers = _leftover_files(tmp_path)
        print(f"leftover files: {leftovers}")
        assert leftovers == []

    def test_youtube_source_too_short_fails_before_download(self, tmp_path: Path) -> None:
        stub = StubSectionDownloader(MP3_FIXTURE)
        gen = SnippetGenerator(tmp_path, section_downloader=stub)
        song = _youtube_song(duration_sec=30.0)

        with pytest.raises(SnippetSourceError, match="too short"):
            gen.ensure_snippets(song, challenge_id=17, offset=20.0, lengths=LENGTHS)
        assert stub.calls == [], "no download may happen for an impossible offset"


class TestFullAudio:
    """ensure_full_audio (issue #7): the solver's whole track at full.mp3."""

    def test_local_mp3_is_copied_byte_identical(self, tmp_path: Path) -> None:
        gen = SnippetGenerator(tmp_path)
        song = _local_song()

        path = gen.ensure_full_audio(song, challenge_id=21)

        assert path == tmp_path / "21" / "full.mp3"
        assert path.read_bytes() == MP3_FIXTURE.read_bytes()

    def test_local_m4a_reencodes_to_mp3(self, tmp_path: Path) -> None:
        """Non-mp3 containers re-encode so songbot-full.mp3 is truthful."""
        gen = SnippetGenerator(tmp_path)
        song = _local_song(M4A_FIXTURE)

        path = gen.ensure_full_audio(song, challenge_id=22)

        assert path == tmp_path / "22" / "full.mp3"
        assert _probe_codec(path) == "mp3"
        duration = _probe_duration(path)
        print(f"m4a -> mp3 full audio: {duration:.4f}s")
        assert abs(duration - 30.0) <= 0.5  # whole track, no section cut

    def test_local_cache_hit_survives_source_removal(self, tmp_path: Path) -> None:
        source = tmp_path / "source.mp3"
        source.write_bytes(MP3_FIXTURE.read_bytes())
        gen = SnippetGenerator(tmp_path / "cache")
        song = _local_song(source)
        first = gen.ensure_full_audio(song, challenge_id=23)

        source.unlink()  # a cache hit never touches the source again
        again = gen.ensure_full_audio(song, challenge_id=23)

        assert again == first
        assert _file_state(again) == _file_state(first)

    def test_zero_byte_full_file_is_regenerated(self, tmp_path: Path) -> None:
        target = tmp_path / "24" / "full.mp3"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"")  # a partial artifact from a crashed run

        gen = SnippetGenerator(tmp_path)
        path = gen.ensure_full_audio(_local_song(), challenge_id=24)

        assert path == target
        assert path.read_bytes() == MP3_FIXTURE.read_bytes()

    def test_youtube_downloads_once_into_the_challenge_dir(self, tmp_path: Path) -> None:
        stub = StubFullDownloader(MP3_FIXTURE)
        gen = SnippetGenerator(tmp_path, full_downloader=stub)
        song = _youtube_song()

        path = gen.ensure_full_audio(song, challenge_id=25)

        assert stub.calls == [song.audio_ref]
        assert path == tmp_path / "25" / "full.mp3"
        assert _probe_codec(path) == "mp3"

        def boom(url: str, dest_base: Path, timeout: float) -> Path:
            raise AssertionError("cache hit must not re-download")

        again = SnippetGenerator(tmp_path, full_downloader=boom).ensure_full_audio(
            song, challenge_id=25
        )
        assert again == path

    def test_youtube_download_failure_leaves_no_partials(self, tmp_path: Path) -> None:
        def failing_downloader(url: str, dest_base: Path, timeout: float) -> Path:
            raise RuntimeError("network boom")

        gen = SnippetGenerator(tmp_path, full_downloader=failing_downloader)
        with pytest.raises(SnippetGenerationError, match="network boom"):
            gen.ensure_full_audio(_youtube_song(), challenge_id=26)

        assert _leftover_files(tmp_path) == []

    def test_unreadable_download_is_removed(self, tmp_path: Path) -> None:
        def corrupt_downloader(url: str, dest_base: Path, timeout: float) -> Path:
            dest = dest_base.parent / f"{dest_base.name}.mp3"
            dest.write_bytes(b"not actually audio")
            return dest

        gen = SnippetGenerator(tmp_path, full_downloader=corrupt_downloader)
        with pytest.raises(SnippetGenerationError):
            gen.ensure_full_audio(_youtube_song(), challenge_id=27)

        assert _leftover_files(tmp_path) == []

    def test_purge_removes_the_full_audio_too(self, tmp_path: Path) -> None:
        gen = SnippetGenerator(tmp_path)
        full = gen.ensure_full_audio(_local_song(), challenge_id=28)
        assert full.is_file()

        gen.purge_challenge(28)

        assert not full.exists()
        assert not (tmp_path / "28").exists()


@pytest.fixture(scope="module")
def real_youtube_song() -> Song:
    """The named contract video from the real playlist (YouTube network, allowed)."""
    songs = YouTubePlaylistProvider(PLAYLIST_URL).fetch()
    song = next((s for s in songs if s.source_id == REAL_VIDEO_ID), None)
    if song is None:
        pytest.skip(f"video {REAL_VIDEO_ID} is no longer present in the playlist")
    return song


_FULL_AUDIO_ATTEMPTS = 3
"""Full-audio download attempts before giving up (fresh extraction per attempt)."""


def _download_full_audio(url: str, dest_base: Path) -> Path:
    """Fetch the complete bestaudio file via yt-dlp's native downloader.

    Mirrors the retry pattern of ``snippets.download_section_with_timeout``:
    up to ``_FULL_AUDIO_ATTEMPTS`` attempts with a FRESH extraction (hence a
    fresh googlevideo URL) per attempt, because googlevideo intermittently
    answers a transient 403 that only a fresh URL clears. Partial output of
    failed attempts is removed so no stray artifacts survive.
    """
    last_error: Exception | None = None
    for attempt in range(1, _FULL_AUDIO_ATTEMPTS + 1):
        try:
            return _download_full_audio_once(url, dest_base)
        except Exception as exc:
            last_error = exc
            print(
                f"full-audio download attempt {attempt}/{_FULL_AUDIO_ATTEMPTS} "
                f"for {url!r} failed: {exc}"
            )
            for leftover in dest_base.parent.glob(f"{dest_base.name}.*"):
                leftover.unlink(missing_ok=True)
            if attempt < _FULL_AUDIO_ATTEMPTS:
                time.sleep(min(2.0 ** (attempt - 1), 4.0))
    raise AssertionError(
        f"full-audio download failed after {_FULL_AUDIO_ATTEMPTS} attempts: {last_error}"
    )


def _download_full_audio_once(url: str, dest_base: Path) -> Path:
    """Single full-audio download attempt (one fresh yt-dlp extraction)."""
    options: dict[str, object] = {
        "format": "bestaudio/best",
        "outtmpl": f"{dest_base}.%(ext)s",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30.0,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])
    matches = [
        p
        for p in dest_base.parent.glob(f"{dest_base.name}.*")
        if p.suffix != ".part" and p.is_file() and p.stat().st_size > 0
    ]
    if not matches:
        raise RuntimeError(f"full-audio download produced no file at {dest_base}.*")
    return sorted(matches)[0]


def _best_alignment_shift(
    needle: array, window: array, center: int, span: int, step: int = 24
) -> tuple[int, float]:
    """Shift (samples, relative to ``center``) minimizing needle/window MAD."""
    best: tuple[int, float] | None = None
    for shift in range(-span, span + 1, step):
        pos = center + shift
        segment = window[pos : pos + len(needle)]
        if len(segment) < len(needle):
            continue
        mad = _mean_abs_diff(needle, segment)
        if best is None or mad < best[1]:
            best = (shift, mad)
    assert best is not None, "alignment scan window too small"
    return best


class TestFullAudioDownloadRetry:
    """The full-audio helper mirrors the section downloader's retry pattern.

    Transient googlevideo 403s must be absorbed by retrying with a fresh
    extraction (fresh URL) per attempt — never by skipping the assertion.
    Fully local: yt-dlp is stubbed out.
    """

    def test_retries_with_fresh_extraction_until_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        downloads: list[list[str]] = []

        class FakeYoutubeDL:
            def __init__(self, options: dict[str, object]) -> None:
                self.options = options

            def __enter__(self) -> FakeYoutubeDL:
                return self

            def __exit__(self, *exc: object) -> None:
                return None

            def download(self, urls: list[str]) -> None:
                downloads.append(list(urls))
                if len(downloads) < 3:
                    # A failed attempt leaves a .part file the retry must clean up
                    # (a real successful yt-dlp run renames it to the final name).
                    (tmp_path / "full.webm.part").write_bytes(b"partial")
                    raise RuntimeError("HTTP Error 403: Forbidden")
                (tmp_path / "full.webm").write_bytes(b"full audio")

        monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYoutubeDL)
        monkeypatch.setattr(time, "sleep", lambda _seconds: None)

        result = _download_full_audio("https://www.youtube.com/watch?v=x", tmp_path / "full")

        assert result == tmp_path / "full.webm"
        assert downloads == [["https://www.youtube.com/watch?v=x"]] * 3
        assert list(tmp_path.glob("*.part")) == [], "stale .part file survived the retries"

    def test_gives_up_after_three_attempts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = 0

        class AlwaysForbidden:
            def __init__(self, options: dict[str, object]) -> None:
                self.options = options

            def __enter__(self) -> AlwaysForbidden:
                return self

            def __exit__(self, *exc: object) -> None:
                return None

            def download(self, urls: list[str]) -> None:
                nonlocal calls
                calls += 1
                raise RuntimeError("HTTP Error 403: Forbidden")

        monkeypatch.setattr(yt_dlp, "YoutubeDL", AlwaysForbidden)
        monkeypatch.setattr(time, "sleep", lambda _seconds: None)

        with pytest.raises(AssertionError, match="after 3 attempts"):
            _download_full_audio("https://www.youtube.com/watch?v=x", tmp_path / "full")
        assert calls == 3


class TestSectionDownloadClientRotation:
    """Issue #11: section downloads rotate YouTube player clients per attempt.

    googlevideo 403s a URL flavor tied to the EXTRACTING client (observed
    2026-08-27: every ANDROID_VR URL 403ed for hours), so each attempt
    re-extracts with a DIFFERENT player client: attempt 1 uses yt-dlp's
    defaults (no extractor_args), the following attempts pin player_client
    from `_PLAYER_CLIENT_SETS`. Fully local: yt-dlp is stubbed out.
    """

    @staticmethod
    def _fake_ytdl(tmp_path: Path, seen_clients: list[object], fail_attempts: int) -> type:
        """A YoutubeDL stub recording player_client per construction."""

        class FakeYoutubeDL:
            def __init__(self, options: dict[str, object]) -> None:
                extractor_args = options.get("extractor_args")
                clients = (
                    extractor_args.get("youtube", {}).get("player_client")
                    if isinstance(extractor_args, dict)
                    and isinstance(extractor_args.get("youtube"), dict)
                    else None
                )
                seen_clients.append(None if clients is None else tuple(clients))

            def __enter__(self) -> FakeYoutubeDL:
                return self

            def __exit__(self, *exc: object) -> None:
                return None

            def download(self, urls: list[str]) -> None:
                if len(seen_clients) <= fail_attempts:
                    # A failed attempt leaves a .part file the retry must clean up.
                    (tmp_path / "section.webm.part").write_bytes(b"partial")
                    raise RuntimeError("HTTP Error 403: Forbidden")
                (tmp_path / "section.webm").write_bytes(b"audio")

        return FakeYoutubeDL

    def test_rotates_player_clients_until_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen_clients: list[object] = []
        monkeypatch.setattr(
            yt_dlp, "YoutubeDL", self._fake_ytdl(tmp_path, seen_clients, fail_attempts=2)
        )
        monkeypatch.setattr(time, "sleep", lambda _seconds: None)

        result = download_section_with_timeout(
            "https://www.youtube.com/watch?v=x", 1.0, 5.0, tmp_path / "section", 30.0
        )

        assert result == tmp_path / "section.webm"
        expected = [
            None if clients is None else tuple(clients)
            for clients in snippets_module._PLAYER_CLIENT_SETS
        ]
        assert seen_clients == expected[:3], (
            "attempt 1 must use yt-dlp defaults, then rotate through the client sets"
        )
        assert list(tmp_path.glob("*.part")) == [], "stale .part file survived the retries"

    def test_exhausts_all_player_clients_then_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen_clients: list[object] = []
        monkeypatch.setattr(
            yt_dlp,
            "YoutubeDL",
            self._fake_ytdl(tmp_path, seen_clients, fail_attempts=10**9),
        )
        monkeypatch.setattr(time, "sleep", lambda _seconds: None)

        attempts = len(snippets_module._PLAYER_CLIENT_SETS)
        with pytest.raises(SnippetGenerationError, match=f"after {attempts} attempts"):
            download_section_with_timeout(
                "https://www.youtube.com/watch?v=x", 1.0, 5.0, tmp_path / "section", 30.0
            )

        expected = [
            None if clients is None else tuple(clients)
            for clients in snippets_module._PLAYER_CLIENT_SETS
        ]
        assert seen_clients == expected
        assert list(tmp_path.glob("*")) == [], "no artifacts survive a fully failed download"


@pytest.mark.flaky(reruns=3, reruns_delay=30, reruns_delay_backoff_factor=2.0)
class TestRealYouTube:
    """VAL-SNIP-010 generator-level shape against the real YouTube video.

    Bounded auto-rerun (pytest-rerunfailures): googlevideo throttling comes in
    windows lasting minutes (observed 2026-08-13: a window outlasted 3 test
    executions with 30s gaps — every in-helper retry attempt 403ed), so the
    rerun delays back off exponentially (30s/60s/120s) to span such a window.
    One transient 403 — or one multi-minute throttle window — must not fail
    the full-suite gate. Reruns re-execute the REAL network assertions:
    nothing is ever skipped, and a genuine regression fails consistently
    across all 4 executions and still fails the suite.
    """

    def test_real_youtube_section_download(self, tmp_path: Path, real_youtube_song: Song) -> None:
        offset = 30.0  # safe: every playlist entry is >= 105s
        gen = SnippetGenerator(tmp_path)
        result = gen.ensure_snippets(
            real_youtube_song, challenge_id=15, offset=offset, lengths=LENGTHS
        )

        for level, target in enumerate(LENGTHS):
            measured = _probe_duration(result[level])
            codec = _probe_codec(result[level])
            print(f"level {level}: codec={codec} duration={measured:.4f}s target={target}s")
            assert codec == "mp3"
            assert abs(measured - target) <= TOLERANCE_SEC
        sections = list((tmp_path / "sections").glob("15.*"))
        assert len(sections) == 1
        print(f"cached intermediate: {sections[0]} ({sections[0].stat().st_size} bytes)")

        # Second run must not re-download: a generator whose downloader always
        # fails still succeeds because every level file is cached.
        def boom(url: str, start: float, end: float, dest_base: Path, timeout: float) -> Path:
            raise AssertionError("second run must not re-download")

        gen2 = SnippetGenerator(tmp_path, section_downloader=boom)
        again = gen2.ensure_snippets(
            real_youtube_song, challenge_id=15, offset=offset, lengths=LENGTHS
        )
        assert again == result

    def test_real_youtube_intermediate_starts_at_section_start(
        self, tmp_path: Path, real_youtube_song: Song
    ) -> None:
        """Offset fidelity end-to-end: the fetched section starts at offset-2s.

        A stream-copied section would start at a cluster boundary seconds
        early (measured ~8s on this video); force-keyframed re-encode must
        land within 25ms of the requested start.
        """
        offset = 30.0
        section_start = offset - 2.0
        gen = SnippetGenerator(tmp_path)
        gen.ensure_snippets(real_youtube_song, challenge_id=15, offset=offset, lengths=LENGTHS)
        intermediate = next((tmp_path / "sections").glob("15.*"))

        full = _download_full_audio(real_youtube_song.audio_ref, tmp_path / "full")
        rate = 48000  # YouTube bestaudio (opus) is 48kHz
        window = _decode_pcm(full, ss=section_start - 2.0, t=8.0, rate=rate)
        needle = _decode_pcm(intermediate, ss=0.5, t=2.0, rate=rate)
        center = int(2.5 * rate)  # needle begins 0.5s into the section
        shift, mad = _best_alignment_shift(needle, window, center, span=int(0.05 * rate))
        print(
            f"intermediate alignment vs full audio: shift {shift} samples "
            f"({shift / rate * 1000:.1f}ms), MAD {mad:.6f}"
        )
        assert abs(shift) <= int(0.025 * rate), (
            f"section starts {shift / rate * 1000:.1f}ms off the requested start"
        )
        assert mad < 0.02, "intermediate content does not match the source audio"
