"""Unit tests for SnippetGenerator's pure logic: argument validation, cache
layout, and purge semantics. No ffmpeg invocations and no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from songbot.catalog import Song
from songbot.snippets import SnippetGenerator, SnippetSourceError

LENGTHS = (1.0, 2.0, 4.0, 8.0, 16.0)


def _song(audio_ref: str = "/nonexistent/x.mp3") -> Song:
    return Song(
        source="local",
        source_id="x.mp3",
        title="T",
        artist="A",
        duration_sec=30.0,
        audio_ref=audio_ref,
        raw_title="x",
    )


class TestArgumentValidation:
    """Invalid arguments raise ValueError before any source file is touched."""

    def test_negative_offset_rejected(self, tmp_path: Path) -> None:
        gen = SnippetGenerator(tmp_path)
        with pytest.raises(ValueError, match="offset"):
            gen.ensure_snippets(_song(), challenge_id=1, offset=-0.5, lengths=LENGTHS)

    def test_empty_lengths_rejected(self, tmp_path: Path) -> None:
        gen = SnippetGenerator(tmp_path)
        with pytest.raises(ValueError, match="lengths"):
            gen.ensure_snippets(_song(), challenge_id=1, offset=0.0, lengths=())

    def test_non_positive_length_rejected(self, tmp_path: Path) -> None:
        gen = SnippetGenerator(tmp_path)
        with pytest.raises(ValueError, match="lengths"):
            gen.ensure_snippets(_song(), challenge_id=1, offset=0.0, lengths=(1.0, 0.0))

    def test_non_finite_length_rejected(self, tmp_path: Path) -> None:
        gen = SnippetGenerator(tmp_path)
        with pytest.raises(ValueError, match="lengths"):
            gen.ensure_snippets(_song(), challenge_id=1, offset=0.0, lengths=(1.0, float("inf")))

    @pytest.mark.parametrize("bad_id", ["", "a/b", "..", ".", "a\\b"])
    def test_unsafe_challenge_id_rejected(self, tmp_path: Path, bad_id: str) -> None:
        gen = SnippetGenerator(tmp_path)
        with pytest.raises(ValueError, match="challenge_id"):
            gen.ensure_snippets(_song(), challenge_id=bad_id, offset=0.0, lengths=LENGTHS)

    def test_non_positive_timeouts_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="timeout"):
            SnippetGenerator(tmp_path, ffmpeg_timeout_sec=0)
        with pytest.raises(ValueError, match="timeout"):
            SnippetGenerator(tmp_path, download_timeout_sec=-1)


class TestFullAudioValidation:
    """ensure_full_audio argument/source validation — no ffmpeg, no network."""

    @pytest.mark.parametrize("bad_id", ["", "a/b", "..", ".", "a\\b"])
    def test_unsafe_challenge_id_rejected(self, tmp_path: Path, bad_id: str) -> None:
        gen = SnippetGenerator(tmp_path)
        with pytest.raises(ValueError, match="challenge_id"):
            gen.ensure_full_audio(_song(), challenge_id=bad_id)

    def test_missing_local_source_raises_named_error(self, tmp_path: Path) -> None:
        gen = SnippetGenerator(tmp_path)
        with pytest.raises(SnippetSourceError, match="does not exist"):
            gen.ensure_full_audio(_song(), challenge_id=1)
        assert list(tmp_path.rglob("*")) == []  # no cache artifacts left behind

    def test_empty_local_source_raises_named_error(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.mp3"
        empty.write_bytes(b"")
        gen = SnippetGenerator(tmp_path / "cache")
        with pytest.raises(SnippetSourceError, match="does not exist or is empty"):
            gen.ensure_full_audio(_song(str(empty)), challenge_id=1)
        assert not (tmp_path / "cache").exists()


class TestPurgeChallenge:
    """purge_challenge removes the challenge dir and any section intermediate."""

    def test_purge_removes_challenge_dir_and_intermediate(self, tmp_path: Path) -> None:
        challenge_dir = tmp_path / "5"
        challenge_dir.mkdir()
        (challenge_dir / "0.mp3").write_bytes(b"audio")
        (challenge_dir / "full.mp3").write_bytes(b"full audio")
        sections = tmp_path / "sections"
        sections.mkdir()
        intermediate = sections / "5.webm"
        intermediate.write_bytes(b"section")

        SnippetGenerator(tmp_path).purge_challenge(5)

        assert not challenge_dir.exists()
        assert not intermediate.exists()

    def test_purge_missing_challenge_is_a_noop(self, tmp_path: Path) -> None:
        SnippetGenerator(tmp_path).purge_challenge(999)  # must not raise

    def test_purge_rejects_unsafe_challenge_id(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="challenge_id"):
            SnippetGenerator(tmp_path).purge_challenge("../escape")
