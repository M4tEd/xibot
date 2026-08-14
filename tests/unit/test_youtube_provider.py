"""Unit tests for YouTubePlaylistProvider with a stubbed yt-dlp flat dump (no network)."""

from __future__ import annotations

import time
from typing import Any

import pytest

from songbot.catalog import CatalogProvider
from songbot.catalog.youtube import (
    MAX_DURATION_SEC,
    MIN_DURATION_SEC,
    YouTubeCatalogError,
    YouTubePlaylistProvider,
    load_entries_with_timeout,
)

PLAYLIST_URL = "https://youtube.com/playlist?list=PL_TEST"


def _entry(
    video_id: str,
    *,
    title: str = "Some Artist - Some Title",
    duration: Any = 200,
    uploader: Any = "Some Channel",
    url: Any = None,
) -> dict[str, Any]:
    return {
        "id": video_id,
        "title": title,
        "duration": duration,
        "uploader": uploader,
        "url": url if url is not None else f"https://www.youtube.com/watch?v={video_id}",
    }


def _provider(entries: list[Any]) -> YouTubePlaylistProvider:
    return YouTubePlaylistProvider(PLAYLIST_URL, entry_loader=lambda url, timeout: entries)


class TestProtocol:
    def test_provider_satisfies_catalog_protocol(self) -> None:
        assert isinstance(_provider([]), CatalogProvider)


class TestEntryMapping:
    def test_maps_flat_entry_fields(self) -> None:
        songs = _provider([_entry("abc123XYZ_-")]).fetch()
        assert len(songs) == 1
        song = songs[0]
        print(f"mapped song: {song!r}")
        assert song.source == "youtube"
        assert song.source_id == "abc123XYZ_-"
        assert song.audio_ref == "https://www.youtube.com/watch?v=abc123XYZ_-"
        assert song.raw_title == "Some Artist - Some Title"
        assert song.title == "Some Title"
        assert song.artist == "Some Artist"
        assert song.duration_sec == 200.0
        assert isinstance(song.duration_sec, float)

    def test_uses_entry_url_field_as_audio_ref(self) -> None:
        # Flat entries carry the watch URL in `url` (there is no `webpage_url`).
        entry = _entry("vid", url="https://www.youtube.com/watch?v=vid")
        songs = _provider([entry]).fetch()
        assert songs[0].audio_ref == "https://www.youtube.com/watch?v=vid"

    def test_non_absolute_url_rebuilt_from_video_id(self) -> None:
        # Defensive: some extractors put a bare video id in `url`.
        songs = _provider([_entry("dQw4w9WgXcQ", url="dQw4w9WgXcQ")]).fetch()
        assert songs[0].audio_ref == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_missing_url_rebuilt_from_video_id(self) -> None:
        entry = _entry("dQw4w9WgXcQ")
        del entry["url"]
        songs = _provider([entry]).fetch()
        assert songs[0].audio_ref == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_raw_title_preserves_original_title_verbatim(self) -> None:
        songs = _provider([_entry("vid", title="  Padded Title  ")]).fetch()
        assert songs[0].raw_title == "  Padded Title  "
        assert songs[0].title == "Padded Title"

    def test_bracket_and_slash_forms_parsed(self) -> None:
        songs = _provider(
            [
                _entry("a", title="[Official] ANiMA / xi [World Fragments]"),
                _entry("b", title="【Paradigm: Reboot】xi VS Sakuzyo - Abyssgazer"),
            ]
        ).fetch()
        by_id = {song.source_id: song for song in songs}
        for song in songs:
            print(f"{song.raw_title!r} -> ({song.artist!r}, {song.title!r})")
        assert (by_id["a"].artist, by_id["a"].title) == ("xi", "ANiMA")
        assert by_id["b"].title == "Abyssgazer"
        assert by_id["b"].artist


class TestUploaderFallback:
    def test_bare_title_falls_back_to_uploader_with_topic_stripped(self) -> None:
        songs = _provider([_entry("vid", title="Agartha", uploader="xi - Topic")]).fetch()
        print(f"Agartha (uploader 'xi - Topic') -> {songs[0].artist!r}/{songs[0].title!r}")
        assert songs[0].title == "Agartha"
        assert songs[0].artist == "xi"

    def test_topic_suffix_stripped_case_insensitively(self) -> None:
        songs = _provider([_entry("vid", title="Agartha", uploader="XI - TOPIC")]).fetch()
        assert songs[0].artist == "XI"

    def test_bare_title_without_uploader_yields_none_artist(self) -> None:
        songs = _provider([_entry("vid", title="Agartha", uploader=None)]).fetch()
        assert songs[0].title == "Agartha"
        assert songs[0].artist is None

    def test_uploader_that_is_only_a_topic_suffix_yields_none(self) -> None:
        songs = _provider([_entry("vid", title="Agartha", uploader="- Topic")]).fetch()
        assert songs[0].artist is None

    def test_channel_used_when_uploader_missing(self) -> None:
        entry = _entry("vid", title="Agartha", uploader=None)
        entry["channel"] = "xi - Topic"
        songs = _provider([entry]).fetch()
        assert songs[0].artist == "xi"

    def test_parsed_artist_beats_uploader(self) -> None:
        songs = _provider([_entry("vid", title="XI - Akasha", uploader="Raudi")]).fetch()
        assert songs[0].artist == "XI"
        assert songs[0].title == "Akasha"


class TestDurationFilter:
    """VAL-CATALOG-010: only 45-600s (inclusive) is admitted; missing duration excluded."""

    def test_bound_constants(self) -> None:
        assert MIN_DURATION_SEC == 45.0
        assert MAX_DURATION_SEC == 600.0

    def test_duration_boundaries(self) -> None:
        entries = [
            _entry("d44", duration=44),
            _entry("d45", duration=45),
            _entry("d600", duration=600),
            _entry("d601", duration=601),
            _entry("dNone", duration=None),
            _entry("dMissing"),  # duration key removed below
        ]
        del entries[5]["duration"]

        songs = _provider(entries).fetch()
        kept = {song.source_id for song in songs}
        for entry in entries:
            included = entry.get("id") in kept
            print(f"duration={entry.get('duration', '<missing>')!r:>10} -> included={included}")

        assert kept == {"d45", "d600"}

    def test_fractional_duration_at_bounds(self) -> None:
        entries = [_entry("lo", duration=44.9), _entry("hi", duration=600.0)]
        songs = _provider(entries).fetch()
        assert [song.source_id for song in songs] == ["hi"]

    def test_non_numeric_duration_excluded_without_raising(self) -> None:
        entries = [_entry("bad", duration="abc"), _entry("ok", duration=100)]
        songs = _provider(entries).fetch()
        assert [song.source_id for song in songs] == ["ok"]

    def test_zero_and_negative_durations_excluded(self) -> None:
        entries = [_entry("zero", duration=0), _entry("neg", duration=-5)]
        assert _provider(entries).fetch() == []


class TestSkipping:
    def test_missing_title_skipped(self) -> None:
        entry = _entry("vid")
        del entry["title"]
        assert _provider([entry]).fetch() == []

    def test_blank_title_skipped(self) -> None:
        assert _provider([_entry("vid", title="   ")]).fetch() == []

    def test_missing_id_skipped(self) -> None:
        entry = _entry("vid")
        del entry["id"]
        assert _provider([entry]).fetch() == []

    def test_non_mapping_entries_skipped(self) -> None:
        songs = _provider([None, _entry("ok")]).fetch()
        assert [song.source_id for song in songs] == ["ok"]

    def test_duplicate_video_ids_deduped(self) -> None:
        entries = [_entry("dup", title="A - One"), _entry("dup", title="A - Two")]
        songs = _provider(entries).fetch()
        assert len(songs) == 1
        assert songs[0].title == "One"


class TestErrors:
    def test_loader_error_wrapped_with_playlist_url(self) -> None:
        def boom(url: str, timeout: float) -> list[Any]:
            raise RuntimeError("connection reset")

        provider = YouTubePlaylistProvider(PLAYLIST_URL, entry_loader=boom)
        with pytest.raises(YouTubeCatalogError, match="PL_TEST"):
            provider.fetch()

    def test_catalog_error_passes_through_unwrapped(self) -> None:
        def fail(url: str, timeout: float) -> list[Any]:
            raise YouTubeCatalogError(f"no playlist at {url}")

        provider = YouTubePlaylistProvider(PLAYLIST_URL, entry_loader=fail)
        with pytest.raises(YouTubeCatalogError, match="no playlist at"):
            provider.fetch()

    def test_timeout_raises_named_error_promptly(self) -> None:
        def slow_extract(url: str, timeout: float) -> list[Any]:
            time.sleep(1.0)
            return []

        start = time.monotonic()
        with pytest.raises(YouTubeCatalogError, match=r"[Tt]imed out.*PL_TEST"):
            load_entries_with_timeout(PLAYLIST_URL, 0.05, extract_fn=slow_extract)
        elapsed = time.monotonic() - start
        print(f"timeout raised after {elapsed:.3f}s (bound was 0.05s)")
        assert elapsed < 1.0

    def test_empty_playlist_url_rejected(self) -> None:
        with pytest.raises(ValueError, match="playlist URL"):
            YouTubePlaylistProvider("  ")

    def test_non_positive_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            YouTubePlaylistProvider(PLAYLIST_URL, timeout_sec=0)
