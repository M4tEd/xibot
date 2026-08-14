"""Integration tests for YouTubePlaylistProvider against the real playlist.

Network: YouTube only (allowed by mission boundaries). The real playlist
``PLDzqiyJzN_jBRIJhUB_vmD5jPIHNkflc1`` ("FULL xi song list") had 226 entries,
all within 105-561s, at mission start; assertions tolerate drift (>= 200).
"""

from __future__ import annotations

import time

import pytest

from songbot.catalog import CatalogProvider, Song
from songbot.catalog.parsing import parse_artist_title
from songbot.catalog.youtube import (
    MAX_DURATION_SEC,
    MIN_DURATION_SEC,
    YouTubeCatalogError,
    YouTubePlaylistProvider,
)

PLAYLIST_URL = "https://youtube.com/playlist?list=PLDzqiyJzN_jBRIJhUB_vmD5jPIHNkflc1"


@pytest.fixture(scope="module")
def songs() -> list[Song]:
    """Fetch the real playlist once per module (flat dump takes ~2.5s)."""
    provider = YouTubePlaylistProvider(PLAYLIST_URL)
    assert isinstance(provider, CatalogProvider)
    return provider.fetch()


def _find(songs: list[Song], video_id: str) -> Song:
    song = next((s for s in songs if s.source_id == video_id), None)
    if song is None:
        pytest.skip(f"video {video_id} is no longer present in the playlist")
    return song


class TestRealPlaylist:
    """VAL-CATALOG-006: well-formed songs from the real playlist."""

    def test_returns_at_least_200_wellformed_songs(self, songs: list[Song]) -> None:
        print(f"fetched {len(songs)} songs from {PLAYLIST_URL}")
        assert len(songs) >= 200

        seen: set[tuple[str, str]] = set()
        for song in songs:
            assert song.source == "youtube"
            assert song.source_id, "source_id must be the YouTube video id"
            assert song.audio_ref.startswith("https://")
            assert "watch" in song.audio_ref
            assert song.source_id in song.audio_ref, "audio_ref must be a watch URL with the id"
            assert song.title, "every song must have a non-empty title"
            assert MIN_DURATION_SEC <= song.duration_sec <= MAX_DURATION_SEC
            assert song.raw_title, "raw_title must preserve the original video title"
            pair = (song.source, song.source_id)
            assert pair not in seen, f"duplicate (source, source_id): {pair}"
            seen.add(pair)

        print("sample raw_title -> (artist, title) mappings:")
        for song in songs[:8]:
            print(f"  {song.raw_title!r} -> ({song.artist!r}, {song.title!r})")

    def test_all_titles_non_empty(self, songs: list[Song]) -> None:
        """VAL-CATALOG-009 (playlist-wide): 100% non-empty titles."""
        empty = [song.source_id for song in songs if not song.title]
        print(f"songs with empty title: {len(empty)} / {len(songs)}")
        assert empty == []


class TestDashTitles:
    """VAL-CATALOG-007: 'Artist - Title' titles are actually split."""

    def test_xi_akasha(self, songs: list[Song]) -> None:
        song = _find(songs, "jO4fTcziVRM")  # raw title: "XI - Akasha"
        print(f"{song.raw_title!r} -> ({song.artist!r}, {song.title!r})")
        assert song.title == "Akasha"
        assert song.artist is not None
        assert song.artist.lower() == "xi"

    def test_every_dash_title_is_split_unless_heuristic_declines(
        self, songs: list[Song]
    ) -> None:
        checked = declined = 0
        for song in songs:
            if " - " not in song.raw_title:
                continue
            checked += 1
            assert song.title, f"empty title for {song.raw_title!r}"
            artist, _ = parse_artist_title(song.raw_title)
            if artist is None:
                declined += 1  # heuristic legitimately declined (empty split side)
                continue
            assert song.title != song.raw_title, (
                f"title containing ' - ' was not split: {song.raw_title!r}"
            )
        print(f"dash-title check: {checked} titles containing ' - ', {declined} declined")
        assert checked > 0


class TestDecoratedTitles:
    """VAL-CATALOG-008: 'Title / Artist' form and bracket decorations."""

    def test_anima_slash_form_with_brackets(self, songs: list[Song]) -> None:
        song = _find(songs, "9F2sK2aO8-U")  # "[Official] ANiMA / xi [World Fragments]"
        print(f"{song.raw_title!r} -> ({song.artist!r}, {song.title!r})")
        assert song.title == "ANiMA"
        assert song.artist == "xi"

    def test_abyssgazer_fullwidth_brackets(self, songs: list[Song]) -> None:
        song = _find(songs, "XLRzISm_Y18")  # "【Paradigm: Reboot】xi VS Sakuzyo - Abyssgazer"
        print(f"{song.raw_title!r} -> ({song.artist!r}, {song.title!r})")
        assert song.title == "Abyssgazer"
        assert song.artist
        for value in (song.title, song.artist or ""):
            assert "【" not in value
            assert "】" not in value
            assert "[" not in value
            assert "]" not in value


class TestBareTitles:
    """VAL-CATALOG-009: bare titles still yield a non-empty title, no raise."""

    def test_agartha_bare_title(self, songs: list[Song]) -> None:
        song = _find(songs, "1Zn2uTmLo3Q")  # raw title "Agartha", uploader "xi - Topic"
        print(f"{song.raw_title!r} -> ({song.artist!r}, {song.title!r})")
        assert song.title == "Agartha"
        # This implementation falls back to the uploader, so ' - Topic' must be stripped.
        assert song.artist == "xi"


class TestInvalidPlaylist:
    """VAL-CATALOG-014: unreachable/invalid playlist fails clearly and promptly."""

    def test_bogus_playlist_raises_named_error_within_bound(self) -> None:
        url = "https://youtube.com/playlist?list=PL_INVALID_SONGBOT_TEST"
        start = time.monotonic()
        with pytest.raises(YouTubeCatalogError) as excinfo:
            YouTubePlaylistProvider(url).fetch()
        elapsed = time.monotonic() - start
        print(
            f"invalid playlist raised {type(excinfo.value).__name__} after "
            f"{elapsed:.1f}s: {excinfo.value}"
        )
        assert elapsed <= 60.0, "the fetch must be bounded by a timeout (<= 60s)"
        assert "PL_INVALID_SONGBOT_TEST" in str(excinfo.value)
