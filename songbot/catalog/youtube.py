"""YouTubePlaylistProvider: catalog provider backed by a YouTube playlist via yt-dlp.

The playlist is fetched with the yt-dlp Python API as a flat dump
(``extract_flat="in_playlist"``, no download, no auth). Flat entries carry
``id``, ``url`` (the full watch URL — flat entries have no ``webpage_url``),
``title``, ``duration``, and ``uploader``/``channel``.

Mapping to `Song`:

- ``source`` = ``"youtube"``, ``source_id`` = the video id, ``raw_title`` =
  the original video title (verbatim).
- ``audio_ref`` = the entry's ``url`` field; if it is missing or not an
  absolute http(s) URL (some extractors put a bare id there), the watch URL
  is rebuilt from the video id.
- ``artist``/``title`` come from the shared `parse_artist_title` heuristics.
  When they yield no artist (bare titles), the provider falls back to the
  uploader/channel name with any trailing ``" - Topic"`` suffix stripped
  (YouTube auto-generated music channels are named ``"<Artist> - Topic"``).
  If no usable uploader exists either, ``artist`` stays ``None``.
- Duplicate video ids are de-duplicated (first occurrence wins) so
  ``(source, source_id)`` is always unique.

Duration filter: only entries with ``MIN_DURATION_SEC <= duration <=
MAX_DURATION_SEC`` (45-600s, inclusive) are kept. Entries with a missing or
non-numeric duration are EXCLUDED — a song of unknown length cannot be
duration-filtered or safely snippeted — and never cause an error.

Fetching is bounded: the yt-dlp extraction runs on a worker thread with a
wall-clock timeout (default 60s) and yt-dlp's own ``socket_timeout`` caps
individual socket operations. Any failure — network error, invalid or
non-playlist URL, timeout — raises `YouTubeCatalogError` naming the playlist
URL. On timeout the worker thread may linger briefly, but ``socket_timeout``
bounds its network waits, so it dies shortly after.
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import yt_dlp

from songbot.catalog import Song
from songbot.catalog.parsing import parse_artist_title

__all__ = [
    "DEFAULT_TIMEOUT_SEC",
    "MAX_DURATION_SEC",
    "MIN_DURATION_SEC",
    "FlatEntry",
    "YouTubeCatalogError",
    "YouTubePlaylistProvider",
    "load_entries_with_timeout",
]

logger = logging.getLogger(__name__)

MIN_DURATION_SEC = 45.0
"""Minimum admitted song duration in seconds (inclusive)."""

MAX_DURATION_SEC = 600.0
"""Maximum admitted song duration in seconds (inclusive)."""

DEFAULT_TIMEOUT_SEC = 60.0
"""Default wall-clock bound for a playlist fetch."""

FlatEntry = Mapping[str, Any]
"""One raw entry of a yt-dlp flat playlist dump."""

EntryLoader = Callable[[str, float], Sequence[FlatEntry]]
"""Loads flat entries for a playlist URL within a timeout. Injectable for tests."""

_WATCH_URL_TEMPLATE = "https://www.youtube.com/watch?v={video_id}"
_TOPIC_SUFFIX_RE = re.compile(r"\s*-\s*Topic$", re.IGNORECASE)


class YouTubeCatalogError(Exception):
    """Raised when a YouTube playlist cannot be fetched or parsed."""


class YouTubePlaylistProvider:
    """Catalog provider backed by a public YouTube playlist (no auth).

    ``entry_loader`` is injectable for tests; the default performs the real
    yt-dlp flat dump bounded by ``timeout_sec``.
    """

    def __init__(
        self,
        playlist_url: str,
        *,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        entry_loader: EntryLoader | None = None,
    ) -> None:
        if not playlist_url or not playlist_url.strip():
            raise ValueError("YouTube playlist URL must be non-empty")
        if timeout_sec <= 0:
            raise ValueError(f"timeout_sec must be positive, got {timeout_sec}")
        self._playlist_url = playlist_url.strip()
        self._timeout_sec = float(timeout_sec)
        self._entry_loader: EntryLoader = (
            entry_loader if entry_loader is not None else load_entries_with_timeout
        )

    @property
    def playlist_url(self) -> str:
        """The playlist URL this provider fetches."""
        return self._playlist_url

    @property
    def timeout_sec(self) -> float:
        """The wall-clock bound applied to each fetch."""
        return self._timeout_sec

    def fetch(self) -> list[Song]:
        """Fetch the playlist and return the duration-filtered songs.

        Raises:
            YouTubeCatalogError: if the playlist cannot be fetched (network
                failure, invalid/non-playlist URL, or timeout). The message
                always identifies the playlist URL.
        """
        try:
            entries = list(self._entry_loader(self._playlist_url, self._timeout_sec))
        except YouTubeCatalogError:
            raise
        except Exception as exc:
            raise YouTubeCatalogError(
                f"Failed to fetch YouTube playlist {self._playlist_url!r}: {exc}"
            ) from exc

        songs: list[Song] = []
        seen_ids: set[str] = set()
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            song = self._song_from_entry(entry)
            if song is None or song.source_id in seen_ids:
                continue
            seen_ids.add(song.source_id)
            songs.append(song)
        return songs

    @staticmethod
    def _song_from_entry(entry: FlatEntry) -> Song | None:
        """Map one flat entry to a `Song`, or `None` to skip it (logged at DEBUG)."""
        video_id = entry.get("id")
        if not isinstance(video_id, str) or not video_id.strip():
            logger.debug("Skipping playlist entry without a usable id: %r", video_id)
            return None
        video_id = video_id.strip()

        raw_title = entry.get("title")
        if not isinstance(raw_title, str) or not raw_title.strip():
            logger.debug("Skipping playlist entry %s without a usable title", video_id)
            return None

        duration = entry.get("duration")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            # Documented behavior: missing/invalid duration excludes the entry
            # (an unknown-length song cannot be duration-filtered or snippeted).
            logger.debug(
                "Skipping playlist entry %s: missing or invalid duration %r",
                video_id,
                duration,
            )
            return None
        duration_sec = float(duration)
        if not MIN_DURATION_SEC <= duration_sec <= MAX_DURATION_SEC:
            logger.debug(
                "Skipping playlist entry %s: duration %.1fs outside %.0f-%.0fs",
                video_id,
                duration_sec,
                MIN_DURATION_SEC,
                MAX_DURATION_SEC,
            )
            return None

        url = entry.get("url")
        if not isinstance(url, str) or not url.startswith("http"):
            url = _WATCH_URL_TEMPLATE.format(video_id=video_id)

        artist, title = parse_artist_title(raw_title)
        if artist is None:
            artist = _clean_uploader(entry.get("uploader") or entry.get("channel"))

        return Song(
            source="youtube",
            source_id=video_id,
            title=title,
            artist=artist,
            duration_sec=duration_sec,
            audio_ref=url,
            raw_title=raw_title,
        )


def load_entries_with_timeout(
    playlist_url: str,
    timeout_sec: float,
    extract_fn: EntryLoader | None = None,
) -> Sequence[FlatEntry]:
    """Run ``extract_fn`` (default: the real yt-dlp dump) with a wall-clock bound.

    The extraction runs on a single worker thread; if it does not finish
    within ``timeout_sec`` seconds a `YouTubeCatalogError` is raised without
    waiting for the thread (yt-dlp's ``socket_timeout`` ensures the orphaned
    thread's network operations also time out). Failures from the extraction
    itself are wrapped in `YouTubeCatalogError` naming the playlist URL.
    """
    extract = extract_fn if extract_fn is not None else _extract_flat_entries
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(extract, playlist_url, timeout_sec)
    try:
        return future.result(timeout=timeout_sec)
    except concurrent.futures.TimeoutError as exc:
        future.cancel()
        raise YouTubeCatalogError(
            f"Timed out after {timeout_sec:.0f}s fetching YouTube playlist {playlist_url!r}"
        ) from exc
    except YouTubeCatalogError:
        raise
    except Exception as exc:
        raise YouTubeCatalogError(
            f"Failed to fetch YouTube playlist {playlist_url!r}: {exc}"
        ) from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _extract_flat_entries(playlist_url: str, timeout_sec: float) -> Sequence[FlatEntry]:
    """Fetch a playlist's flat entry list via the yt-dlp Python API (network)."""
    options: dict[str, Any] = {
        "extract_flat": "in_playlist",
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": False,
        "noplaylist": False,
        "socket_timeout": min(timeout_sec, 30.0),
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
    if not isinstance(info, Mapping) or "entries" not in info:
        raise YouTubeCatalogError(
            f"URL {playlist_url!r} did not resolve to a playlist (no entries returned)"
        )
    return [entry for entry in info["entries"] or [] if isinstance(entry, Mapping)]


def _clean_uploader(value: object) -> str | None:
    """Return a usable artist name from an uploader/channel string, else `None`.

    Strips a trailing ``" - Topic"`` (case-insensitive) from YouTube's
    auto-generated music channel names.
    """
    if not isinstance(value, str):
        return None
    name = _TOPIC_SUFFIX_RE.sub("", value).strip()
    return name or None
