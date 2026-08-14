"""LocalDirectoryProvider: mutagen tags with 'Artist - Title.ext' filename fallback.

Scans a local music directory (recursively) for supported audio files and
builds one `Song` per file:

- Supported extensions: .mp3, .m4a, .flac, .ogg (case-insensitive). Anything
  else is ignored.
- Embedded tags take precedence over the filename. Tags are read with
  mutagen's easy interface, which maps exactly the frames the architecture
  pins: ID3 ``TIT2``/``TPE1`` (mp3), MP4 ``©nam``/``©ART`` (m4a), and Vorbis
  ``TITLE``/``ARTIST`` (flac/ogg).
- Untagged (or partially tagged) files fall back to parsing the filename stem
  with `parse_artist_title` ("Artist - Title.ext").
- ``duration_sec`` comes from mutagen's stream info.
- ``source`` is ``"local"``, ``source_id`` is the path relative to the music
  dir (POSIX separators), ``audio_ref`` is the absolute file path, and
  ``raw_title`` is the original filename stem.

Unreadable or corrupt files are skipped with a logged warning naming the
file; they never abort the scan.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mutagen

from songbot.catalog import Song
from songbot.catalog.parsing import parse_artist_title

__all__ = ["SUPPORTED_EXTENSIONS", "LocalDirectoryProvider"]

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".mp3", ".m4a", ".flac", ".ogg")


class LocalDirectoryProvider:
    """Catalog provider backed by a local directory of audio files."""

    def __init__(self, music_dir: Path | str) -> None:
        self._music_dir = Path(music_dir).expanduser().resolve()

    @property
    def music_dir(self) -> Path:
        """The resolved directory this provider scans."""
        return self._music_dir

    def fetch(self) -> list[Song]:
        """Scan the music directory and return one `Song` per supported file.

        Raises:
            FileNotFoundError: if the music directory does not exist.
        """
        if not self._music_dir.is_dir():
            raise FileNotFoundError(
                f"Local music directory '{self._music_dir}' does not exist"
            )
        songs: list[Song] = []
        for path in sorted(self._music_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            song = self._song_from_file(path)
            if song is not None:
                songs.append(song)
        return songs

    def _song_from_file(self, path: Path) -> Song | None:
        """Build a `Song` from one audio file, or `None` (with a warning) to skip it."""
        try:
            audio = mutagen.File(path, easy=True)
        except Exception as exc:  # corrupt files raise various MutagenErrors
            logger.warning("Skipping unreadable audio file '%s': %s", path, exc)
            return None
        if audio is None:
            logger.warning(
                "Skipping audio file '%s': format not recognized by mutagen", path
            )
            return None

        length = getattr(audio.info, "length", None)
        if not isinstance(length, (int, float)) or length <= 0:
            logger.warning("Skipping audio file '%s': no readable duration", path)
            return None

        tag_title = _first_text(audio.get("title"))
        tag_artist = _first_text(audio.get("artist"))
        parsed_artist, parsed_title = parse_artist_title(path.stem)
        title = tag_title if tag_title is not None else parsed_title
        artist = tag_artist if tag_artist is not None else parsed_artist
        if not title:
            logger.warning("Skipping audio file '%s': could not determine a title", path)
            return None

        return Song(
            source="local",
            source_id=path.relative_to(self._music_dir).as_posix(),
            title=title,
            artist=artist,
            duration_sec=float(length),
            audio_ref=str(path),
            raw_title=path.stem,
        )


def _first_text(values: object) -> str | None:
    """Return the first non-empty string of a mutagen tag value list, if any."""
    if not isinstance(values, (list, tuple)):
        return None
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
