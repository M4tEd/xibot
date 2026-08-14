"""SnippetGenerator: ffmpeg exact-duration cuts at a fixed offset + disk cache.

Local sources are cut directly with ffmpeg using input seeking and a full
RE-ENCODE to mp3 (``-ss <offset> -t <len> -i <file> -c:a libmp3lame``) —
stream copy overshoots on mp3 frame boundaries (~8ms; verified). YouTube
sources are fetched ONCE per challenge as a padded section
(``offset-2s .. offset+max_len+2s``, clamped at 0) via yt-dlp's
download-ranges API under a wall-clock timeout, cached as an intermediate,
and then cut locally per level with the same ffmpeg re-encode. The section
fetch uses yt-dlp's ``force_keyframes_at_cuts`` (re-encode): with stream copy
the section starts at the nearest container cluster boundary BEFORE the
requested start (measured ~8s early on a real video), which would silently
shift every snippet; the re-encode makes the section start sample-accurate
(verified against the full audio). Transient googlevideo 403s observed during
ffmpeg-based section fetches are absorbed by retrying with fresh URLs.

Cache layout (rooted at ``Settings.snippet_cache_dir``)::

    <cache_dir>/<challenge_id>/<level>.mp3      # one file per snippet level
    <cache_dir>/sections/<challenge_id>.<ext>   # YouTube section intermediate

Idempotency: existing non-empty level files are skipped untouched (a 0-byte
file is a partial artifact and is regenerated). Every generated file is
ffprobe-verified and raises on >50ms duration deviation. Failures raise a
clear named error (`SnippetSourceError` for missing/corrupt/too-short source
audio, `SnippetGenerationError` for ffmpeg/yt-dlp/verification failures) and
leave no partial cache files: anything the failed call created is removed,
while complete pre-existing files are kept.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import math
import os
import secrets
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yt_dlp

from songbot.catalog import Song

__all__ = [
    "DEFAULT_DOWNLOAD_TIMEOUT_SEC",
    "DEFAULT_FFMPEG_TIMEOUT_SEC",
    "DURATION_TOLERANCE_SEC",
    "SECTION_PADDING_SEC",
    "SectionDownloader",
    "SnippetError",
    "SnippetGenerationError",
    "SnippetGenerator",
    "SnippetSourceError",
    "download_section_with_timeout",
]

logger = logging.getLogger(__name__)

DURATION_TOLERANCE_SEC = 0.05
"""Maximum allowed deviation between target and probed snippet duration."""

SECTION_PADDING_SEC = 2.0
"""Padding on each side of the YouTube section download (absorbs cut overshoot)."""

DEFAULT_FFMPEG_TIMEOUT_SEC = 60.0
"""Wall-clock bound for each ffmpeg/ffprobe invocation."""

DEFAULT_DOWNLOAD_TIMEOUT_SEC = 120.0
"""Wall-clock bound for the yt-dlp section download (all attempts combined)."""

_DOWNLOAD_ATTEMPTS = 3
"""Section-download attempts before giving up (fresh URL per attempt)."""

SectionDownloader = Callable[[str, float, float, Path, float], Path]
"""Fetches ``url``'s audio from ``start`` to ``end`` seconds into a file.

Arguments: ``(url, start_sec, end_sec, dest_base, timeout_sec)``. The file is
written as ``<dest_base>.<ext>`` (extension chosen by the downloader) and its
path returned. The production default is `download_section_with_timeout`;
tests inject a stub.
"""


class SnippetError(Exception):
    """Base class for snippet generation failures."""


class SnippetSourceError(SnippetError):
    """The source audio is missing, corrupt, or too short for the request."""


class SnippetGenerationError(SnippetError):
    """ffmpeg/yt-dlp failed or a generated snippet failed verification."""


class SnippetGenerator:
    """Generates and caches exact-duration mp3 snippets for daily challenges.

    ``section_downloader`` is injectable for tests; the default performs the
    real yt-dlp section download bounded by ``download_timeout_sec``.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        ffmpeg_timeout_sec: float = DEFAULT_FFMPEG_TIMEOUT_SEC,
        download_timeout_sec: float = DEFAULT_DOWNLOAD_TIMEOUT_SEC,
        section_downloader: SectionDownloader | None = None,
    ) -> None:
        if ffmpeg_timeout_sec <= 0:
            raise ValueError(f"ffmpeg_timeout_sec must be positive, got {ffmpeg_timeout_sec}")
        if download_timeout_sec <= 0:
            raise ValueError(f"download_timeout_sec must be positive, got {download_timeout_sec}")
        self._cache_dir = Path(cache_dir)
        self._ffmpeg_timeout_sec = float(ffmpeg_timeout_sec)
        self._download_timeout_sec = float(download_timeout_sec)
        self._ffmpeg = shutil.which("ffmpeg")
        self._ffprobe = shutil.which("ffprobe")
        if self._ffmpeg is None or self._ffprobe is None:
            raise SnippetError(
                "ffmpeg and ffprobe must be on PATH "
                f"(found ffmpeg={self._ffmpeg!r}, ffprobe={self._ffprobe!r})"
            )
        self._section_downloader: SectionDownloader = (
            section_downloader if section_downloader is not None else download_section_with_timeout
        )

    @property
    def cache_dir(self) -> Path:
        """The snippet cache root this generator works under."""
        return self._cache_dir

    def ensure_snippets(
        self,
        song: Song,
        challenge_id: int | str,
        offset: float,
        lengths: Sequence[float],
    ) -> dict[int, Path]:
        """Ensure every snippet level exists for ``challenge_id``; return their paths.

        ``lengths`` maps level index -> target duration in seconds (level 0 is
        the shortest). Existing non-empty files are skipped untouched; missing
        or 0-byte levels are (re)generated. When nothing is missing the source
        audio is never touched (no re-download, no re-encode).

        Raises:
            ValueError: on invalid arguments (bad offset/lengths/challenge_id).
            SnippetSourceError: source audio missing, corrupt, or too short.
            SnippetGenerationError: ffmpeg/yt-dlp/verification failure.
        """
        cid = self._validate_args(challenge_id, offset, lengths)
        offset = float(offset)
        targets = {level: self._level_path(cid, level) for level in range(len(lengths))}
        needed = {
            level: path for level, path in targets.items() if not self._is_complete(path)
        }
        if not needed:
            logger.debug("challenge %s: all %d snippet levels cached", cid, len(targets))
            return targets

        max_len = max(float(v) for v in lengths)
        created: list[Path] = []
        intermediate: Path | None = None
        try:
            if song.source == "youtube":
                intermediate = self._ensure_intermediate(song, cid, offset, max_len)
                cut_source = intermediate
                cut_base = offset - _section_start(offset)
            else:
                cut_source = self._validate_local_source(song, offset, max_len)
                cut_base = offset

            self._challenge_dir(cid).mkdir(parents=True, exist_ok=True)
            for level in sorted(needed):
                self._generate_level(cut_source, cut_base, float(lengths[level]), targets[level])
                created.append(targets[level])
        except BaseException:
            self._cleanup_after_failure(created, self._challenge_dir(cid), intermediate)
            raise
        logger.info(
            "challenge %s: generated %d snippet level(s) for %r at offset %.3fs",
            cid,
            len(created),
            song.audio_ref,
            offset,
        )
        return targets

    def purge_challenge(self, challenge_id: int | str) -> None:
        """Delete a challenge's snippet cache dir and any section intermediate.

        Used by skip-song (pinned decision #5). Missing entries are ignored.
        """
        cid = self._validate_challenge_id(challenge_id)
        shutil.rmtree(self._challenge_dir(cid), ignore_errors=True)
        for candidate in self._sections_dir.glob(f"{cid}.*"):
            candidate.unlink(missing_ok=True)

    # -- source preparation -------------------------------------------------

    def _validate_local_source(self, song: Song, offset: float, max_len: float) -> Path:
        """Return the source path, raising `SnippetSourceError` if unusable."""
        source = Path(song.audio_ref)
        if not source.is_file():
            raise SnippetSourceError(
                f"Source audio file does not exist: {source} "
                f"(song {song.title!r}, source_id {song.source_id!r})"
            )
        try:
            duration = self._probe_duration(source)
        except SnippetGenerationError as exc:
            raise SnippetSourceError(
                f"Source audio file is unreadable or corrupt: {source} ({exc})"
            ) from exc
        self._check_length(source, song, duration, offset, max_len)
        return source

    def _ensure_intermediate(
        self, song: Song, cid: str, offset: float, max_len: float
    ) -> Path:
        """Return a usable cached section file for the challenge, downloading once."""
        start = _section_start(offset)
        end = offset + max_len + SECTION_PADDING_SEC
        required = (offset - start) + max_len
        if offset + max_len > song.duration_sec + DURATION_TOLERANCE_SEC:
            raise SnippetSourceError(
                f"Source too short: {song.audio_ref!r} is {song.duration_sec:.3f}s but "
                f"offset {offset:.3f}s + longest snippet {max_len:.3f}s does not fit"
            )

        for candidate in sorted(self._sections_dir.glob(f"{cid}.*")):
            if self._intermediate_usable(candidate, required):
                logger.debug("challenge %s: reusing cached section %s", cid, candidate)
                return candidate
            logger.warning("challenge %s: discarding unusable section %s", cid, candidate)
            candidate.unlink(missing_ok=True)

        self._sections_dir.mkdir(parents=True, exist_ok=True)
        dest_base = self._sections_dir / f".tmp-{cid}-{os.getpid()}-{secrets.token_hex(4)}"
        try:
            downloaded = self._section_downloader(
                song.audio_ref, start, end, dest_base, self._download_timeout_sec
            )
        except SnippetError:
            raise
        except Exception as exc:
            raise SnippetGenerationError(
                f"Section download failed for {song.audio_ref!r}: {exc}"
            ) from exc
        downloaded = Path(downloaded)
        try:
            if not downloaded.is_file() or downloaded.stat().st_size == 0:
                raise SnippetGenerationError(
                    f"Section download for {song.audio_ref!r} produced no file "
                    f"(expected {dest_base}.*)"
                )
            duration = self._probe_duration(downloaded)
            if duration < required - DURATION_TOLERANCE_SEC:
                raise SnippetSourceError(
                    f"Source too short: {song.audio_ref!r} yielded a {duration:.3f}s section "
                    f"but {required:.3f}s is needed from offset {offset:.3f}s"
                )
        except BaseException:
            downloaded.unlink(missing_ok=True)
            raise
        final = self._sections_dir / f"{cid}{downloaded.suffix}"
        os.replace(downloaded, final)
        logger.info(
            "challenge %s: downloaded section %.3f-%.3fs of %r -> %s",
            cid,
            start,
            end,
            song.audio_ref,
            final,
        )
        return final

    def _intermediate_usable(self, candidate: Path, required: float) -> bool:
        """True iff the cached section file is readable and long enough."""
        if not candidate.is_file() or candidate.stat().st_size == 0:
            return False
        try:
            duration = self._probe_duration(candidate)
        except SnippetGenerationError:
            return False
        return duration >= required - DURATION_TOLERANCE_SEC

    def _check_length(
        self, source: Path, song: Song, duration: float, offset: float, max_len: float
    ) -> None:
        if offset + max_len > duration + DURATION_TOLERANCE_SEC:
            raise SnippetSourceError(
                f"Source too short: {source} is {duration:.3f}s but offset {offset:.3f}s + "
                f"longest snippet {max_len:.3f}s does not fit "
                f"(song {song.title!r}, source_id {song.source_id!r})"
            )

    # -- cutting and verification -------------------------------------------

    def _generate_level(self, source: Path, base: float, length: float, target: Path) -> None:
        """Cut one level to a temp file, verify it, then atomically rename."""
        fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".tmp-", suffix=".mp3")
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            self._run(
                [
                    self._ffmpeg_or_raise(),
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-ss",
                    _format_seconds(base),
                    "-t",
                    _format_seconds(length),
                    "-i",
                    str(source),
                    "-vn",
                    "-c:a",
                    "libmp3lame",
                    str(tmp),
                ],
                what=f"ffmpeg cut of {source} at {base:.3f}s for {target.name}",
            )
            self._verify_duration(tmp, length, target)
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)

    def _verify_duration(self, path: Path, target_len: float, final: Path) -> None:
        measured = self._probe_duration(path)
        delta = abs(measured - target_len)
        if delta > DURATION_TOLERANCE_SEC:
            raise SnippetGenerationError(
                f"Generated snippet {final.name} duration {measured:.3f}s deviates "
                f"{delta * 1000:.0f}ms from target {target_len:.3f}s "
                f"(tolerance {DURATION_TOLERANCE_SEC * 1000:.0f}ms)"
            )

    def _probe_duration(self, path: Path) -> float:
        result = self._run(
            [
                self._ffprobe_or_raise(),
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            what=f"ffprobe of {path}",
        )
        try:
            return float(result.stdout.strip())
        except ValueError as exc:
            raise SnippetGenerationError(
                f"ffprobe of {path} returned no duration: {result.stdout!r}"
            ) from exc

    def _run(self, cmd: list[str], *, what: str) -> subprocess.CompletedProcess[str]:
        """Run ``cmd`` with a timeout, raising `SnippetGenerationError` on failure."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._ffmpeg_timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise SnippetGenerationError(
                f"{what} timed out after {self._ffmpeg_timeout_sec:.0f}s"
            ) from exc
        except OSError as exc:
            raise SnippetGenerationError(f"{what} could not be started: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise SnippetGenerationError(
                f"{what} failed (exit {result.returncode}): {detail[:500]}"
            )
        return result

    # -- cleanup and layout ---------------------------------------------------

    def _cleanup_after_failure(
        self, created: list[Path], challenge_dir: Path, intermediate: Path | None
    ) -> None:
        """Remove everything the failed call created; keep pre-existing files.

        A failed YouTube run also discards the section intermediate: any cut
        failure implicates it, and the next call re-downloads a fresh copy.
        """
        for path in created:
            path.unlink(missing_ok=True)
        if intermediate is not None:
            intermediate.unlink(missing_ok=True)
        # rmdir only succeeds if the failed call left the directory empty.
        with contextlib.suppress(OSError):
            challenge_dir.rmdir()

    @staticmethod
    def _is_complete(path: Path) -> bool:
        """A cached level file counts as done iff it exists and is non-empty."""
        return path.is_file() and path.stat().st_size > 0

    def _challenge_dir(self, cid: str) -> Path:
        return self._cache_dir / cid

    def _level_path(self, cid: str, level: int) -> Path:
        return self._challenge_dir(cid) / f"{level}.mp3"

    @property
    def _sections_dir(self) -> Path:
        return self._cache_dir / "sections"

    def _ffmpeg_or_raise(self) -> str:
        if self._ffmpeg is None:  # pragma: no cover - guarded at __init__
            raise SnippetError("ffmpeg is not available on PATH")
        return self._ffmpeg

    def _ffprobe_or_raise(self) -> str:
        if self._ffprobe is None:  # pragma: no cover - guarded at __init__
            raise SnippetError("ffprobe is not available on PATH")
        return self._ffprobe

    # -- argument validation ----------------------------------------------------

    @classmethod
    def _validate_args(
        cls, challenge_id: int | str, offset: float, lengths: Sequence[float]
    ) -> str:
        cid = cls._validate_challenge_id(challenge_id)
        if not lengths:
            raise ValueError("lengths must be a non-empty sequence of seconds")
        for value in lengths:
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"lengths entries must be positive finite seconds, got {value!r}")
        if not math.isfinite(offset) or offset < 0:
            raise ValueError(f"offset must be a finite number >= 0, got {offset!r}")
        return cid

    @staticmethod
    def _validate_challenge_id(challenge_id: int | str) -> str:
        cid = str(challenge_id)
        if not cid or cid in (".", "..") or "/" in cid or "\\" in cid:
            raise ValueError(f"challenge_id must be a single path segment, got {challenge_id!r}")
        return cid


def _section_start(offset: float) -> float:
    """Start of the YouTube section download: 2s before the offset, clamped at 0."""
    return max(0.0, offset - SECTION_PADDING_SEC)


def _format_seconds(value: float) -> str:
    """Format seconds for ffmpeg at full round-trip precision (repr of float)."""
    return repr(float(value))


def download_section_with_timeout(
    url: str, start_sec: float, end_sec: float, dest_base: Path, timeout_sec: float
) -> Path:
    """Download ``url``'s audio section ``[start_sec, end_sec]`` via yt-dlp.

    Runs yt-dlp on a worker thread with a wall-clock bound (same pattern as
    the catalog provider); yt-dlp's ``socket_timeout`` caps the orphaned
    thread's network waits after a timeout. The output file is
    ``<dest_base>.<ext>`` (container chosen by the selected format).

    Raises:
        SnippetGenerationError: on timeout, download failure, or missing output.
    """
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(_download_section, url, start_sec, end_sec, dest_base, timeout_sec)
    try:
        return future.result(timeout=timeout_sec)
    except concurrent.futures.TimeoutError as exc:
        future.cancel()
        raise SnippetGenerationError(
            f"Timed out after {timeout_sec:.0f}s downloading audio section from {url!r}"
        ) from exc
    except SnippetGenerationError:
        raise
    except Exception as exc:
        raise SnippetGenerationError(
            f"Failed to download audio section from {url!r}: {exc}"
        ) from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _download_section(
    url: str, start_sec: float, end_sec: float, dest_base: Path, timeout_sec: float
) -> Path:
    """Fetch one audio section with the yt-dlp Python API (network: YouTube).

    Retried a few times with a fresh extraction (hence a fresh googlevideo
    URL) per attempt: ffmpeg-based section fetches intermittently get a
    transient 403 from googlevideo that a retry clears. Partial output of
    failed attempts is removed so no stray artifacts survive.
    """
    last_error: SnippetGenerationError | None = None
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            return _download_section_once(url, start_sec, end_sec, dest_base, timeout_sec)
        except SnippetGenerationError as exc:
            last_error = exc
            logger.warning(
                "section download attempt %d/%d for %r failed: %s",
                attempt,
                _DOWNLOAD_ATTEMPTS,
                url,
                exc,
            )
            for leftover in dest_base.parent.glob(f"{dest_base.name}.*"):
                leftover.unlink(missing_ok=True)
            if attempt < _DOWNLOAD_ATTEMPTS:
                time.sleep(min(2.0 ** (attempt - 1), 4.0))
    raise SnippetGenerationError(
        f"Failed to download audio section from {url!r} after "
        f"{_DOWNLOAD_ATTEMPTS} attempts: {last_error}"
    ) from last_error


def _download_section_once(
    url: str, start_sec: float, end_sec: float, dest_base: Path, timeout_sec: float
) -> Path:
    """Single yt-dlp section-download attempt (network: YouTube)."""
    options: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": f"{dest_base}.%(ext)s",
        "download_ranges": lambda _info, _ydl: [
            {"start_time": start_sec, "end_time": end_sec}
        ],
        # Re-encode at the cut points: stream copy would start the section at
        # the nearest cluster boundary seconds BEFORE start_sec.
        "force_keyframes_at_cuts": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        # Keep stdout pristine: the harness CLI prints machine-readable JSON.
        "noprogress": True,
        "socket_timeout": min(timeout_sec, 30.0),
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([url])
    except Exception as exc:
        raise SnippetGenerationError(
            f"yt-dlp could not download section {start_sec:.3f}-{end_sec:.3f}s of {url!r}: {exc}"
        ) from exc
    matches = [
        path
        for path in dest_base.parent.glob(f"{dest_base.name}.*")
        if path.suffix != ".part" and path.is_file() and path.stat().st_size > 0
    ]
    if not matches:
        raise SnippetGenerationError(
            f"yt-dlp produced no output file for {url!r} (expected {dest_base}.*)"
        )
    return sorted(matches)[0]
