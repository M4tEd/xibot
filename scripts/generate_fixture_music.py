#!/usr/bin/env python3
"""Regenerate SongBot's synthetic fixture music library at ``data/fixture-music/``.

The library is gitignored, so a fresh clone must recreate it before the demo
harness chain or the integration tests that read the fixtures can run:

    .venv/bin/python scripts/generate_fixture_music.py

Each of the 8 songs is a distinct 30-second synthesized melody rendered with
ffmpeg's ``aevalsrc`` filter (mono, 44.1 kHz): a piecewise-constant frequency
pattern (the looped melody notes) with a per-note pluck envelope and a second
harmonic for timbre. Seven files are encoded as MP3 and one as M4A/AAC; tags
are written with mutagen (ID3 ``TIT2``/``TPE1`` for mp3, MP4 ``©nam``/``©ART``
for m4a) so they read back exactly the way ``LocalDirectoryProvider`` reads
them. ``Retro Waves - Sunset Drive.mp3`` is deliberately left UNTAGGED so the
catalog's ``Artist - Title.ext`` filename fallback is exercised.

The script is idempotent: files that already exist are skipped (pass
``--force`` to rebuild them). Requires ``ffmpeg`` and ``ffprobe`` on PATH plus
mutagen (a project dependency — run it with the venv python).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import mutagen

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "fixture-music"

DURATION_SEC = 30.0
SAMPLE_RATE = 44100
DURATION_TOLERANCE_SEC = 0.25
FFMPEG_TIMEOUT_SEC = 120
FFPROBE_TIMEOUT_SEC = 30


class FixtureGenerationError(RuntimeError):
    """A fixture file could not be rendered, verified, or tagged."""


@dataclass(frozen=True)
class FixtureSong:
    """One fixture song: identity, container, tag policy, and melody shape."""

    filename: str
    artist: str
    title: str
    container: str  # "mp3" or "m4a"
    tagged: bool  # False => no title/artist tags (filename-fallback fixture)
    notes: tuple[float, ...]  # melody frequencies in Hz, looped for DURATION_SEC
    note_dur: float  # seconds per note
    harmonic_amp: float  # second-harmonic level relative to the fundamental

    @property
    def stem(self) -> str:
        return Path(self.filename).stem


# The 8-song library. Tags deliberately match the "Artist - Title" filename
# stems; melodies differ in scale, register, rhythm, and timbre so every file
# has distinct audio content.
SONGS: tuple[FixtureSong, ...] = (
    # C-major pentatonic sunrise ascent.
    FixtureSong(
        "Aurora Fields - Solar Bloom.mp3", "Aurora Fields", "Solar Bloom", "mp3", True,
        (261.626, 329.628, 391.995, 440.000, 523.251, 440.000, 391.995, 329.628), 0.5, 0.30,
    ),
    # Flowing A-minor descent.
    FixtureSong(
        "Echo Cartography - Glass Rivers.mp3", "Echo Cartography", "Glass Rivers", "mp3", True,
        (440.000, 391.995, 329.628, 293.665, 261.626, 293.665, 329.628, 261.626), 0.5, 0.25,
    ),
    # Slow low E-miner freight groove.
    FixtureSong(
        "Iron Meadow - Midnight Freight.mp3", "Iron Meadow", "Midnight Freight", "mp3", True,
        (164.814, 195.998, 246.942, 220.000, 195.998, 164.814), 1.0, 0.35,
    ),
    # Fast synthwave minor arpeggio.
    FixtureSong(
        "Midnight Circuit - Neon Skyline.mp3", "Midnight Circuit", "Neon Skyline", "mp3", True,
        (220.000, 261.626, 329.628, 440.000, 329.628, 261.626), 0.25, 0.40,
    ),
    # Whole-tone climb.
    FixtureSong(
        "Quantum Drift - Digital Horizon.mp3", "Quantum Drift", "Digital Horizon", "mp3", True,
        (261.626, 293.665, 329.628, 369.994, 415.305, 466.164), 0.4, 0.20,
    ),
    # Retro fifths groove — UNTAGGED on purpose (filename-fallback fixture).
    FixtureSong(
        "Retro Waves - Sunset Drive.mp3", "Retro Waves", "Sunset Drive", "mp3", False,
        (146.832, 220.000, 293.665, 220.000, 329.628, 220.000), 0.5, 0.30,
    ),
    # Melancholy slow sway.
    FixtureSong(
        "The Analog Ghosts - Velvet Static.mp3", "The Analog Ghosts", "Velvet Static", "mp3",
        True, (246.942, 220.000, 184.997, 164.814, 184.997, 195.998), 0.75, 0.22,
    ),
    # 3/4 waltz, encoded as M4A/AAC to cover the MP4 tag path.
    FixtureSong(
        "The Cartographers - Paper Moons.m4a", "The Cartographers", "Paper Moons", "m4a", True,
        (195.998, 261.626, 329.628, 293.665, 261.626, 220.000), 0.6, 0.28,
    ),
)


def melody_expression(song: FixtureSong) -> str:
    """Build the ffmpeg ``aevalsrc`` expression for a song's looped melody.

    The frequency is piecewise-constant over the looped note pattern
    (``between`` on ``mod(t, loop)``); a fast-attack/decay envelope on the
    note-local time softens the transitions, and a quiet second harmonic gives
    each song a bit of timbre. Commas are escaped (``\\,``) for the ffmpeg
    filtergraph parser.
    """
    loop = len(song.notes) * song.note_dur
    freq = "+".join(
        f"{note:.3f}*between(mod(t\\,{loop:.4f})\\,"
        f"{i * song.note_dur:.4f}\\,{(i + 1) * song.note_dur:.4f})"
        for i, note in enumerate(song.notes)
    )
    local = f"mod(t\\,{song.note_dur:.4f})"
    envelope = f"(1-exp(-60*{local}))*exp(-2.5*{local})"
    return (
        f"0.6*{envelope}*(sin(2*PI*({freq})*t)"
        f"+{song.harmonic_amp:.2f}*sin(2*PI*({freq})*2*t))"
    )


def render_song(song: FixtureSong, dest: Path, ffmpeg: str) -> None:
    """Render one song to ``dest`` via ffmpeg aevalsrc (atomic via .part file)."""
    expr = melody_expression(song)
    input_arg = f"aevalsrc={expr}:d={DURATION_SEC}:s={SAMPLE_RATE}"
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", input_arg]
    if song.container == "mp3":
        cmd += ["-c:a", "libmp3lame", "-q:a", "4"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "128k"]
    tmp = dest.with_name(f"{dest.stem}.part{dest.suffix}")
    cmd.append(str(tmp))
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_SEC, check=False,
        )
        if result.returncode != 0:
            raise FixtureGenerationError(
                f"ffmpeg failed for {song.filename}: {result.stderr.strip()}"
            )
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def probe_duration(path: Path, ffprobe: str) -> float:
    """Return the ffprobe format duration of ``path`` in seconds."""
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True, text=True, timeout=FFPROBE_TIMEOUT_SEC, check=False,
    )
    if result.returncode != 0:
        raise FixtureGenerationError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    return float(result.stdout.strip())


def tag_file(path: Path, title: str, artist: str) -> None:
    """Write title/artist tags with mutagen (ID3 for mp3, MP4 atoms for m4a)."""
    audio = mutagen.File(path, easy=True)
    if audio is None:
        raise FixtureGenerationError(f"mutagen could not parse {path}")
    if audio.tags is None:
        audio.add_tags()
    audio["title"] = title
    audio["artist"] = artist
    audio.save()


def generate_fixtures(output_dir: Path, *, force: bool = False) -> list[str]:
    """Generate the full fixture library into ``output_dir``.

    Existing files are skipped unless ``force`` is set. Returns one status line
    per song. Raises FixtureGenerationError on any render/verify/tag failure.
    """
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise FixtureGenerationError(
            "ffmpeg and ffprobe must be on PATH (macOS: brew install ffmpeg)"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[str] = []
    for song in SONGS:
        dest = output_dir / song.filename
        if dest.exists() and not force:
            results.append(f"skipped (exists): {song.filename}")
            continue
        render_song(song, dest, ffmpeg)
        duration = probe_duration(dest, ffprobe)
        if abs(duration - DURATION_SEC) > DURATION_TOLERANCE_SEC:
            dest.unlink(missing_ok=True)
            raise FixtureGenerationError(
                f"{song.filename}: duration {duration:.3f}s deviates from "
                f"{DURATION_SEC}s by more than {DURATION_TOLERANCE_SEC}s"
            )
        if song.tagged:
            tag_file(dest, song.title, song.artist)
            results.append(f"generated (tagged): {song.filename} [{duration:.3f}s]")
        else:
            results.append(f"generated (untagged, filename fallback): {song.filename} "
                           f"[{duration:.3f}s]")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write the library (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate files even if they already exist.",
    )
    args = parser.parse_args(argv)
    try:
        lines = generate_fixtures(args.output_dir, force=args.force)
    except FixtureGenerationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for line in lines:
        print(line)
    print(f"fixture library ready at {args.output_dir} ({len(SONGS)} songs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
