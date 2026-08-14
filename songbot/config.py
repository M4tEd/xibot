"""Configuration loading and validation for SongBot.

`load_settings()` reads `.env` first, then applies `os.environ` on top —
environment variables OVERRIDE `.env` values, including empty-string
overrides (pinned design decision #10). Validation collects EVERY problem
and raises a single `ConfigError` listing them all (never fail-fast).

An empty ``YOUTUBE_PLAYLIST_URL`` or ``LOCAL_MUSIC_DIR`` is valid and simply
disables that catalog provider (exposed as ``None``).
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import dotenv_values

__all__ = ["ConfigError", "Settings", "load_settings"]

DEFAULT_ENV_FILE = ".env"

DEFAULT_DAILY_POST_TIME = "12:00"
DEFAULT_TIMEZONE = "America/Halifax"
DEFAULT_MAX_GUESSES_PER_DAY = 6
DEFAULT_SNIPPET_LENGTHS = (1.0, 2.0, 4.0, 8.0, 16.0)
DEFAULT_SNIPPET_POINTS = (100, 75, 50, 30, 15)
DEFAULT_BOTH_CORRECT_MULTIPLIER = 1.5
DEFAULT_DATABASE_PATH = Path("./data/songbot.db")
DEFAULT_SNIPPET_CACHE_DIR = Path("./data/snippets")
DEFAULT_HEALTH_PORT = 3108
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_DISCORD_API_BASE = "https://discord.com/api/v10"

_VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


class ConfigError(Exception):
    """Raised when configuration is invalid. The message lists ALL problems."""


@dataclass(frozen=True)
class Settings:
    """Validated, immutable SongBot configuration.

    ``youtube_playlist_url`` / ``local_music_dir`` are ``None`` when the
    corresponding catalog provider is disabled (empty value in config).
    """

    discord_token: str
    guild_id: str
    channel_id: str
    youtube_playlist_url: str | None
    local_music_dir: Path | None
    daily_post_time: str  # strict "HH:MM", 00:00-23:59
    timezone: str  # IANA name, validated via zoneinfo
    max_guesses_per_day: int
    snippet_lengths: tuple[float, ...]  # seconds, parallel to snippet_points
    snippet_points: tuple[int, ...]
    both_correct_multiplier: float
    database_path: Path
    snippet_cache_dir: Path
    health_port: int
    log_level: str
    discord_api_base: str

    @property
    def tz(self) -> ZoneInfo:
        """The configured timezone as a `ZoneInfo` (validity guaranteed at load)."""
        return ZoneInfo(self.timezone)


def load_settings(
    env_file: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Load and validate settings from `.env` and the environment.

    Values from ``environ`` (default: ``os.environ``) override values from
    ``env_file`` (default: ``.env`` in the current working directory), so
    per-invocation overrides like ``YOUTUBE_PLAYLIST_URL=""`` work.

    Raises:
        ConfigError: listing every detected problem at once.
    """
    raw: dict[str, str] = {}
    path = Path(env_file) if env_file is not None else Path(DEFAULT_ENV_FILE)
    if path.is_file():
        for key, value in dotenv_values(path).items():
            raw[key] = value if value is not None else ""
    for key, value in (os.environ if environ is None else environ).items():
        raw[key] = value

    problems: list[str] = []

    def required(key: str) -> str | None:
        value = raw.get(key, "").strip()
        if not value:
            problems.append(f"{key} is required but missing or empty")
            return None
        return value

    def optional(key: str, default: str) -> str:
        # Defaults apply only when the key is ABSENT; an explicitly set value
        # (even an empty string) is used as-is so it gets validated.
        if key not in raw:
            return default
        return raw[key].strip()

    discord_token = required("DISCORD_BOT_TOKEN")
    guild_id = required("DISCORD_GUILD_ID")
    channel_id = required("DISCORD_CHANNEL_ID")

    youtube_playlist_url = optional("YOUTUBE_PLAYLIST_URL", "") or None

    local_music_dir_raw = optional("LOCAL_MUSIC_DIR", "")
    local_music_dir = Path(local_music_dir_raw) if local_music_dir_raw else None
    if local_music_dir is not None:
        _check_dir_creatable(local_music_dir, "LOCAL_MUSIC_DIR", problems)

    daily_post_time = optional("DAILY_POST_TIME", DEFAULT_DAILY_POST_TIME)
    if not _TIME_RE.match(daily_post_time):
        problems.append(
            f"DAILY_POST_TIME '{daily_post_time}' is not a valid HH:MM time (00:00-23:59)"
        )

    timezone = optional("TIMEZONE", DEFAULT_TIMEZONE)
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        problems.append(f"TIMEZONE '{timezone}' is not a valid IANA timezone")

    max_guesses_per_day = _parse_int(
        optional("MAX_GUESSES_PER_DAY", str(DEFAULT_MAX_GUESSES_PER_DAY)),
        "MAX_GUESSES_PER_DAY",
        problems,
        minimum=1,
    )
    snippet_lengths = _parse_number_list(
        optional(
            "SNIPPET_LENGTHS",
            ",".join(_fmt_num(v) for v in DEFAULT_SNIPPET_LENGTHS),
        ),
        "SNIPPET_LENGTHS",
        problems,
        integer=False,
    )
    snippet_points = _parse_number_list(
        optional("SNIPPET_POINTS", ",".join(_fmt_num(v) for v in DEFAULT_SNIPPET_POINTS)),
        "SNIPPET_POINTS",
        problems,
        integer=True,
    )
    if (
        snippet_lengths is not None
        and snippet_points is not None
        and len(snippet_lengths) != len(snippet_points)
    ):
        problems.append(
            "SNIPPET_LENGTHS and SNIPPET_POINTS must have the same number of entries "
            f"(got {len(snippet_lengths)} lengths vs {len(snippet_points)} points)"
        )

    both_correct_multiplier = _parse_float(
        optional("BOTH_CORRECT_MULTIPLIER", str(DEFAULT_BOTH_CORRECT_MULTIPLIER)),
        "BOTH_CORRECT_MULTIPLIER",
        problems,
        minimum_exclusive=0.0,
    )

    database_path = Path(optional("DATABASE_PATH", str(DEFAULT_DATABASE_PATH)))
    if database_path.exists() and database_path.is_dir():
        problems.append(f"DATABASE_PATH '{database_path}' exists but is a directory")
    else:
        _check_dir_creatable(
            database_path.parent,
            "DATABASE_PATH parent directory",
            problems,
            must_be_writable=True,
        )

    snippet_cache_dir = Path(optional("SNIPPET_CACHE_DIR", str(DEFAULT_SNIPPET_CACHE_DIR)))
    _check_dir_creatable(snippet_cache_dir, "SNIPPET_CACHE_DIR", problems, must_be_writable=True)

    health_port = _parse_int(
        optional("HEALTH_PORT", str(DEFAULT_HEALTH_PORT)),
        "HEALTH_PORT",
        problems,
        minimum=1,
        maximum=65535,
    )

    log_level = optional("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    if log_level not in _VALID_LOG_LEVELS:
        problems.append(
            f"LOG_LEVEL '{log_level}' is invalid; expected one of {', '.join(_VALID_LOG_LEVELS)}"
        )

    discord_api_base = optional("DISCORD_API_BASE", DEFAULT_DISCORD_API_BASE)

    if problems:
        details = "\n".join(f"  - {problem}" for problem in problems)
        raise ConfigError(f"Invalid configuration ({len(problems)} problem(s)):\n{details}")

    # Everything below is provably non-None: any parse failure above appends
    # to `problems`, and a non-empty `problems` raises before this point.
    return Settings(
        discord_token=_assert_present(discord_token),
        guild_id=_assert_present(guild_id),
        channel_id=_assert_present(channel_id),
        youtube_playlist_url=youtube_playlist_url,
        local_music_dir=local_music_dir,
        daily_post_time=daily_post_time,
        timezone=timezone,
        max_guesses_per_day=_assert_present(max_guesses_per_day),
        snippet_lengths=_assert_present(snippet_lengths),
        snippet_points=_assert_present(snippet_points),
        both_correct_multiplier=_assert_present(both_correct_multiplier),
        database_path=database_path,
        snippet_cache_dir=snippet_cache_dir,
        health_port=_assert_present(health_port),
        log_level=log_level,
        discord_api_base=discord_api_base,
    )


def _assert_present[T](value: T | None) -> T:
    # Guaranteed non-None by the problems check in load_settings().
    assert value is not None
    return value


def _fmt_num(value: float) -> str:
    """Format a numeric default without a trailing '.0' when integral."""
    return str(int(value)) if float(value).is_integer() else str(value)


def _parse_int(
    raw: str,
    field: str,
    problems: list[str],
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    try:
        value = int(raw)
    except ValueError:
        problems.append(f"{field} '{raw}' is not an integer")
        return None
    if minimum is not None and value < minimum:
        problems.append(f"{field} must be >= {minimum}, got {value}")
        return None
    if maximum is not None and value > maximum:
        problems.append(f"{field} must be <= {maximum}, got {value}")
        return None
    return value


def _parse_float(
    raw: str,
    field: str,
    problems: list[str],
    *,
    minimum_exclusive: float | None = None,
) -> float | None:
    try:
        value = float(raw)
    except ValueError:
        problems.append(f"{field} '{raw}' is not a number")
        return None
    if minimum_exclusive is not None and value <= minimum_exclusive:
        problems.append(f"{field} must be > {minimum_exclusive}, got {value}")
        return None
    return value


@overload
def _parse_number_list(
    raw: str, field: str, problems: list[str], *, integer: Literal[True]
) -> tuple[int, ...] | None: ...


@overload
def _parse_number_list(
    raw: str, field: str, problems: list[str], *, integer: Literal[False]
) -> tuple[float, ...] | None: ...


def _parse_number_list(
    raw: str,
    field: str,
    problems: list[str],
    *,
    integer: bool,
) -> tuple[float, ...] | tuple[int, ...] | None:
    """Parse a comma-separated list of positive numbers for `field`."""
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        problems.append(f"{field} must be a non-empty comma-separated list")
        return None
    values: list[float] = []
    ok = True
    kind = "integer" if integer else "number"
    for part in parts:
        try:
            value = float(int(part)) if integer else float(part)
        except ValueError:
            problems.append(f"{field} entry '{part}' is not a valid {kind}")
            ok = False
            continue
        if not math.isfinite(value):
            problems.append(f"{field} entry '{part}' is not a finite {kind}")
            ok = False
            continue
        if value <= 0:
            problems.append(f"{field} entries must be positive, got {part}")
            ok = False
            continue
        values.append(value)
    if not ok:
        return None
    if integer:
        return tuple(int(v) for v in values)
    return tuple(values)


def _check_dir_creatable(
    path: Path, field: str, problems: list[str], *, must_be_writable: bool = False
) -> None:
    """Append a problem if `path` cannot serve as a usable directory.

    A missing directory must be creatable (nearest existing ancestor is a
    writable directory). An existing path must be a directory, and when
    ``must_be_writable`` is true — for runtime directories the bot writes
    into (snippet cache dir, database parent dir) — it must also be
    writable, so a read-only directory fails at load instead of at runtime.

    Does NOT create anything (creation happens on first use by callers).
    """
    if path.exists():
        if not path.is_dir():
            problems.append(f"{field} '{path}' exists but is not a directory")
        elif must_be_writable and not os.access(path, os.W_OK):
            problems.append(f"{field} '{path}' exists but is not writable")
        return
    ancestor = path
    while not ancestor.exists():
        parent = ancestor.parent
        if parent == ancestor:  # filesystem root reached
            break
        ancestor = parent
    if not ancestor.is_dir():
        problems.append(
            f"{field} '{path}' cannot be created: '{ancestor}' exists but is not a directory"
        )
    elif not os.access(ancestor, os.W_OK):
        problems.append(
            f"{field} '{path}' cannot be created: '{ancestor}' is not writable"
        )
