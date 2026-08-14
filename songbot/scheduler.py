"""Pure time logic: next post time, day boundaries. No I/O.

All functions are deterministic and DST-safe via ``zoneinfo``: results are
built with ``datetime.combine(local_date, post_time, tzinfo=zone)`` so the
UTC offset is always the one in effect at that local wall-clock time (never
a pinned offset). A naive ``now`` is interpreted as UTC.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from songbot.config import DEFAULT_DAILY_POST_TIME

__all__ = ["is_post_due", "next_post_datetime"]

_TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


def _zone(tz: str | ZoneInfo) -> ZoneInfo:
    """Accept an IANA timezone name or an existing ZoneInfo."""
    return tz if isinstance(tz, ZoneInfo) else ZoneInfo(tz)


def _parse_post_time(time_str: str) -> time:
    """Parse a strict HH:MM post time (00:00-23:59)."""
    match = _TIME_RE.match(time_str)
    if match is None:
        raise ValueError(f"invalid post time {time_str!r}; expected HH:MM (00:00-23:59)")
    return time(hour=int(match.group(1)), minute=int(match.group(2)))


def _aware(now: datetime) -> datetime:
    """Return ``now`` timezone-aware; naive input is interpreted as UTC."""
    return now if now.tzinfo is not None else now.replace(tzinfo=UTC)


def _local_post_datetime(day: date, post_time: time, zone: ZoneInfo) -> datetime:
    """The aware datetime of ``post_time`` on local date ``day`` in ``zone``.

    Combining a plain date with the zone (rather than adding timedeltas to an
    aware datetime) keeps the wall clock exact across DST transitions: the
    offset is resolved by zoneinfo for that local time.
    """
    return datetime.combine(day, post_time, tzinfo=zone)


def next_post_datetime(now: datetime, time_str: str, tz: str | ZoneInfo) -> datetime:
    """The next local datetime at or after ``now`` when a daily post is due.

    At-or-after convention (pinned): when ``now`` is exactly the post time,
    the result is TODAY at the post time; one second later it is tomorrow's.
    The result always carries the configured zone's tzinfo and has the exact
    post-time wall clock (e.g. 12:00:00), correct across DST transitions.

    Args:
        now: reference time (naive is treated as UTC).
        time_str: post time as strict "HH:MM".
        tz: IANA timezone name or ZoneInfo.
    """
    zone = _zone(tz)
    post_time = _parse_post_time(time_str)
    local_now = _aware(now).astimezone(zone)
    candidate = _local_post_datetime(local_now.date(), post_time, zone)
    if local_now > candidate:
        candidate = _local_post_datetime(
            local_now.date() + timedelta(days=1), post_time, zone
        )
    return candidate


def is_post_due(
    last_post_date: date | None,
    now: datetime,
    tz: str | ZoneInfo,
    time_str: str = DEFAULT_DAILY_POST_TIME,
) -> bool:
    """Whether a daily post is due right now (pinned gating semantics).

    Due iff no post is recorded (``last_post_date is None``), or
    ``last_post_date`` is before today's LOCAL date AND ``now`` has reached
    today's post time. A post recorded for today (or a skewed future date)
    is never due again, and nothing posts early after a midnight restart.

    Args:
        last_post_date: local date of the most recent post, or None.
        now: reference time (naive is treated as UTC).
        tz: IANA timezone name or ZoneInfo.
        time_str: post time as strict "HH:MM" (defaults to the configured
            default, 12:00).
    """
    if last_post_date is None:
        return True
    zone = _zone(tz)
    post_time = _parse_post_time(time_str)
    local_now = _aware(now).astimezone(zone)
    today = local_now.date()
    if last_post_date >= today:
        return False
    return local_now >= _local_post_datetime(today, post_time, zone)
