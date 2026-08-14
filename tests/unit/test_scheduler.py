"""Unit tests for songbot.scheduler: pure post-time logic, no I/O.

Covers contract assertions VAL-DAILY-010 (boundary cases), VAL-DAILY-011
(DST transitions) and VAL-DAILY-012 (is_post_due gating), all with
America/Halifax and the default 12:00 post time.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from songbot.scheduler import is_post_due, next_post_datetime

HALIFAX = ZoneInfo("America/Halifax")
POST_TIME = "12:00"


def _halifax(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=HALIFAX)


class TestNextPostDatetimeBoundaries:
    """VAL-DAILY-010: next 12:00 local, at-or-after exact-boundary convention."""

    def test_one_second_before_post_time_is_today(self) -> None:
        result = next_post_datetime(_halifax(2025, 6, 10, 11, 59, 59), POST_TIME, "America/Halifax")
        assert result == _halifax(2025, 6, 10, 12, 0, 0)

    def test_exactly_at_post_time_is_today(self) -> None:
        """The pinned at-or-after convention: now == post time -> today, not tomorrow."""
        result = next_post_datetime(_halifax(2025, 6, 10, 12, 0, 0), POST_TIME, "America/Halifax")
        assert result == _halifax(2025, 6, 10, 12, 0, 0)

    def test_one_second_after_post_time_is_tomorrow(self) -> None:
        result = next_post_datetime(_halifax(2025, 6, 10, 12, 0, 1), POST_TIME, "America/Halifax")
        assert result == _halifax(2025, 6, 11, 12, 0, 0)

    def test_midnight_is_today(self) -> None:
        result = next_post_datetime(_halifax(2025, 6, 10, 0, 0, 0), POST_TIME, "America/Halifax")
        assert result == _halifax(2025, 6, 10, 12, 0, 0)

    def test_last_second_of_day_rolls_to_tomorrow(self) -> None:
        result = next_post_datetime(_halifax(2025, 6, 10, 23, 59, 59), POST_TIME, "America/Halifax")
        assert result == _halifax(2025, 6, 11, 12, 0, 0)

    def test_result_carries_configured_zone_and_exact_wall_clock(self) -> None:
        result = next_post_datetime(_halifax(2025, 6, 10, 15, 42, 17), POST_TIME, "America/Halifax")
        assert result.tzinfo == HALIFAX
        assert (result.hour, result.minute, result.second, result.microsecond) == (12, 0, 0, 0)

    def test_aware_utc_input_is_converted_to_local(self) -> None:
        """15:00 UTC == 12:00 ADT exactly -> today's 12:00 Halifax (at-or-after)."""
        now = datetime(2025, 6, 10, 15, 0, 0, tzinfo=UTC)
        result = next_post_datetime(now, POST_TIME, "America/Halifax")
        assert result == _halifax(2025, 6, 10, 12, 0, 0)

    def test_naive_input_is_treated_as_utc(self) -> None:
        now = datetime(2025, 6, 10, 15, 0, 0)  # naive == 12:00 ADT
        result = next_post_datetime(now, POST_TIME, "America/Halifax")
        assert result == _halifax(2025, 6, 10, 12, 0, 0)

    def test_zoneinfo_instance_accepted(self) -> None:
        result = next_post_datetime(_halifax(2025, 6, 10, 13, 0, 0), POST_TIME, HALIFAX)
        assert result == _halifax(2025, 6, 11, 12, 0, 0)

    def test_month_boundary_rolls(self) -> None:
        result = next_post_datetime(_halifax(2025, 1, 31, 13, 0, 0), POST_TIME, "America/Halifax")
        assert result == _halifax(2025, 2, 1, 12, 0, 0)

    def test_non_default_post_time(self) -> None:
        result = next_post_datetime(_halifax(2025, 6, 10, 8, 30, 0), "09:45", "America/Halifax")
        assert result == _halifax(2025, 6, 10, 9, 45, 0)


class TestNextPostDatetimeDst:
    """VAL-DAILY-011: wall-clock 12:00 local with the correct post-transition offset."""

    def test_spring_forward_keeps_wall_clock_with_adt_offset(self) -> None:
        """2025-03-09 is the spring-forward day; 12:00 that day is ADT (UTC-3)."""
        now = _halifax(2025, 3, 8, 13, 0, 0)  # AST, UTC-4
        assert now.utcoffset() == timedelta(hours=-4)
        result = next_post_datetime(now, POST_TIME, "America/Halifax")
        assert result == _halifax(2025, 3, 9, 12, 0, 0)
        assert result.utcoffset() == timedelta(hours=-3)  # ADT, i.e. 15:00 UTC — not 16:00
        assert result.astimezone(UTC) == datetime(2025, 3, 9, 15, 0, 0, tzinfo=UTC)

    def test_fall_back_keeps_wall_clock_with_ast_offset(self) -> None:
        """2025-11-02 is the fall-back day; 12:00 that day is AST (UTC-4).

        NOTE: VAL-DAILY-011's parenthetical says "i.e., 17:00 UTC" but 12:00
        at UTC-4 is 16:00 UTC — the contract's binding condition (wall-clock
        12:00 local with the correct post-transition offset, utcoffset == -4h)
        is what is asserted here; 17:00 UTC would contradict it.
        """
        now = _halifax(2025, 11, 1, 13, 0, 0)  # ADT, UTC-3
        assert now.utcoffset() == timedelta(hours=-3)
        result = next_post_datetime(now, POST_TIME, "America/Halifax")
        assert result == _halifax(2025, 11, 2, 12, 0, 0)
        assert result.utcoffset() == timedelta(hours=-4)  # AST, i.e. 16:00 UTC
        assert result.astimezone(UTC) == datetime(2025, 11, 2, 16, 0, 0, tzinfo=UTC)


class TestIsPostDue:
    """VAL-DAILY-012: due iff last_post_date < today AND now >= today's post time."""

    def test_never_posted_is_due(self) -> None:
        assert is_post_due(None, _halifax(2025, 6, 10, 0, 1, 0), "America/Halifax") is True
        assert is_post_due(None, _halifax(2025, 6, 10, 23, 59, 59), "America/Halifax") is True

    def test_posted_today_is_never_due(self) -> None:
        today = date(2025, 6, 10)
        assert is_post_due(today, _halifax(2025, 6, 10, 12, 5, 0), "America/Halifax") is False
        assert is_post_due(today, _halifax(2025, 6, 10, 23, 59, 0), "America/Halifax") is False

    def test_posted_yesterday_not_due_before_post_time(self) -> None:
        """No early posts after a midnight restart."""
        yesterday = date(2025, 6, 9)
        assert is_post_due(yesterday, _halifax(2025, 6, 10, 11, 59, 0), "America/Halifax") is False
        assert is_post_due(yesterday, _halifax(2025, 6, 10, 0, 30, 0), "America/Halifax") is False

    def test_posted_yesterday_due_at_exact_post_time(self) -> None:
        yesterday = date(2025, 6, 9)
        assert is_post_due(yesterday, _halifax(2025, 6, 10, 12, 0, 0), "America/Halifax") is True

    def test_posted_yesterday_due_after_post_time(self) -> None:
        yesterday = date(2025, 6, 9)
        assert is_post_due(yesterday, _halifax(2025, 6, 10, 12, 1, 0), "America/Halifax") is True

    def test_last_post_in_the_future_is_not_due(self) -> None:
        tomorrow = date(2025, 6, 11)
        assert is_post_due(tomorrow, _halifax(2025, 6, 10, 18, 0, 0), "America/Halifax") is False

    def test_today_is_local_not_utc(self) -> None:
        """01:30 UTC on the 10th is still the 9th in Halifax: posting for the 9th
        at 22:30 local is long past, so a last post on the 8th IS due."""
        now = datetime(2025, 6, 10, 1, 30, 0, tzinfo=UTC)  # 2025-06-09 22:30 ADT
        assert is_post_due(date(2025, 6, 9), now, "America/Halifax") is False  # posted "today"
        assert is_post_due(date(2025, 6, 8), now, "America/Halifax") is True
