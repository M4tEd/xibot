"""Unit tests for refresh_catalog: combined upsert, removal rules, failure isolation.

All tests use in-memory stub providers and an in-memory database — no
network, no fixture files, fully deterministic. Pinned design decisions
covered: #10 (provider enablement), #12 (upsert semantics, referenced-row
retention, per-source failure isolation).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from songbot.catalog import CatalogProvider, CatalogSource, Song
from songbot.catalog.refresh import RefreshResult, SourceRefresh, refresh_catalog
from songbot.catalog.youtube import YouTubeCatalogError
from songbot.config import Settings
from songbot.db import Database, SongRow


def _settings(
    tmp_path: Path,
    *,
    local_music_dir: Path | None = None,
    youtube_playlist_url: str | None = None,
) -> Settings:
    """A valid Settings with both catalog providers disabled unless specified."""
    return Settings(
        discord_token="test-token",
        guild_id="guild-1",
        channel_id="channel-1",
        youtube_playlist_url=youtube_playlist_url,
        local_music_dir=local_music_dir,
        daily_post_time="12:00",
        timezone="America/Halifax",
        max_guesses_per_day=6,
        snippet_lengths=(1.0, 2.0, 4.0, 8.0, 16.0),
        snippet_points=(100, 75, 50, 30, 15),
        both_correct_multiplier=1.5,
        database_path=tmp_path / "songbot.db",
        snippet_cache_dir=tmp_path / "snippets",
        health_port=3108,
        log_level="INFO",
        discord_api_base="https://discord.com/api/v10",
    )


def _song(
    source_id: str,
    *,
    source: CatalogSource = "local",
    title: str = "A Title",
    artist: str | None = "An Artist",
    duration_sec: float = 200.0,
    audio_ref: str | None = None,
    raw_title: str | None = None,
) -> Song:
    return Song(
        source=source,
        source_id=source_id,
        title=title,
        artist=artist,
        duration_sec=duration_sec,
        audio_ref=audio_ref if audio_ref is not None else f"/music/{source_id}",
        raw_title=raw_title if raw_title is not None else f"raw {source_id}",
    )


def _youtube_song(source_id: str, **kwargs: object) -> Song:
    kwargs.setdefault("audio_ref", f"https://www.youtube.com/watch?v={source_id}")
    return _song(source_id, source="youtube", **kwargs)  # type: ignore[arg-type]


class StubProvider:
    """In-memory CatalogProvider stub; mutate `songs`/`error` between fetches."""

    def __init__(
        self, songs: list[Song] | None = None, error: Exception | None = None
    ) -> None:
        self.songs = list(songs) if songs is not None else []
        self.error = error
        self.fetch_calls = 0

    def fetch(self) -> list[Song]:
        self.fetch_calls += 1
        if self.error is not None:
            raise self.error
        return list(self.songs)


@pytest.fixture
def db() -> Iterator[Database]:
    database = Database.open(":memory:")
    yield database
    database.close()


def _songs_table(db: Database) -> list[SongRow]:
    rows = db.query("SELECT * FROM songs ORDER BY source, source_id")
    return [SongRow.from_row(row) for row in rows]


def _insert_song_row(db: Database, source: str, source_id: str) -> None:
    db.execute(
        "INSERT INTO songs"
        " (source, source_id, title, artist, duration_sec, audio_ref, raw_title, created_at)"
        " VALUES (?, ?, 'T', 'A', 100.0, 'ref', 'raw', ?)",
        (source, source_id, datetime.now(UTC).isoformat()),
    )


def _insert_challenge(db: Database, song_id: int, date: str = "2026-08-13") -> None:
    db.execute(
        "INSERT INTO challenges"
        " (guild_id, channel_id, song_id, date, snippet_offset_sec, status, created_at)"
        " VALUES (?, ?, ?, ?, ?, 'active', ?)",
        ("guild-1", "channel-1", song_id, date, 0.0, datetime.now(UTC).isoformat()),
    )


class TestCombinedUpsert:
    """refresh_catalog upserts every enabled provider's songs (pinned #12)."""

    def test_stub_satisfies_provider_protocol(self) -> None:
        assert isinstance(StubProvider(), CatalogProvider)

    def test_both_providers_populate_songs_table(
        self, db: Database, tmp_path: Path
    ) -> None:
        local = StubProvider([_song("a.mp3"), _song("b.mp3", title="B Title")])
        youtube = StubProvider(
            [
                _youtube_song("vid1", title="YT One"),
                _youtube_song("vid2", title="YT Two", artist=None),
            ]
        )
        result = refresh_catalog(
            db, _settings(tmp_path), providers={"local": local, "youtube": youtube}
        )
        print(f"result: {result}")
        assert result.ok
        assert [s.source for s in result.sources] == ["local", "youtube"]
        assert result.by_source("local") == SourceRefresh(source="local", added=2)
        assert result.by_source("youtube") == SourceRefresh(source="youtube", added=2)

        rows = _songs_table(db)
        assert [(r.source, r.source_id) for r in rows] == [
            ("local", "a.mp3"),
            ("local", "b.mp3"),
            ("youtube", "vid1"),
            ("youtube", "vid2"),
        ]
        by_pair = {(r.source, r.source_id): r for r in rows}
        assert by_pair[("local", "b.mp3")].title == "B Title"
        assert by_pair[("youtube", "vid2")].artist is None
        assert by_pair[("youtube", "vid1")].audio_ref.endswith("watch?v=vid1")
        for row in rows:
            assert row.created_at

    def test_by_source_unknown_raises_key_error(
        self, db: Database, tmp_path: Path
    ) -> None:
        result = refresh_catalog(db, _settings(tmp_path), providers={"local": StubProvider()})
        with pytest.raises(KeyError, match="youtube"):
            result.by_source("youtube")


class TestIdempotency:
    """Re-running refresh inserts nothing: stable ids, no duplicates (VAL-CATALOG-012)."""

    def test_second_run_keeps_ids_and_counts(self, db: Database, tmp_path: Path) -> None:
        provider = StubProvider([_song("a.mp3"), _song("b.mp3")])
        settings = _settings(tmp_path)
        first = refresh_catalog(db, settings, providers={"local": provider})
        before = _songs_table(db)

        second = refresh_catalog(db, settings, providers={"local": provider})
        after = _songs_table(db)

        assert first.by_source("local") == SourceRefresh(source="local", added=2)
        assert second.by_source("local") == SourceRefresh(source="local", updated=2)
        assert [(r.id, r.source, r.source_id) for r in after] == [
            (r.id, r.source, r.source_id) for r in before
        ]
        assert [r.created_at for r in after] == [r.created_at for r in before]
        assert max(r.id for r in after) == max(r.id for r in before)


class TestUpdateInPlace:
    """Changed source metadata updates the existing row (VAL-CATALOG-013 shape)."""

    def test_metadata_update_preserves_row_id(self, db: Database, tmp_path: Path) -> None:
        provider = StubProvider([_song("a.mp3", title="Old Title", duration_sec=100.0)])
        settings = _settings(tmp_path)
        refresh_catalog(db, settings, providers={"local": provider})
        before = _songs_table(db)[0]
        assert before.title == "Old Title"

        provider.songs = [
            _song("a.mp3", title="New Title", duration_sec=123.0, raw_title="a raw")
        ]
        result = refresh_catalog(db, settings, providers={"local": provider})

        after = _songs_table(db)
        assert len(after) == 1
        row = after[0]
        print(f"before: {before}\nafter:  {row}")
        assert row.id == before.id, "upsert must update in place, not delete+insert"
        assert row.created_at == before.created_at
        assert row.title == "New Title"
        assert row.duration_sec == 123.0
        assert row.raw_title == "a raw"
        assert result.by_source("local") == SourceRefresh(source="local", updated=1)


class TestRemoval:
    """Vanished songs are deleted only when unreferenced by challenges (pinned #12)."""

    def test_vanished_unreferenced_song_is_deleted(
        self, db: Database, tmp_path: Path
    ) -> None:
        provider = StubProvider([_song("a.mp3"), _song("b.mp3")])
        settings = _settings(tmp_path)
        refresh_catalog(db, settings, providers={"local": provider})

        provider.songs = [_song("a.mp3")]
        result = refresh_catalog(db, settings, providers={"local": provider})

        assert result.by_source("local") == SourceRefresh(source="local", updated=1, removed=1)
        assert [(r.source, r.source_id) for r in _songs_table(db)] == [("local", "a.mp3")]

    def test_vanished_referenced_song_is_retained(
        self, db: Database, tmp_path: Path
    ) -> None:
        provider = StubProvider([_song("a.mp3"), _song("b.mp3")])
        settings = _settings(tmp_path)
        refresh_catalog(db, settings, providers={"local": provider})
        b_id = next(r.id for r in _songs_table(db) if r.source_id == "b.mp3")
        _insert_challenge(db, b_id)

        provider.songs = [_song("a.mp3")]
        result = refresh_catalog(db, settings, providers={"local": provider})

        src = result.by_source("local")
        print(f"referenced-removal result: {src}")
        assert src.removed == 0
        assert src.retained == 1
        assert {(r.source, r.source_id) for r in _songs_table(db)} == {
            ("local", "a.mp3"),
            ("local", "b.mp3"),
        }

    def test_empty_fetch_removes_all_unreferenced_rows(
        self, db: Database, tmp_path: Path
    ) -> None:
        provider = StubProvider([_song("a.mp3"), _song("b.mp3")])
        settings = _settings(tmp_path)
        refresh_catalog(db, settings, providers={"local": provider})

        provider.songs = []
        result = refresh_catalog(db, settings, providers={"local": provider})

        assert result.by_source("local") == SourceRefresh(source="local", removed=2)
        assert _songs_table(db) == []

    def test_rows_from_other_sources_are_never_touched(
        self, db: Database, tmp_path: Path
    ) -> None:
        # A source no enabled provider reports on (e.g. a future spotify source)
        # is out of scope for both upsert and removal.
        _insert_song_row(db, "spotify", "xyz")
        result = refresh_catalog(
            db, _settings(tmp_path), providers={"local": StubProvider([_song("a.mp3")])}
        )
        assert result.ok
        assert {(r.source, r.source_id) for r in _songs_table(db)} == {
            ("local", "a.mp3"),
            ("spotify", "xyz"),
        }


class TestFailureIsolation:
    """One provider's failure neither rolls back the other nor loses its own rows."""

    def test_failing_source_records_named_error_other_source_commits(
        self, db: Database, tmp_path: Path
    ) -> None:
        local = StubProvider([_song("a.mp3")])
        youtube = StubProvider(
            error=YouTubeCatalogError("Failed to fetch YouTube playlist 'bogus'")
        )
        result = refresh_catalog(
            db, _settings(tmp_path), providers={"local": local, "youtube": youtube}
        )
        print(f"result: {result}")

        assert not result.ok
        yt = result.by_source("youtube")
        assert yt.error is not None
        assert "YouTubeCatalogError" in yt.error, "error must name the exception type"
        assert "bogus" in yt.error, "error must carry the provider's message"
        assert yt.added == yt.updated == yt.removed == yt.retained == 0

        assert result.by_source("local") == SourceRefresh(source="local", added=1)
        assert [(r.source, r.source_id) for r in _songs_table(db)] == [("local", "a.mp3")]

    def test_failed_source_keeps_its_existing_rows(
        self, db: Database, tmp_path: Path
    ) -> None:
        youtube = StubProvider([_youtube_song("vid1", title="Kept Title")])
        settings = _settings(tmp_path)
        refresh_catalog(db, settings, providers={"youtube": youtube})
        assert len(_songs_table(db)) == 1

        youtube.error = YouTubeCatalogError(
            "Timed out after 60s fetching YouTube playlist 'bogus'"
        )
        result = refresh_catalog(db, settings, providers={"youtube": youtube})

        yt = result.by_source("youtube")
        assert yt.error is not None
        assert "YouTubeCatalogError" in yt.error
        rows = _songs_table(db)
        assert len(rows) == 1, "a failed fetch must not delete the source's stored rows"
        assert (rows[0].source, rows[0].source_id, rows[0].title) == (
            "youtube",
            "vid1",
            "Kept Title",
        )

    def test_upsert_error_rolls_back_only_that_source(
        self, db: Database, tmp_path: Path
    ) -> None:
        # title=None violates songs.title NOT NULL -> the local transaction fails.
        bad_song = Song(
            source="local",
            source_id="bad.mp3",
            title=None,  # type: ignore[arg-type]
            artist="A",
            duration_sec=1.0,
            audio_ref="/x",
            raw_title="y",
        )
        local = StubProvider([_song("good.mp3"), bad_song])
        youtube = StubProvider([_youtube_song("vid1")])
        result = refresh_catalog(
            db, _settings(tmp_path), providers={"local": local, "youtube": youtube}
        )
        print(f"result: {result}")

        loc = result.by_source("local")
        assert loc.error is not None
        assert "IntegrityError" in loc.error
        # The whole local transaction rolled back: not even good.mp3 was committed,
        # while the independent youtube source committed fine.
        assert [(r.source, r.source_id) for r in _songs_table(db)] == [("youtube", "vid1")]


class TestProviderEnablement:
    """Pinned #10: empty playlist URL / empty local dir disables that provider."""

    def test_no_providers_enabled_is_a_noop(self, db: Database, tmp_path: Path) -> None:
        result = refresh_catalog(db, _settings(tmp_path))
        assert result == RefreshResult(sources=())
        assert result.ok
        assert _songs_table(db) == []

    def test_disabled_provider_rows_left_untouched(
        self, db: Database, tmp_path: Path
    ) -> None:
        _insert_song_row(db, "youtube", "vid-old")
        music_dir = tmp_path / "music"
        music_dir.mkdir()

        # Only the local provider is enabled; it builds the real
        # LocalDirectoryProvider from settings (empty dir -> no songs).
        result = refresh_catalog(db, _settings(tmp_path, local_music_dir=music_dir))

        assert [s.source for s in result.sources] == ["local"]
        assert result.by_source("local") == SourceRefresh(source="local")
        assert {(r.source, r.source_id) for r in _songs_table(db)} == {("youtube", "vid-old")}


class TestDuplicates:
    def test_duplicate_source_ids_in_one_fetch_first_wins(
        self, db: Database, tmp_path: Path
    ) -> None:
        provider = StubProvider(
            [_song("a.mp3", title="First"), _song("a.mp3", title="Second")]
        )
        result = refresh_catalog(db, _settings(tmp_path), providers={"local": provider})
        assert result.by_source("local") == SourceRefresh(source="local", added=1)
        rows = _songs_table(db)
        assert len(rows) == 1
        assert rows[0].title == "First"
