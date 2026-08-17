"""Unit tests for songbot.config (Settings / load_settings / ConfigError)."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from songbot.config import ConfigError, Settings, load_settings

skip_if_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses filesystem permission checks",
)

REQUIRED_ENV = {
    "DISCORD_BOT_TOKEN": "test-token",
    "DISCORD_GUILD_ID": "123456789",
    "DISCORD_CHANNEL_ID": "987654321",
}

VALID_ENV = {
    **REQUIRED_ENV,
    "YOUTUBE_PLAYLIST_URL": "https://youtube.com/playlist?list=PL_TEST",
    "LOCAL_MUSIC_DIR": "./data/fixture-music",
    "DAILY_POST_TIME": "12:00",
    "TIMEZONE": "America/Halifax",
    "MAX_GUESSES_PER_DAY": "6",
    "SNIPPET_LENGTHS": "1,2,4,8,16",
    "SNIPPET_POINTS": "100,75,50,30,15",
    "BOTH_CORRECT_MULTIPLIER": "1.5",
    "DATABASE_PATH": "./data/songbot.db",
    "SNIPPET_CACHE_DIR": "./data/snippets",
    "HEALTH_PORT": "3108",
    "LOG_LEVEL": "INFO",
}


def write_env(tmp_path: Path, values: dict[str, str]) -> Path:
    env_file = tmp_path / ".env"
    env_file.write_text("\n".join(f"{k}={v}" for k, v in values.items()) + "\n")
    return env_file


def load(env_file: Path, environ: dict[str, str] | None = None) -> Settings:
    """Load settings with a fully controlled environment (no real os.environ leakage)."""
    return load_settings(env_file=env_file, environ=environ if environ is not None else {})


class TestHappyPath:
    def test_loads_valid_env_file(self, tmp_path: Path) -> None:
        env_file = write_env(tmp_path, VALID_ENV)
        settings = load(env_file)
        assert isinstance(settings, Settings)
        assert settings.discord_token == "test-token"
        assert settings.guild_id == "123456789"
        assert settings.channel_id == "987654321"
        assert settings.youtube_playlist_url == "https://youtube.com/playlist?list=PL_TEST"
        assert settings.local_music_dir == Path("./data/fixture-music")
        assert settings.daily_post_time == "12:00"
        assert settings.timezone == "America/Halifax"
        assert settings.max_guesses_per_day == 6
        assert settings.snippet_lengths == (1.0, 2.0, 4.0, 8.0, 16.0)
        assert settings.snippet_points == (100, 75, 50, 30, 15)
        assert settings.both_correct_multiplier == 1.5
        assert settings.database_path == Path("./data/songbot.db")
        assert settings.snippet_cache_dir == Path("./data/snippets")
        assert settings.health_port == 3108
        assert settings.log_level == "INFO"
        assert settings.discord_api_base == "https://discord.com/api/v10"

    def test_defaults_applied_when_optional_keys_missing(self, tmp_path: Path) -> None:
        env_file = write_env(tmp_path, REQUIRED_ENV)
        settings = load(env_file)
        assert settings.daily_post_time == "12:00"
        assert settings.timezone == "America/Halifax"
        assert settings.max_guesses_per_day == 6
        assert settings.snippet_lengths == (1.0, 2.0, 4.0, 8.0, 16.0)
        assert settings.snippet_points == (100, 75, 50, 30, 15)
        assert settings.both_correct_multiplier == 1.5
        assert settings.database_path == Path("./data/songbot.db")
        assert settings.snippet_cache_dir == Path("./data/snippets")
        assert settings.health_port == 3108
        assert settings.log_level == "INFO"
        assert settings.discord_api_base == "https://discord.com/api/v10"

    def test_settings_is_frozen(self, tmp_path: Path) -> None:
        settings = load(write_env(tmp_path, VALID_ENV))
        with pytest.raises(dataclasses.FrozenInstanceError):
            settings.health_port = 9999  # type: ignore[misc]

    def test_fractional_snippet_lengths_accepted(self, tmp_path: Path) -> None:
        env_file = write_env(
            tmp_path, {**VALID_ENV, "SNIPPET_LENGTHS": "0.5,1.5", "SNIPPET_POINTS": "100,50"}
        )
        settings = load(env_file)
        assert settings.snippet_lengths == (0.5, 1.5)
        assert settings.snippet_points == (100, 50)


class TestPrecedence:
    """Pinned decision #10: os.environ OVERRIDES .env, including empty-string overrides."""

    def test_environ_overrides_env_file(self, tmp_path: Path) -> None:
        env_file = write_env(tmp_path, VALID_ENV)
        settings = load(env_file, environ={"TIMEZONE": "UTC", "HEALTH_PORT": "4000"})
        assert settings.timezone == "UTC"
        assert settings.health_port == 4000

    def test_empty_environ_value_overrides_env_file(self, tmp_path: Path) -> None:
        env_file = write_env(tmp_path, VALID_ENV)
        settings = load(env_file, environ={"YOUTUBE_PLAYLIST_URL": ""})
        assert settings.youtube_playlist_url is None

    def test_missing_env_file_is_not_an_error(self, tmp_path: Path) -> None:
        settings = load(tmp_path / "does-not-exist.env", environ=dict(VALID_ENV))
        assert settings.timezone == "America/Halifax"


class TestDisabledProviders:
    def test_empty_youtube_playlist_url_disables_provider(self, tmp_path: Path) -> None:
        env_file = write_env(tmp_path, {**VALID_ENV, "YOUTUBE_PLAYLIST_URL": ""})
        settings = load(env_file)
        assert settings.youtube_playlist_url is None

    def test_empty_local_music_dir_disables_provider(self, tmp_path: Path) -> None:
        env_file = write_env(tmp_path, {**VALID_ENV, "LOCAL_MUSIC_DIR": ""})
        settings = load(env_file)
        assert settings.local_music_dir is None


class TestMissingRequired:
    def test_missing_required_keys_all_listed(self, tmp_path: Path) -> None:
        env_file = write_env(tmp_path, {})
        with pytest.raises(ConfigError) as exc_info:
            load(env_file)
        message = str(exc_info.value)
        assert "DISCORD_BOT_TOKEN" in message

    def test_empty_required_key_is_missing(self, tmp_path: Path) -> None:
        env_file = write_env(tmp_path, {**VALID_ENV, "DISCORD_BOT_TOKEN": ""})
        with pytest.raises(ConfigError, match="DISCORD_BOT_TOKEN"):
            load(env_file)


class TestGuildChannelBootstrapPair:
    """Multi-guild: the env pair is an optional bootstrap — both or neither."""

    def test_neither_set_is_valid_and_yields_none(self, tmp_path: Path) -> None:
        env_file = write_env(
            tmp_path,
            {k: v for k, v in VALID_ENV.items() if "GUILD" not in k and "CHANNEL" not in k},
        )
        settings = load(env_file)
        assert settings.guild_id is None
        assert settings.channel_id is None

    def test_guild_without_channel_is_an_error(self, tmp_path: Path) -> None:
        env_file = write_env(tmp_path, {**VALID_ENV, "DISCORD_CHANNEL_ID": ""})
        with pytest.raises(ConfigError, match="set together"):
            load(env_file)

    def test_channel_without_guild_is_an_error(self, tmp_path: Path) -> None:
        env_file = write_env(tmp_path, {**VALID_ENV, "DISCORD_GUILD_ID": ""})
        with pytest.raises(ConfigError, match="set together"):
            load(env_file)

    def test_both_set_is_valid(self, tmp_path: Path) -> None:
        env_file = write_env(tmp_path, VALID_ENV)
        settings = load(env_file)
        assert settings.guild_id == "123456789"
        assert settings.channel_id == "987654321"


class TestTimezoneValidation:
    def test_invalid_timezone_rejected(self, tmp_path: Path) -> None:
        """VAL-OPS-002: bad timezone raises ConfigError naming the field/value."""
        env_file = write_env(tmp_path, {**VALID_ENV, "TIMEZONE": "Mars/Olympus_Mons"})
        with pytest.raises(ConfigError) as exc_info:
            load(env_file)
        message = str(exc_info.value)
        assert "TIMEZONE" in message
        assert "Mars/Olympus_Mons" in message

    def test_invalid_timezone_error_type_is_config_error(self, tmp_path: Path) -> None:
        env_file = write_env(tmp_path, {**VALID_ENV, "TIMEZONE": "Mars/Olympus_Mons"})
        with pytest.raises(ConfigError) as exc_info:
            load(env_file)
        assert type(exc_info.value) is ConfigError


class TestDailyPostTime:
    @pytest.mark.parametrize("bad", ["25:99", "noon", "9:00", "12:60", "24:00", "12", "12:0"])
    def test_bad_time_format_rejected(self, tmp_path: Path, bad: str) -> None:
        env_file = write_env(tmp_path, {**VALID_ENV, "DAILY_POST_TIME": bad})
        with pytest.raises(ConfigError, match="DAILY_POST_TIME"):
            load(env_file)

    @pytest.mark.parametrize("good", ["00:00", "09:05", "12:00", "23:59"])
    def test_good_time_format_accepted(self, tmp_path: Path, good: str) -> None:
        env_file = write_env(tmp_path, {**VALID_ENV, "DAILY_POST_TIME": good})
        assert load(env_file).daily_post_time == good


class TestSnippetLists:
    def test_mismatched_lengths_rejected(self, tmp_path: Path) -> None:
        """VAL-OPS-003 case A: mismatch names both fields."""
        env_file = write_env(
            tmp_path,
            {**VALID_ENV, "SNIPPET_LENGTHS": "1,2,4,8,16", "SNIPPET_POINTS": "100,75,50"},
        )
        with pytest.raises(ConfigError) as exc_info:
            load(env_file)
        message = str(exc_info.value)
        assert "SNIPPET_LENGTHS" in message
        assert "SNIPPET_POINTS" in message

    def test_non_positive_points_rejected(self, tmp_path: Path) -> None:
        """VAL-OPS-003 case B: non-positive value is named."""
        env_file = write_env(
            tmp_path,
            {**VALID_ENV, "SNIPPET_POINTS": "100,0,50,30,15"},
        )
        with pytest.raises(ConfigError) as exc_info:
            load(env_file)
        message = str(exc_info.value)
        assert "SNIPPET_POINTS" in message
        assert "0" in message

    def test_non_positive_lengths_rejected(self, tmp_path: Path) -> None:
        env_file = write_env(
            tmp_path,
            {**VALID_ENV, "SNIPPET_LENGTHS": "1,-2,4,8,16"},
        )
        with pytest.raises(ConfigError, match="SNIPPET_LENGTHS"):
            load(env_file)

    def test_empty_lists_rejected(self, tmp_path: Path) -> None:
        env_file = write_env(
            tmp_path, {**VALID_ENV, "SNIPPET_LENGTHS": "", "SNIPPET_POINTS": ""}
        )
        with pytest.raises(ConfigError, match="SNIPPET_LENGTHS"):
            load(env_file)

    def test_non_numeric_entries_rejected(self, tmp_path: Path) -> None:
        env_file = write_env(tmp_path, {**VALID_ENV, "SNIPPET_LENGTHS": "1,two,4"})
        with pytest.raises(ConfigError, match="SNIPPET_LENGTHS"):
            load(env_file)


class TestMultiplier:
    @pytest.mark.parametrize("bad", ["0", "-1.5", "abc"])
    def test_non_positive_or_invalid_multiplier_rejected(self, tmp_path: Path, bad: str) -> None:
        env_file = write_env(tmp_path, {**VALID_ENV, "BOTH_CORRECT_MULTIPLIER": bad})
        with pytest.raises(ConfigError, match="BOTH_CORRECT_MULTIPLIER"):
            load(env_file)


class TestNumericFields:
    @pytest.mark.parametrize("bad", ["0", "65536", "abc", "-1"])
    def test_invalid_health_port_rejected(self, tmp_path: Path, bad: str) -> None:
        env_file = write_env(tmp_path, {**VALID_ENV, "HEALTH_PORT": bad})
        with pytest.raises(ConfigError, match="HEALTH_PORT"):
            load(env_file)

    @pytest.mark.parametrize("bad", ["0", "-3", "six"])
    def test_invalid_max_guesses_rejected(self, tmp_path: Path, bad: str) -> None:
        env_file = write_env(tmp_path, {**VALID_ENV, "MAX_GUESSES_PER_DAY": bad})
        with pytest.raises(ConfigError, match="MAX_GUESSES_PER_DAY"):
            load(env_file)


class TestDirectories:
    def test_missing_but_creatable_dirs_accepted_and_not_eagerly_created(
        self, tmp_path: Path
    ) -> None:
        """VAL-OPS-005 case A: load succeeds; feature pins NO eager creation."""
        cache_dir = tmp_path / "newdir" / "snippets"
        db_path = tmp_path / "newdir" / "songbot.db"
        music_dir = tmp_path / "newdir" / "music"
        env_file = write_env(
            tmp_path,
            {
                **VALID_ENV,
                "SNIPPET_CACHE_DIR": str(cache_dir),
                "DATABASE_PATH": str(db_path),
                "LOCAL_MUSIC_DIR": str(music_dir),
            },
        )
        settings = load(env_file)
        assert settings.snippet_cache_dir == cache_dir
        assert not cache_dir.exists(), "load_settings must not eagerly create directories"
        assert not music_dir.exists(), "load_settings must not eagerly create directories"

    def test_cache_dir_under_regular_file_rejected(self, tmp_path: Path) -> None:
        """VAL-OPS-005 case B: uncreatable directory raises ConfigError naming it."""
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file")
        env_file = write_env(
            tmp_path, {**VALID_ENV, "SNIPPET_CACHE_DIR": str(blocker / "snippets")}
        )
        with pytest.raises(ConfigError) as exc_info:
            load(env_file)
        assert "SNIPPET_CACHE_DIR" in str(exc_info.value)

    def test_cache_dir_that_is_a_regular_file_rejected(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file")
        env_file = write_env(tmp_path, {**VALID_ENV, "SNIPPET_CACHE_DIR": str(blocker)})
        with pytest.raises(ConfigError, match="SNIPPET_CACHE_DIR"):
            load(env_file)

    def test_database_path_under_regular_file_rejected(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file")
        env_file = write_env(
            tmp_path, {**VALID_ENV, "DATABASE_PATH": str(blocker / "songbot.db")}
        )
        with pytest.raises(ConfigError, match="DATABASE_PATH"):
            load(env_file)

    def test_local_music_dir_under_regular_file_rejected(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file")
        env_file = write_env(
            tmp_path, {**VALID_ENV, "LOCAL_MUSIC_DIR": str(blocker / "music")}
        )
        with pytest.raises(ConfigError, match="LOCAL_MUSIC_DIR"):
            load(env_file)

    @skip_if_root
    def test_read_only_existing_cache_dir_rejected(self, tmp_path: Path) -> None:
        """Regression: an existing but read-only SNIPPET_CACHE_DIR must not load."""
        cache_dir = tmp_path / "snippets"
        cache_dir.mkdir()
        cache_dir.chmod(0o500)
        try:
            env_file = write_env(tmp_path, {**VALID_ENV, "SNIPPET_CACHE_DIR": str(cache_dir)})
            with pytest.raises(ConfigError) as exc_info:
                load(env_file)
            message = str(exc_info.value)
            assert "SNIPPET_CACHE_DIR" in message
            assert str(cache_dir) in message
        finally:
            cache_dir.chmod(0o700)

    @skip_if_root
    def test_read_only_existing_database_parent_rejected(self, tmp_path: Path) -> None:
        """Regression: a DATABASE_PATH whose existing parent is read-only must not load."""
        parent = tmp_path / "db"
        parent.mkdir()
        parent.chmod(0o500)
        try:
            env_file = write_env(
                tmp_path, {**VALID_ENV, "DATABASE_PATH": str(parent / "songbot.db")}
            )
            with pytest.raises(ConfigError) as exc_info:
                load(env_file)
            message = str(exc_info.value)
            assert "DATABASE_PATH" in message
            assert str(parent) in message
        finally:
            parent.chmod(0o700)

    def test_writable_existing_runtime_dirs_accepted(self, tmp_path: Path) -> None:
        """Existing WRITABLE runtime dirs still load (the writability check has teeth
        only for non-writable dirs)."""
        cache_dir = tmp_path / "snippets"
        cache_dir.mkdir()
        db_dir = tmp_path / "db"
        db_dir.mkdir()
        env_file = write_env(
            tmp_path,
            {
                **VALID_ENV,
                "SNIPPET_CACHE_DIR": str(cache_dir),
                "DATABASE_PATH": str(db_dir / "songbot.db"),
            },
        )
        settings = load(env_file)
        assert settings.snippet_cache_dir == cache_dir
        assert settings.database_path == db_dir / "songbot.db"


class TestAllProblemsListed:
    def test_single_config_error_lists_all_problems(self, tmp_path: Path) -> None:
        """VAL-OPS-004: bad timezone + bad time + mismatched lists + uncreatable dir."""
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file")
        env_file = write_env(
            tmp_path,
            {
                **VALID_ENV,
                "TIMEZONE": "Mars/Olympus_Mons",
                "DAILY_POST_TIME": "25:99",
                "SNIPPET_LENGTHS": "1,2,4,8,16",
                "SNIPPET_POINTS": "100,75,50",
                "SNIPPET_CACHE_DIR": str(blocker / "snippets"),
            },
        )
        with pytest.raises(ConfigError) as exc_info:
            load(env_file)
        message = str(exc_info.value)
        assert "TIMEZONE" in message
        assert "DAILY_POST_TIME" in message
        assert "SNIPPET_LENGTHS" in message
        assert "SNIPPET_CACHE_DIR" in message
