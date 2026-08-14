"""Unit tests for songbot.catalog.parsing.parse_artist_title."""

from __future__ import annotations

from songbot.catalog.parsing import parse_artist_title


class TestArtistDashTitle:
    def test_simple_split(self) -> None:
        assert parse_artist_title("Midnight Circuit - Neon Skyline") == (
            "Midnight Circuit",
            "Neon Skyline",
        )

    def test_uppercase_artist_preserved(self) -> None:
        # Real playlist entry jO4fTcziVRM: "XI - Akasha" (VAL-CATALOG-007).
        artist, title = parse_artist_title("XI - Akasha")
        assert artist == "XI"
        assert title == "Akasha"

    def test_extra_whitespace_collapsed(self) -> None:
        assert parse_artist_title("  Artist   -   Title  ") == ("Artist", "Title")

    def test_multiple_dashes_split_on_first(self) -> None:
        artist, title = parse_artist_title("Artist - Title - Remix")
        assert artist == "Artist"
        assert title == "Title - Remix"

    def test_empty_split_side_declines(self) -> None:
        artist, title = parse_artist_title("Artist - ")
        assert artist is None
        assert title  # title is always non-empty for non-empty input


class TestBracketStripping:
    def test_square_brackets_prefix_and_suffix(self) -> None:
        # Real playlist entry 9F2sK2aO8-U (VAL-CATALOG-008a).
        artist, title = parse_artist_title("[Official] ANiMA / xi [World Fragments]")
        assert artist == "xi"
        assert title == "ANiMA"

    def test_fullwidth_brackets_stripped(self) -> None:
        # Real playlist entry XLRzISm_Y18 (VAL-CATALOG-008b).
        artist, title = parse_artist_title("【Paradigm: Reboot】xi VS Sakuzyo - Abyssgazer")
        assert artist == "xi VS Sakuzyo"
        assert title == "Abyssgazer"

    def test_paren_decoration_stripped(self) -> None:
        assert parse_artist_title("Artist - Title (Official Video)") == ("Artist", "Title")

    def test_multiple_bracket_groups_stripped(self) -> None:
        artist, title = parse_artist_title("Artist - Title [Official] (HD) 【Audio】")
        assert artist == "Artist"
        assert title == "Title"

    def test_only_brackets_falls_back_to_raw(self) -> None:
        # Title must never become empty (VAL-CATALOG-009 shape).
        artist, title = parse_artist_title("[Official]")
        assert artist is None
        assert title == "[Official]"


class TestSlashForm:
    def test_title_slash_artist(self) -> None:
        assert parse_artist_title("ANiMA / xi") == ("xi", "ANiMA")

    def test_slash_without_spaces_not_split(self) -> None:
        assert parse_artist_title("AC/DC - Thunderstruck") == ("AC/DC", "Thunderstruck")

    def test_dash_takes_precedence_over_slash(self) -> None:
        artist, title = parse_artist_title("Artist - Title / Remix")
        assert artist == "Artist"
        assert title == "Title / Remix"


class TestFeatNormalization:
    def test_ft_with_dot_normalized(self) -> None:
        assert parse_artist_title("Artist - Title ft. Someone") == (
            "Artist",
            "Title feat. Someone",
        )

    def test_ft_without_dot_normalized(self) -> None:
        assert parse_artist_title("Artist - Title ft Someone") == (
            "Artist",
            "Title feat. Someone",
        )

    def test_featuring_normalized(self) -> None:
        assert parse_artist_title("Artist - Title featuring Someone") == (
            "Artist",
            "Title feat. Someone",
        )

    def test_feat_bracket_stripped_as_decoration(self) -> None:
        assert parse_artist_title("Artist - Title (feat. Someone)") == ("Artist", "Title")

    def test_feat_in_artist_normalized(self) -> None:
        assert parse_artist_title("Artist ft. Someone - Title") == (
            "Artist feat. Someone",
            "Title",
        )


class TestBareTitles:
    def test_bare_title_returns_none_artist(self) -> None:
        # Real playlist entry 1Zn2uTmLo3Q: "Agartha" (VAL-CATALOG-009 shape).
        assert parse_artist_title("Agartha") == (None, "Agartha")

    def test_empty_string_does_not_raise(self) -> None:
        assert parse_artist_title("") == (None, "")
