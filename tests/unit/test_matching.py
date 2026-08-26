"""Unit tests for songbot.matching: normalize() + match_guess().

Pure logic, fully deterministic. Contract coverage:
- VAL-GUESS-015: normalization matrix (case, whitespace, punctuation,
  brackets, feat./ft., remaster markers, unicode-fold, "The " prefix, subset)
- VAL-GUESS-016: fuzzy threshold boundary at token_set_ratio >= 85, applied
  AFTER normalization (``>=``, not ``>``)
- VAL-GUESS-017: raw_title fallback rescues guesses on poorly-parsed
  YouTube entries
"""

from __future__ import annotations

import pytest
from rapidfuzz import fuzz

from songbot.catalog import Song
from songbot.matching import MATCH_THRESHOLD, match_guess, normalize

QUEEN = Song(
    source="local",
    source_id="queen-bohemian",
    title="Bohemian Rhapsody",
    artist="Queen",
    duration_sec=30.0,
    audio_ref="/music/queen-bohemian.mp3",
    raw_title="Queen - Bohemian Rhapsody",
)

BEYONCE = Song(
    source="local",
    source_id="beyonce-halo",
    title="Halo",
    artist="Beyoncé",
    duration_sec=30.0,
    audio_ref="/music/beyonce-halo.mp3",
    raw_title="Beyoncé - Halo",
)

THE_BAND = Song(
    source="local",
    source_id="europe-countdown",
    title="The Final Countdown",
    artist="Europe",
    duration_sec=30.0,
    audio_ref="/music/europe-countdown.mp3",
    raw_title="Europe - The Final Countdown",
)


class TestNormalize:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Bohemian Rhapsody", "bohemian rhapsody"),
            ("BOHEMIAN RHAPSODY", "bohemian rhapsody"),
            ("  Bohemian   Rhapsody\t", "bohemian rhapsody"),
            ("Bohemian Rhapsody!", "bohemian rhapsody"),
            ("bohemian rhapsody…", "bohemian rhapsody"),
            ("Bohemian Rhapsody (Official Video)", "bohemian rhapsody"),
            ("Bohemian Rhapsody [Remastered 2011]", "bohemian rhapsody"),
            ("Bohemian Rhapsody 【MV】", "bohemian rhapsody"),
            ("Queen feat. X", "queen x"),
            ("Queen ft. X", "queen x"),
            ("Queen featuring X", "queen x"),
            ("Bohemian Rhapsody Remastered", "bohemian rhapsody"),
            ("Bohemian Rhapsody remaster", "bohemian rhapsody"),
            ("Beyoncé", "beyonce"),
            ("The Queen", "the queen"),
            ("", ""),
            ("   ", ""),
            ("!!!…【】", ""),
        ],
        ids=[
            "plain",
            "uppercase",
            "whitespace",
            "exclamation",
            "ellipsis",
            "paren-suffix",
            "bracket-remaster",
            "cjk-brackets",
            "feat-dot",
            "ft-dot",
            "featuring",
            "remastered",
            "remaster",
            "unicode-fold",
            "the-prefix-kept",
            "empty",
            "blank",
            "only-noise",
        ],
    )
    def test_normalize_matrix(self, raw: str, expected: str) -> None:
        assert normalize(raw) == expected


class TestMatchGuessNormalizationMatrix:
    """VAL-GUESS-015: each case must produce the expected matched flags."""

    @pytest.mark.parametrize(
        ("guess", "matched_title", "matched_artist"),
        [
            ("Bohemian Rhapsody", True, False),  # (a) exact title
            ("BOHEMIAN RHAPSODY", True, False),  # (b) uppercase
            ("  Bohemian   Rhapsody  ", True, False),  # (c) whitespace
            ("Bohemian Rhapsody!", True, False),  # (d) punctuation
            ("bohemian rhapsody…", True, False),  # (d) unicode ellipsis
            ("Bohemian Rhapsody (Official Video)", True, False),  # (e) parens
            ("Bohemian Rhapsody [Remastered 2011]", True, False),  # (e) brackets
            ("Bohemian Rhapsody 【MV】", True, False),  # (e) CJK brackets
            ("Queen feat. X", False, True),  # (f) feat. marker
            ("Queen ft. X", False, True),  # (f) ft. marker
            ("Bohemian Rhapsody Remastered", True, False),  # (g) remastered
            ("Bohemian Rhapsody remaster", True, False),  # (g) remaster
            ("The Queen", False, True),  # (i) "The " prefix added
            ("Bohemian", True, False),  # (j) partial multi-word subset
            ("Some Other Song", False, False),  # negative control
        ],
        ids=[
            "exact-title",
            "uppercase-title",
            "whitespace-variant",
            "punctuation-bang",
            "punctuation-ellipsis",
            "bracket-parens",
            "bracket-remastered",
            "bracket-cjk",
            "feat-marker",
            "ft-marker",
            "remastered-suffix",
            "remaster-suffix",
            "the-prefix-added",
            "title-subset",
            "unrelated",
        ],
    )
    def test_queen_matrix(
        self, guess: str, matched_title: bool, matched_artist: bool
    ) -> None:
        result = match_guess(guess, QUEEN)
        assert result.matched_title is matched_title
        assert result.matched_artist is matched_artist
        assert result.is_correct is (matched_title or matched_artist)
        assert result.is_both is (matched_title and matched_artist)

    def test_unicode_fold_artist_both_directions(self) -> None:
        """VAL-GUESS-015(h): 'Beyoncé' <-> 'Beyonce' against artist 'Beyoncé'."""
        assert match_guess("Beyonce", BEYONCE).matched_artist is True
        assert match_guess("Beyoncé", BEYONCE).matched_artist is True

    def test_the_prefix_removed_against_titled_song(self) -> None:
        """VAL-GUESS-015(i): 'The ' prefix removed ('Final Countdown')."""
        result = match_guess("Final Countdown", THE_BAND)
        assert result.matched_title is True
        assert result.matched_artist is False

    def test_exact_artist(self) -> None:
        result = match_guess("Queen", QUEEN)
        assert result.matched_artist is True
        assert result.matched_title is False
        assert result.is_correct is True
        assert result.is_both is False

    def test_combined_artist_title_matches_both(self) -> None:
        """The '<artist> - <title>' combined guess underpins the 1.5x bonus."""
        result = match_guess("Queen - Bohemian Rhapsody", QUEEN)
        assert result.matched_title is True
        assert result.matched_artist is True
        assert result.is_both is True

    def test_empty_guess_never_matches(self) -> None:
        result = match_guess("   ", QUEEN)
        assert result == match_guess("", QUEEN)
        assert result.is_correct is False
        assert result.matched_title is False
        assert result.matched_artist is False


class TestMatchGuessThreshold:
    """VAL-GUESS-016: threshold is token_set_ratio >= 85 AFTER normalization."""

    def test_single_char_typo_in_long_title_matches(self) -> None:
        # token_set_ratio("bohemian rhapsodu", "bohemian rhapsody") == 94.12
        guess = "Bohemian Rhapsodu"
        ratio = fuzz.token_set_ratio(normalize(guess), normalize(QUEEN.title))
        assert ratio >= MATCH_THRESHOLD
        assert match_guess(guess, QUEEN).matched_title is True

    def test_unrelated_string_rejected(self) -> None:
        # token_set_ratio("zxqv unrelated noise", "bohemian rhapsody") == 32.43
        guess = "zxqv unrelated noise"
        ratio = fuzz.token_set_ratio(normalize(guess), normalize(QUEEN.title))
        assert ratio < MATCH_THRESHOLD
        assert match_guess(guess, QUEEN).is_correct is False

    def test_boundary_pair_straddles_85_exactly(self) -> None:
        """A constructed pair at EXACTLY 85.0 must match; 80.0 must not.

        Both strings are 20 chars; token_set_ratio reduces to fuzz.ratio:
        LCS 17 -> indel distance 6 -> (40-6)/40 == 0.85 -> 85.0 (accepted,
        proving the comparison is ``>= 85`` and not ``> 85``); LCS 16 ->
        distance 8 -> 80.0 (rejected).
        """
        song = Song(
            source="local",
            source_id="boundary",
            title="a" * 20,
            artist="c" * 20,
            duration_sec=30.0,
            audio_ref="/music/boundary.mp3",
            raw_title="d" * 20,
        )
        at_threshold = "a" * 17 + "b" * 3
        below_threshold = "a" * 16 + "b" * 4

        ratio_at = fuzz.token_set_ratio(normalize(at_threshold), normalize(song.title))
        ratio_below = fuzz.token_set_ratio(normalize(below_threshold), normalize(song.title))
        assert ratio_at == 85.0
        assert ratio_below == 80.0

        assert match_guess(at_threshold, song).matched_title is True
        assert match_guess(below_threshold, song).is_correct is False

    def test_threshold_applied_after_normalization(self) -> None:
        """Decorations are stripped BEFORE the ratio: '(Official Video)' and
        case/punctuation differences must not lower the score below 85."""
        guess = "BOHEMIAN   RHAPSODY (Official Video)!"
        result = match_guess(guess, QUEEN)
        assert result.matched_title is True


class TestPerTokenFallback:
    """The per-token fallback: a field also matches when every one of its
    tokens has a close twin among the guess tokens (ratio >= 85, or one
    Levenshtein edit for 4+ char tokens).

    Exists because ``token_set_ratio``'s subset bonus is all-or-nothing per
    token: one typo breaks the exact-token intersection and the score
    collapses below 85 — which used to kill the both-fields bonus on
    combined guesses and made short titles/artists tolerate zero typos.
    """

    def test_combined_guess_with_typo_in_artist_still_matches_both(self) -> None:
        """The reported bug: 'Quen' alone passes (88.9), but inside a combined
        guess the artist check used to fail -> bonus lost."""
        result = match_guess("Bohemian Rhapsody Quen", QUEEN)
        assert result.matched_title is True
        assert result.matched_artist is True
        assert result.is_both is True

    def test_combined_guess_with_typo_in_title_still_matches_both(self) -> None:
        """'Bohemian Rhapsodu' alone passes (94.1); adding the artist used to
        drag the title check below 85 -> bonus lost."""
        result = match_guess("Bohemian Rhapsodu Queen", QUEEN)
        assert result.matched_title is True
        assert result.matched_artist is True
        assert result.is_both is True

    def test_short_title_single_substitution_matches(self) -> None:
        # fuzz.ratio("halp", "halo") == 75.0 — below threshold, one edit apart.
        assert match_guess("Halp", BEYONCE).matched_title is True

    def test_short_artist_single_substitution_matches(self) -> None:
        # fuzz.ratio("qween", "queen") == 80.0 — below threshold, one edit apart.
        assert match_guess("Qween", QUEEN).matched_artist is True

    def test_one_edit_allowance_requires_four_chars(self) -> None:
        """1-3 char tokens stay effectively exact: one edit changes too much."""
        tiny = Song(
            source="local",
            source_id="tiny",
            title="ABC",
            artist="U2",
            duration_sec=30.0,
            audio_ref="/music/tiny.mp3",
            raw_title="U2 - ABC",
        )
        assert match_guess("abd", tiny).is_correct is False  # ABC, one sub
        assert match_guess("u3", tiny).is_correct is False  # U2, one sub
        assert match_guess("abc", tiny).matched_title is True
        assert match_guess("u2", tiny).matched_artist is True

    def test_near_miss_on_one_token_of_multi_token_field_still_fails(self) -> None:
        """Every candidate token needs a twin: 'Bohemia' covers 'bohemian'
        (93.3) but nothing covers 'rhapsody'."""
        result = match_guess("Bohemia", QUEEN)
        assert result.is_correct is False

    def test_unrelated_guess_still_rejected(self) -> None:
        assert match_guess("zxqv unrelated noise", QUEEN).is_correct is False

    def test_fallback_respects_custom_threshold(self) -> None:
        """The per-token ratio gate uses the same threshold as the field level.

        "rhapsoyd"/"rhapsody" is a transposition: fuzz.ratio 93.75 but two
        Levenshtein edits, so the one-edit clause cannot rescue it — it
        passes only while the threshold is <= 93.75.
        """
        guess = "Bohemian Rhapsoyd Queen"
        assert match_guess(guess, QUEEN).is_both is True
        strict = match_guess(guess, QUEEN, threshold=95)
        assert strict.matched_title is False
        assert strict.matched_artist is True
        assert strict.is_both is False


class TestRawTitleFallback:
    """VAL-GUESS-017: raw_title rescues guesses on poorly-parsed YouTube entries."""

    POORLY_PARSED = Song(
        source="youtube",
        source_id="vid123",
        title="Official Video",  # parse disaster: real identity only in raw_title
        artist=None,
        duration_sec=213.0,
        audio_ref="https://www.youtube.com/watch?v=vid123",
        raw_title="Real Artist - Real Title (Official Video)",
    )

    def test_raw_title_rescues_title_guess(self) -> None:
        result = match_guess("Real Title", self.POORLY_PARSED)
        assert result.is_correct is True
        # The fallback credits the rescue as a title match (documented choice);
        # it never fabricates a both-match bonus.
        assert result.matched_title is True
        assert result.matched_artist is False
        assert result.is_both is False

    def test_raw_title_rescues_artist_guess(self) -> None:
        result = match_guess("Real Artist", self.POORLY_PARSED)
        assert result.is_correct is True

    def test_guess_matching_nothing_still_fails(self) -> None:
        result = match_guess("Unrelated Thing", self.POORLY_PARSED)
        assert result.is_correct is False
        assert result.matched_title is False
        assert result.matched_artist is False

    def test_parsed_fields_win_before_fallback(self) -> None:
        """When the parsed title matches, the fallback is not consulted."""
        result = match_guess("Official Video", self.POORLY_PARSED)
        assert result.matched_title is True
        assert result.is_correct is True
