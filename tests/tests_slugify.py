"""Tests for stats_engine.utils.slugify."""

from __future__ import annotations

import pytest

from stats_engine.utils import slugify


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 1. Happy path: simple words separated by spaces.
        ("Volkanovski vs Lopes", "volkanovski-vs-lopes"),
        # 2. Period + space collapses to a single hyphen.
        ("Volkanovski vs. Lopes", "volkanovski-vs-lopes"),
        # 3. Apostrophe drops entirely (no hyphen substitution).
        ("O'Malley vs Dvalishvili", "omalley-vs-dvalishvili"),
        # 4. Ampersand becomes a hyphen separator.
        ("O'Malley & Dvalishvili", "omalley-dvalishvili"),
        # 5. Accents fold to plain ASCII via NFKD.
        ("Conor McGregor vs José Aldo", "conor-mcgregor-vs-jose-aldo"),
        # 6. Numbers are preserved.
        ("UFC 314: Volkanovski vs. Lopes", "ufc-314-volkanovski-vs-lopes"),
        # 7. Already-slugged input is idempotent.
        ("ufc-314-volkanovski-vs-lopes", "ufc-314-volkanovski-vs-lopes"),
        # 8. Leading/trailing junk strips cleanly.
        ("   ...UFC 1...   ", "ufc-1"),
        # 9. Emoji and other non-ASCII symbols become separators.
        ("🥊 UFC 100 🥊", "ufc-100"),
        # 10. Curly apostrophe drops the same way as straight.
        ("O\u2019Malley", "omalley"),
        # 11. Real-world title with colon, period, parens.
        (
            "UFC Fight Night: Holloway vs. Allen (Main Event)",
            "ufc-fight-night-holloway-vs-allen-main-event",
        ),
        # 12. Multiple consecutive separators of different kinds collapse to one hyphen.
        ("a---b___c   d", "a-b-c-d"),
    ],
)
def test_slugify_happy_paths(text: str, expected: str) -> None:
    assert slugify(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",            # empty string
        "   ",         # whitespace only
        "!!!",         # all punctuation
        "...",         # dots only
        "🥊🥊🥊",       # emoji only (fold to nothing)
        "'''",         # apostrophes only (drop to empty)
    ],
)
def test_slugify_empty_result_raises(text: str) -> None:
    with pytest.raises(ValueError):
        slugify(text)


def test_slugify_rejects_non_str() -> None:
    with pytest.raises(TypeError):
        slugify(123)  # type: ignore[arg-type]


def test_slugify_is_idempotent() -> None:
    """slugify(slugify(x)) == slugify(x) for any valid x."""
    inputs = [
        "UFC 314: Volkanovski vs. Lopes",
        "O'Malley & Dvalishvili",
        "Conor McGregor vs José Aldo",
    ]
    for text in inputs:
        once = slugify(text)
        twice = slugify(once)
        assert once == twice