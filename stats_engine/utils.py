"""Cross-cutting helpers for the stats_engine package.

Dependency-free utilities shared across the Elo engine, cleaners, and (later)
other stats_engine submodules. Add new helpers here when the alternative is
duplicating logic in two or more files.
"""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# Straight, curly, and modifier apostrophes are all dropped (no hyphen)
# so names like "O'Malley" slug to "omalley" rather than "o-malley".
_APOSTROPHES = "'\u2019\u2018\u02bc\u0060"

# One or more characters that are NOT lowercase ASCII alphanum.
# Each run collapses to a single hyphen in the output slug.
_NON_ALPHANUM_RUN = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Convert a free-text string into a URL-safe slug.

    Pipeline:
        1. NFKD-normalise + drop combining marks (ASCII-fold accents).
        2. Drop apostrophes entirely (no hyphen substitution).
        3. Lowercase.
        4. Replace every run of non-alphanumeric characters with a single ``-``.
        5. Strip leading/trailing hyphens.

    Args:
        text: The input string. Must be a ``str``.

    Returns:
        The slug. Always non-empty.

    Raises:
        TypeError: If ``text`` is not a ``str``.
        ValueError: If the slug would be empty (input was empty,
            whitespace-only, all punctuation, or contained no characters
            that survive ASCII folding).

    Examples:
        >>> slugify("Volkanovski vs. Lopes")
        'volkanovski-vs-lopes'
        >>> slugify("O'Malley & Dvalishvili")
        'omalley-dvalishvili'
        >>> slugify("UFC 314: Volkanovski vs. Lopes")
        'ufc-314-volkanovski-vs-lopes'
    """
    if not isinstance(text, str):
        raise TypeError(f"slugify expected str, got {type(text).__name__}")

    # 1. ASCII-fold via NFKD + drop combining marks.
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))

    # 2. Drop apostrophes entirely.
    for apostrophe in _APOSTROPHES:
        folded = folded.replace(apostrophe, "")

    # 3. Lowercase.
    folded = folded.lower()

    # 4. Replace non-alphanum runs with a single hyphen.
    slug = _NON_ALPHANUM_RUN.sub("-", folded)

    # 5. Strip leading/trailing hyphens.
    slug = slug.strip("-")

    if not slug:
        raise ValueError(
            f"slugify produced an empty slug from input {text!r}; "
            "input must contain at least one alphanumeric character."
        )

    return slug