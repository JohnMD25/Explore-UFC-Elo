"""
weight_class_cleaner.py
=========================================================================
Normalise raw UFC WEIGHTCLASS strings into four orthogonal fields:
    weight_class   -- canonical division (lowercase snake_case)
    title_type     -- "regular" | "title" | "interim"
    is_womens      -- bool
    is_tournament  -- bool (TUF finals, Road to UFC, pre-weight-class UFC)

Examples:
    "UFC Middleweight Title Bout"                            -> ("middleweight",      "title",   False, False)
    "Bantamweight Bout"                                      -> ("bantamweight",      "regular", False, False)
    "Women's Strawweight Title Bout"                         -> ("strawweight",       "title",   True,  False)
    "UFC Interim Lightweight Title Bout"                     -> ("lightweight",       "interim", False, False)
    "Open Weight Bout"                                       -> ("open_weight",       "regular", False, False)
    "Catch Weight 165 lbs Bout"                              -> ("catch_weight",      "regular", False, False)
    "UFC Light Heavyweight Title Bout"                       -> ("light_heavyweight", "title",   False, False)
    "Ultimate Fighter 14 Bantamweight Tournament Title Bout" -> ("bantamweight",      "title",   False, True)
    "UFC 3 Tournament Title Bout"                            -> ("open_weight",       "title",   False, True)
    "UFC Superfight Championship Bout"                       -> ("open_weight",       "title",   False, False)
    "Road to UFC 3 Flyweight Tournament TitleBout"           -> ("flyweight",         "title",   False, True)

Usage (run from project root, standalone inspection):
    python data/weight_class_cleaner.py --source local
    python data/weight_class_cleaner.py --source url
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Literal

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UPSTREAM_URL = (
    "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/"
    "ufc_fight_results.csv"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_RAW    = PROJECT_ROOT / "data" / "raw" / "ufc_stats" / "ufc_fight_results.csv"
DEFAULT_OUT  = PROJECT_ROOT / "data" / "cleaned" / "ufc" / "weightclass_cleaned.csv"

# Canonical divisions, lightest to heaviest. Useful for ordering UI axes.
CANONICAL_WEIGHT_CLASSES: tuple[str, ...] = (
    "strawweight",
    "flyweight",
    "bantamweight",
    "featherweight",
    "lightweight",
    "super_lightweight",
    "welterweight",
    "super_welterweight",
    "middleweight",
    "super_middleweight",
    "light_heavyweight",
    "heavyweight",
    "super_heavyweight",
    "open_weight",
    "catch_weight",
)

# Map "phrase as it appears in the raw string, lowercased" -> canonical key.
# Substring search is used, with longest-match-wins ordering at parse time so
# "super lightweight" is tried before "lightweight" and "light heavyweight"
# before "heavyweight".
_PHRASE_TO_CANONICAL: dict[str, str] = {
    "strawweight":        "strawweight",
    "flyweight":          "flyweight",
    "bantamweight":       "bantamweight",
    "featherweight":      "featherweight",
    "lightweight":        "lightweight",
    "super lightweight":  "super_lightweight",
    "welterweight":       "welterweight",
    "super welterweight": "super_welterweight",
    "middleweight":       "middleweight",
    "super middleweight": "super_middleweight",
    "light heavyweight":  "light_heavyweight",
    "heavyweight":        "heavyweight",
    "super heavyweight":  "super_heavyweight",
}

# Pre-sorted at module load time, longest first, so we don't re-sort per row.
_PHRASES_LONGEST_FIRST: tuple[str, ...] = tuple(
    sorted(_PHRASE_TO_CANONICAL.keys(), key=len, reverse=True)
)


# ---------------------------------------------------------------------------
# Single-string parser
# ---------------------------------------------------------------------------
def clean_weight_class(raw: object) -> tuple[object, object, object, object]:
    """
    Parse one raw WEIGHTCLASS string into
    (weight_class, title_type, is_womens, is_tournament).

    Returns (pd.NA, pd.NA, pd.NA, pd.NA) if input is null/blank.
    Returns ("unknown:<raw>", title_type, is_womens, is_tournament) if no
    canonical division is found, so unmapped strings surface loudly instead
    of being silently nulled.
    """
    if pd.isna(raw) or str(raw).strip() == "":
        return (pd.NA, pd.NA, pd.NA, pd.NA)

    raw_str = str(raw).strip()
    s = raw_str.lower()

    # Pre-normalise upstream typo: "TitleBout" -> "Title Bout"
    # Seen on 3 "Road to UFC 3 ... Tournament TitleBout" rows.
    s = s.replace("titlebout", "title bout")

    # Flag detection (run on the full string, BEFORE weight-class matching)
    is_womens     = bool(re.search(r"\bwomen'?s?\b", s))
    is_interim    = "interim" in s
    is_title      = ("title" in s) or ("championship" in s)
    is_tournament = "tournament" in s

    if is_interim:
        title_type = "interim"
    elif is_title:
        title_type = "title"
    else:
        title_type = "regular"

    # Weight-class detection: substring search with longest-match-wins.
    # This is robust to any event prefix or modifier ordering:
    #   "UFC Middleweight Title Bout"                          -> middleweight
    #   "Ultimate Fighter 14 Bantamweight Tournament Title Bout" -> bantamweight
    #   "Road To UFC 1 Featherweight Tournament Title Bout"    -> featherweight
    weight_class: object = None
    for phrase in _PHRASES_LONGEST_FIRST:
        if re.search(rf"\b{re.escape(phrase)}\b", s):
            weight_class = _PHRASE_TO_CANONICAL[phrase]
            break

    if weight_class is None:
        # Catch/open weight may carry suffixes like "165 lbs" — substring OK.
        if "catch weight" in s or "catchweight" in s:
            weight_class = "catch_weight"
        elif "open weight" in s or "openweight" in s:
            weight_class = "open_weight"
        elif is_tournament or "superfight" in s:
            # Pre-weight-class UFC tournaments (UFC 2-17) + the early UFC
            # Superfight title both predated formal weight classes.
            # Historically these were open-weight bouts.
            weight_class = "open_weight"
        else:
            weight_class = f"unknown:{raw_str}"

    return (weight_class, title_type, is_womens, is_tournament)


# ---------------------------------------------------------------------------
# DataFrame transform
# ---------------------------------------------------------------------------
def clean_weight_class_column(
    df: pd.DataFrame, col: str = "WEIGHTCLASS"
) -> pd.DataFrame:
    """Return a copy of `df` with the four parsed columns added."""
    parsed = df[col].apply(clean_weight_class)
    out = df.copy()
    out["weight_class"]  = parsed.map(lambda t: t[0])
    out["title_type"]    = parsed.map(lambda t: t[1])
    out["is_womens"]     = parsed.map(lambda t: t[2])
    out["is_tournament"] = parsed.map(lambda t: t[3])
    return out


# ---------------------------------------------------------------------------
# Standalone CLI for inspection
# ---------------------------------------------------------------------------
def _load_fight_results(source: Literal["url", "local"]) -> pd.DataFrame:
    if source == "url":
        return pd.read_csv(UPSTREAM_URL)
    if not LOCAL_RAW.exists():
        raise FileNotFoundError(
            f"Local raw file missing: {LOCAL_RAW}\nUse --source url, or curl it."
        )
    return pd.read_csv(LOCAL_RAW)


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean ufc_fight_results WEIGHTCLASS")
    ap.add_argument("--source", choices=["url", "local"], default="url")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading ufc_fight_results from {args.source}...")
    fr = _load_fight_results(args.source)
    print(f"      {len(fr):,} rows")

    print("[2/3] Cleaning WEIGHTCLASS...")
    cleaned = clean_weight_class_column(fr)

    wc_str = cleaned["weight_class"].astype("string")
    unknowns = cleaned[wc_str.fillna("").str.startswith("unknown:")]
    if len(unknowns):
        print(f"      WARNING: {len(unknowns):,} rows with unmapped WEIGHTCLASS:")
        for val, n in unknowns["WEIGHTCLASS"].value_counts().items():
            print(f"        {n:5,}  {val}")
    else:
        print("      all WEIGHTCLASS values mapped to a canonical class")

    print("      title_type distribution:")
    for v, n in cleaned["title_type"].value_counts(dropna=False).items():
        print(f"        {n:5,}  {v}")
    womens_count     = int(cleaned["is_womens"].fillna(False).sum())
    tournament_count = int(cleaned["is_tournament"].fillna(False).sum())
    print(f"      is_womens:     {womens_count:,} rows")
    print(f"      is_tournament: {tournament_count:,} rows")

    print("      weight_class distribution (top 15):")
    for v, n in cleaned["weight_class"].value_counts(dropna=False).head(15).items():
        print(f"        {n:5,}  {v}")

    keep = [
        "EVENT", "BOUT", "WEIGHTCLASS",
        "weight_class", "title_type", "is_womens", "is_tournament",
    ]
    print(f"[3/3] Writing -> {args.out}")
    cleaned[keep].to_csv(args.out, index=False)


if __name__ == "__main__":
    main()