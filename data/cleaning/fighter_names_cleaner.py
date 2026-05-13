"""
fighter_names_cleaner.py
=========================================================================
Small shared utility for attaching fighter_id columns to any UFC fight
DataFrame, using the canonical name->id lookup produced by
build_fighter_dim.py.

This module does NOT have a CLI. It is imported by:
    - fight_results_cleaner.py   (uses attach_bout_fighter_ids on the BOUT col)
    - fight_stats_cleaner.py     (uses attach_fighter_ids on the FIGHTER col)
    - run_cleaning.py            (orchestrator)

Public API
----------
    load_name_lookup(path: Path | None = None) -> pd.DataFrame
    attach_fighter_ids(df, name_col, lookup=None, id_col=None) -> pd.DataFrame
    attach_bout_fighter_ids(df, bout_col="BOUT", lookup=None) -> pd.DataFrame
    report_unmatched(df, id_cols, label="") -> None
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT       = Path(__file__).resolve().parent.parent.parent
DEFAULT_LOOKUP_CSV = PROJECT_ROOT / "data" / "derived" / "fighter_name_to_id.csv"


def load_name_lookup(path: Path | None = None) -> pd.DataFrame:
    """
    Load the canonical name->id lookup from disk.

    Returns a DataFrame with columns: name, fighter_id, match_type, score.
    The 'name' values are exactly the strings seen in upstream BOUT/FIGHTER
    columns (exact + RapidFuzz-accepted matches).
    """
    p = path or DEFAULT_LOOKUP_CSV
    if not p.exists():
        raise FileNotFoundError(
            f"Name lookup missing: {p}\n"
            "Run `python data/build_fighter_dim.py` first to generate it."
        )
    return pd.read_csv(p)


def attach_fighter_ids(
    df: pd.DataFrame,
    name_col: str,
    lookup: pd.DataFrame | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """
    Add a {id_col} column to df by left-merging on name_col -> lookup.name.

    The new id column is nullable Int64. Rows where name_col can't be matched
    will have <NA> in the id column (caller decides whether to drop them).

    Args:
        df:        DataFrame with a column of fighter-name strings.
        name_col:  Name of the column in df holding the fighter name.
        lookup:    Optional pre-loaded name lookup; otherwise loaded from disk.
        id_col:    Name for the new id column. Defaults to f"{name_col}_id".

    Returns:
        A copy of df with the new id column appended.
    """
    if lookup is None:
        lookup = load_name_lookup()
    id_col = id_col or f"{name_col}_id"

    merged = df.merge(
        lookup[["name", "fighter_id"]],
        how="left",
        left_on=name_col,
        right_on="name",
    )
    if "name" in merged.columns and "name" != name_col:
        merged = merged.drop(columns=["name"])
    merged = merged.rename(columns={"fighter_id": id_col})
    merged[id_col] = merged[id_col].astype("Int64")
    return merged


def attach_bout_fighter_ids(
    df: pd.DataFrame,
    bout_col: str = "BOUT",
    lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Split BOUT column on ' vs. ' into fighter_a/fighter_b name columns,
    then attach fighter_a_id and fighter_b_id via the name lookup.

    Adds four new columns:
        fighter_a       (str)              -- left side of the BOUT string
        fighter_b       (str)              -- right side of the BOUT string
        fighter_a_id    (Int64, nullable)
        fighter_b_id    (Int64, nullable)
    """
    if lookup is None:
        lookup = load_name_lookup()

    out = df.copy()
    split = out[bout_col].astype(str).str.split(" vs. ", n=1, expand=True)
    out["fighter_a"] = split[0].str.strip()
    out["fighter_b"] = (
        split[1].str.strip() if 1 in split.columns else pd.Series(pd.NA, index=out.index)
    )

    out = attach_fighter_ids(out, "fighter_a", lookup, id_col="fighter_a_id")
    out = attach_fighter_ids(out, "fighter_b", lookup, id_col="fighter_b_id")
    return out


def report_unmatched(
    df: pd.DataFrame,
    id_cols: Iterable[str],
    label: str = "",
) -> None:
    """Print a per-id-column count of rows where the id is NA."""
    prefix = f"[{label}] " if label else ""
    for col in id_cols:
        total = len(df)
        n_na = int(df[col].isna().sum())
        pct = (n_na / total * 100) if total else 0
        marker = "OK" if n_na == 0 else "WARNING"
        print(
            f"      {prefix}{col}: {n_na:,} unmatched / {total:,} total "
            f"({pct:.2f}%)  [{marker}]"
        )