"""Tests for data.cleaning.fight_results_cleaner (v0.2.0)."""

from __future__ import annotations

import pandas as pd
import pytest

from data.cleaning.fight_results_cleaner import (
    OUTPUT_COLUMNS,
    REQUIRED_EVENT_DETAILS_COLUMNS,
    REQUIRED_FIGHT_RESULTS_COLUMNS,
    REQUIRED_WEIGHTCLASS_COLUMNS,
    _build_fight_id,
    _map_bout_type,
    _map_method,
    _map_outcome,
    clean_fight_results,
)


# ---------------------------------------------------------------------------
# Helpers to build minimal test inputs
# ---------------------------------------------------------------------------
def _make_fight_results(rows):
    return pd.DataFrame(rows, columns=REQUIRED_FIGHT_RESULTS_COLUMNS)


def _make_event_details(rows):
    return pd.DataFrame(rows, columns=REQUIRED_EVENT_DETAILS_COLUMNS)


def _make_weightclass(rows):
    return pd.DataFrame(rows, columns=REQUIRED_WEIGHTCLASS_COLUMNS)


def _make_lookup(rows):
    """Lookup shape per attach_bout_fighter_ids: name, fighter_id, match_type, score."""
    return pd.DataFrame(rows, columns=["name", "fighter_id", "match_type", "score"])


def _happy_inputs():
    """Construct a valid 3-event / 5-bout test frame."""
    fr = _make_fight_results([
        # Event A: 2 bouts (row 0 = main, row 1 = prelim)
        {"EVENT": "UFC Test 1: A vs B", "BOUT": "Fighter Alpha vs. Fighter Bravo",
         "OUTCOME": "W/L", "METHOD": "KO/TKO", "ROUND": 2,
         "TIME": "3:45", "TIME FORMAT": "5 Rnd (5-5-5-5-5)",
         "REFEREE": "Ref One", "DETAILS": "Punch"},
        {"EVENT": "UFC Test 1: A vs B", "BOUT": "Fighter Charlie vs. Fighter Delta",
         "OUTCOME": "L/W", "METHOD": "Submission", "ROUND": 1,
         "TIME": "4:20", "TIME FORMAT": "3 Rnd (5-5-5)",
         "REFEREE": "Ref Two", "DETAILS": "RNC"},
        # Event B: 2 bouts, draws and NCs
        {"EVENT": "UFC Test 2: C vs D", "BOUT": "Fighter Echo vs. Fighter Foxtrot",
         "OUTCOME": "D/D", "METHOD": "Decision - Majority", "ROUND": 3,
         "TIME": "5:00", "TIME FORMAT": "3 Rnd (5-5-5)",
         "REFEREE": "Ref Three", "DETAILS": "Draw"},
        {"EVENT": "UFC Test 2: C vs D", "BOUT": "Fighter Golf vs. Fighter Hotel",
         "OUTCOME": "NC/NC", "METHOD": "No Contest", "ROUND": 2,
         "TIME": "1:30", "TIME FORMAT": "3 Rnd (5-5-5)",
         "REFEREE": "Ref Four", "DETAILS": "Illegal knee"},
        # Event C: 1 bout, title fight
        {"EVENT": "UFC Test 3: Title", "BOUT": "Fighter India vs. Fighter Juliet",
         "OUTCOME": "W/L", "METHOD": "Decision - Unanimous", "ROUND": 5,
         "TIME": "5:00", "TIME FORMAT": "5 Rnd (5-5-5-5-5)",
         "REFEREE": "Ref Five", "DETAILS": "50-45 50-45 50-45"},
    ])
    ed = _make_event_details([
        {"event_name": "UFC Test 1: A vs B", "event_date": "2025-01-15"},
        {"event_name": "UFC Test 2: C vs D", "event_date": "2025-02-20"},
        {"event_name": "UFC Test 3: Title",  "event_date": "2025-03-10"},
    ])
    wc = _make_weightclass([
        {"EVENT": "UFC Test 1: A vs B", "BOUT": "Fighter Alpha vs. Fighter Bravo",
         "weight_class": "Lightweight", "title_type": "Regular",
         "is_womens": False, "is_tournament": False},
        {"EVENT": "UFC Test 1: A vs B", "BOUT": "Fighter Charlie vs. Fighter Delta",
         "weight_class": "Welterweight", "title_type": "Regular",
         "is_womens": False, "is_tournament": False},
        {"EVENT": "UFC Test 2: C vs D", "BOUT": "Fighter Echo vs. Fighter Foxtrot",
         "weight_class": "Featherweight", "title_type": "Regular",
         "is_womens": False, "is_tournament": False},
        {"EVENT": "UFC Test 2: C vs D", "BOUT": "Fighter Golf vs. Fighter Hotel",
         "weight_class": "Bantamweight", "title_type": "Regular",
         "is_womens": False, "is_tournament": False},
        {"EVENT": "UFC Test 3: Title", "BOUT": "Fighter India vs. Fighter Juliet",
         "weight_class": "Middleweight", "title_type": "Title",
         "is_womens": False, "is_tournament": False},
    ])
    lookup = _make_lookup([
        {"name": "fighter alpha",   "fighter_id": 1,  "match_type": "exact", "score": 100},
        {"name": "fighter bravo",   "fighter_id": 2,  "match_type": "exact", "score": 100},
        {"name": "fighter charlie", "fighter_id": 3,  "match_type": "exact", "score": 100},
        {"name": "fighter delta",   "fighter_id": 4,  "match_type": "exact", "score": 100},
        {"name": "fighter echo",    "fighter_id": 5,  "match_type": "exact", "score": 100},
        {"name": "fighter foxtrot", "fighter_id": 6,  "match_type": "exact", "score": 100},
        {"name": "fighter golf",    "fighter_id": 7,  "match_type": "exact", "score": 100},
        {"name": "fighter hotel",   "fighter_id": 8,  "match_type": "exact", "score": 100},
        {"name": "fighter india",   "fighter_id": 9,  "match_type": "exact", "score": 100},
        {"name": "fighter juliet",  "fighter_id": 10, "match_type": "exact", "score": 100},
    ])
    return fr, ed, wc, lookup


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_happy_path_produces_expected_columns():
    fr, ed, wc, lookup = _happy_inputs()
    out = clean_fight_results(fr, ed, wc, lookup)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert len(out) == 5


def test_happy_path_canonical_values():
    fr, ed, wc, lookup = _happy_inputs()
    out = clean_fight_results(fr, ed, wc, lookup)
    title_row = out[out["EVENT"] == "UFC Test 3: Title"].iloc[0]
    assert title_row["fight_id"] == "ufc-test-3-title__fighter-india-vs-fighter-juliet__1"
    assert title_row["fight_seq"] == 1
    assert title_row["event_date"] == "2025-03-10"
    assert title_row["outcome"] == "a_win"
    assert title_row["method"] == "Decision"
    assert title_row["bout_type"] == "title"
    assert title_row["round_no"] == 5
    assert title_row["total_rounds"] == 5
    assert title_row["weight_class"] == "Middleweight"


# ---------------------------------------------------------------------------
# _map_outcome
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("W/L",     "a_win"),
    ("L/W",     "b_win"),
    ("D/D",     "draw"),
    ("NC/NC",   "nc"),
    ("NC",      "nc"),
    (None,      "nc"),
    ("unknown", "nc"),
    ("garbage", "nc"),
])
def test_map_outcome(raw, expected):
    assert _map_outcome(raw) == expected


# ---------------------------------------------------------------------------
# _map_method
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bucket, outcome, expected", [
    ("ko_tko",     "a_win", "KO/TKO"),
    ("submission", "b_win", "Submission"),
    ("decision",   "a_win", "Decision"),
    ("dq",         "a_win", "DQ"),
    ("no_contest", "nc",    "NC"),
    ("other",      "a_win", "Decision"),  # coerced
    ("ko_tko",     "draw",  "Draw"),       # draw override
    ("submission", "draw",  "Draw"),       # draw override
])
def test_map_method(bucket, outcome, expected):
    assert _map_method(bucket, outcome) == expected


# ---------------------------------------------------------------------------
# _map_bout_type
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("title_type, expected", [
    # Lowercase forms (real weightclass_cleaned.csv values)
    ("title",          "title"),
    ("interim",        "interim_title"),
    ("interim_title",  "interim_title"),
    ("interim title",  "interim_title"),
    ("regular",        "non_title"),
    # Capitalised forms (defensive, case-insensitive lookup)
    ("Title",          "title"),
    ("Interim Title",  "interim_title"),
    ("Regular",        "non_title"),
    # NA + unknown -> non_title
    (None,             "non_title"),
    ("Unknown String", "non_title"),
])
def test_map_bout_type(title_type, expected):
    assert _map_bout_type(title_type) == expected


# ---------------------------------------------------------------------------
# _build_fight_id
# ---------------------------------------------------------------------------
def test_build_fight_id_basic():
    assert _build_fight_id("UFC 314", "Volkanovski vs. Lopes") == \
        "ufc-314__volkanovski-vs-lopes"


def test_build_fight_id_apostrophes_drop():
    assert _build_fight_id("UFC 281: Adesanya vs. Pereira",
                           "Sean O'Malley vs. Petr Yan") == \
        "ufc-281-adesanya-vs-pereira__sean-omalley-vs-petr-yan"


# ---------------------------------------------------------------------------
# total_rounds
# ---------------------------------------------------------------------------
def test_total_rounds_derived_from_bout_length():
    fr, ed, wc, lookup = _happy_inputs()
    out = clean_fight_results(fr, ed, wc, lookup)
    three_rd = out[out["bout_length_minutes"] == 15]
    five_rd  = out[out["bout_length_minutes"] == 25]
    assert (three_rd["total_rounds"] == 3).all()
    assert (five_rd["total_rounds"] == 5).all()


# ---------------------------------------------------------------------------
# bout_order
# ---------------------------------------------------------------------------
def test_bout_order_reverse_within_event():
    fr, ed, wc, lookup = _happy_inputs()
    out = clean_fight_results(fr, ed, wc, lookup)
    event_a = out[out["EVENT"] == "UFC Test 1: A vs B"].sort_values("bout_order", ascending=False)
    assert event_a.iloc[0]["BOUT"] == "Fighter Alpha vs. Fighter Bravo"  # main
    assert event_a.iloc[0]["bout_order"] == 2
    assert event_a.iloc[1]["bout_order"] == 1


def test_bout_order_independent_per_event():
    fr, ed, wc, lookup = _happy_inputs()
    out = clean_fight_results(fr, ed, wc, lookup)
    a_orders = sorted(out[out["EVENT"] == "UFC Test 1: A vs B"]["bout_order"].tolist())
    b_orders = sorted(out[out["EVENT"] == "UFC Test 2: C vs D"]["bout_order"].tolist())
    c_orders = sorted(out[out["EVENT"] == "UFC Test 3: Title"]["bout_order"].tolist())
    assert a_orders == [1, 2]
    assert b_orders == [1, 2]
    assert c_orders == [1]


# ---------------------------------------------------------------------------
# event_date merge
# ---------------------------------------------------------------------------
def test_event_date_merged_correctly():
    fr, ed, wc, lookup = _happy_inputs()
    out = clean_fight_results(fr, ed, wc, lookup)
    a_dates = out[out["EVENT"] == "UFC Test 1: A vs B"]["event_date"].unique()
    assert list(a_dates) == ["2025-01-15"]


def test_event_date_missing_event_raises():
    fr, ed, wc, lookup = _happy_inputs()
    ed_short = ed[ed["event_name"] != "UFC Test 3: Title"].reset_index(drop=True)
    with pytest.raises(ValueError, match="no event_date after merge"):
        clean_fight_results(fr, ed_short, wc, lookup)


# ---------------------------------------------------------------------------
# fight_id uniqueness via fight_seq
# ---------------------------------------------------------------------------
def test_same_event_same_bout_gets_unique_fight_seq():
    """Tournament-era same-night rematch (Sakuraba vs. Silveira at UFC Ultimate
    Japan): same EVENT + same BOUT must produce distinct fight_seq values and
    distinct fight_ids with __1 / __2 suffixes."""
    fr, ed, wc, lookup = _happy_inputs()
    dup_fr = pd.concat([fr, fr.iloc[[0]]], ignore_index=True)
    out = clean_fight_results(dup_fr, ed, wc, lookup)
    rematches = out[out["BOUT"] == "Fighter Alpha vs. Fighter Bravo"]
    assert len(rematches) == 2
    assert sorted(rematches["fight_seq"].tolist()) == [1, 2]
    assert rematches["fight_id"].nunique() == 2
    # fight_ids should end in __1 and __2 respectively
    suffixes = {fid.rsplit("__", 1)[1] for fid in rematches["fight_id"]}
    assert suffixes == {"1", "2"}


def test_all_unique_bouts_get_fight_seq_1():
    """In the absence of same-night rematches, every fight_seq is 1
    (the consistent always-suffix decision means __1 on every clean row)."""
    fr, ed, wc, lookup = _happy_inputs()
    out = clean_fight_results(fr, ed, wc, lookup)
    assert (out["fight_seq"] == 1).all()
    assert out["fight_id"].str.endswith("__1").all()


def test_rematches_at_different_events_unique_fight_ids():
    """Same bout string at different events -> different fight_ids (rematch case)."""
    fr = _make_fight_results([
        {"EVENT": "UFC 257", "BOUT": "McGregor vs. Poirier",
         "OUTCOME": "L/W", "METHOD": "KO/TKO", "ROUND": 2,
         "TIME": "2:32", "TIME FORMAT": "5 Rnd (5-5-5-5-5)",
         "REFEREE": "R", "DETAILS": "D"},
        {"EVENT": "UFC 264", "BOUT": "McGregor vs. Poirier",
         "OUTCOME": "L/W", "METHOD": "TKO - Doctor's Stoppage", "ROUND": 1,
         "TIME": "5:00", "TIME FORMAT": "5 Rnd (5-5-5-5-5)",
         "REFEREE": "R", "DETAILS": "D"},
    ])
    ed = _make_event_details([
        {"event_name": "UFC 257", "event_date": "2021-01-23"},
        {"event_name": "UFC 264", "event_date": "2021-07-10"},
    ])
    wc = _make_weightclass([
        {"EVENT": "UFC 257", "BOUT": "McGregor vs. Poirier",
         "weight_class": "Lightweight", "title_type": "Regular",
         "is_womens": False, "is_tournament": False},
        {"EVENT": "UFC 264", "BOUT": "McGregor vs. Poirier",
         "weight_class": "Lightweight", "title_type": "Regular",
         "is_womens": False, "is_tournament": False},
    ])
    lookup = _make_lookup([
        {"name": "mcgregor", "fighter_id": 1, "match_type": "exact", "score": 100},
        {"name": "poirier",  "fighter_id": 2, "match_type": "exact", "score": 100},
    ])
    out = clean_fight_results(fr, ed, wc, lookup)
    assert len(out) == 2
    assert out["fight_id"].nunique() == 2  # different events -> different keys
    # Both rematches are seq=1 within their respective events.
    assert (out["fight_seq"] == 1).all()


# ---------------------------------------------------------------------------
# Required-column validation
# ---------------------------------------------------------------------------
def test_missing_fight_results_column_raises():
    fr, ed, wc, lookup = _happy_inputs()
    fr_bad = fr.drop(columns=["METHOD"])
    with pytest.raises(ValueError, match="fight_results is missing required columns"):
        clean_fight_results(fr_bad, ed, wc, lookup)


def test_missing_event_details_column_raises():
    fr, ed, wc, lookup = _happy_inputs()
    ed_bad = ed.drop(columns=["event_date"])
    with pytest.raises(ValueError, match="event_details is missing required columns"):
        clean_fight_results(fr, ed_bad, wc, lookup)


def test_missing_weightclass_column_raises():
    fr, ed, wc, lookup = _happy_inputs()
    wc_bad = wc.drop(columns=["title_type"])
    with pytest.raises(ValueError, match="weightclass is missing required columns"):
        clean_fight_results(fr, ed, wc_bad, lookup)


# ---------------------------------------------------------------------------
# Trace column preservation
# ---------------------------------------------------------------------------
def test_trace_columns_preserved():
    fr, ed, wc, lookup = _happy_inputs()
    out = clean_fight_results(fr, ed, wc, lookup)
    for col in ["EVENT", "BOUT", "fighter_a", "fighter_b",
                "winner_id", "loser_id", "method_bucket", "method_detail",
                "time_seconds", "bout_length_minutes", "referee", "details"]:
        assert col in out.columns, f"Trace column {col!r} missing"


# ---------------------------------------------------------------------------
# round -> round_no rename
# ---------------------------------------------------------------------------
def test_round_no_present_round_absent():
    fr, ed, wc, lookup = _happy_inputs()
    out = clean_fight_results(fr, ed, wc, lookup)
    assert "round_no" in out.columns
    assert "round" not in out.columns


# ---------------------------------------------------------------------------
# Draw method override
# ---------------------------------------------------------------------------
def test_draw_method_overrides_bucket():
    fr, ed, wc, lookup = _happy_inputs()
    out = clean_fight_results(fr, ed, wc, lookup)
    draw_row = out[out["outcome"] == "draw"].iloc[0]
    assert draw_row["method"] == "Draw"
    # The underlying bucket should still be "decision" (preserved as trace)
    assert draw_row["method_bucket"] == "decision"


# ---------------------------------------------------------------------------
# Defensive whitespace stripping on merge keys
# ---------------------------------------------------------------------------
def test_trailing_whitespace_on_event_does_not_break_merge():
    """Regression: upstream ufc_fight_results.csv had trailing spaces on EVENT
    that silently broke the event_date join. Cleaner must strip merge keys."""
    fr, ed, wc, lookup = _happy_inputs()
    # Inject trailing whitespace on EVENT and BOUT in the raw fight_results frame.
    fr["EVENT"] = fr["EVENT"] + "  "
    fr["BOUT"]  = fr["BOUT"]  + " "
    # event_details and weightclass are stripped already (clean upstream),
    # so the merge will only succeed if the cleaner strips fight_results too.
    out = clean_fight_results(fr, ed, wc, lookup)
    assert len(out) == 5
    assert out["event_date"].notna().all()
    # And the stripped EVENT/BOUT should be what's preserved in the trace columns.
    assert (out["EVENT"] == "UFC Test 1: A vs B").sum() == 2