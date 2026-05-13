"""Tests for `stats_engine.elo_engine`.

Run with: `pytest tests/test_elo_engine.py -v`.
Design reference: the Elo engine's Notion module page and the
2026-05-13 Decision Log entry.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stats_engine.elo_engine import (
    DEFAULT_BASE,
    DEFAULT_DECAY_PER_YEAR,
    DEFAULT_K,
    EloEngine,
    FightRow,
    _apply_inactivity_decay,
    _expected_score,
    _finish_bonus,
    compute_elo_history,
)


# --- helpers --------------------------------------------------------------

def make_fight(
    fight_id: str = "f1",
    event_date: str = "2024-01-01",
    bout_order: int = 1,
    fighter_a_id: str = "A",
    fighter_b_id: str = "B",
    outcome: str = "a_win",
    method: str = "Decision",
    round_no: int = 3,
    total_rounds: int = 3,
    weight_class: str = "Lightweight",
    bout_type: str = "Regular",
) -> FightRow:
    return FightRow(
        fight_id=fight_id,
        event_date=pd.Timestamp(event_date),
        bout_order=bout_order,
        fighter_a_id=fighter_a_id,
        fighter_b_id=fighter_b_id,
        outcome=outcome,
        method=method,
        round_no=round_no,
        total_rounds=total_rounds,
        weight_class=weight_class,
        bout_type=bout_type,
    )


def make_fight_df(fights: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(fights)


DECISION_FIGHTS: list[dict] = [
    {
        "fight_id": "f1", "event_date": "2020-01-01", "bout_order": 1,
        "fighter_a_id": "A", "fighter_b_id": "B",
        "outcome": "a_win", "method": "Decision",
        "round_no": 3, "total_rounds": 3,
        "weight_class": "LW", "bout_type": "Regular",
    },
    {
        "fight_id": "f2", "event_date": "2020-06-01", "bout_order": 1,
        "fighter_a_id": "C", "fighter_b_id": "A",
        "outcome": "b_win", "method": "Submission",
        "round_no": 2, "total_rounds": 3,
        "weight_class": "LW", "bout_type": "Regular",
    },
    {
        "fight_id": "f3", "event_date": "2021-01-01", "bout_order": 1,
        "fighter_a_id": "A", "fighter_b_id": "D",
        "outcome": "draw", "method": "Draw",
        "round_no": 3, "total_rounds": 3,
        "weight_class": "LW", "bout_type": "Regular",
    },
]


# --- pure helper unit tests -----------------------------------------------

def test_expected_score_equal_ratings_is_half() -> None:
    assert _expected_score(1500, 1500) == pytest.approx(0.5)


def test_expected_score_is_zero_sum() -> None:
    a = _expected_score(1700, 1500)
    b = _expected_score(1500, 1700)
    assert a > 0.5 > b
    assert a + b == pytest.approx(1.0)


def test_finish_bonus_non_finish_methods_are_one() -> None:
    assert _finish_bonus("Decision", 3, 3) == 1.0
    assert _finish_bonus("DQ", 1, 5) == 1.0
    assert _finish_bonus("Draw", 3, 3) == 1.0
    assert _finish_bonus("NC", 1, 3) == 1.0


def test_finish_bonus_ko_r1_three_round() -> None:
    assert _finish_bonus("KO/TKO", 1, 3) == pytest.approx(1.20)


def test_finish_bonus_ko_r1_five_round() -> None:
    assert _finish_bonus("Submission", 1, 5) == pytest.approx(1.20)


def test_finish_bonus_last_round_is_floor() -> None:
    assert _finish_bonus("KO/TKO", 3, 3) == pytest.approx(1.10)
    assert _finish_bonus("Submission", 5, 5) == pytest.approx(1.10)


def test_finish_bonus_middle_round_interpolates() -> None:
    # R3 of a 5R fight: fraction_remaining = (5-3)/(5-1) = 0.5
    # bonus = 1.10 + 0.10 * 0.5 = 1.15
    assert _finish_bonus("KO/TKO", 3, 5) == pytest.approx(1.15)


def test_finish_bonus_pre_1999_single_round() -> None:
    assert _finish_bonus("KO/TKO", 1, 1) == pytest.approx(1.20)
    assert _finish_bonus("Submission", 1, 1) == pytest.approx(1.20)


def test_inactivity_decay_compounds_toward_base() -> None:
    expected = 1500.0 + 200.0 * (0.99 ** 5)
    got = _apply_inactivity_decay(
        1700.0, 5.0, DEFAULT_DECAY_PER_YEAR, DEFAULT_BASE
    )
    assert got == pytest.approx(expected)
    assert got == pytest.approx(1690.197, abs=0.01)


def test_inactivity_decay_no_time_is_noop() -> None:
    assert _apply_inactivity_decay(
        1700.0, 0.0, DEFAULT_DECAY_PER_YEAR, DEFAULT_BASE
    ) == 1700.0


def test_inactivity_decay_below_base_returns_toward_base() -> None:
    # A 1400-rated fighter who took 10 years off should drift up toward 1500.
    got = _apply_inactivity_decay(
        1400.0, 10.0, DEFAULT_DECAY_PER_YEAR, DEFAULT_BASE
    )
    assert 1400.0 < got < DEFAULT_BASE


# --- engine: per-fight ----------------------------------------------------

def test_symmetric_updates_equal_ratings() -> None:
    engine = EloEngine()
    result = engine.process_fight(make_fight())
    assert result.delta_a + result.delta_b == pytest.approx(0.0)
    assert result.delta_a > 0
    assert result.delta_b < 0


def test_decision_baseline_delta() -> None:
    """A=1600 beats B=1500 by decision, K=32 ==> A gains approx 11.518."""
    engine = EloEngine()
    engine.seed_rating("A", 1600.0, pd.Timestamp("2024-01-01"))
    engine.seed_rating("B", 1500.0, pd.Timestamp("2024-01-01"))

    result = engine.process_fight(make_fight(event_date="2024-01-01"))

    expected_score_a = _expected_score(1600.0, 1500.0)
    expected_delta = DEFAULT_K * (1.0 - expected_score_a)
    assert result.delta_a == pytest.approx(expected_delta)
    assert result.delta_a == pytest.approx(11.518, abs=0.001)


def test_finish_bonus_r1_is_1_20_times_decision_delta() -> None:
    decision = EloEngine().process_fight(
        make_fight(method="Decision", round_no=3, total_rounds=3)
    )
    finish = EloEngine().process_fight(
        make_fight(method="KO/TKO", round_no=1, total_rounds=3)
    )
    assert finish.delta_a == pytest.approx(decision.delta_a * 1.20)


def test_finish_bonus_last_round_is_1_10_times_decision_delta() -> None:
    decision = EloEngine().process_fight(
        make_fight(method="Decision", round_no=3, total_rounds=3)
    )
    finish = EloEngine().process_fight(
        make_fight(method="KO/TKO", round_no=3, total_rounds=3)
    )
    assert finish.delta_a == pytest.approx(decision.delta_a * 1.10)


def test_draw_zero_delta_but_state_advances() -> None:
    engine = EloEngine()
    result = engine.process_fight(make_fight(outcome="draw", method="Draw"))
    assert result.delta_a == 0.0
    assert result.delta_b == 0.0
    assert result.elo_a_before == result.elo_a_after == DEFAULT_BASE
    assert result.elo_b_before == result.elo_b_after == DEFAULT_BASE
    # Fight count and last-fight-date should still update so a follow-up
    # bout has a starting point.
    current = engine.current_ratings().set_index("fighter_id")
    assert current.loc["A", "fight_count"] == 1
    assert current.loc["B", "fight_count"] == 1


def test_nc_zero_delta() -> None:
    result = EloEngine().process_fight(make_fight(outcome="nc", method="NC"))
    assert result.delta_a == 0.0
    assert result.delta_b == 0.0


def test_first_fight_starts_at_base() -> None:
    result = EloEngine().process_fight(make_fight())
    assert result.elo_a_before == DEFAULT_BASE
    assert result.elo_b_before == DEFAULT_BASE


def test_inactivity_decay_applied_pre_fight() -> None:
    """After a winning first fight, then a 5-year layoff, the fighter's
    pre-fight rating in their second bout should be decayed toward 1500."""
    engine = EloEngine()
    engine.process_fight(make_fight(fight_id="f1", event_date="2019-01-01"))
    rating_after_first = (
        engine.current_ratings().set_index("fighter_id").loc["A", "rating"]
    )
    assert rating_after_first > DEFAULT_BASE

    second = engine.process_fight(make_fight(
        fight_id="f2",
        event_date="2024-01-01",
        fighter_a_id="A",
        fighter_b_id="C",   # fresh opponent
    ))
    days = (pd.Timestamp("2024-01-01") - pd.Timestamp("2019-01-01")).days
    years = days / 365.25
    decayed = DEFAULT_BASE + (rating_after_first - DEFAULT_BASE) * (
        (1 - DEFAULT_DECAY_PER_YEAR) ** years
    )
    assert second.elo_a_before == pytest.approx(decayed, rel=1e-6)
    assert second.elo_a_before < rating_after_first


# --- engine: batch / determinism ------------------------------------------

def test_process_fights_returns_two_rows_per_fight() -> None:
    long_df = EloEngine().process_fights(make_fight_df(DECISION_FIGHTS))
    assert len(long_df) == 2 * len(DECISION_FIGHTS)


def test_process_fights_emits_draw_rows_with_zero_delta() -> None:
    long_df = EloEngine().process_fights(make_fight_df(DECISION_FIGHTS))
    draw_rows = long_df[long_df["fight_id"] == "f3"]
    assert len(draw_rows) == 2
    assert (draw_rows["elo_delta"] == 0.0).all()
    assert (draw_rows["result"] == "draw").all()


def test_process_fights_is_deterministic_under_shuffle() -> None:
    a = EloEngine().process_fights(make_fight_df(DECISION_FIGHTS))
    b = EloEngine().process_fights(
        make_fight_df(list(reversed(DECISION_FIGHTS)))
    )
    pd.testing.assert_frame_equal(a, b)


def test_same_day_tournament_fights_chain() -> None:
    """Same-day fights ordered by bout_order should accumulate Elo with
    no spurious inactivity decay between them."""
    fights = [
        {
            "fight_id": "t1", "event_date": "1994-03-11", "bout_order": 1,
            "fighter_a_id": "A", "fighter_b_id": "X",
            "outcome": "a_win", "method": "KO/TKO",
            "round_no": 1, "total_rounds": 1,
            "weight_class": "OW", "bout_type": "Regular",
        },
        {
            "fight_id": "t2", "event_date": "1994-03-11", "bout_order": 2,
            "fighter_a_id": "A", "fighter_b_id": "Y",
            "outcome": "a_win", "method": "Submission",
            "round_no": 1, "total_rounds": 1,
            "weight_class": "OW", "bout_type": "Regular",
        },
    ]
    long_df = EloEngine().process_fights(make_fight_df(fights))
    a_rows = long_df[long_df["fighter_id"] == "A"].sort_values("fight_no")
    # The second fight starts from where the first ended (no decay between).
    assert a_rows.iloc[1]["elo_before"] == pytest.approx(a_rows.iloc[0]["elo_after"])


# --- compute_elo_history: full pipeline ----------------------------------

def test_compute_elo_history_returns_three_dataframes() -> None:
    long_df, wide_df, peak_df = compute_elo_history(make_fight_df(DECISION_FIGHTS))
    assert len(long_df) == 2 * len(DECISION_FIGHTS)
    assert len(wide_df) == len(DECISION_FIGHTS)
    assert set(peak_df["fighter_id"]) == {"A", "B", "C", "D"}


def test_compute_elo_history_wide_is_zero_sum() -> None:
    """delta_a + delta_b == 0 on every decisive row of wide."""
    _, wide_df, _ = compute_elo_history(make_fight_df(DECISION_FIGHTS))
    decisive = wide_df[~wide_df["outcome"].isin(["draw", "nc"])]
    assert (decisive["delta_a"] + decisive["delta_b"]).abs().max() == pytest.approx(0.0)


def test_compute_elo_history_peak_matches_max_elo_after() -> None:
    long_df, _, peak_df = compute_elo_history(make_fight_df(DECISION_FIGHTS))
    for fid in peak_df["fighter_id"]:
        fighter_rows = long_df[long_df["fighter_id"] == fid]
        peak_row = peak_df.loc[peak_df["fighter_id"] == fid, "peak_elo"].iloc[0]
        assert peak_row == pytest.approx(fighter_rows["elo_after"].max())