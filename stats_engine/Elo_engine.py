"""Compute UFC fighter Elo ratings from cleaned fight results.

Pure compute module: no file IO, no Streamlit, no plotting. The refresh
pipeline calls `compute_elo_history` once per run; `data/loaders.py` consumes
the resulting CSVs. Design locked 2026-05-13 — see the module's Notion page
for the full spec and the Decision Log entry of the same date.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

# --- constants ------------------------------------------------------------

DEFAULT_K: float = 32.0
DEFAULT_BASE: float = 1500.0
DEFAULT_DECAY_PER_YEAR: float = 0.01

OUTCOME_A_WIN = "a_win"
OUTCOME_B_WIN = "b_win"
OUTCOME_DRAW = "draw"
OUTCOME_NC = "nc"
ZERO_DELTA_OUTCOMES = frozenset({OUTCOME_DRAW, OUTCOME_NC})

METHOD_KO_TKO = "KO/TKO"
METHOD_SUBMISSION = "Submission"
FINISH_METHODS = frozenset({METHOD_KO_TKO, METHOD_SUBMISSION})

LONG_COLUMNS = [
    "fight_id", "event_date", "fighter_id", "opponent_id",
    "weight_class", "bout_type", "result", "method",
    "round_no", "total_rounds",
    "opponent_elo_before", "elo_before", "elo_after", "elo_delta",
    "fight_no",
]

WIDE_COLUMNS = [
    "fight_id", "event_date", "fighter_a_id", "fighter_b_id",
    "outcome", "method", "bout_type", "weight_class",
    "round_no", "total_rounds",
    "elo_a_before", "elo_a_after", "elo_b_before", "elo_b_after",
    "delta_a", "delta_b",
]

PEAK_COLUMNS = [
    "fighter_id", "peak_elo", "peak_date",
    "peak_fight_id", "opponent_id", "fight_no_at_peak",
]


# --- data classes ---------------------------------------------------------

@dataclass(frozen=True)
class FightRow:
    """One bout from `fight_results_cleaned.csv`. See the Data Contracts
    page in Notion for the cleaner's full output schema."""
    fight_id: str
    event_date: pd.Timestamp
    bout_order: int
    fighter_a_id: str
    fighter_b_id: str
    outcome: str        # OUTCOME_A_WIN / OUTCOME_B_WIN / OUTCOME_DRAW / OUTCOME_NC
    method: str         # KO/TKO / Submission / Decision / DQ / Draw / NC
    round_no: int       # 1-based
    total_rounds: int   # 1 / 3 / 5
    weight_class: str
    bout_type: str


@dataclass(frozen=True)
class FightEloResult:
    """Output of `EloEngine.process_fight` — both sides of one bout."""
    fight_id: str
    event_date: pd.Timestamp
    fighter_a_id: str
    fighter_b_id: str
    weight_class: str
    bout_type: str
    outcome: str
    method: str
    round_no: int
    total_rounds: int
    elo_a_before: float
    elo_a_after: float
    elo_b_before: float
    elo_b_after: float
    delta_a: float
    delta_b: float
    fight_no_a: int
    fight_no_b: int


# --- private helpers ------------------------------------------------------

def _expected_score(rating_a: float, rating_b: float) -> float:
    """Standard Elo expected score for A given both ratings."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _finish_bonus(method: str, round_no: int, total_rounds: int) -> float:
    """Multiplier applied to the raw Elo delta.

    1.0 for non-finishes (Decision / DQ / Draw / NC). For KO/TKO and
    Submission: linear decay from 1.20x at R1 to a 1.10x floor at the final
    scheduled round. Pre-1999 single-round bouts collapse to a flat 1.20x.
    """
    if method not in FINISH_METHODS:
        return 1.0
    if total_rounds <= 1:
        return 1.20
    fraction_remaining = (total_rounds - round_no) / (total_rounds - 1)
    return 1.10 + 0.10 * fraction_remaining


def _apply_inactivity_decay(
    rating: float,
    years_since_last: float,
    decay_per_year: float,
    base: float,
) -> float:
    """Compounding decay back toward `base`.

    ``rating' = base + (rating - base) * (1 - decay_per_year) ** years``.
    A fighter's first-ever bout has ``years_since_last`` undefined; callers
    must skip decay in that case (handled in `EloEngine._decayed_rating`).
    """
    if years_since_last <= 0:
        return rating
    return base + (rating - base) * ((1.0 - decay_per_year) ** years_since_last)


def _result_label(outcome: str, side: str) -> str:
    """Map (bout outcome, side in {'a','b'}) to that fighter's result label."""
    if outcome == OUTCOME_DRAW:
        return "draw"
    if outcome == OUTCOME_NC:
        return "nc"
    won = (outcome == OUTCOME_A_WIN and side == "a") or (
        outcome == OUTCOME_B_WIN and side == "b"
    )
    return "win" if won else "loss"


# --- engine ---------------------------------------------------------------

class EloEngine:
    """Stateful Elo engine. Feed fights in chronological order via
    `process_fight`, or pass the full cleaned DataFrame to `process_fights`.
    Internal state is private; treat the engine as a black box once seeded.
    """

    def __init__(
        self,
        k: float = DEFAULT_K,
        base: float = DEFAULT_BASE,
        decay_per_year: float = DEFAULT_DECAY_PER_YEAR,
    ) -> None:
        self.k = k
        self.base = base
        self.decay_per_year = decay_per_year
        self._ratings: dict[str, float] = {}
        self._last_fight_date: dict[str, pd.Timestamp] = {}
        self._fight_count: dict[str, int] = {}
        self._peak: dict[str, dict] = {}

    # ---- internal ----

    def _decayed_rating(self, fighter_id: str, event_date: pd.Timestamp) -> float:
        rating = self._ratings.get(fighter_id, self.base)
        last = self._last_fight_date.get(fighter_id)
        if last is None:
            return rating
        years = (event_date - last).days / 365.25
        return _apply_inactivity_decay(
            rating, years, self.decay_per_year, self.base
        )

    def _record_peak(
        self,
        fighter_id: str,
        elo_after: float,
        event_date: pd.Timestamp,
        fight_id: str,
        opponent_id: str,
        fight_no: int,
    ) -> None:
        current = self._peak.get(fighter_id)
        if current is None or elo_after > current["peak_elo"]:
            self._peak[fighter_id] = {
                "peak_elo": elo_after,
                "peak_date": event_date,
                "peak_fight_id": fight_id,
                "opponent_id": opponent_id,
                "fight_no_at_peak": fight_no,
            }

    # ---- public ----

    def seed_rating(
        self,
        fighter_id: str,
        rating: float,
        last_fight_date: pd.Timestamp | None = None,
    ) -> None:
        """Inject a fighter's pre-existing rating without processing a fight.

        Intended for tests and scenario analysis (e.g. "what if Fighter X
        entered the UFC at 1650?"). Pass ``last_fight_date`` if you want
        inactivity decay measured from that point; leave it ``None`` to skip
        decay until the fighter's first real bout. Does not update fight
        count or peak history — those are populated by real bouts only.
        """
        self._ratings[fighter_id] = rating
        if last_fight_date is not None:
            self._last_fight_date[fighter_id] = pd.Timestamp(last_fight_date)

    def process_fight(self, fight: FightRow) -> FightEloResult:
        """Apply inactivity decay → compute raw delta → apply finish bonus
        → commit state. Returns both fighters' before/after ratings."""
        rating_a_before = self._decayed_rating(fight.fighter_a_id, fight.event_date)
        rating_b_before = self._decayed_rating(fight.fighter_b_id, fight.event_date)

        if fight.outcome in ZERO_DELTA_OUTCOMES:
            delta_a = 0.0
            delta_b = 0.0
        else:
            score_a = 1.0 if fight.outcome == OUTCOME_A_WIN else 0.0
            expected_a = _expected_score(rating_a_before, rating_b_before)
            raw_delta_a = self.k * (score_a - expected_a)
            bonus = _finish_bonus(
                fight.method, fight.round_no, fight.total_rounds
            )
            delta_a = raw_delta_a * bonus
            delta_b = -delta_a

        rating_a_after = rating_a_before + delta_a
        rating_b_after = rating_b_before + delta_b

        # Commit state only after both sides are computed so the second read
        # isn't polluted by the first write.
        self._ratings[fight.fighter_a_id] = rating_a_after
        self._ratings[fight.fighter_b_id] = rating_b_after
        self._last_fight_date[fight.fighter_a_id] = fight.event_date
        self._last_fight_date[fight.fighter_b_id] = fight.event_date
        fight_no_a = self._fight_count.get(fight.fighter_a_id, 0) + 1
        fight_no_b = self._fight_count.get(fight.fighter_b_id, 0) + 1
        self._fight_count[fight.fighter_a_id] = fight_no_a
        self._fight_count[fight.fighter_b_id] = fight_no_b

        self._record_peak(
            fight.fighter_a_id, rating_a_after, fight.event_date,
            fight.fight_id, fight.fighter_b_id, fight_no_a,
        )
        self._record_peak(
            fight.fighter_b_id, rating_b_after, fight.event_date,
            fight.fight_id, fight.fighter_a_id, fight_no_b,
        )

        return FightEloResult(
            fight_id=fight.fight_id,
            event_date=fight.event_date,
            fighter_a_id=fight.fighter_a_id,
            fighter_b_id=fight.fighter_b_id,
            weight_class=fight.weight_class,
            bout_type=fight.bout_type,
            outcome=fight.outcome,
            method=fight.method,
            round_no=fight.round_no,
            total_rounds=fight.total_rounds,
            elo_a_before=rating_a_before,
            elo_a_after=rating_a_after,
            elo_b_before=rating_b_before,
            elo_b_after=rating_b_after,
            delta_a=delta_a,
            delta_b=delta_b,
            fight_no_a=fight_no_a,
            fight_no_b=fight_no_b,
        )

    def process_fights(self, fights: pd.DataFrame) -> pd.DataFrame:
        """Process a full DataFrame in deterministic order and return the
        long-format Elo history. Sort key: (event_date asc, bout_order asc).
        """
        sorted_fights = _sort_fights(fights)
        logger.info(
            "Processing %d fights through Elo engine.", len(sorted_fights)
        )
        long_rows: list[dict] = []
        for row in sorted_fights.itertuples(index=False):
            result = self.process_fight(_row_to_fight(row))
            long_rows.extend(_long_rows_from_result(result))
        return pd.DataFrame(long_rows, columns=LONG_COLUMNS)

    def current_ratings(self) -> pd.DataFrame:
        """Snapshot of every seen fighter's current rating, last-fight
        date, and cumulative UFC fight count. Cheap; derived from state."""
        rows = [
            {
                "fighter_id": fid,
                "rating": rating,
                "last_fight_date": self._last_fight_date.get(fid),
                "fight_count": self._fight_count.get(fid, 0),
            }
            for fid, rating in self._ratings.items()
        ]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("rating", ascending=False).reset_index(drop=True)
        return df

    def peak_ratings(self) -> pd.DataFrame:
        """Career-best post-fight rating per fighter (sorted desc)."""
        rows = [
            {"fighter_id": fid, **info}
            for fid, info in self._peak.items()
        ]
        df = pd.DataFrame(rows, columns=PEAK_COLUMNS)
        if not df.empty:
            df = df.sort_values(
                by=["peak_elo", "peak_date"],
                ascending=[False, True],
            ).reset_index(drop=True)
        return df


# --- row / result helpers -------------------------------------------------

def _sort_fights(fights: pd.DataFrame) -> pd.DataFrame:
    return fights.sort_values(
        by=["event_date", "bout_order"], kind="mergesort"
    ).reset_index(drop=True)


def _row_to_fight(row) -> FightRow:
    return FightRow(
        fight_id=row.fight_id,
        event_date=pd.Timestamp(row.event_date),
        bout_order=int(row.bout_order),
        fighter_a_id=row.fighter_a_id,
        fighter_b_id=row.fighter_b_id,
        outcome=row.outcome,
        method=row.method,
        round_no=int(row.round_no),
        total_rounds=int(row.total_rounds),
        weight_class=row.weight_class,
        bout_type=row.bout_type,
    )


def _long_rows_from_result(r: FightEloResult) -> list[dict]:
    return [
        {
            "fight_id": r.fight_id,
            "event_date": r.event_date,
            "fighter_id": r.fighter_a_id,
            "opponent_id": r.fighter_b_id,
            "weight_class": r.weight_class,
            "bout_type": r.bout_type,
            "result": _result_label(r.outcome, "a"),
            "method": r.method,
            "round_no": r.round_no,
            "total_rounds": r.total_rounds,
            "opponent_elo_before": r.elo_b_before,
            "elo_before": r.elo_a_before,
            "elo_after": r.elo_a_after,
            "elo_delta": r.delta_a,
            "fight_no": r.fight_no_a,
        },
        {
            "fight_id": r.fight_id,
            "event_date": r.event_date,
            "fighter_id": r.fighter_b_id,
            "opponent_id": r.fighter_a_id,
            "weight_class": r.weight_class,
            "bout_type": r.bout_type,
            "result": _result_label(r.outcome, "b"),
            "method": r.method,
            "round_no": r.round_no,
            "total_rounds": r.total_rounds,
            "opponent_elo_before": r.elo_a_before,
            "elo_before": r.elo_b_before,
            "elo_after": r.elo_b_after,
            "elo_delta": r.delta_b,
            "fight_no": r.fight_no_b,
        },
    ]


def _wide_row_from_result(r: FightEloResult) -> dict:
    return {
        "fight_id": r.fight_id,
        "event_date": r.event_date,
        "fighter_a_id": r.fighter_a_id,
        "fighter_b_id": r.fighter_b_id,
        "outcome": r.outcome,
        "method": r.method,
        "bout_type": r.bout_type,
        "weight_class": r.weight_class,
        "round_no": r.round_no,
        "total_rounds": r.total_rounds,
        "elo_a_before": r.elo_a_before,
        "elo_a_after": r.elo_a_after,
        "elo_b_before": r.elo_b_before,
        "elo_b_after": r.elo_b_after,
        "delta_a": r.delta_a,
        "delta_b": r.delta_b,
    }


# --- module-level convenience --------------------------------------------

def compute_elo_history(
    fights: pd.DataFrame,
    k: float = DEFAULT_K,
    base: float = DEFAULT_BASE,
    decay_per_year: float = DEFAULT_DECAY_PER_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute all three Elo outputs in one deterministic pass.

    Returns ``(long_df, wide_df, peak_df)`` matching the Data Contracts
    pages ``elo_history_long.csv``, ``elo_history_wide.csv`` and
    ``peak_ratings.csv`` respectively. This module performs no file IO;
    the caller (the refresh pipeline) is responsible for writing CSVs.
    """
    engine = EloEngine(k=k, base=base, decay_per_year=decay_per_year)
    sorted_fights = _sort_fights(fights)
    logger.info(
        "compute_elo_history: processing %d fights.", len(sorted_fights)
    )
    long_rows: list[dict] = []
    wide_rows: list[dict] = []
    for row in sorted_fights.itertuples(index=False):
        result = engine.process_fight(_row_to_fight(row))
        long_rows.extend(_long_rows_from_result(result))
        wide_rows.append(_wide_row_from_result(result))
    long_df = pd.DataFrame(long_rows, columns=LONG_COLUMNS)
    wide_df = pd.DataFrame(wide_rows, columns=WIDE_COLUMNS)
    peak_df = engine.peak_ratings()
    return long_df, wide_df, peak_df