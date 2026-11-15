"""
features.py  –  PART A: Feature Engineering for Demand Forecasting.

This module transforms the raw historical demand table into a feature matrix
suitable for supervised learning (regression).

KEY FEATURE GROUPS:
  1. Hostel characteristics  – static properties like AC, distance, fee
  2. Context features        – cohort size, rent index, semester indicator
  3. Lag features            – previous semester's applications, same semester
                               last year → these capture serial correlation
  4. Rolling statistics      – 2-year rolling mean of applications
  5. Derived ratios          – fee-to-max-fee ratio, capacity utilisation

WHY LAG FEATURES?
  Hostel demand tomorrow is partly predictable from demand today.  A student
  decides to apply partly because "everyone went there last year."  Lags
  encode this time dependency without needing a full time-series model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import RANDOM_SEED


def build_demand_features(demand_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling features to the demand panel.

    Parameters
    ----------
    demand_df : pd.DataFrame
        Raw output from load_historical_demand().

    Returns
    -------
    pd.DataFrame
        Original columns + engineered features.  Rows with NaN lags are dropped
        (these are the very first year/semester for each hostel).
    """
    df = demand_df.copy()

    # ── Create a numeric time index for sorting ──────────────────────────────
    # time_idx = year * 10 + semester  (e.g., 20101, 20102, 20111, …)
    df["time_idx"] = df["year"] * 10 + df["semester"]
    df = df.sort_values(["hostel_id", "time_idx"]).reset_index(drop=True)

    # ── Lag 1: previous semester (for the same hostel) ────────────────────────
    df["lag1_applications"] = df.groupby("hostel_id")["applications"].shift(1)

    # ── Lag 2: same semester one year ago (shift by 2 because 2 semesters/year)
    df["lag2_applications"] = df.groupby("hostel_id")["applications"].shift(2)

    # ── Rolling mean: last 2 years (4 semester records) ──────────────────────
    df["rolling4_mean"] = (
        df.groupby("hostel_id")["applications"]
        .transform(lambda x: x.shift(1).rolling(4, min_periods=2).mean())
    )

    # ── Lag occupancy rate ───────────────────────────────────────────────────
    df["lag1_occ_rate"] = df.groupby("hostel_id")["occupancy_rate"].shift(1)

    # ── Derived features ─────────────────────────────────────────────────────
    max_fee = df["annual_fee_inr"].max()
    df["fee_ratio"]    = df["annual_fee_inr"] / max_fee           # 0-1
    df["beds_per_floor"]= df["total_beds"] / df["floors"]
    df["sem2_flag"]    = (df["semester"] == 2).astype(int)        # 1 if Sem 2

    # ── Drop rows where we cannot compute lags (first 1-2 rows per hostel) ───
    feature_cols = [
        "lag1_applications", "lag2_applications",
        "rolling4_mean", "lag1_occ_rate",
    ]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    return df


def get_demand_feature_columns() -> list:
    """
    Return the ordered list of feature columns used by the demand model.
    Call this to ensure the training and prediction pipelines use identical features.
    """
    return [
        # Hostel static features
        "total_beds",
        "ac",
        "distance_km",
        "mess_rating",
        "wifi_rating",
        "fee_ratio",
        "floors",
        "beds_per_floor",
        # Context
        "cohort_size",
        "rent_index",
        "sem2_flag",
        # Lag features
        "lag1_applications",
        "lag2_applications",
        "rolling4_mean",
        "lag1_occ_rate",
    ]


def get_demand_target_column() -> str:
    """Return the name of the regression target for demand forecasting."""
    return "applications"


# ─────────────────────────────────────────────
# SATISFACTION / PREFERENCE FEATURES
# ─────────────────────────────────────────────

def build_preference_features(survey_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for the student-room satisfaction regression model.

    The survey table already has both student and hostel columns side by side.
    We just compute a few cross-interaction terms that are meaningful:
      - budget_gap:    (budget - annual_fee) / budget       (positive = comfortable)
      - ac_match:      1 if student wants AC and hostel has AC, else 0
      - distance_score: exp(-distance) so closer = higher score
      - first_choice:  1 if this hostel is the student's first choice

    Parameters
    ----------
    survey_df : pd.DataFrame
        Output from load_survey().

    Returns
    -------
    pd.DataFrame
        With additional engineered columns.
    """
    df = survey_df.copy()

    # Budget comfort: positive if student can afford the hostel
    df["budget_gap"] = (df["budget_inr"] - df["annual_fee_inr"]) / df["budget_inr"].clip(lower=1)

    # AC match: both want AC and hostel has AC, OR neither
    df["ac_match"] = ((df["ac_preference"] == df["ac"]).astype(int))

    # Distance score (closer = better; exponential decay)
    df["distance_score"] = np.exp(-0.5 * df["distance_km"])

    # Whether this hostel was first choice
    df["first_choice_match"] = (
        (df["hostel_id"] == df["first_choice_hostel"]).astype(int)
    )

    # Room type match
    df["room_type_match"] = (
        (df["room_type"] == df["room_type_pref"]).astype(int)
    )

    # Sleep schedule encoded to numeric for the model
    sleep_map = {"early": 0, "normal": 1, "night": 2}
    df["sleep_num"] = df["sleep_schedule"].map(sleep_map).fillna(1)

    # Room type encoded
    room_map = {"single": 0, "double": 1, "triple": 2}
    df["room_type_num"] = df["room_type"].map(room_map).fillna(1)

    return df


def get_preference_feature_columns() -> list:
    """Return the ordered list of feature columns used by the preference model."""
    return [
        "year_of_study",
        "budget_gap",
        "ac_match",
        "distance_score",
        "mess_rating",
        "wifi_rating",
        "fee_ratio" if "fee_ratio" in [] else "annual_fee_inr",  # raw fee
        "first_choice_match",
        "room_type_match",
        "disability",
        "scholarship",
        "sleep_num",
        "room_type_num",
        "home_distance_km",
    ]


def get_preference_feature_columns_v2() -> list:
    """
    Correct list of preference features (no fee_ratio – that's a demand feature).
    """
    return [
        "year_of_study",
        "budget_gap",
        "ac_match",
        "distance_score",
        "mess_rating",
        "wifi_rating",
        "annual_fee_inr",
        "first_choice_match",
        "room_type_match",
        "disability",
        "scholarship",
        "sleep_num",
        "room_type_num",
        "home_distance_km",
    ]


def get_preference_target_column() -> str:
    """Return the name of the regression target for satisfaction prediction."""
    return "satisfaction_rating"
