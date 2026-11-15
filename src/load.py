"""
load.py  –  Data loaders for the Smart Hostel System.

These functions read CSVs from the data/ folder.  They can be used by:
  - The ML pipeline (src/prediction/)
  - The optimizer (src/optimization/)
  - The Streamlit dashboard (pages/)

DESIGN PRINCIPLE:
  If the CSV files already exist (generated or uploaded by the admin), they are
  read directly.  If a file is missing, the generator is called automatically.
  This makes the dashboard work even on first launch without any manual steps.
"""

import pandas as pd
from pathlib import Path
import sys

# Ensure src/ is importable wherever this is called from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    DATA_DIR, HOSTEL_CSV, DEMAND_CSV, STUDENTS_CSV, SURVEY_CSV,
)


def _ensure_data():
    """If any required CSV is missing, generate all data files."""
    required = [HOSTEL_CSV, DEMAND_CSV, STUDENTS_CSV, SURVEY_CSV]
    if not all(p.exists() for p in required):
        print("[load] Data files missing – generating synthetic data …")
        from src.data.generate import run_all
        run_all()


def load_hostels() -> pd.DataFrame:
    """
    Returns the hostel master table.
    Columns: hostel_id, name, gender, single_rooms, double_rooms, triple_rooms,
             total_beds, ac, distance_km, mess_rating, wifi_rating,
             annual_fee_inr, floors
    """
    _ensure_data()
    return pd.read_csv(HOSTEL_CSV)


def load_historical_demand() -> pd.DataFrame:
    """
    Returns the panel dataset of historical hostel demand.
    Key columns: year, semester, hostel_id, applications (target),
                 occupancy, occupancy_rate, cohort_size, rent_index, …
    """
    _ensure_data()
    df = pd.read_csv(DEMAND_CSV)
    df["year"]     = df["year"].astype(int)
    df["semester"] = df["semester"].astype(int)
    return df


def load_students() -> pd.DataFrame:
    """
    Returns the student applicants for the current semester.
    Key columns: student_id, gender, year_of_study, budget_inr,
                 ac_preference, sleep_schedule, room_type_pref,
                 home_distance_km, disability, scholarship,
                 first_choice_hostel, ranked_preferences,
                 far_away, first_year
    """
    _ensure_data()
    return pd.read_csv(STUDENTS_CSV)


def load_survey() -> pd.DataFrame:
    """
    Returns historical satisfaction survey records.
    Key columns: student_id, hostel_id, room_type, satisfaction_rating,
                 plus all student-profile and hostel-feature columns.
    """
    _ensure_data()
    return pd.read_csv(SURVEY_CSV)


def load_all() -> dict:
    """Convenience: load all four DataFrames into a dict."""
    return {
        "hostels":  load_hostels(),
        "demand":   load_historical_demand(),
        "students": load_students(),
        "survey":   load_survey(),
    }
