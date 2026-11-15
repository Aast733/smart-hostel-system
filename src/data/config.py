"""
config.py  –  Central configuration for the Smart Hostel System.

This file defines ALL constants used across the project:
  - File paths (relative, using pathlib so they work on any machine)
  - Hostel master data
  - Simulation / generation parameters
  - ML hyper-parameters
  - Allocation rules

Beginners: import anything you need from here instead of hard-coding values.
"""

from pathlib import Path

# ─────────────────────────────────────────────
# ROOT & DIRECTORY PATHS  (all relative)
# ─────────────────────────────────────────────
# __file__ is src/config.py  →  parent = src/  →  parent.parent = project root
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR   = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR= ROOT / "outputs"
DOCS_DIR   = ROOT / "docs"

# Raw CSV file paths
HOSTEL_CSV     = DATA_DIR / "hostels.csv"
DEMAND_CSV     = DATA_DIR / "historical_demand.csv"
STUDENTS_CSV   = DATA_DIR / "students.csv"
SURVEY_CSV     = DATA_DIR / "satisfaction_survey.csv"

# Output file paths
ALLOCATION_CSV    = OUTPUTS_DIR / "allocation_results.csv"
WAITLIST_CSV      = OUTPUTS_DIR / "waitlist.csv"
EVAL_CSV          = OUTPUTS_DIR / "evaluation_metrics.csv"
REPORT_TXT        = OUTPUTS_DIR / "report.txt"
DEMAND_METRICS_CSV= OUTPUTS_DIR / "demand_model_metrics.csv"
PREF_METRICS_CSV  = OUTPUTS_DIR / "preference_model_metrics.csv"
FORECAST_CSV      = OUTPUTS_DIR / "demand_forecast.csv"

# Saved model paths
DEMAND_MODEL_PATH = MODELS_DIR / "demand_model.joblib"
PREF_MODEL_PATH   = MODELS_DIR / "preference_model.joblib"

# ─────────────────────────────────────────────
# RANDOM SEED  (fixed so all teammates get same results)
# ─────────────────────────────────────────────
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# HOSTEL MASTER DATA
# Each entry is one hostel.  Fields:
#   hostel_id, name, gender (M/F), total_beds,
#   single_rooms, double_rooms, triple_rooms,
#   ac (bool), distance_km, mess_rating (1-5),
#   wifi_rating (1-5), annual_fee_inr, floors
# ─────────────────────────────────────────────
HOSTELS = [
    {
        "hostel_id": "H1",
        "name": "Saraswati Boys Hostel",
        "gender": "M",
        "single_rooms": 20,
        "double_rooms": 40,
        "triple_rooms": 10,
        "ac": False,
        "distance_km": 0.3,
        "mess_rating": 3.8,
        "wifi_rating": 4.0,
        "annual_fee_inr": 45000,
        "floors": 4,
    },
    {
        "hostel_id": "H2",
        "name": "Chanakya Boys Hostel",
        "gender": "M",
        "single_rooms": 10,
        "double_rooms": 60,
        "triple_rooms": 20,
        "ac": True,
        "distance_km": 0.6,
        "mess_rating": 4.2,
        "wifi_rating": 4.5,
        "annual_fee_inr": 75000,
        "floors": 5,
    },
    {
        "hostel_id": "H3",
        "name": "Viveka Boys Hostel",
        "gender": "M",
        "single_rooms": 5,
        "double_rooms": 30,
        "triple_rooms": 30,
        "ac": False,
        "distance_km": 1.2,
        "mess_rating": 3.5,
        "wifi_rating": 3.8,
        "annual_fee_inr": 38000,
        "floors": 3,
    },
    {
        "hostel_id": "H4",
        "name": "Lakshmi Girls Hostel",
        "gender": "F",
        "single_rooms": 15,
        "double_rooms": 50,
        "triple_rooms": 15,
        "ac": False,
        "distance_km": 0.4,
        "mess_rating": 4.0,
        "wifi_rating": 4.2,
        "annual_fee_inr": 48000,
        "floors": 4,
    },
    {
        "hostel_id": "H5",
        "name": "Ambika Girls Hostel",
        "gender": "F",
        "single_rooms": 10,
        "double_rooms": 40,
        "triple_rooms": 20,
        "ac": True,
        "distance_km": 0.8,
        "mess_rating": 4.5,
        "wifi_rating": 4.8,
        "annual_fee_inr": 82000,
        "floors": 6,
    },
    {
        "hostel_id": "H6",
        "name": "Sarojini Girls Hostel",
        "gender": "F",
        "single_rooms": 8,
        "double_rooms": 35,
        "triple_rooms": 25,
        "ac": False,
        "distance_km": 1.5,
        "mess_rating": 3.6,
        "wifi_rating": 3.5,
        "annual_fee_inr": 36000,
        "floors": 3,
    },
]

# ─────────────────────────────────────────────
# DERIVED HOSTEL CONSTANTS
# ─────────────────────────────────────────────
def compute_hostel_beds(h: dict) -> int:
    """Return total bed count for a hostel definition dict."""
    return h["single_rooms"] * 1 + h["double_rooms"] * 2 + h["triple_rooms"] * 3

for _h in HOSTELS:
    _h["total_beds"] = compute_hostel_beds(_h)

HOSTEL_IDS   = [h["hostel_id"] for h in HOSTELS]
TOTAL_BEDS   = sum(h["total_beds"] for h in HOSTELS)

# ─────────────────────────────────────────────
# HISTORICAL DEMAND SIMULATION PARAMETERS
# ─────────────────────────────────────────────
SIM_YEARS      = list(range(2010, 2026))   # 2010 … 2025 (16 years)
SEMESTERS      = [1, 2]
BASE_COHORT    = 800           # students admitted per year in 2010
COHORT_GROWTH  = 0.04          # ~4% annual growth
COVID_YEARS    = [2020, 2021]  # demand dip years
COVID_FACTOR   = 0.55          # fraction of normal applications during COVID
CAPACITY_EXPANSION_HOSTEL = "H2"
CAPACITY_EXPANSION_YEAR   = 2018   # H2 added 50 beds in 2018
CAPACITY_EXPANSION_BEDS   = 50

# ─────────────────────────────────────────────
# STUDENT APPLICANT SIMULATION
# ─────────────────────────────────────────────
N_STUDENTS     = 1400          # students applying for the upcoming semester
GENDER_SPLIT   = 0.52          # fraction male
DISABILITY_RATE= 0.04
SCHOLARSHIP_RATE = 0.20
FAR_DISTANCE_THRESHOLD_KM = 200   # home > 200 km → "far-away" priority

# ─────────────────────────────────────────────
# ALLOCATION RULES
# ─────────────────────────────────────────────
BUDGET_TOLERANCE_INR = 5000    # student can be placed up to 5000 above budget
GROUND_FLOOR_FOR_DISABLED = True

# Priority weights (higher = more important)
PRIORITY_WEIGHTS = {
    "first_year":   3.0,
    "disabled":     4.0,
    "scholarship":  2.0,
    "far_away":     2.5,
    "other":        1.0,
}

ROOMMATE_WEIGHT = 0.15   # fraction of satisfaction from roommate compatibility

# ─────────────────────────────────────────────
# EVALUATION THRESHOLDS
# ─────────────────────────────────────────────
HAPPY_THRESHOLD = 7.0    # satisfaction >= 7 counts as "happy"
