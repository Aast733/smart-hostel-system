"""
generate.py  -  Synthetic data generator for the Smart Hostel System.

This module creates realistic CSV files that simulate 15+ years of hostel
demand history plus a cohort of student applicants for the current semester.

WHY SYNTHETIC DATA?
  Real hostel data is private (personal student records).  We generate data
  with known statistical properties so we can verify our models honestly.

KEY IDEAS:
  - Hostel demand grows with the admitted cohort size (more students -> more
    applications).
  - Semester 2 always has fewer applicants than Semester 1 (most students
    settle in).
  - COVID-19 (2020-21) caused a large dip in residential demand.
  - One hostel expanded capacity in 2018, attracting more applicants.
  - Gaussian noise makes the data look realistic (not perfectly smooth).
  - A hidden ground-truth satisfaction function lets us evaluate allocations
    honestly without leaking the function to the model.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import our central configuration
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    DATA_DIR, HOSTELS, SIM_YEARS, SEMESTERS, BASE_COHORT, COHORT_GROWTH,
    COVID_YEARS, COVID_FACTOR, CAPACITY_EXPANSION_HOSTEL,
    CAPACITY_EXPANSION_YEAR, CAPACITY_EXPANSION_BEDS,
    N_STUDENTS, GENDER_SPLIT, DISABILITY_RATE, SCHOLARSHIP_RATE,
    FAR_DISTANCE_THRESHOLD_KM, RANDOM_SEED,
)


# ─────────────────────────────────────────────
# HELPER: save a DataFrame and print confirmation
# ─────────────────────────────────────────────
def _save(df: pd.DataFrame, path: Path, name: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  [generate] Saved {name}: {len(df)} rows -> {path.name}")


# ─────────────────────────────────────────────
# 1.  HOSTEL MASTER DATA
# ─────────────────────────────────────────────
def generate_hostel_data() -> pd.DataFrame:
    """
    Create the hostel master table from the constant definitions in config.py.
    This is deterministic (no randomness needed).
    """
    rows = []
    for h in HOSTELS:
        rows.append({
            "hostel_id":      h["hostel_id"],
            "name":           h["name"],
            "gender":         h["gender"],
            "single_rooms":   h["single_rooms"],
            "double_rooms":   h["double_rooms"],
            "triple_rooms":   h["triple_rooms"],
            "total_beds":     h["total_beds"],
            "ac":             int(h["ac"]),         # 1 = AC, 0 = non-AC
            "distance_km":    h["distance_km"],
            "mess_rating":    h["mess_rating"],
            "wifi_rating":    h["wifi_rating"],
            "annual_fee_inr": h["annual_fee_inr"],
            "floors":         h["floors"],
        })
    df = pd.DataFrame(rows)
    _save(df, DATA_DIR / "hostels.csv", "hostels")
    return df


# ─────────────────────────────────────────────
# 2.  HISTORICAL DEMAND PANEL  (2010 - 2025)
# ─────────────────────────────────────────────
def generate_historical_demand(hostel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a panel dataset:  (hostel x semester x year) -> applications.

    Logic for each row:
      base_demand   = beds x base_occupancy_rate (hostel-specific)
      cohort_scale  = cohort_size / base_cohort_size   (demand grows with cohort)
      fee_penalty   = 1 - 0.3 x (fee / max_fee)         (pricier -> fewer apps)
      sem2_discount = 0.75 for semester 2 (students settled)
      covid_factor  = 0.55 during 2020-21
      expansion     = +50 beds -> base goes up for that hostel after 2018
      noise         = N(0, 10%)

    We also record occupancy = min(applications, beds) and occupancy_rate.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    hostel_map = {h["hostel_id"]: h for h in HOSTELS}
    max_fee    = max(h["annual_fee_inr"] for h in HOSTELS)

    # Each hostel has a slightly different base occupancy rate
    base_occ = {
        "H1": 0.88, "H2": 0.82, "H3": 0.75,
        "H4": 0.90, "H5": 0.85, "H6": 0.72,
    }

    # Annual rent index: slow increase over years
    rent_index = {y: 1.0 + 0.03 * (y - 2010) for y in SIM_YEARS}

    rows = []
    for year in SIM_YEARS:
        # Cohort grows ~4% per year
        cohort = int(BASE_COHORT * (1 + COHORT_GROWTH) ** (year - 2010))

        for sem in SEMESTERS:
            for h_id, h in hostel_map.items():
                beds = h["total_beds"]
                # Capacity expansion in H2 after 2018
                if h_id == CAPACITY_EXPANSION_HOSTEL and year >= CAPACITY_EXPANSION_YEAR:
                    beds += CAPACITY_EXPANSION_BEDS

                # Base demand proportional to bed count and hostel attractiveness
                base = beds * base_occ[h_id]

                # Scale with cohort size
                demand = base * (cohort / BASE_COHORT)

                # Fee penalty: expensive hostels get fewer applicants
                fee_frac = h["annual_fee_inr"] / max_fee
                demand *= (1 - 0.30 * fee_frac)

                # Semester-2 sees ~25% fewer applications
                if sem == 2:
                    demand *= 0.75

                # COVID dip
                if year in COVID_YEARS:
                    demand *= COVID_FACTOR

                # Add realistic Gaussian noise (std = 10% of demand)
                noise = rng.normal(0, 0.10 * demand)
                applications = max(5, int(round(demand + noise)))

                # Occupancy = how many actually got a room (min of applications, beds)
                occupancy     = min(applications, beds)
                occupancy_rate= round(occupancy / beds, 4)

                rows.append({
                    "year":           year,
                    "semester":       sem,
                    "hostel_id":      h_id,
                    "hostel_name":    h["name"],
                    "gender":         h["gender"],
                    "total_beds":     beds,
                    "ac":             int(h["ac"]),
                    "distance_km":    h["distance_km"],
                    "mess_rating":    h["mess_rating"],
                    "wifi_rating":    h["wifi_rating"],
                    "annual_fee_inr": h["annual_fee_inr"],
                    "floors":         h["floors"],
                    "cohort_size":    cohort,
                    "rent_index":     round(rent_index[year], 4),
                    "applications":   applications,    # ← PREDICTION TARGET
                    "occupancy":      occupancy,
                    "occupancy_rate": occupancy_rate,
                })

    df = pd.DataFrame(rows).sort_values(["hostel_id", "year", "semester"]).reset_index(drop=True)
    _save(df, DATA_DIR / "historical_demand.csv", "historical_demand")
    return df


# ─────────────────────────────────────────────
# 3.  STUDENT APPLICANTS  (current semester)
# ─────────────────────────────────────────────
def generate_students() -> pd.DataFrame:
    """
    Simulate N_STUDENTS applying for hostel accommodation.

    Each student has:
      - demographics (gender, year of study, home distance)
      - preferences  (budget, AC pref, sleep schedule, first-choice hostel)
      - flags        (disability, scholarship)
      - ranked hostel preferences (list of hostel IDs, gender-appropriate)
    """
    rng = np.random.default_rng(RANDOM_SEED + 1)

    # Gender lists for preference assignment
    boy_hostels  = [h["hostel_id"] for h in HOSTELS if h["gender"] == "M"]
    girl_hostels = [h["hostel_id"] for h in HOSTELS if h["gender"] == "F"]

    rows = []
    for i in range(N_STUDENTS):
        gender   = "M" if rng.random() < GENDER_SPLIT else "F"
        year_of_study = int(rng.choice([1, 2, 3, 4], p=[0.35, 0.30, 0.20, 0.15]))
        disabled = rng.random() < DISABILITY_RATE
        scholar  = rng.random() < SCHOLARSHIP_RATE
        home_km  = float(rng.exponential(scale=200))   # most students < 400 km

        # Budget correlated with scholarship status
        if scholar:
            budget = int(rng.integers(35000, 65000))
        else:
            budget = int(rng.integers(40000, 100000))

        ac_pref  = int(rng.random() < 0.45)    # 45% prefer AC
        sleep_sch= rng.choice(["early", "normal", "night"], p=[0.25, 0.50, 0.25])

        # Room type preference
        room_pref= rng.choice(["single", "double", "triple"], p=[0.40, 0.40, 0.20])

        # Generate ranked hostel preferences (gender-appropriate only)
        relevant = boy_hostels if gender == "M" else girl_hostels
        shuffled = rng.permutation(relevant).tolist()
        first_choice = shuffled[0]
        pref_str = ",".join(shuffled)    # stored as comma-separated string

        rows.append({
            "student_id":     f"S{i+1:04d}",
            "gender":         gender,
            "year_of_study":  year_of_study,
            "budget_inr":     budget,
            "ac_preference":  ac_pref,
            "sleep_schedule": sleep_sch,
            "room_type_pref": room_pref,
            "home_distance_km": round(home_km, 1),
            "disability":     int(disabled),
            "scholarship":    int(scholar),
            "first_choice_hostel": first_choice,
            "ranked_preferences":  pref_str,
            # Derived flags
            "far_away":       int(home_km > FAR_DISTANCE_THRESHOLD_KM),
            "first_year":     int(year_of_study == 1),
        })

    df = pd.DataFrame(rows)
    _save(df, DATA_DIR / "students.csv", "students")
    return df


# ─────────────────────────────────────────────
# 4.  SATISFACTION SURVEY  (historical pairings)
# ─────────────────────────────────────────────
def _true_satisfaction(student: dict, hostel: dict, room_type: str, rng) -> float:
    """
    Hidden ground-truth satisfaction function.
    This function is the "oracle" - it computes the TRUE satisfaction a student
    would get in a given hostel/room combination.  The model never sees this
    function directly; it can only learn from noisy observations.

    Scoring (each component 0-1, then scaled to 1-10):
      - budget fit:     gets full marks if hostel fee << budget
      - AC match:       +1 if student wants AC and hostel has it
      - distance:       closer is better (exponential decay)
      - mess rating:    hostel mess quality (normalized)
      - first-choice:   small bonus if this was the student's top pick
      - room type:      bonus for getting the preferred room type
    """
    fee    = hostel["annual_fee_inr"]
    budget = student["budget_inr"]
    budget_gap = (budget - fee) / max(budget, 1)    # positive = affordable
    budget_score = min(max(budget_gap, 0), 1)        # clip 0-1

    ac_score = 1.0 if (student["ac_preference"] == hostel["ac"]) else 0.3

    dist_score = float(np.exp(-0.5 * hostel["distance_km"]))  # exponential decay

    mess_score = (hostel["mess_rating"] - 1) / 4     # normalise 1-5 -> 0-1

    first_choice_bonus = 0.15 if hostel["hostel_id"] == student["first_choice_hostel"] else 0.0

    room_score = 0.8 if student["room_type_pref"] == room_type else 0.5

    # Weighted sum -> 0-1 scale -> 1-10
    raw = (
        0.30 * budget_score +
        0.20 * ac_score +
        0.15 * dist_score +
        0.15 * mess_score +
        0.10 * room_score +
        first_choice_bonus
    )
    # Add first_choice_bonus already included; renormalise
    score_10 = 1 + 9 * min(raw, 1.0)
    return score_10


def generate_satisfaction_survey(student_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate historical (student, hostel, room_type) -> satisfaction_rating records.
    
    We sample a random subset of past (student-like, hostel) pairs and apply the
    hidden ground-truth function + Gaussian noise.  The ML model trains on these
    noisy observations and hopefully recovers the underlying pattern.
    """
    rng = np.random.default_rng(RANDOM_SEED + 2)
    hostel_map = {h["hostel_id"]: h for h in HOSTELS}
    room_types = ["single", "double", "triple"]

    # Build a pool of "students" for historical survey (re-use student data structure)
    # We'll generate 3000 historical survey entries
    n_survey = 3000
    rows = []
    student_records = student_df.to_dict("records")

    for _ in range(n_survey):
        # Pick a random student profile
        stu = rng.choice(student_records)
        # Only assign gender-appropriate hostel
        gender = stu["gender"]
        valid_hostels = [h for h in HOSTELS if h["gender"] == gender]
        hostel = rng.choice(valid_hostels)
        room_type = rng.choice(room_types)

        true_sat = _true_satisfaction(stu, hostel, room_type, rng)
        # Add noise (std = 0.8 on 1-10 scale)
        observed = true_sat + rng.normal(0, 0.8)
        observed = float(np.clip(observed, 1.0, 10.0))

        rows.append({
            "student_id":        stu["student_id"],
            "hostel_id":         hostel["hostel_id"],
            "room_type":         room_type,
            "gender":            gender,
            "year_of_study":     stu["year_of_study"],
            "budget_inr":        stu["budget_inr"],
            "ac_preference":     stu["ac_preference"],
            "sleep_schedule":    stu["sleep_schedule"],
            "room_type_pref":    stu["room_type_pref"],
            "home_distance_km":  stu["home_distance_km"],
            "disability":        stu["disability"],
            "scholarship":       stu["scholarship"],
            "first_choice_hostel": stu["first_choice_hostel"],
            # Hostel features
            "hostel_gender":     hostel["gender"],
            "ac":                int(hostel["ac"]),
            "distance_km":       hostel["distance_km"],
            "mess_rating":       hostel["mess_rating"],
            "wifi_rating":       hostel["wifi_rating"],
            "annual_fee_inr":    hostel["annual_fee_inr"],
            # Target
            "satisfaction_rating": round(observed, 2),
        })

    df = pd.DataFrame(rows)
    _save(df, DATA_DIR / "satisfaction_survey.csv", "satisfaction_survey")
    return df


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────
def run_all() -> dict:
    """Generate all data files and return DataFrames in a dict."""
    print("\n=== DATA GENERATION ===")
    hostel_df   = generate_hostel_data()
    demand_df   = generate_historical_demand(hostel_df)
    student_df  = generate_students()
    survey_df   = generate_satisfaction_survey(student_df)
    print("  [generate] All data files written to data/")
    return {
        "hostels":  hostel_df,
        "demand":   demand_df,
        "students": student_df,
        "survey":   survey_df,
    }


if __name__ == "__main__":
    run_all()
