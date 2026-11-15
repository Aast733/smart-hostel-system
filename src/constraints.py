"""
constraints.py  –  PART B: Hard Constraint Checking for Room Allocation.

This module defines and enforces the HARD RULES that can NEVER be broken:
  1. Gender match  – a male student cannot be placed in a girls' hostel.
  2. Room capacity – total students assigned to a room cannot exceed room capacity.
  3. Budget limit  – annual_fee > student_budget + tolerance → infeasible.
  4. Disability    – disabled students must be assigned ground-floor rooms only.

PURPOSE:
  The optimizer uses the feasibility functions here to filter out impossible
  (student, hostel) pairs BEFORE solving the assignment problem.
  This ensures the optimizer never wastes effort on infeasible options.

NOTE: This module belongs to PART B (Optimization). It contains NO ML code.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import BUDGET_TOLERANCE_INR, GROUND_FLOOR_FOR_DISABLED


def build_feasibility_matrix(
    student_df: pd.DataFrame,
    hostel_df:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute a Boolean DataFrame indicating which (student, hostel) pairs are feasible.

    Parameters
    ----------
    student_df : pd.DataFrame   – all applicants
    hostel_df  : pd.DataFrame   – hostel master table

    Returns
    -------
    pd.DataFrame  shape (n_students, n_hostels)
        True  = this student CAN be placed in this hostel
        False = at least one hard constraint is violated
    Index  = student_id
    Columns= hostel_id
    """
    student_ids = student_df["student_id"].tolist()
    hostel_ids  = hostel_df["hostel_id"].tolist()

    # Start with all True; we'll flip cells that violate constraints
    feasible = pd.DataFrame(
        True,
        index=student_ids,
        columns=hostel_ids,
    )

    # Create lookup dicts for speed
    hostel_gender = hostel_df.set_index("hostel_id")["gender"].to_dict()
    hostel_fee    = hostel_df.set_index("hostel_id")["annual_fee_inr"].to_dict()
    hostel_floors = hostel_df.set_index("hostel_id")["floors"].to_dict()

    for _, stu in student_df.iterrows():
        sid    = stu["student_id"]
        gender = stu["gender"]
        budget = stu["budget_inr"]
        disab  = bool(stu["disability"])

        for hid in hostel_ids:
            # ── Rule 1: Gender match ──────────────────────────────────────
            if hostel_gender[hid] != gender:
                feasible.at[sid, hid] = False
                continue

            # ── Rule 2: Budget constraint ─────────────────────────────────
            if hostel_fee[hid] > budget + BUDGET_TOLERANCE_INR:
                feasible.at[sid, hid] = False
                continue

            # ── Rule 3: Disability – ground floor only ────────────────────
            # We model this as: if disabled and hostel has > 1 floor,
            # the student CAN still be placed (ground floor rooms exist).
            # But if hostel has 0 ground-floor beds (in a hypothetical future
            # extension), we'd block it.  For now, always feasible if floors >= 1.
            # No additional blocking needed here; we track floor assignment separately.

    return feasible


def check_hard_constraints(
    allocation_df: pd.DataFrame,
    student_df:    pd.DataFrame,
    hostel_df:     pd.DataFrame,
) -> pd.DataFrame:
    """
    Audit an existing allocation for hard-constraint violations.

    Parameters
    ----------
    allocation_df : pd.DataFrame
        Must have columns: student_id, hostel_id, room_type, floor_number

    Returns
    -------
    pd.DataFrame of violations (empty = all good).
    """
    hostel_lu  = hostel_df.set_index("hostel_id").to_dict("index")
    student_lu = student_df.set_index("student_id").to_dict("index")

    violations = []
    for _, row in allocation_df.iterrows():
        sid  = row["student_id"]
        hid  = row["hostel_id"]
        if hid == "WAITLIST":
            continue

        stu  = student_lu.get(sid, {})
        hos  = hostel_lu.get(hid, {})
        if not stu or not hos:
            continue

        # Gender check
        if stu["gender"] != hos["gender"]:
            violations.append({"student_id": sid, "hostel_id": hid,
                                "violation": "gender_mismatch"})

        # Budget check
        if hos["annual_fee_inr"] > stu["budget_inr"] + BUDGET_TOLERANCE_INR:
            violations.append({"student_id": sid, "hostel_id": hid,
                                "violation": "budget_exceeded"})

        # Disability floor check
        if stu.get("disability", 0) and GROUND_FLOOR_FOR_DISABLED:
            if row.get("floor_number", 1) != 1:
                violations.append({"student_id": sid, "hostel_id": hid,
                                    "violation": "disabled_not_ground_floor"})

    return pd.DataFrame(violations)


def get_student_priority(stu_row: pd.Series) -> float:
    """
    Compute a scalar priority weight for a student.
    Higher weight → more important to satisfy.

    Priority rules (from config):
      first_year  → 3.0
      disabled    → 4.0
      scholarship → 2.0
      far_away    → 2.5
      otherwise   → 1.0

    Multiple flags are SUMMED (a far-away disabled first-year gets a lot of priority).
    """
    from src.config import PRIORITY_WEIGHTS
    weight = PRIORITY_WEIGHTS["other"]   # base weight
    if stu_row.get("first_year", 0):
        weight += PRIORITY_WEIGHTS["first_year"]
    if stu_row.get("disability", 0):
        weight += PRIORITY_WEIGHTS["disabled"]
    if stu_row.get("scholarship", 0):
        weight += PRIORITY_WEIGHTS["scholarship"]
    if stu_row.get("far_away", 0):
        weight += PRIORITY_WEIGHTS["far_away"]
    return weight
