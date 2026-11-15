"""
metrics.py  -  Evaluation Module for the Smart Hostel System.

This module computes ALL evaluation metrics for both:
  A. Prediction quality (demand + satisfaction models)
  B. Allocation quality (comparing methods using TRUE satisfaction)

IMPORTANT: Allocation evaluation uses the TRUE hidden satisfaction function,
NOT the model's own predictions.  This prevents circular evaluation.

ALLOCATION METRICS:
  - Allocation rate:        % of students who received a room
  - Mean / Median satisfaction (true score)
  - 10th percentile satisfaction (fairness for worst-off students)
  - Happy % :               % of housed students with true_sat >= 7.0
  - Gini coefficient:       inequality of satisfaction scores (0 = perfect equality)
  - Priority group coverage: % of priority students (disabled, scholars, etc.) housed
  - First-choice match:     % of housed students in their first-choice hostel
  - Roommate mismatch:      % of rooms with sleep-schedule conflict
  - Hostel occupancy %:     beds used / total beds per hostel
  - Constraint violations:  count (must be 0)

REPORT:
  A human-readable summary is written to outputs/report.txt.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    HAPPY_THRESHOLD, EVAL_CSV, REPORT_TXT, OUTPUTS_DIR, HOSTELS,
)
from src.optimization.constraints import check_hard_constraints


# ─────────────────────────────────────────────
# TRUE SATISFACTION ORACLE (same function as in generate.py)
# ─────────────────────────────────────────────
def _true_satisfaction_eval(student: dict, hostel: dict, room_type: str) -> float:
    """
    Compute the TRUE (hidden) satisfaction for one (student, hostel, room_type) triple.
    This mirrors the ground-truth function in generate.py exactly.
    """
    fee    = hostel["annual_fee_inr"]
    budget = student["budget_inr"]
    budget_gap   = (budget - fee) / max(budget, 1)
    budget_score = min(max(budget_gap, 0), 1)

    ac_score  = 1.0 if (int(student["ac_preference"]) == int(hostel["ac"])) else 0.3
    dist_score= float(np.exp(-0.5 * hostel["distance_km"]))
    mess_score= (hostel["mess_rating"] - 1) / 4

    first_choice_bonus = 0.15 if hostel["hostel_id"] == student["first_choice_hostel"] else 0.0

    room_pref  = student.get("room_type_pref", "double")
    room_score = 0.8 if room_pref == room_type else 0.5

    raw = (
        0.30 * budget_score +
        0.20 * ac_score +
        0.15 * dist_score +
        0.15 * mess_score +
        0.10 * room_score +
        first_choice_bonus
    )
    return 1 + 9 * min(raw, 1.0)


# ─────────────────────────────────────────────
# GINI COEFFICIENT
# ─────────────────────────────────────────────
def _gini(scores: np.ndarray) -> float:
    """
    Compute the Gini coefficient of a satisfaction score array.
    Gini = 0  -> perfect equality (everyone equally satisfied)
    Gini = 1  -> maximum inequality
    """
    if len(scores) == 0:
        return 0.0
    scores = np.sort(scores)
    n      = len(scores)
    cumsum = np.cumsum(scores)
    return float((2 * np.dot(np.arange(1, n + 1), scores) - (n + 1) * cumsum[-1]) /
                 (n * cumsum[-1]) if cumsum[-1] > 0 else 0)


# ─────────────────────────────────────────────
# COMPUTE TRUE SATISFACTION FOR AN ALLOCATION
# ─────────────────────────────────────────────
def compute_true_satisfaction(
    alloc_df:   pd.DataFrame,
    student_df: pd.DataFrame,
    hostel_df:  pd.DataFrame,
) -> pd.Series:
    """
    For each housed student, compute their TRUE satisfaction using the hidden function.

    Returns a Series indexed by student_id with values in [1, 10].
    """
    hostel_map = {
        hid: {**row, "hostel_id": hid}
        for hid, row in hostel_df.set_index("hostel_id").to_dict("index").items()
    }
    student_lu = {
        sid: {**row, "student_id": sid}
        for sid, row in student_df.set_index("student_id").to_dict("index").items()
    }
    results    = {}

    for _, row in alloc_df.iterrows():
        sid  = row["student_id"]
        hid  = row["hostel_id"]
        rt   = row.get("room_type", "double")
        stu  = student_lu.get(sid, {})
        hos  = hostel_map.get(hid, {})
        if not stu or not hos:
            continue
        results[sid] = _true_satisfaction_eval(stu, hos, rt)

    return pd.Series(results, name="true_satisfaction")


# ─────────────────────────────────────────────
# MAIN EVALUATION FUNCTION
# ─────────────────────────────────────────────
def evaluate_allocation(
    alloc_df:    pd.DataFrame,
    waitlist_df: pd.DataFrame,
    student_df:  pd.DataFrame,
    hostel_df:   pd.DataFrame,
    method_name: str = "Method",
) -> dict:
    """
    Evaluate one allocation (alloc_df + waitlist_df) against all metrics.

    Returns a dict of scalar metric values.
    """
    n_applicants = len(student_df)
    n_housed     = len(alloc_df)
    n_waitlisted = len(waitlist_df)
    alloc_rate   = n_housed / max(n_applicants, 1)

    # ── True satisfaction for all housed students ─────────────────────────────
    true_sat = compute_true_satisfaction(alloc_df, student_df, hostel_df)

    if len(true_sat) > 0:
        mean_sat    = float(true_sat.mean())
        median_sat  = float(true_sat.median())
        p10_sat     = float(np.percentile(true_sat.values, 10))
        happy_pct   = float((true_sat >= HAPPY_THRESHOLD).mean() * 100)
        gini        = _gini(true_sat.values)
    else:
        mean_sat = median_sat = p10_sat = happy_pct = gini = 0.0

    # ── Priority group coverage ───────────────────────────────────────────────
    priority_students = student_df[
        (student_df["disability"] == 1) |
        (student_df["scholarship"] == 1) |
        (student_df["first_year"]  == 1) |
        (student_df["far_away"]    == 1)
    ]["student_id"].tolist()
    housed_ids = set(alloc_df["student_id"].tolist())
    if priority_students:
        priority_coverage = len([s for s in priority_students if s in housed_ids]) / len(priority_students)
    else:
        priority_coverage = 1.0

    # ── First-choice match ────────────────────────────────────────────────────
    if len(alloc_df) > 0:
        fc_map = student_df.set_index("student_id")["first_choice_hostel"].to_dict()
        alloc_fc = alloc_df["student_id"].map(fc_map)
        first_choice_match = float(
            (alloc_df["hostel_id"].values == alloc_fc.values).mean() * 100
        )
    else:
        first_choice_match = 0.0

    # ── Roommate mismatch ─────────────────────────────────────────────────────
    if "sleep_schedule" in alloc_df.columns:
        multi_occ = alloc_df[alloc_df["room_type"].isin(["double", "triple"])]
        if len(multi_occ) > 0:
            room_groups = multi_occ.groupby("room_id")["sleep_schedule"].apply(list)
            mismatch_rooms = sum(
                1 for scheds in room_groups
                if "early" in scheds and "night" in scheds
            )
            total_multi_rooms = len(room_groups)
            roommate_mismatch_pct = mismatch_rooms / max(total_multi_rooms, 1) * 100
        else:
            roommate_mismatch_pct = 0.0
    else:
        roommate_mismatch_pct = 0.0

    # ── Hostel occupancy ──────────────────────────────────────────────────────
    hostel_occ = {}
    hostel_bed_totals = {h["hostel_id"]: h["total_beds"] for h in HOSTELS}
    if len(alloc_df) > 0:
        housed_counts = alloc_df.groupby("hostel_id")["student_id"].count()
        for hid, total in hostel_bed_totals.items():
            count = housed_counts.get(hid, 0)
            hostel_occ[hid] = round(count / max(total, 1) * 100, 1)
    else:
        hostel_occ = {hid: 0.0 for hid in hostel_bed_totals}

    # ── Constraint violations ──────────────────────────────────────────────────
    if len(alloc_df) > 0:
        violations_df = check_hard_constraints(alloc_df, student_df, hostel_df)
        n_violations  = len(violations_df)
    else:
        n_violations = 0

    metrics = {
        "method":             method_name,
        "n_applicants":       n_applicants,
        "n_housed":           n_housed,
        "n_waitlisted":       n_waitlisted,
        "allocation_rate_%":  round(alloc_rate * 100, 2),
        "mean_satisfaction":  round(mean_sat, 3),
        "median_satisfaction":round(median_sat, 3),
        "p10_satisfaction":   round(p10_sat, 3),
        "happy_%":            round(happy_pct, 2),
        "gini":               round(gini, 4),
        "priority_coverage_%":round(priority_coverage * 100, 2),
        "first_choice_match_%":round(first_choice_match, 2),
        "roommate_mismatch_%": round(roommate_mismatch_pct, 2),
        "constraint_violations": n_violations,
    }
    metrics.update({f"occ_{k}_%": v for k, v in hostel_occ.items()})
    return metrics


# ─────────────────────────────────────────────
# EVALUATE ALL METHODS AND SAVE
# ─────────────────────────────────────────────
def evaluate_all_methods(
    results:    dict,   # {method_name: (alloc_df, waitlist_df)}
    student_df: pd.DataFrame,
    hostel_df:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Run evaluate_allocation for every method, assemble comparison table,
    save CSV and text report.

    Parameters
    ----------
    results : {"Random": (alloc_df, wl_df), "Greedy": ..., ...}

    Returns
    -------
    pd.DataFrame : comparison table
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    for name, (alloc_df, waitlist_df) in results.items():
        m = evaluate_allocation(alloc_df, waitlist_df, student_df, hostel_df, name)
        all_metrics.append(m)
        print(f"  [{name}] Allocation rate: {m['allocation_rate_%']}%  "
              f"Mean sat: {m['mean_satisfaction']}  Violations: {m['constraint_violations']}")

    eval_df = pd.DataFrame(all_metrics)
    eval_df.to_csv(EVAL_CSV, index=False)
    print(f"  Evaluation saved -> {EVAL_CSV.name}")

    # ── Write text report ─────────────────────────────────────────────────────
    _write_report(eval_df)

    return eval_df


def _write_report(eval_df: pd.DataFrame) -> None:
    """Write a plain-text conclusion report for the project."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 70,
        "  SMART HOSTEL SYSTEM - EVALUATION REPORT",
        "=" * 70,
        "",
        "PART A: PREDICTION",
        "  - Demand model trained on 2010-2023; tested on 2024-2025.",
        "  - Best model selected by TimeSeriesSplit CV (no test leakage).",
        "  - Metrics saved to outputs/demand_model_metrics.csv",
        "",
        "  - Preference model trained on 3000 historical survey records.",
        "  - 80/20 train-test split with 5-fold CV.",
        "  - Metrics saved to outputs/preference_model_metrics.csv",
        "",
        "PART B: ALLOCATION COMPARISON",
        "  (All scores use the TRUE hidden satisfaction; NOT model predictions)",
        "",
    ]
    # Find best method by mean satisfaction
    best_row = eval_df.loc[eval_df["mean_satisfaction"].idxmax()]
    for _, row in eval_df.iterrows():
        lines.append(f"  Method: {row['method']}")
        lines.append(f"    Allocation rate:    {row['allocation_rate_%']}%")
        lines.append(f"    Mean satisfaction:  {row['mean_satisfaction']}")
        lines.append(f"    Happy (>=7):        {row['happy_%']}%")
        lines.append(f"    Gini (lower=fairer):{row['gini']}")
        lines.append(f"    First-choice match: {row['first_choice_match_%']}%")
        lines.append(f"    Constraint violations: {row['constraint_violations']}")
        lines.append("")

    lines += [
        f"CONCLUSION: {best_row['method']} achieves the highest mean true satisfaction "
        f"({best_row['mean_satisfaction']}) with {best_row['constraint_violations']} "
        f"constraint violations.",
        "",
        "LIMITATIONS:",
        "  - All data is synthetic (generated from a known formula).",
        "  - The preference model is trained on noisy observations; real surveys",
        "    would yield better predictions.",
        "  - Room-level assignment within a hostel is simplified (single bed per slot).",
        "",
        "FUTURE SCOPE:",
        "  - Integrate real application data via an institutional portal.",
        "  - Add dynamic reallocation when a student cancels.",
        "  - Use multi-objective optimisation (satisfaction + fairness Pareto front).",
        "  - Include semester-2 rollover (priority for staying students).",
        "=" * 70,
    ]

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report written -> {REPORT_TXT.name}")
