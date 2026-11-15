"""
allocators.py  -  PART B: Room Allocation Algorithms.

This module implements THREE allocation strategies so we can compare them:

  1. RANDOM BASELINE
     - Randomly shuffles students and assigns them to the first available
       feasible bed.  Fast but ignores preferences entirely.

  2. GREEDY FIRST-COME-FIRST-SERVED (FCFS)
     - Processes students in application order (priority-weighted).
     - Each student gets the highest predicted-satisfaction hostel that still
       has capacity.

  3. OPTIMAL HUNGARIAN ALLOCATION  (our proposed method)
     - Builds a Student x Bed cost matrix using:
         cost = -(priority_weight x predicted_satisfaction)
       so the optimizer maximises priority-weighted satisfaction.
     - Solves the Linear Sum Assignment Problem (LSAP) via
       scipy.optimize.linear_sum_assignment  (Hungarian algorithm).
     - Infeasible pairs get cost = +∞ so they're never assigned.
     - After assignment, a roommate local-search swap pass is run
       (see roommate.py) to reduce sleep-schedule conflicts.

  4. ORACLE UPPER BOUND
     - Same as Hungarian but uses TRUE satisfaction scores instead of
       model predictions.  This shows the ceiling performance.

NOTE: This module belongs to PART B (Optimization). No ML training happens here.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    RANDOM_SEED, HOSTELS, BUDGET_TOLERANCE_INR, ALLOCATION_CSV, WAITLIST_CSV,
    OUTPUTS_DIR,
)
from src.optimization.constraints import (
    build_feasibility_matrix,
    get_student_priority,
)


# ─────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────

def _build_bed_list(hostel_df: pd.DataFrame) -> list:
    """
    Expand the hostel table into a flat list of individual BED SLOTS.
    Each slot: {hostel_id, room_id, room_type, capacity, floor_number}

    Floor assignment strategy (simple round-robin):
      - Triple rooms: floors 1..2  (ground-floor accessible for disabled)
      - Double rooms: floors 2..N-1 (middle floors)
      - Single rooms: top floors N-1..N
    All assignments are min-clamped to [1, floors].
    """
    beds = []

    for _, row in hostel_df.iterrows():
        hid    = row["hostel_id"]
        floors = max(int(row["floors"]), 1)
        room_no = 1

        # ── Triples on low floors (ground accessible) ──────────────────────
        for i in range(int(row["triple_rooms"])):
            floor = 1 + (i % max(floors // 2, 1))
            floor = min(floor, floors)
            for seat in range(3):
                beds.append({
                    "hostel_id":   hid,
                    "room_id":     f"{hid}_T{room_no:03d}",
                    "room_type":   "triple",
                    "capacity":    3,
                    "floor_number": floor,
                    "seat":        seat,
                })
            room_no += 1

        # ── Doubles on middle floors ────────────────────────────────────────
        mid_base = max(floors // 2, 1)
        for i in range(int(row["double_rooms"])):
            floor = mid_base + (i % max(floors - mid_base + 1, 1))
            floor = min(floor, floors)
            for seat in range(2):
                beds.append({
                    "hostel_id":   hid,
                    "room_id":     f"{hid}_D{room_no:03d}",
                    "room_type":   "double",
                    "capacity":    2,
                    "floor_number": floor,
                    "seat":        seat,
                })
            room_no += 1

        # ── Singles on upper floors ─────────────────────────────────────────
        for i in range(int(row["single_rooms"])):
            floor = floors - (i % max(floors, 1))
            floor = max(floor, 1)
            beds.append({
                "hostel_id":   hid,
                "room_id":     f"{hid}_S{room_no:03d}",
                "room_type":   "single",
                "capacity":    1,
                "floor_number": floor,
            })
            room_no += 1

    return beds



def _make_output_df(student_df, assignments: dict, bed_list: list, reason: str) -> tuple:
    """
    Build the final allocation DataFrame and the waitlist.

    Parameters
    ----------
    assignments : {student_id -> bed_index}  or {student_id -> None if waitlisted}
    bed_list    : list of bed slot dicts
    reason      : method name (for logging)

    Returns (allocation_df, waitlist_df)
    """
    alloc_rows   = []
    waitlist_rows= []

    student_lu = student_df.set_index("student_id").to_dict("index")

    for _, stu in student_df.iterrows():
        sid   = stu["student_id"]
        bidx  = assignments.get(sid)

        if bidx is None or bidx == -1:
            wl_reason = assignments.get(f"{sid}_reason", "no_feasible_bed")
            waitlist_rows.append({
                "student_id":    sid,
                "gender":        stu["gender"],
                "year_of_study": stu["year_of_study"],
                "disability":    stu["disability"],
                "scholarship":   stu["scholarship"],
                "reason":        wl_reason,
            })
        else:
            bed = bed_list[bidx]
            alloc_rows.append({
                "student_id":    sid,
                "hostel_id":     bed["hostel_id"],
                "room_id":       bed["room_id"],
                "room_type":     bed["room_type"],
                "floor_number":  bed["floor_number"],
                "gender":        stu["gender"],
                "year_of_study": stu["year_of_study"],
                "budget_inr":    stu["budget_inr"],
                "disability":    stu["disability"],
                "scholarship":   stu["scholarship"],
                "first_choice_hostel": stu["first_choice_hostel"],
                "sleep_schedule":stu.get("sleep_schedule", "normal"),
            })

    alloc_df    = pd.DataFrame(alloc_rows)
    waitlist_df = pd.DataFrame(waitlist_rows)
    return alloc_df, waitlist_df


# ─────────────────────────────────────────────
# 1.  RANDOM BASELINE
# ─────────────────────────────────────────────
def allocate_random(
    student_df:  pd.DataFrame,
    hostel_df:   pd.DataFrame,
    sat_matrix:  pd.DataFrame = None,  # not used, but accepted for uniform interface
    seed:        int = RANDOM_SEED,
) -> tuple:
    """
    Randomly assign students to feasible beds.
    Students who cannot be housed are waitlisted.
    """
    rng = np.random.default_rng(seed)
    feasibility = build_feasibility_matrix(student_df, hostel_df)
    bed_list    = _build_bed_list(hostel_df)

    # Shuffle students and beds
    student_order = rng.permutation(student_df.index.tolist())
    bed_order     = rng.permutation(len(bed_list)).tolist()

    bed_assigned  = [False] * len(bed_list)
    assignments   = {}

    for sidx in student_order:
        stu  = student_df.loc[sidx]
        sid  = stu["student_id"]
        placed = False

        for bidx in bed_order:
            if bed_assigned[bidx]:
                continue
            bed = bed_list[bidx]
            hid = bed["hostel_id"]

            if not feasibility.at[sid, hid]:
                continue
            # Disability floor check
            if stu["disability"] and bed["floor_number"] != 1:
                continue

            bed_assigned[bidx]  = True
            assignments[sid]    = bidx
            placed = True
            break

        if not placed:
            assignments[sid]           = -1
            assignments[f"{sid}_reason"] = "no_feasible_bed"

    return _make_output_df(student_df, assignments, bed_list, "Random")


# ─────────────────────────────────────────────
# 2.  GREEDY FCFS BASELINE
# ─────────────────────────────────────────────
def allocate_greedy(
    student_df:  pd.DataFrame,
    hostel_df:   pd.DataFrame,
    sat_matrix:  pd.DataFrame,
) -> tuple:
    """
    Greedy First-Come-First-Served:
      - Process students from highest priority to lowest.
      - Each student gets the feasible hostel with highest predicted satisfaction
        that still has available beds.
    """
    feasibility = build_feasibility_matrix(student_df, hostel_df)
    bed_list    = _build_bed_list(hostel_df)

    # Count available beds per hostel
    hostel_beds = {}
    for i, bed in enumerate(bed_list):
        hostel_beds.setdefault(bed["hostel_id"], []).append(i)

    # Sort students by priority (high first)
    priorities = student_df.apply(get_student_priority, axis=1)
    sorted_idx = priorities.sort_values(ascending=False).index.tolist()

    bed_assigned = [False] * len(bed_list)
    assignments  = {}

    hostel_ids = hostel_df["hostel_id"].tolist()

    for sidx in sorted_idx:
        stu  = student_df.loc[sidx]
        sid  = stu["student_id"]

        # Get predicted satisfaction for each hostel (if sat_matrix provided)
        if sat_matrix is not None and sid in sat_matrix.index:
            sat_row = sat_matrix.loc[sid]
        else:
            sat_row = pd.Series(5.0, index=hostel_ids)

        # Sort hostels by satisfaction, descending
        hostel_order = (
            sat_row[hostel_ids]
            .dropna()
            .sort_values(ascending=False)
            .index.tolist()
        )

        placed = False
        for hid in hostel_order:
            if not feasibility.at[sid, hid]:
                continue

            # Find first available bed in this hostel
            for bidx in hostel_beds.get(hid, []):
                if bed_assigned[bidx]:
                    continue
                bed = bed_list[bidx]
                # Disability check
                if stu["disability"] and bed["floor_number"] != 1:
                    continue

                bed_assigned[bidx]   = True
                assignments[sid]     = bidx
                placed = True
                break
            if placed:
                break

        if not placed:
            assignments[sid]           = -1
            # Determine reason
            feasible_hostels = [h for h in hostel_ids if feasibility.at[sid, h]]
            if not feasible_hostels:
                reason = "no_feasible_hostel"
            else:
                reason = "all_beds_full"
            assignments[f"{sid}_reason"] = reason

    return _make_output_df(student_df, assignments, bed_list, "Greedy")


# ─────────────────────────────────────────────
# 3.  OPTIMAL HUNGARIAN ALLOCATION
# ─────────────────────────────────────────────
def allocate_hungarian(
    student_df:  pd.DataFrame,
    hostel_df:   pd.DataFrame,
    sat_matrix:  pd.DataFrame,
    priority_weights_override: dict = None,
    roommate_weight_override:  float = None,
) -> tuple:
    """
    Solve the Linear Sum Assignment Problem (Hungarian algorithm) on a
    Student x Bed cost matrix to maximise priority-weighted satisfaction.

    Cost formula:
        cost[i, j] = -(priority_i x predicted_satisfaction_i_hostel_j)
    Infeasible pairs -> cost = +1e9 (effectively infinity).

    After solving, a swap-based roommate pass is applied (roommate.py).

    Parameters
    ----------
    sat_matrix : pd.DataFrame  rows=student_id, cols=hostel_id, values=predicted_sat

    Returns
    -------
    (allocation_df, waitlist_df)
    """
    from src.optimization.roommate import improve_roommate_compatibility
    from src.config import ROOMMATE_WEIGHT

    rw = roommate_weight_override if roommate_weight_override is not None else ROOMMATE_WEIGHT

    feasibility = build_feasibility_matrix(student_df, hostel_df)
    bed_list    = _build_bed_list(hostel_df)
    n_students  = len(student_df)
    n_beds      = len(bed_list)

    # Index mappings
    student_ids = student_df["student_id"].tolist()
    hostel_ids  = hostel_df["hostel_id"].tolist()
    student_idx = {sid: i for i, sid in enumerate(student_ids)}
    bed_hid     = [b["hostel_id"] for b in bed_list]
    bed_floor   = [b["floor_number"] for b in bed_list]

    # Priority weights per student
    priorities = student_df.apply(get_student_priority, axis=1).values  # shape (n_students,)

    # ── Build cost matrix (n_students x n_beds) ───────────────────────────────
    INF_COST = 1e9
    cost_mat = np.full((n_students, n_beds), INF_COST)

    for i, sid in enumerate(student_ids):
        stu = student_df.loc[student_df["student_id"] == sid].iloc[0]
        for j, bed in enumerate(bed_list):
            hid = bed["hostel_id"]
            # Hard constraint checks
            if not feasibility.at[sid, hid]:
                continue
            if stu["disability"] and bed["floor_number"] != 1:
                continue

            # Get predicted satisfaction for this (student, hostel) pair
            if sat_matrix is not None and sid in sat_matrix.index and hid in sat_matrix.columns:
                pred_sat = float(sat_matrix.at[sid, hid])
            else:
                pred_sat = 5.0   # neutral fallback

            cost_mat[i, j] = -(priorities[i] * pred_sat)

    # ── Solve LSAP ───────────────────────────────────────────────────────────
    # linear_sum_assignment minimises cost; we used negative satisfaction -> maximises
    # Handle rectangular matrices: pad if more beds than students
    if n_beds >= n_students:
        row_ind, col_ind = linear_sum_assignment(cost_mat)
    else:
        # More students than beds: transpose trick
        row_ind, col_ind = linear_sum_assignment(cost_mat[:, :n_beds])

    # ── Extract assignments ───────────────────────────────────────────────────
    bed_taken    = {}   # bed_idx -> student_id
    assignments  = {}

    for r, c in zip(row_ind, col_ind):
        if cost_mat[r, c] >= INF_COST:
            # LSAP had no choice but to use an infeasible pair -> waitlist
            sid = student_ids[r]
            assignments[sid]           = -1
            assignments[f"{sid}_reason"] = "no_feasible_bed"
        else:
            sid = student_ids[r]
            assignments[sid] = c
            bed_taken[c]     = sid

    # Students not in row_ind (if more students than beds)
    assigned_students = set(student_ids[r] for r in row_ind)
    for sid in student_ids:
        if sid not in assigned_students:
            assignments[sid]           = -1
            assignments[f"{sid}_reason"] = "insufficient_beds"

    alloc_df, waitlist_df = _make_output_df(student_df, assignments, bed_list, "Hungarian")

    # ── Roommate compatibility improvement ────────────────────────────────────
    if len(alloc_df) > 0:
        alloc_df = improve_roommate_compatibility(alloc_df, student_df, rw)

    return alloc_df, waitlist_df


# ─────────────────────────────────────────────
# 4.  ORACLE UPPER BOUND
# ─────────────────────────────────────────────
def allocate_oracle(
    student_df:  pd.DataFrame,
    hostel_df:   pd.DataFrame,
    true_sat_fn,
) -> tuple:
    """
    Oracle allocator uses the TRUE (hidden) satisfaction function, not the model.
    This gives the theoretical upper bound on allocation quality.

    Parameters
    ----------
    true_sat_fn : callable(student_dict, hostel_dict, room_type) -> float (1-10)
    """
    from src.optimization.roommate import improve_roommate_compatibility
    from src.config import ROOMMATE_WEIGHT

    # Build hostel_map with hostel_id included inside each dict
    # set_index drops hostel_id as a column, so we inject it back explicitly
    hostel_map = {
        hid: {**row, "hostel_id": hid}
        for hid, row in hostel_df.set_index("hostel_id").to_dict("index").items()
    }

    # Build true satisfaction matrix (student x hostel)
    import numpy as _np
    records = []
    for _, stu in student_df.iterrows():
        row = {"student_id": stu["student_id"]}
        for hid, hos in hostel_map.items():
            rt = stu.get("room_type_pref", "double")
            ts = true_sat_fn(stu.to_dict(), hos, rt, _np.random.default_rng(RANDOM_SEED))
            row[hid] = ts
        records.append(row)

    true_matrix = pd.DataFrame(records).set_index("student_id")

    # Re-use Hungarian with the true matrix
    return allocate_hungarian(student_df, hostel_df, true_matrix)


def save_allocation(alloc_df: pd.DataFrame, waitlist_df: pd.DataFrame, method: str = "") -> None:
    """Persist allocation and waitlist to the outputs folder."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{method.lower().replace(' ', '_')}" if method else ""
    alloc_path = OUTPUTS_DIR / f"allocation{suffix}.csv"
    wl_path    = OUTPUTS_DIR / f"waitlist{suffix}.csv"
    alloc_df.to_csv(alloc_path, index=False)
    waitlist_df.to_csv(wl_path, index=False)
    print(f"  Allocation saved -> {alloc_path.name}  ({len(alloc_df)} housed, {len(waitlist_df)} waitlisted)")
