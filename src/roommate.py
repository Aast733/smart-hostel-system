"""
roommate.py  -  PART B: Roommate Compatibility Local Search.

After the Hungarian algorithm assigns students to beds, this module runs a
local search (swap-based) to reduce ROOMMATE SLEEP-SCHEDULE CONFLICTS.

PROBLEM:
  If a night-owl and an early-bird share a room, both are unhappy.
  The Hungarian algorithm optimizes individual satisfaction, not pairings.

SOLUTION (Local Search):
  1. Group allocated students by room_id.
  2. For multi-occupancy rooms (double, triple), compute the "mismatch score":
       - Mismatch = 1 if room contains both "early" and "night" sleepers.
       - Mismatch = 0 if all sleepers are in the same category.
  3. Try every possible swap of two students between different rooms.
  4. Accept the swap if it reduces total mismatch AND does not violate
     any hard constraints (gender, budget, floor for disability).
  5. Repeat until no improving swap is found (local optimum).

COMPLEXITY: O(n²) swaps per iteration.  For ~1000 students this is ~500k
  checks per pass, which is fast in practice.

NOTE: This module is PART B (Optimization).  No ML here.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import RANDOM_SEED, BUDGET_TOLERANCE_INR


def _sleep_mismatch(schedules: list) -> int:
    """
    Compute mismatch for a room's occupant sleep schedules.
    Returns 1 if both 'early' and 'night' are present, else 0.
    """
    sched_set = set(schedules)
    if "early" in sched_set and "night" in sched_set:
        return 1
    return 0


def _total_mismatch(alloc_df: pd.DataFrame) -> int:
    """Count total mismatch rooms across the whole allocation."""
    if "sleep_schedule" not in alloc_df.columns:
        return 0
    total = 0
    for room_id, group in alloc_df.groupby("room_id"):
        if len(group) > 1:
            total += _sleep_mismatch(group["sleep_schedule"].tolist())
    return total


def improve_roommate_compatibility(
    alloc_df:   pd.DataFrame,
    student_df: pd.DataFrame,
    roommate_weight: float = 0.15,
    max_passes:  int = 20,
) -> pd.DataFrame:
    """
    Run swap-based local search to reduce sleep-schedule mismatches.

    Parameters
    ----------
    alloc_df    : allocation table (output of one of the allocators)
    student_df  : student profiles (for looking up sleep_schedule)
    roommate_weight : unused numerically here but noted in output
    max_passes  : maximum number of full-scan passes before stopping

    Returns
    -------
    Improved allocation DataFrame (same schema).
    """
    if "sleep_schedule" not in alloc_df.columns:
        # Merge sleep_schedule from student_df
        sleep_map = student_df.set_index("student_id")["sleep_schedule"].to_dict()
        alloc_df  = alloc_df.copy()
        alloc_df["sleep_schedule"] = alloc_df["student_id"].map(sleep_map).fillna("normal")

    alloc_df = alloc_df.copy().reset_index(drop=True)

    # Build hostel lookup for constraint checks
    # (we only check gender since budget was already verified at assignment)
    hostel_gender_map = (
        alloc_df[["student_id", "hostel_id"]]
        .set_index("student_id")["hostel_id"]
        .to_dict()
    )

    initial_mismatch = _total_mismatch(alloc_df)
    if initial_mismatch == 0:
        return alloc_df   # Nothing to improve

    improved = True
    passes   = 0

    while improved and passes < max_passes:
        improved = False
        passes  += 1
        current_mm = _total_mismatch(alloc_df)
        if current_mm == 0:
            break

        indices = alloc_df.index.tolist()
        swap_found = False

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_i = indices[i]
                idx_j = indices[j]
                row_i = alloc_df.loc[idx_i]
                row_j = alloc_df.loc[idx_j]

                # Only swap rooms within the same hostel (preserves fee, budget, and hostel preference)
                if row_i["hostel_id"] != row_j["hostel_id"]:
                    continue

                # Skip if same room (no point swapping within same room)
                if row_i["room_id"] == row_j["room_id"]:
                    continue

                # Skip if different genders (rooms are gender-segregated by hostel)
                if row_i["gender"] != row_j["gender"]:
                    continue

                # Try swapping hostel/room assignments
                # We swap room_id, hostel_id, room_type, floor_number
                # CONSTRAINT: after swap, disability rule must still hold
                if row_i["disability"] and row_j["floor_number"] != 1:
                    continue
                if row_j["disability"] and row_i["floor_number"] != 1:
                    continue

                # Compute mismatch before and after swap
                # Current room composition
                room_i_members = alloc_df[alloc_df["room_id"] == row_i["room_id"]]
                room_j_members = alloc_df[alloc_df["room_id"] == row_j["room_id"]]

                before_mm = (
                    _sleep_mismatch(room_i_members["sleep_schedule"].tolist()) +
                    _sleep_mismatch(room_j_members["sleep_schedule"].tolist())
                )

                # Simulate swap
                scheds_i_new = (
                    room_i_members[room_i_members.index != idx_i]["sleep_schedule"].tolist() +
                    [row_j["sleep_schedule"]]
                )
                scheds_j_new = (
                    room_j_members[room_j_members.index != idx_j]["sleep_schedule"].tolist() +
                    [row_i["sleep_schedule"]]
                )
                after_mm = _sleep_mismatch(scheds_i_new) + _sleep_mismatch(scheds_j_new)

                if after_mm < before_mm:
                    # Accept swap
                    cols_to_swap = ["hostel_id", "room_id", "room_type", "floor_number"]
                    tmp = alloc_df.loc[idx_i, cols_to_swap].copy()
                    alloc_df.loc[idx_i, cols_to_swap] = alloc_df.loc[idx_j, cols_to_swap].values
                    alloc_df.loc[idx_j, cols_to_swap] = tmp.values
                    improved    = True
                    swap_found  = True
                    break
            if swap_found:
                break

    final_mismatch = _total_mismatch(alloc_df)
    print(f"  [roommate] Mismatch rooms: {initial_mismatch} -> {final_mismatch} "
          f"(passes={passes})")
    return alloc_df
