"""
pages/7_Student_Lookup.py  –  Per-student allocation lookup.

Enter a student ID to see:
  - Allocated hostel and room
  - True and predicted satisfaction score
  - Reason for allocation or waitlisting
  - Student's priority flags
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import OUTPUTS_DIR, STUDENTS_CSV, HOSTEL_CSV, PREF_MODEL_PATH

st.set_page_config(page_title="Student Lookup", page_icon="👤", layout="wide")
st.markdown("# 👤 Student Lookup")
st.caption("Enter a student ID to see their allocation, satisfaction score, and details.")

@st.cache_data
def load_students():
    return pd.read_csv(STUDENTS_CSV)

@st.cache_data
def load_hostels():
    return pd.read_csv(HOSTEL_CSV)

@st.cache_data
def load_alloc(method_key):
    p = OUTPUTS_DIR / f"allocation_{method_key}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_data
def load_waitlist(method_key):
    p = OUTPUTS_DIR / f"waitlist_{method_key}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

student_df = load_students()
hostel_df  = load_hostels()

method_choice = st.selectbox(
    "View results from allocation method:",
    ["Hungarian (Optimal)", "Greedy FCFS", "Random", "Oracle"],
)
method_key_map = {
    "Hungarian (Optimal)": "hungarian",
    "Greedy FCFS":         "greedy",
    "Random":              "random",
    "Oracle":              "oracle",
}
mkey      = method_key_map[method_choice]
alloc_df  = load_alloc(mkey)
wl_df     = load_waitlist(mkey)

# ── Search box ─────────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([3, 1])
with col_input:
    student_id = st.text_input(
        "Student ID (e.g. S0001):",
        placeholder="S0001",
        key="student_id_input",
    )
with col_btn:
    search = st.button("🔍 Look Up", use_container_width=True)

# Quick random example
if st.button("🎲 Random Example", use_container_width=False):
    import random
    random.seed(42)
    if not alloc_df.empty:
        student_id = random.choice(alloc_df["student_id"].tolist())
        st.session_state["student_id_input"] = student_id

if not student_id:
    st.info("Enter a Student ID above (e.g. S0001, S0042, S1399) and click Look Up.")
    st.stop()

# ── Look up student profile ────────────────────────────────────────────────────
student_id = student_id.strip().upper()
stu_rows = student_df[student_df["student_id"] == student_id]
if stu_rows.empty:
    st.error(f"Student **{student_id}** not found in the student database.")
    st.stop()

stu = stu_rows.iloc[0]
st.divider()

# ── Layout ────────────────────────────────────────────────────────────────────
col_profile, col_result = st.columns([1, 1.5])

with col_profile:
    st.subheader("🎓 Student Profile")
    gender_icon = "🧑" if stu["gender"] == "M" else "👩"
    st.markdown(f"""
    | Field | Value |
    |---|---|
    | Student ID | `{stu['student_id']}` |
    | Gender | {gender_icon} {stu['gender']} |
    | Year of Study | Year {stu['year_of_study']} |
    | Budget | ₹{int(stu['budget_inr']):,} |
    | AC Preference | {'✅ Yes' if stu['ac_preference'] else '❌ No'} |
    | Sleep Schedule | {stu.get('sleep_schedule', 'N/A')} |
    | Room Type Pref | {stu.get('room_type_pref', 'N/A')} |
    | Home Distance | {stu['home_distance_km']:.1f} km |
    | First Choice Hostel | {stu['first_choice_hostel']} |
    """)

    st.markdown("**Priority Flags:**")
    flags = []
    if stu.get("first_year"):   flags.append("🔵 First Year")
    if stu.get("disability"):   flags.append("♿ Disabled")
    if stu.get("scholarship"):  flags.append("🎓 Scholarship")
    if stu.get("far_away"):     flags.append("🌍 Far Away (>200km)")
    if flags:
        for f in flags:
            st.markdown(f"  - {f}")
    else:
        st.markdown("  - No priority flags")

    preferences = str(stu.get("ranked_preferences", "")).split(",")
    st.markdown(f"**Ranked Hostel Preferences:** {' → '.join(preferences)}")

with col_result:
    st.subheader(f"🏠 Allocation Result ({method_choice})")

    # Check allocation
    if not alloc_df.empty:
        alloc_row = alloc_df[alloc_df["student_id"] == student_id]
    else:
        alloc_row = pd.DataFrame()

    if not wl_df.empty:
        wl_row = wl_df[wl_df["student_id"] == student_id]
    else:
        wl_row = pd.DataFrame()

    if not alloc_row.empty:
        ar = alloc_row.iloc[0]
        # Get hostel details
        hos_row = hostel_df[hostel_df["hostel_id"] == ar["hostel_id"]]
        hos = hos_row.iloc[0] if not hos_row.empty else pd.Series()

        # True satisfaction
        try:
            from src.evaluation.metrics import _true_satisfaction_eval
            true_sat = _true_satisfaction_eval(stu.to_dict(), hos.to_dict(), ar.get("room_type", "double"))
        except Exception:
            true_sat = None

        # Predicted satisfaction
        pred_sat = None
        try:
            import joblib
            from src.prediction.preference_model import predict_satisfaction_matrix
            pref_model = joblib.load(PREF_MODEL_PATH)
            sat_mat = predict_satisfaction_matrix(stu_rows, hostel_df, pref_model)
            if not sat_mat.empty and ar["hostel_id"] in sat_mat.columns:
                pred_sat = float(sat_mat.at[student_id, ar["hostel_id"]])
        except Exception:
            pass

        st.success("✅ Student is HOUSED")
        st.markdown(f"""
        | Field | Value |
        |---|---|
        | Allocated Hostel | **{ar['hostel_id']}** – {hos.get('name', '')} |
        | Room ID | `{ar.get('room_id', 'N/A')}` |
        | Room Type | {ar.get('room_type', 'N/A')} |
        | Floor | {ar.get('floor_number', 'N/A')} |
        | Annual Fee | ₹{int(hos.get('annual_fee_inr', 0)):,} |
        | AC | {'✅ Yes' if hos.get('ac') else '❌ No'} |
        | First Choice Match | {'✅ Yes' if ar['hostel_id'] == stu['first_choice_hostel'] else '❌ No'} |
        """)

        col_s1, col_s2 = st.columns(2)
        if true_sat is not None:
            sat_color = "#22c55e" if true_sat >= 7 else "#f59e0b" if true_sat >= 5 else "#ef4444"
            with col_s1:
                st.markdown(
                    f'<div style="background:#1e293b;border-radius:8px;padding:1rem;text-align:center;">'
                    f'<div style="font-size:2rem;font-weight:700;color:{sat_color}">{true_sat:.2f}/10</div>'
                    '<div style="color:#94a3b8;font-size:0.8rem">True Satisfaction (hidden oracle)</div></div>',
                    unsafe_allow_html=True,
                )
        if pred_sat is not None:
            with col_s2:
                st.markdown(
                    f'<div style="background:#1e293b;border-radius:8px;padding:1rem;text-align:center;">'
                    f'<div style="font-size:2rem;font-weight:700;color:#6366f1">{pred_sat:.2f}/10</div>'
                    '<div style="color:#94a3b8;font-size:0.8rem">Predicted Satisfaction (model)</div></div>',
                    unsafe_allow_html=True,
                )

        if not hos.empty:
            st.markdown("**Hostel Details:**")
            st.markdown(f"""
            - Distance: {hos.get('distance_km', 'N/A')} km from campus
            - Mess Rating: {'⭐' * int(round(hos.get('mess_rating', 0)))} ({hos.get('mess_rating', 'N/A')})
            - WiFi Rating: {'⭐' * int(round(hos.get('wifi_rating', 0)))} ({hos.get('wifi_rating', 'N/A')})
            - Floors: {hos.get('floors', 'N/A')}
            """)

    elif not wl_row.empty:
        wr = wl_row.iloc[0]
        st.error(f"⏳ Student is on the **WAITLIST**")
        st.markdown(f"""
        | Field | Value |
        |---|---|
        | Student ID | `{student_id}` |
        | Reason | **{wr.get('reason', 'Unknown')}** |
        | Gender | {wr.get('gender', 'N/A')} |
        | Disability | {'Yes' if wr.get('disability') else 'No'} |
        """)
        reason = wr.get("reason", "")
        st.markdown(f"""
        **What this means:**
        - `no_feasible_hostel` → No gender-appropriate hostel within budget
        - `all_beds_full` → All feasible hostels are at capacity
        - `no_feasible_bed` → Could not satisfy constraints
        - `insufficient_beds` → More students than total beds

        **Recommendation:** This student should explore off-campus accommodation or apply again next semester.
        """)
    else:
        if alloc_df.empty and wl_df.empty:
            st.warning(f"Allocation not found for method **{method_choice}**. Run the pipeline first.")
        else:
            st.warning(f"Student **{student_id}** is not in the allocation or waitlist for {method_choice}.")
