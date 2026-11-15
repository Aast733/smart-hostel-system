"""
pages/5_Room_Allocation.py  –  PART B: Room Allocation dashboard.

Allows users to:
  - Adjust rule settings (budget tolerance, priority weights, roommate weight)
  - Run any allocation method on demand
  - View the allocation table and waitlist
  - See per-hostel occupancy breakdown
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import OUTPUTS_DIR, STUDENTS_CSV, HOSTEL_CSV, PREF_MODEL_PATH, HOSTELS

st.set_page_config(page_title="Room Allocation", page_icon="🏨", layout="wide")
st.markdown("# 🏨 Room Allocation — Part B")
st.caption(
    "Intelligent room assignment using the Hungarian algorithm + roommate compatibility. "
    "Hard constraints are never violated."
)

@st.cache_data
def load_students():
    return pd.read_csv(STUDENTS_CSV)

@st.cache_data
def load_hostels():
    return pd.read_csv(HOSTEL_CSV)

@st.cache_resource
def load_pref_model():
    import joblib
    return joblib.load(PREF_MODEL_PATH) if PREF_MODEL_PATH.exists() else None

student_df = load_students()
hostel_df  = load_hostels()
pref_model = load_pref_model()

# ── Sidebar controls ───────────────────────────────────────────────────────────
st.sidebar.subheader("⚙️ Allocation Settings")
budget_tolerance = st.sidebar.slider(
    "Budget Tolerance (₹)", 0, 20000, 5000, step=500,
    help="Students can be placed in hostels up to this much above their budget.",
)
roommate_weight = st.sidebar.slider(
    "Roommate Weight", 0.0, 0.5, 0.15, step=0.05,
    help="Weight of roommate compatibility in the local search pass.",
)

st.sidebar.subheader("Priority Weights")
pw_firstyr   = st.sidebar.slider("First Year",   1.0, 8.0, 3.0, step=0.5)
pw_disabled  = st.sidebar.slider("Disabled",      1.0, 8.0, 4.0, step=0.5)
pw_scholar   = st.sidebar.slider("Scholarship",   1.0, 8.0, 2.0, step=0.5)
pw_faraway   = st.sidebar.slider("Far Away",      1.0, 8.0, 2.5, step=0.5)

method_choice = st.sidebar.selectbox(
    "Run Allocation Method",
    ["Hungarian (Optimal)", "Greedy FCFS", "Random"],
)

run_btn = st.sidebar.button("🚀 Run Allocation", type="primary", use_container_width=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 Allocation Table", "⏳ Waitlist", "📊 Occupancy"])

# ── Try loading existing results ───────────────────────────────────────────────
def _load_existing_alloc(method_key: str):
    p = OUTPUTS_DIR / f"allocation_{method_key}.csv"
    w = OUTPUTS_DIR / f"waitlist_{method_key}.csv"
    if p.exists():
        alloc = pd.read_csv(p)
        wl    = pd.read_csv(w) if w.exists() else pd.DataFrame()
        return alloc, wl
    return pd.DataFrame(), pd.DataFrame()

method_key_map = {
    "Hungarian (Optimal)": "hungarian",
    "Greedy FCFS":         "greedy",
    "Random":              "random",
}

# ── Run allocation if button pressed ──────────────────────────────────────────
if run_btn:
    with st.spinner("Running allocation … (this may take ~30s for Hungarian)"):
        try:
            # Update config values dynamically
            import src.config as cfg
            cfg.BUDGET_TOLERANCE_INR = budget_tolerance
            cfg.PRIORITY_WEIGHTS["first_year"]  = pw_firstyr
            cfg.PRIORITY_WEIGHTS["disabled"]    = pw_disabled
            cfg.PRIORITY_WEIGHTS["scholarship"] = pw_scholar
            cfg.PRIORITY_WEIGHTS["far_away"]    = pw_faraway

            from src.prediction.preference_model import predict_satisfaction_matrix
            from src.optimization.allocators import (
                allocate_random, allocate_greedy, allocate_hungarian, save_allocation,
            )

            sat_matrix = predict_satisfaction_matrix(student_df, hostel_df, pref_model)

            mkey = method_key_map[method_choice]
            if mkey == "random":
                alloc_df, wl_df = allocate_random(student_df, hostel_df)
            elif mkey == "greedy":
                alloc_df, wl_df = allocate_greedy(student_df, hostel_df, sat_matrix)
            else:
                alloc_df, wl_df = allocate_hungarian(
                    student_df, hostel_df, sat_matrix,
                    roommate_weight_override=roommate_weight,
                )
            save_allocation(alloc_df, wl_df, mkey)
            st.success(f"✅ {method_choice} allocation complete: {len(alloc_df)} housed, {len(wl_df)} waitlisted")
            st.cache_data.clear()
        except Exception as e:
            st.error(f"Allocation failed: {e}")

# ── Load results ───────────────────────────────────────────────────────────────
mkey    = method_key_map[method_choice]
alloc_df, wl_df = _load_existing_alloc(mkey)

# ── Tab 1: Allocation table ───────────────────────────────────────────────────
with tab1:
    if alloc_df.empty:
        st.info("No allocation results yet. Click **Run Allocation** in the sidebar.")
    else:
        st.markdown(f"**{len(alloc_df)} students housed** via {method_choice}")

        # Filter controls
        c1, c2, c3 = st.columns(3)
        with c1:
            h_filter = st.multiselect("Hostel", hostel_df["hostel_id"].tolist(),
                                       default=hostel_df["hostel_id"].tolist(), key="alloc_h")
        with c2:
            g_filter = st.multiselect("Gender", ["M", "F"], default=["M", "F"], key="alloc_g")
        with c3:
            rt_filter = st.multiselect("Room Type", ["single", "double", "triple"],
                                        default=["single", "double", "triple"], key="alloc_rt")

        mask = (
            alloc_df["hostel_id"].isin(h_filter) &
            alloc_df["gender"].isin(g_filter) &
            alloc_df["room_type"].isin(rt_filter)
        )
        filtered = alloc_df[mask]
        st.dataframe(filtered, use_container_width=True, height=400)
        st.download_button("⬇ Download allocation CSV",
                           filtered.to_csv(index=False), f"allocation_{mkey}.csv",
                           "text/csv", key="dl_alloc")

# ── Tab 2: Waitlist ────────────────────────────────────────────────────────────
with tab2:
    if wl_df.empty and alloc_df.empty:
        st.info("No allocation results yet.")
    elif wl_df.empty:
        st.success("🎉 No students on the waitlist!")
    else:
        st.markdown(f"**{len(wl_df)} students on waitlist**")

        if "reason" in wl_df.columns:
            reason_counts = wl_df["reason"].value_counts().reset_index()
            reason_counts.columns = ["Reason", "Count"]
            col_w1, col_w2 = st.columns([1, 1])
            with col_w1:
                fig = px.pie(reason_counts, names="Reason", values="Count",
                             title="Waitlist Reasons",
                             color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig, use_container_width=True)
            with col_w2:
                st.dataframe(wl_df, use_container_width=True, height=350)
        else:
            st.dataframe(wl_df, use_container_width=True)

        st.download_button("⬇ Download waitlist", wl_df.to_csv(index=False),
                           f"waitlist_{mkey}.csv", "text/csv", key="dl_wl")

# ── Tab 3: Occupancy ───────────────────────────────────────────────────────────
with tab3:
    if alloc_df.empty:
        st.info("No allocation results yet.")
    else:
        hostel_capacity = {h["hostel_id"]: h["total_beds"] for h in HOSTELS}
        if "hostel_id" in alloc_df.columns:
            housed_counts = alloc_df.groupby("hostel_id")["student_id"].count().reset_index()
            housed_counts.columns = ["hostel_id", "housed"]
            housed_counts["capacity"] = housed_counts["hostel_id"].map(hostel_capacity)
            housed_counts["occupancy_%"] = (housed_counts["housed"] / housed_counts["capacity"] * 100).round(1)
            housed_counts["empty"] = housed_counts["capacity"] - housed_counts["housed"]

            col_o1, col_o2 = st.columns(2)
            with col_o1:
                fig_occ = go.Figure()
                fig_occ.add_trace(go.Bar(
                    x=housed_counts["hostel_id"], y=housed_counts["housed"],
                    name="Housed", marker_color="#6366f1",
                ))
                fig_occ.add_trace(go.Bar(
                    x=housed_counts["hostel_id"], y=housed_counts["empty"],
                    name="Empty Beds", marker_color="#334155",
                ))
                fig_occ.update_layout(
                    barmode="stack", title="Bed Utilisation per Hostel",
                    yaxis_title="Beds", height=380,
                )
                st.plotly_chart(fig_occ, use_container_width=True)

            with col_o2:
                fig_pct = px.bar(
                    housed_counts, x="hostel_id", y="occupancy_%",
                    color="occupancy_%",
                    color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
                    range_color=[0, 100],
                    title="Occupancy Rate %",
                    labels={"occupancy_%": "Occupancy %", "hostel_id": "Hostel"},
                    text="occupancy_%",
                )
                fig_pct.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_pct.update_layout(height=380)
                st.plotly_chart(fig_pct, use_container_width=True)

            st.dataframe(housed_counts, use_container_width=True)
