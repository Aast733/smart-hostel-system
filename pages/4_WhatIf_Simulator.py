"""
pages/4_WhatIf_Simulator.py  –  What-If Demand Simulator.

Lets users change hostel parameters (fee, mess rating, capacity, cohort size)
via sliders and instantly see how the demand prediction changes.

This page demonstrates the practical value of the demand model:
  - University admins can ask "what if we reduce the fee by ₹10,000?" and see
    the predicted impact on applications before making the decision.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import DEMAND_MODEL_PATH, DEMAND_CSV, HOSTELS, OUTPUTS_DIR

st.set_page_config(page_title="What-If Simulator", page_icon="🎛️", layout="wide")
st.markdown("# 🎛️ What-If Demand Simulator")
st.caption(
    "Adjust hostel parameters and instantly see how the demand model predicts changes. "
    "Uses the trained best demand model."
)

@st.cache_resource
def load_model():
    if DEMAND_MODEL_PATH.exists():
        return joblib.load(DEMAND_MODEL_PATH)
    return None

@st.cache_data
def load_demand():
    return pd.read_csv(DEMAND_CSV)

model = load_model()
demand_df = load_demand()

if model is None:
    st.error("Model not found. Run `python run_pipeline.py` first.")
    st.stop()

# ── Get baseline features from most recent year ────────────────────────────────
from src.prediction.features import build_demand_features, get_demand_feature_columns

@st.cache_data
def get_baseline_features():
    feat_df   = build_demand_features(demand_df)
    feat_cols = get_demand_feature_columns()
    latest = (
        feat_df.sort_values(["hostel_id", "year", "semester"])
        .groupby("hostel_id")
        .last()
        .reset_index()
    )
    return latest, feat_cols

latest_df, feat_cols = get_baseline_features()

# ── Sidebar: pick hostel ───────────────────────────────────────────────────────
hostel_options = {h["hostel_id"]: f"{h['hostel_id']} – {h['name']}" for h in HOSTELS}
selected_hid   = st.selectbox(
    "Select Hostel to Simulate",
    list(hostel_options.keys()),
    format_func=lambda k: hostel_options[k],
)

baseline = latest_df[latest_df["hostel_id"] == selected_hid].iloc[0]
hostel_def = next(h for h in HOSTELS if h["hostel_id"] == selected_hid)

st.divider()
st.subheader(f"Adjust parameters for: {hostel_options[selected_hid]}")

col_ctrl, col_vis = st.columns([1, 2])

with col_ctrl:
    st.markdown("**📋 Parameter Sliders**")

    fee = st.slider(
        "Annual Fee (₹)",
        min_value=25000, max_value=150000,
        value=int(baseline["annual_fee_inr"]),
        step=1000,
        help="Higher fees typically reduce applications.",
    )
    mess = st.slider(
        "Mess Rating (1-5)",
        min_value=1.0, max_value=5.0,
        value=float(baseline["mess_rating"]),
        step=0.1,
        help="Better mess → more applicants.",
    )
    capacity = st.slider(
        "Total Beds",
        min_value=50, max_value=500,
        value=int(baseline["total_beds"]),
        step=10,
        help="More beds = higher base demand potential.",
    )
    cohort = st.slider(
        "Admitted Cohort Size",
        min_value=600, max_value=2000,
        value=int(baseline["cohort_size"]),
        step=50,
        help="More students admitted → more hostel demand.",
    )
    rent_idx = st.slider(
        "Rent Index (1.0 = base)",
        min_value=0.8, max_value=2.0,
        value=float(baseline["rent_index"]),
        step=0.05,
        help="Higher general rent → more students prefer hostel.",
    )
    sem2 = st.checkbox("Semester 2 (lower demand)", value=False)

# ── Compute prediction ─────────────────────────────────────────────────────────
max_fee_global = max(h["annual_fee_inr"] for h in HOSTELS)
max_fee_global = max(max_fee_global, fee)   # allow slider to exceed current max

def make_feature_row(fee_v, mess_v, cap_v, cohort_v, rent_v, sem2_v, baseline_row, feat_cols_):
    """Build a single feature vector for prediction."""
    return {
        "total_beds":       cap_v,
        "ac":               baseline_row["ac"],
        "distance_km":      baseline_row["distance_km"],
        "mess_rating":      mess_v,
        "wifi_rating":      baseline_row["wifi_rating"],
        "fee_ratio":        fee_v / max_fee_global,
        "floors":           baseline_row["floors"],
        "beds_per_floor":   cap_v / max(baseline_row["floors"], 1),
        "cohort_size":      cohort_v,
        "rent_index":       rent_v,
        "sem2_flag":        int(sem2_v),
        "lag1_applications":baseline_row["applications"],
        "lag2_applications":baseline_row.get("lag1_applications", baseline_row["applications"]),
        "rolling4_mean":    baseline_row.get("rolling4_mean", baseline_row["applications"]),
        "lag1_occ_rate":    baseline_row["occupancy_rate"],
    }

baseline_feat = make_feature_row(
    baseline["annual_fee_inr"], baseline["mess_rating"],
    baseline["total_beds"], baseline["cohort_size"],
    baseline["rent_index"], False, baseline, feat_cols,
)
X_base = np.array([[baseline_feat[c] for c in feat_cols]])
pred_base = float(model.predict(X_base)[0])

current_feat = make_feature_row(fee, mess, capacity, cohort, rent_idx, sem2, baseline, feat_cols)
X_curr = np.array([[current_feat[c] for c in feat_cols]])
pred_curr = float(model.predict(X_curr)[0])

delta = pred_curr - pred_base
delta_pct = (delta / max(pred_base, 1)) * 100

with col_vis:
    st.markdown("**📊 Predicted Demand Change**")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Baseline Prediction", f"{pred_base:.0f} applicants")
    with m2:
        st.metric("New Prediction", f"{pred_curr:.0f} applicants",
                  delta=f"{delta:+.0f}",
                  delta_color="normal")
    with m3:
        occ_ratio = pred_curr / max(capacity, 1)
        status    = "🔴 OVER" if occ_ratio > 1.1 else ("🟢 BALANCED" if occ_ratio >= 0.8 else "🟡 UNDER")
        st.metric("Demand/Capacity Ratio", f"{occ_ratio:.2f}", delta=status)

    st.divider()

    # ── Sensitivity sweep: vary fee ────────────────────────────────────────────
    st.markdown("**Fee Sensitivity (all other params at current slider values)**")
    fees_sweep = list(range(25000, 155000, 5000))
    preds_sweep = []
    for f_val in fees_sweep:
        feat_sw = make_feature_row(f_val, mess, capacity, cohort, rent_idx, sem2, baseline, feat_cols)
        X_sw    = np.array([[feat_sw[c] for c in feat_cols]])
        preds_sweep.append(float(model.predict(X_sw)[0]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fees_sweep, y=preds_sweep, mode="lines+markers",
        name="Predicted Applications",
        line=dict(color="#6366f1", width=2),
    ))
    fig.add_hline(y=capacity, line_dash="dash", line_color="#ef4444",
                  annotation_text="Bed Capacity", annotation_position="top left")
    fig.add_vline(x=fee, line_dash="dot", line_color="#f59e0b",
                  annotation_text="Current Fee", annotation_position="top right")
    fig.update_layout(
        title="Predicted Applications vs Annual Fee",
        xaxis_title="Annual Fee (₹)", yaxis_title="Predicted Applications",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Sensitivity sweep: vary mess rating ────────────────────────────────────
    st.markdown("**Mess Rating Sensitivity**")
    mess_sweep = [round(m, 1) for m in np.arange(1.0, 5.1, 0.2)]
    preds_mess = []
    for m_val in mess_sweep:
        feat_sw = make_feature_row(fee, m_val, capacity, cohort, rent_idx, sem2, baseline, feat_cols)
        X_sw    = np.array([[feat_sw[c] for c in feat_cols]])
        preds_mess.append(float(model.predict(X_sw)[0]))

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=mess_sweep, y=preds_mess, mode="lines+markers",
        line=dict(color="#8b5cf6", width=2),
    ))
    fig2.add_hline(y=capacity, line_dash="dash", line_color="#ef4444")
    fig2.update_layout(
        title="Predicted Applications vs Mess Rating",
        xaxis_title="Mess Rating (1-5)", yaxis_title="Predicted Applications",
        height=280,
    )
    st.plotly_chart(fig2, use_container_width=True)
