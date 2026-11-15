"""
pages/6_Evaluation.py  –  Side-by-side comparison of all allocation methods.

Shows:
  - Comparison table: allocation rate, satisfaction, Gini, first-choice, violations
  - Bar chart comparison of all metrics
  - Satisfaction distributions for each method
  - Clear conclusion statement on which method wins and why
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import EVAL_CSV, OUTPUTS_DIR, REPORT_TXT

st.set_page_config(page_title="Evaluation", page_icon="📊", layout="wide")
st.markdown("# 📊 Allocation Evaluation")
st.caption(
    "All methods are evaluated using the **TRUE hidden satisfaction** — "
    "not the model's own predictions.  This prevents circular evaluation."
)

@st.cache_data
def load_eval():
    return pd.read_csv(EVAL_CSV) if EVAL_CSV.exists() else pd.DataFrame()

@st.cache_data
def load_alloc(method_key):
    p = OUTPUTS_DIR / f"allocation_{method_key}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

eval_df = load_eval()

if eval_df.empty:
    st.warning("Evaluation results not found. Run `python run_pipeline.py` first.")
    st.stop()

# ── Highlight best values ──────────────────────────────────────────────────────
def style_eval_table(df):
    styled = df.copy()
    return styled

# ── Main metrics table ─────────────────────────────────────────────────────────
st.subheader("📋 Method Comparison Table")

display_cols = [
    "method", "n_housed", "allocation_rate_%",
    "mean_satisfaction", "median_satisfaction", "p10_satisfaction",
    "happy_%", "gini", "priority_coverage_%",
    "first_choice_match_%", "roommate_mismatch_%", "constraint_violations",
]
available = [c for c in display_cols if c in eval_df.columns]

# Color-code best values
best_method_idx = eval_df["mean_satisfaction"].idxmax()
best_method     = eval_df.loc[best_method_idx, "method"]

def highlight_row(row):
    if row["method"] == best_method:
        return ["background-color: #1a2f1a; color: #86efac"] * len(row)
    elif row["method"] == "Oracle":
        return ["background-color: #1a1a3e; color: #a5b4fc"] * len(row)
    return [""] * len(row)

styled_table = eval_df[available].style.apply(highlight_row, axis=1).format(
    {
        "allocation_rate_%":    "{:.1f}%",
        "mean_satisfaction":    "{:.3f}",
        "median_satisfaction":  "{:.3f}",
        "p10_satisfaction":     "{:.3f}",
        "happy_%":              "{:.1f}%",
        "gini":                 "{:.4f}",
        "priority_coverage_%":  "{:.1f}%",
        "first_choice_match_%": "{:.1f}%",
        "roommate_mismatch_%":  "{:.1f}%",
    }
)
st.dataframe(styled_table, use_container_width=True)
st.download_button("⬇ Download evaluation CSV", eval_df.to_csv(index=False),
                   "evaluation.csv", "text/csv", key="dl_eval")

st.divider()

# ── Bar chart comparison ───────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Mean True Satisfaction")
    fig1 = px.bar(
        eval_df, x="method", y="mean_satisfaction",
        color="method",
        color_discrete_map={
            "Random": "#64748b", "Greedy": "#f59e0b",
            "Hungarian": "#6366f1", "Oracle": "#22c55e",
        },
        text="mean_satisfaction",
        labels={"mean_satisfaction": "Mean True Satisfaction (1-10)", "method": "Method"},
    )
    fig1.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig1.update_layout(height=380, showlegend=False, yaxis_range=[0, 10])
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Happy Students (≥ 7.0)")
    fig2 = px.bar(
        eval_df, x="method", y="happy_%",
        color="method",
        color_discrete_map={
            "Random": "#64748b", "Greedy": "#f59e0b",
            "Hungarian": "#6366f1", "Oracle": "#22c55e",
        },
        text="happy_%",
        labels={"happy_%": "% Students with Satisfaction ≥ 7", "method": "Method"},
    )
    fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig2.update_layout(height=380, showlegend=False, yaxis_range=[0, 100])
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Gini Coefficient (lower = fairer)")
    fig3 = px.bar(
        eval_df, x="method", y="gini",
        color="method",
        color_discrete_map={
            "Random": "#64748b", "Greedy": "#f59e0b",
            "Hungarian": "#6366f1", "Oracle": "#22c55e",
        },
        text="gini",
        labels={"gini": "Gini (0=equal, 1=unequal)", "method": "Method"},
    )
    fig3.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig3.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Priority Group Coverage")
    fig4 = px.bar(
        eval_df, x="method", y="priority_coverage_%",
        color="method",
        color_discrete_map={
            "Random": "#64748b", "Greedy": "#f59e0b",
            "Hungarian": "#6366f1", "Oracle": "#22c55e",
        },
        text="priority_coverage_%",
        labels={"priority_coverage_%": "% Priority Students Housed", "method": "Method"},
    )
    fig4.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig4.update_layout(height=350, showlegend=False, yaxis_range=[0, 100])
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ── Satisfaction distribution histograms ──────────────────────────────────────
st.subheader("📉 Satisfaction Score Distributions")
method_key_map = {
    "Random": "random", "Greedy": "greedy",
    "Hungarian": "hungarian", "Oracle": "oracle",
}

@st.cache_data
def load_true_sats(method_key):
    """Load allocation and compute true satisfaction for a method."""
    alloc = load_alloc(method_key)
    if alloc.empty:
        return pd.DataFrame()
    try:
        from src.data.load import load_students, load_hostels
        from src.evaluation.metrics import compute_true_satisfaction
        students = load_students()
        hostels  = load_hostels()
        true_sat = compute_true_satisfaction(alloc, students, hostels)
        return true_sat.reset_index().rename(columns={0: "satisfaction", "index": "student_id"})
    except Exception:
        return pd.DataFrame()

fig_dist = go.Figure()
colors = {"Random": "#64748b", "Greedy": "#f59e0b",
          "Hungarian": "#6366f1", "Oracle": "#22c55e"}
for method_name, mkey in method_key_map.items():
    sat_df = load_true_sats(mkey)
    if not sat_df.empty:
        col_name = sat_df.columns[-1]  # last column = satisfaction values
        fig_dist.add_trace(go.Histogram(
            x=sat_df[col_name], name=method_name,
            opacity=0.6, nbinsx=25,
            marker_color=colors.get(method_name, "#888"),
        ))

fig_dist.add_vline(x=7.0, line_dash="dash", line_color="#ef4444",
                   annotation_text="Happy threshold (7.0)")
fig_dist.update_layout(
    barmode="overlay", title="True Satisfaction Distributions by Method",
    xaxis_title="True Satisfaction (1-10)", yaxis_title="Count of Students",
    height=400,
)
st.plotly_chart(fig_dist, use_container_width=True)

# ── Conclusion ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("🏆 Conclusion")

if not eval_df.empty:
    hung_row = eval_df[eval_df["method"] == "Hungarian"]
    rand_row = eval_df[eval_df["method"] == "Random"]
    oracle_row= eval_df[eval_df["method"] == "Oracle"]

    violations = int(hung_row["constraint_violations"].iloc[0]) if not hung_row.empty else "N/A"
    hung_sat   = float(hung_row["mean_satisfaction"].iloc[0]) if not hung_row.empty else 0
    rand_sat   = float(rand_row["mean_satisfaction"].iloc[0]) if not rand_row.empty else 0
    ora_sat    = float(oracle_row["mean_satisfaction"].iloc[0]) if not oracle_row.empty else 0

    st.markdown(f"""
    **Winner among deployable methods: Hungarian Algorithm**

    | Criterion | Result |
    |---|---|
    | Constraint violations | **{violations}** (zero violations = all hard rules satisfied) |
    | Mean true satisfaction | **{hung_sat:.3f}/10** (vs Random: {rand_sat:.3f}, Oracle ceiling: {ora_sat:.3f}) |
    | Gap to oracle | **{ora_sat - hung_sat:.3f}** points (the "cost" of using predicted vs true satisfaction) |

    The **Hungarian algorithm** outperforms both Random and Greedy baselines because it:
    1. Globally optimises the student–bed assignment (not greedy local decisions)
    2. Weights priority students more heavily (disabled, first-years, scholarship holders)
    3. Post-processes with roommate local search to reduce sleep-schedule conflicts

    The gap to the Oracle upper bound shows how much improvement a better preference model could deliver.
    """)

# ── Full report ────────────────────────────────────────────────────────────────
if REPORT_TXT.exists():
    with st.expander("📄 Full Text Report"):
        st.code(REPORT_TXT.read_text(encoding="utf-8"), language="text")
