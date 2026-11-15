"""
pages/3_Satisfaction_Model.py  –  PART A: Preference Model dashboard page.

Shows:
  - Model comparison table (K-fold CV + hold-out test metrics)
  - Actual vs Predicted scatter plot
  - Feature importance chart
  - Explanation of how the model is used in optimization
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import PREF_METRICS_CSV, OUTPUTS_DIR, SURVEY_CSV, PREF_MODEL_PATH

st.set_page_config(page_title="Satisfaction Model", page_icon="⭐", layout="wide")
st.markdown("# ⭐ Student-Room Satisfaction Model — Part A")
st.caption(
    "Predicts how satisfied a student would be in a given room/hostel. "
    "This predicted score becomes the **benefit** used by the room allocator."
)

@st.cache_data
def load_metrics():
    return pd.read_csv(PREF_METRICS_CSV) if PREF_METRICS_CSV.exists() else pd.DataFrame()

@st.cache_data
def load_importance():
    p = OUTPUTS_DIR / "preference_feature_importance.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_data
def load_test_scatter():
    """Rebuild test scatter from survey data and trained model."""
    try:
        import joblib
        from src.data.load import load_survey
        from src.prediction.features import build_preference_features, get_preference_feature_columns_v2
        from sklearn.model_selection import train_test_split
        from src.config import RANDOM_SEED
        survey = load_survey()
        feat_df = build_preference_features(survey)
        feat_cols = get_preference_feature_columns_v2()
        X = feat_df[feat_cols].values.astype(float)
        y = feat_df["satisfaction_rating"].values
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_SEED)
        model = joblib.load(PREF_MODEL_PATH)
        preds = model.predict(X_test)
        return pd.DataFrame({"actual": y_test, "predicted": np.clip(preds, 1, 10)})
    except Exception:
        return pd.DataFrame()

metrics_df    = load_metrics()
importance_df = load_importance()
scatter_df    = load_test_scatter()

# ── Model comparison ───────────────────────────────────────────────────────────
st.subheader("🏆 Model Comparison (5-fold CV + Hold-out Test)")

if not metrics_df.empty:
    best_idx = metrics_df["MAE"].idxmin()

    def highlight_best(row):
        return ["background-color: #1a1a3e; color: #a5b4fc" if row.name == best_idx else "" for _ in row]

    st.dataframe(
        metrics_df.style.apply(highlight_best, axis=1).format(
            {"MAE": "{:.4f}", "RMSE": "{:.4f}", "R2": "{:.4f}",
             "MAPE_%": "{:.2f}%", "CV_MAE": "{:.4f}"}
        ),
        use_container_width=True,
    )
    st.download_button("⬇ Download metrics", metrics_df.to_csv(index=False),
                       "pref_metrics.csv", "text/csv", key="dl_pref_met")
    best_model = metrics_df.loc[best_idx, "model"]
    st.success(f"✅ Best model: **{best_model}** (MAE = {metrics_df.loc[best_idx,'MAE']:.4f} on 1-10 scale)")
else:
    st.warning("Metrics not found. Run the pipeline first.")

st.divider()

col1, col2 = st.columns(2)

# ── Scatter: Actual vs Predicted ───────────────────────────────────────────────
with col1:
    st.subheader("📉 Actual vs Predicted Satisfaction")
    if not scatter_df.empty:
        fig = px.scatter(
            scatter_df.sample(min(500, len(scatter_df)), random_state=42),
            x="actual", y="predicted",
            opacity=0.5, color_discrete_sequence=["#6366f1"],
            labels={"actual": "True Rating (1-10)", "predicted": "Predicted Rating (1-10)"},
            title="Hold-out Test Set (sample of 500)",
        )
        # Perfect prediction line
        fig.add_shape(type="line", x0=1, y0=1, x1=10, y1=10,
                      line=dict(color="#ef4444", dash="dash"))
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run pipeline to see scatter plot.")

# ── Feature importance ─────────────────────────────────────────────────────────
with col2:
    st.subheader("🔍 Feature Importance (Permutation)")
    if not importance_df.empty:
        imp_top = importance_df.head(14)
        fig2 = px.bar(
            imp_top.sort_values("importance"), x="importance", y="feature",
            orientation="h", error_x="std",
            color="importance",
            color_continuous_scale="Plasma",
            title="Features ranked by importance",
            labels={"importance": "Mean ↑ in MAE when permuted"},
        )
        fig2.update_layout(height=420, coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Residual distribution ──────────────────────────────────────────────────────
if not scatter_df.empty:
    st.subheader("📊 Residual Distribution (Predicted − Actual)")
    scatter_df["residual"] = scatter_df["predicted"] - scatter_df["actual"]
    fig3 = px.histogram(
        scatter_df, x="residual", nbins=40,
        color_discrete_sequence=["#8b5cf6"],
        title="Prediction Errors on Test Set",
        labels={"residual": "Residual (Predicted − Actual)"},
    )
    fig3.add_vline(x=0, line_color="#ef4444", line_dash="dash")
    fig3.update_layout(height=300)
    st.plotly_chart(fig3, use_container_width=True)

# ── How it feeds Part B ────────────────────────────────────────────────────────
st.divider()
with st.expander("🔗 How this model is used in Part B (Optimization)"):
    st.markdown("""
    **The bridge between Prediction (Part A) and Optimization (Part B):**

    1. After training, the model predicts satisfaction for **every (student, hostel) pair**.
       This produces a matrix of shape `(n_students × n_hostels)`.

    2. The optimizer (Hungarian algorithm) uses this matrix as the **benefit score**:
       ```
       cost[student_i, bed_j] = -(priority_weight_i × predicted_satisfaction_i,hostel(bed_j))
       ```
       Minimising cost = **maximising priority-weighted satisfaction**.

    3. The evaluation then measures **true (hidden) satisfaction** for each allocated student.
       This ensures the evaluation is honest and not circular.

    **Key insight:** A better preference model → better benefit scores → better allocation → higher true satisfaction.
    """)
