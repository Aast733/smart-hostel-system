"""
pages/2_Demand_Prediction.py  –  PART A: Demand Forecasting dashboard page.

Shows:
  - Model comparison table (CV MAE, test MAE, RMSE, R², MAPE)
  - Actual vs Predicted chart for the test period (2024-2025)
  - Feature importance bar chart (permutation importance)
  - Next-year forecast table with OVER/BALANCED/UNDER badges
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    DEMAND_METRICS_CSV, FORECAST_CSV, OUTPUTS_DIR, DEMAND_CSV,
    DEMAND_MODEL_PATH,
)

st.set_page_config(page_title="Demand Prediction", page_icon="📈", layout="wide")
st.markdown("# 📈 Demand Prediction — Part A")
st.caption("Supervised regression models to forecast hostel application volumes.")

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_metrics():
    return pd.read_csv(DEMAND_METRICS_CSV) if DEMAND_METRICS_CSV.exists() else pd.DataFrame()

@st.cache_data
def load_forecast():
    return pd.read_csv(FORECAST_CSV) if FORECAST_CSV.exists() else pd.DataFrame()

@st.cache_data
def load_demand():
    return pd.read_csv(DEMAND_CSV)

@st.cache_data
def load_importance():
    p = OUTPUTS_DIR / "demand_feature_importance.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_data
def load_test_predictions():
    """Reconstruct test predictions from the saved model and demand data."""
    try:
        import joblib, numpy as np
        from src.prediction.features import build_demand_features, get_demand_feature_columns
        demand_df = load_demand()
        feat_df   = build_demand_features(demand_df)
        feat_cols = get_demand_feature_columns()
        model     = joblib.load(DEMAND_MODEL_PATH)
        test_mask = feat_df["year"].isin([2024, 2025])
        X_test    = feat_df.loc[test_mask, feat_cols].values
        preds     = model.predict(X_test)
        test_df   = feat_df.loc[test_mask].copy()
        test_df["predicted_applications"] = preds
        return test_df
    except Exception as e:
        return pd.DataFrame()

metrics_df    = load_metrics()
forecast_df   = load_forecast()
importance_df = load_importance()
test_df       = load_test_predictions()

# ── Model comparison ───────────────────────────────────────────────────────────
st.subheader("🏆 Model Comparison (Hold-out Test: 2024-2025)")
if not metrics_df.empty:
    # Highlight best row
    best_idx = metrics_df["MAE"].idxmin()
    def highlight_best(row):
        return ["background-color: #312e81; color: #a5b4fc" if row.name == best_idx else "" for _ in row]

    styled = metrics_df.style.apply(highlight_best, axis=1).format(
        {"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.4f}", "MAPE_%": "{:.2f}%", "CV_MAE": "{:.2f}"}
    )
    st.dataframe(styled, use_container_width=True)
    st.download_button("⬇ Download metrics", metrics_df.to_csv(index=False),
                       "demand_metrics.csv", "text/csv", key="dl_dem_met")

    best_model = metrics_df.loc[best_idx, "model"]
    st.success(f"✅ Best model: **{best_model}** (lowest test MAE = {metrics_df.loc[best_idx,'MAE']})")
else:
    st.warning("Metrics not found. Run the pipeline first.")

st.divider()

# ── Actual vs Predicted ────────────────────────────────────────────────────────
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("📉 Actual vs Predicted (Test Set: 2024-2025)")
    if not test_df.empty:
        fig = go.Figure()
        for hid in test_df["hostel_id"].unique():
            h = test_df[test_df["hostel_id"] == hid]
            x_labels = (h["year"].astype(str) + "-S" + h["semester"].astype(str)).tolist()
            fig.add_trace(go.Scatter(
                x=x_labels, y=h["applications"].tolist(),
                mode="lines+markers", name=f"{hid} actual",
                line=dict(dash="dash"),
            ))
            fig.add_trace(go.Scatter(
                x=x_labels, y=h["predicted_applications"].tolist(),
                mode="lines+markers", name=f"{hid} predicted",
            ))
        fig.update_layout(
            title="Actual vs Predicted Applications",
            xaxis_title="Period", yaxis_title="Applications",
            height=420, legend=dict(orientation="h", y=-0.3),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run pipeline to generate test predictions.")

with col2:
    st.subheader("🔍 Feature Importance (Permutation)")
    if not importance_df.empty:
        imp_top = importance_df.head(12)
        fig2 = px.bar(
            imp_top.sort_values("importance"), x="importance", y="feature",
            orientation="h", error_x="std",
            color="importance",
            color_continuous_scale="Viridis",
            title="Top Features",
            labels={"importance": "Mean ↑ in MAE when permuted"},
        )
        fig2.update_layout(height=420, coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Next-year forecast ─────────────────────────────────────────────────────────
st.subheader("🔮 Next-Year Demand Forecast (2026, Semester 1)")
if not forecast_df.empty:
    def _badge(status):
        colors = {
            "OVERSUBSCRIBED": "🔴", "BALANCED": "🟢", "UNDER-USED": "🟡",
        }
        return colors.get(status, "⚪") + " " + status

    display = forecast_df.copy()
    display["status_display"] = display["status"].apply(_badge)

    col_f1, col_f2 = st.columns([1.3, 1])
    with col_f1:
        st.dataframe(
            display[["hostel_id", "hostel_name", "total_beds",
                      "predicted_applications", "pred_occupancy_rate", "status_display"]],
            use_container_width=True,
        )
        st.download_button("⬇ Download forecast", forecast_df.to_csv(index=False),
                           "forecast.csv", "text/csv", key="dl_forecast")

    with col_f2:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=forecast_df["hostel_id"], y=forecast_df["total_beds"],
            name="Capacity (Beds)", marker_color="#334155",
        ))
        fig3.add_trace(go.Bar(
            x=forecast_df["hostel_id"], y=forecast_df["predicted_applications"],
            name="Predicted Applications",
            marker_color=forecast_df["status"].map(
                {"OVERSUBSCRIBED": "#ef4444", "BALANCED": "#22c55e", "UNDER-USED": "#f59e0b"}
            ).tolist(),
        ))
        fig3.update_layout(
            barmode="group", title="Capacity vs Predicted Demand",
            yaxis_title="Count", height=350,
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Status summary cards
    c1, c2, c3 = st.columns(3)
    for col, status, icon, clr in [
        (c1, "OVERSUBSCRIBED", "🔴", "#fca5a5"),
        (c2, "BALANCED",       "🟢", "#6ee7b7"),
        (c3, "UNDER-USED",     "🟡", "#fde68a"),
    ]:
        n = (forecast_df["status"] == status).sum()
        with col:
            st.markdown(
                f'<div style="background:#1e293b;border-radius:10px;padding:1rem;text-align:center;">'
                f'<span style="font-size:2rem">{icon}</span><br>'
                f'<b style="color:{clr};font-size:1.5rem">{n}</b><br>'
                f'<small style="color:#94a3b8">{status}</small></div>',
                unsafe_allow_html=True,
            )
else:
    st.info("Run pipeline to generate forecast.")

st.divider()
# ── Methodology note ──────────────────────────────────────────────────────────
with st.expander("📖 Methodology Notes"):
    st.markdown("""
    **Validation Strategy:**
    - **Time-based train/test split**: training on 2010-2023, testing on 2024-2025.
    - **TimeSeriesSplit CV**: 5-fold CV within the training set only — no test data leakage.
    - **Model selection**: best model chosen by CV MAE (not test MAE).

    **Feature Engineering:**
    - `lag1_applications`: previous semester's applications (strongest predictor)
    - `lag2_applications`: same semester one year ago
    - `rolling4_mean`: 2-year rolling mean (smooth trend)
    - `fee_ratio`, `cohort_size`, `sem2_flag`: demand context
    - `mess_rating`, `wifi_rating`, `distance_km`: hostel attractiveness

    **Why no XGBoost?** XGBoost had installation issues on some Python 3.13 setups.
    `HistGradientBoostingRegressor` from scikit-learn achieves similar performance.
    """)
