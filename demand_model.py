"""
demand_model.py  -  PART A: Hostel Demand Forecasting (Supervised Learning).

This module trains regression models to predict the NUMBER OF APPLICATIONS
a hostel will receive in a given semester.

MODELS TRAINED:
  1. Naive Baseline   - predict the mean of the training set (simplest possible)
  2. Ridge Regression - linear model with L2 regularisation (prevents overfitting)
  3. Random Forest    - ensemble of decision trees (handles non-linearity well)
  4. Gradient Boosting- boosted trees (often best on tabular data)

VALIDATION STRATEGY:
  We use TIME-BASED splitting to avoid data leakage:
    - Train: 2010 - 2023  (all older data)
    - Test:  2024 - 2025  (last 2 years, held out)
  Within training data we also run TimeSeriesSplit cross-validation
  to select the best model WITHOUT touching the test set.

OUTPUTS:
  - Best model saved to models/demand_model.joblib
  - Metric tables saved to outputs/demand_model_metrics.csv
  - Forecast for next year saved to outputs/demand_forecast.csv
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    RANDOM_SEED, DEMAND_MODEL_PATH, DEMAND_METRICS_CSV, FORECAST_CSV,
    MODELS_DIR, OUTPUTS_DIR, HOSTELS,
)
from src.prediction.features import (
    build_demand_features,
    get_demand_feature_columns,
    get_demand_target_column,
)


# ─────────────────────────────────────────────
# HELPER: compute all regression metrics
# ─────────────────────────────────────────────
def _regression_metrics(y_true, y_pred, name: str) -> dict:
    """Compute MAE, RMSE, R², MAPE for a set of predictions."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    # MAPE: avoid divide-by-zero
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {"model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2),
            "R2": round(r2, 4), "MAPE_%": round(mape, 2)}


# ─────────────────────────────────────────────
# DEFINE CANDIDATE MODELS
# ─────────────────────────────────────────────
def _get_candidate_models() -> dict:
    """
    Return a dict of {name: sklearn Pipeline}.
    Each pipeline first scales features (StandardScaler) then applies the model.
    Scaling is important for Ridge (distance-based), less so for trees but harmless.
    """
    return {
        "Naive Baseline": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  DummyRegressor(strategy="mean")),
        ]),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=10.0, random_state=RANDOM_SEED)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=200, max_depth=8,
                random_state=RANDOM_SEED, n_jobs=-1,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.08,
                random_state=RANDOM_SEED,
            )),
        ]),
    }


# ─────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────
def train_demand_model(demand_df: pd.DataFrame) -> dict:
    """
    Train demand forecasting models, select the best by CV, evaluate on held-out
    test years (2024-2025), save best model and metrics.

    Parameters
    ----------
    demand_df : pd.DataFrame
        Raw historical demand panel from load_historical_demand().

    Returns
    -------
    dict with keys:
        "best_model_name", "best_pipeline", "metrics_df", "test_df"
    """
    print("\n=== PART A: DEMAND MODEL TRAINING ===")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Feature engineering ──────────────────────────────────────────────────
    feat_df  = build_demand_features(demand_df)
    feat_cols= get_demand_feature_columns()
    target   = get_demand_target_column()

    # ── Time-based train / test split ────────────────────────────────────────
    # Test set = last 2 complete years (2024, 2025)
    test_years  = [2024, 2025]
    train_mask  = ~feat_df["year"].isin(test_years)
    test_mask   =  feat_df["year"].isin(test_years)

    X_train = feat_df.loc[train_mask, feat_cols].values
    y_train = feat_df.loc[train_mask, target].values
    X_test  = feat_df.loc[test_mask,  feat_cols].values
    y_test  = feat_df.loc[test_mask,  target].values

    print(f"  Train size: {len(X_train)} rows  |  Test size: {len(X_test)} rows")

    # ── TimeSeriesSplit cross-validation on TRAIN only ───────────────────────
    tscv = TimeSeriesSplit(n_splits=5)
    candidates = _get_candidate_models()
    cv_scores  = {}
    for name, pipe in candidates.items():
        neg_mae = cross_val_score(
            pipe, X_train, y_train, cv=tscv,
            scoring="neg_mean_absolute_error", n_jobs=-1,
        )
        cv_mae = -neg_mae.mean()
        cv_scores[name] = cv_mae
        print(f"  [{name}]  CV MAE = {cv_mae:.2f}")

    # ── Select best model by CV MAE (lower is better) ────────────────────────
    best_name = min(cv_scores, key=cv_scores.get)
    print(f"  [OK] Best model by CV: {best_name}")

    # ── Fit ALL models on full train set; evaluate on test ───────────────────
    all_metrics = []
    fitted_models = {}
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        fitted_models[name] = pipe
        preds = pipe.predict(X_test)
        m = _regression_metrics(y_test, preds, name)
        m["CV_MAE"] = round(cv_scores[name], 2)
        all_metrics.append(m)

    metrics_df = pd.DataFrame(all_metrics).sort_values("MAE")
    metrics_df.to_csv(DEMAND_METRICS_CSV, index=False)
    print(f"  Metrics saved -> {DEMAND_METRICS_CSV.name}")

    # ── Save best model ───────────────────────────────────────────────────────
    best_pipeline = fitted_models[best_name]
    joblib.dump(best_pipeline, DEMAND_MODEL_PATH)
    print(f"  Model saved -> {DEMAND_MODEL_PATH.name}")

    # ── Permutation importance on test set ───────────────────────────────────
    perm = permutation_importance(
        best_pipeline, X_test, y_test,
        n_repeats=10, random_state=RANDOM_SEED,
    )
    imp_df = pd.DataFrame({
        "feature":    feat_cols,
        "importance": perm.importances_mean,
        "std":        perm.importances_std,
    }).sort_values("importance", ascending=False)
    imp_path = OUTPUTS_DIR / "demand_feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"  Feature importance saved -> {imp_path.name}")

    # ── Attach predictions to test rows for plotting ─────────────────────────
    test_df = feat_df.loc[test_mask].copy()
    test_df["predicted_applications"] = best_pipeline.predict(X_test)

    return {
        "best_model_name": best_name,
        "best_pipeline":   best_pipeline,
        "metrics_df":      metrics_df,
        "test_df":         test_df,
        "imp_df":          imp_df,
        "feat_df":         feat_df,
        "feat_cols":       feat_cols,
        "fitted_models":   fitted_models,
    }


# ─────────────────────────────────────────────
# FORECAST NEXT SEMESTER
# ─────────────────────────────────────────────
def forecast_next_year(demand_df: pd.DataFrame, pipeline) -> pd.DataFrame:
    """
    Forecast hostel demand for the NEXT semester (year 2026, semester 1).

    Strategy:
      - Take the most recent row for each hostel (latest year/semester)
      - Shift lag features forward by one step
      - Increment cohort_size by growth rate
      - Increment rent_index by trend rate
      - Predict with the saved best model

    Parameters
    ----------
    demand_df : pd.DataFrame
        Full historical demand (including 2025 data).
    pipeline : sklearn Pipeline
        Trained demand model.

    Returns
    -------
    pd.DataFrame
        Per-hostel forecast with columns: hostel_id, predicted_applications,
        total_beds, status (OVERSUBSCRIBED / BALANCED / UNDER-USED).
    """
    COHORT_GROWTH = 0.04
    feat_df  = build_demand_features(demand_df)
    feat_cols= get_demand_feature_columns()

    # Get most recent row for each hostel
    latest = (
        feat_df
        .sort_values(["hostel_id", "year", "semester"])
        .groupby("hostel_id")
        .last()
        .reset_index()
    )

    # Construct next-period feature row
    rows = []
    for _, row in latest.iterrows():
        feat_row = {
            "total_beds":          row["total_beds"],
            "ac":                  row["ac"],
            "distance_km":         row["distance_km"],
            "mess_rating":         row["mess_rating"],
            "wifi_rating":         row["wifi_rating"],
            "fee_ratio":           row["fee_ratio"],
            "floors":              row["floors"],
            "beds_per_floor":      row["beds_per_floor"],
            "cohort_size":         int(row["cohort_size"] * (1 + COHORT_GROWTH)),
            "rent_index":          round(row["rent_index"] * 1.03, 4),
            "sem2_flag":           0,   # forecasting Sem 1
            # Lags (shift forward)
            "lag1_applications":   row["applications"],
            "lag2_applications":   row["lag1_applications"],
            "rolling4_mean":       row["rolling4_mean"],
            "lag1_occ_rate":       row["occupancy_rate"],
        }
        X = np.array([[feat_row[c] for c in feat_cols]])
        pred = float(pipeline.predict(X)[0])
        beds = int(row["total_beds"])
        ratio= pred / max(beds, 1)
        if ratio > 1.1:
            status = "OVERSUBSCRIBED"
        elif ratio < 0.8:
            status = "UNDER-USED"
        else:
            status = "BALANCED"
        rows.append({
            "hostel_id":             row["hostel_id"],
            "hostel_name":           row["hostel_name"],
            "predicted_applications": round(pred),
            "total_beds":            beds,
            "pred_occupancy_rate":   round(ratio, 3),
            "status":                status,
        })

    forecast_df = pd.DataFrame(rows)
    forecast_df.to_csv(FORECAST_CSV, index=False)
    print(f"  Forecast saved -> {FORECAST_CSV.name}")
    return forecast_df


def load_demand_model():
    """Load and return the saved demand model pipeline."""
    return joblib.load(DEMAND_MODEL_PATH)
