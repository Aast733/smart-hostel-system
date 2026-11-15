"""
preference_model.py  -  PART A: Student-Room Satisfaction Model (Supervised Learning).

PURPOSE:
  Predict how satisfied a student (profile) would be in a particular hostel room.
  The predicted score becomes the "benefit" score used by the optimizer in Part B.

TARGET:  satisfaction_rating  (1-10 scale, continuous)

MODELS:
  1. Ridge Regression  - fast linear baseline
  2. Random Forest     - captures non-linear interactions (e.g., budget x AC)
  3. HistGradientBoosting - scikit-learn's fast boosted tree (Python 3.13 safe)

VALIDATION:
  - 80/20 random hold-out split (K-fold CV within the 80% training portion)
  - 5-fold cross-validation on training set for model selection
  - Report MAE, RMSE, R², MAPE on held-out test set

SAVED OUTPUT:
  - models/preference_model.joblib
  - outputs/preference_model_metrics.csv
  - outputs/preference_feature_importance.csv
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys

from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    RANDOM_SEED, PREF_MODEL_PATH, PREF_METRICS_CSV,
    MODELS_DIR, OUTPUTS_DIR,
)
from src.prediction.features import (
    build_preference_features,
    get_preference_feature_columns_v2,
    get_preference_target_column,
)


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def _regression_metrics(y_true, y_pred, name: str) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {"model": name, "MAE": round(mae, 3), "RMSE": round(rmse, 3),
            "R2": round(r2, 4), "MAPE_%": round(mape, 2)}


# ─────────────────────────────────────────────
# CANDIDATE MODELS
# ─────────────────────────────────────────────
def _get_candidate_models() -> dict:
    return {
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0, random_state=RANDOM_SEED)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=150, max_depth=8,
                random_state=RANDOM_SEED, n_jobs=-1,
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  HistGradientBoostingRegressor(
                max_iter=200, max_depth=5, learning_rate=0.05,
                random_state=RANDOM_SEED,
            )),
        ]),
    }


# ─────────────────────────────────────────────
# MAIN TRAINING
# ─────────────────────────────────────────────
def train_preference_model(survey_df: pd.DataFrame) -> dict:
    """
    Train student-room satisfaction models, select best, save model + metrics.

    Parameters
    ----------
    survey_df : pd.DataFrame
        Historical satisfaction survey from load_survey().

    Returns
    -------
    dict with keys:
        "best_model_name", "best_pipeline", "metrics_df", "test_df"
    """
    print("\n=== PART A: PREFERENCE MODEL TRAINING ===")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Feature engineering ──────────────────────────────────────────────────
    feat_df   = build_preference_features(survey_df)
    feat_cols = get_preference_feature_columns_v2()
    target    = get_preference_target_column()

    X = feat_df[feat_cols].values.astype(float)
    y = feat_df[target].values.astype(float)

    # ── 80 / 20 stratified-ish random split ─────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED,
    )
    print(f"  Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

    # ── 5-fold CV on training portion ────────────────────────────────────────
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    candidates = _get_candidate_models()
    cv_scores  = {}
    for name, pipe in candidates.items():
        neg_mae = cross_val_score(
            pipe, X_train, y_train, cv=kf,
            scoring="neg_mean_absolute_error", n_jobs=-1,
        )
        cv_mae = -neg_mae.mean()
        cv_scores[name] = cv_mae
        print(f"  [{name}]  CV MAE = {cv_mae:.4f}")

    best_name = min(cv_scores, key=cv_scores.get)
    print(f"  [OK] Best model by CV: {best_name}")

    # ── Train all on full training set; evaluate on test ─────────────────────
    all_metrics = []
    fitted_models = {}
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        fitted_models[name] = pipe
        preds = pipe.predict(X_test)
        m = _regression_metrics(y_test, preds, name)
        m["CV_MAE"] = round(cv_scores[name], 4)
        all_metrics.append(m)

    metrics_df = pd.DataFrame(all_metrics).sort_values("MAE")
    metrics_df.to_csv(PREF_METRICS_CSV, index=False)
    print(f"  Metrics saved -> {PREF_METRICS_CSV.name}")

    # ── Save best model ───────────────────────────────────────────────────────
    best_pipeline = fitted_models[best_name]
    joblib.dump(best_pipeline, PREF_MODEL_PATH)
    print(f"  Model saved -> {PREF_MODEL_PATH.name}")

    # ── Permutation importance ───────────────────────────────────────────────
    perm = permutation_importance(
        best_pipeline, X_test, y_test,
        n_repeats=10, random_state=RANDOM_SEED,
    )
    imp_df = pd.DataFrame({
        "feature":    feat_cols,
        "importance": perm.importances_mean,
        "std":        perm.importances_std,
    }).sort_values("importance", ascending=False)
    imp_path = OUTPUTS_DIR / "preference_feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)

    # ── Attach predictions to test rows ─────────────────────────────────────
    test_idx = feat_df.index[len(X_train):]   # approximate; good enough for plotting
    # Rebuild from actual split
    test_df = pd.DataFrame(X_test, columns=feat_cols)
    test_df["actual_satisfaction"]    = y_test
    test_df["predicted_satisfaction"] = best_pipeline.predict(X_test)

    return {
        "best_model_name": best_name,
        "best_pipeline":   best_pipeline,
        "metrics_df":      metrics_df,
        "test_df":         test_df,
        "imp_df":          imp_df,
        "feat_cols":       feat_cols,
        "fitted_models":   fitted_models,
    }


# ─────────────────────────────────────────────
# SCORE STUDENTS FOR OPTIMIZER
# ─────────────────────────────────────────────
def predict_satisfaction_matrix(
    student_df: pd.DataFrame,
    hostel_df:  pd.DataFrame,
    pipeline,
) -> pd.DataFrame:
    """
    Predict satisfaction for EVERY (student, hostel) pair.

    This produces the benefit matrix used by the optimizer.
    Rows = students, columns = hostel IDs, values = predicted satisfaction (1-10).

    Parameters
    ----------
    student_df  : student applicants DataFrame
    hostel_df   : hostel master DataFrame
    pipeline    : trained preference model Pipeline

    Returns
    -------
    pd.DataFrame  shape (n_students, n_hostels)
    """
    from src.prediction.features import build_preference_features

    feat_cols = get_preference_feature_columns_v2()
    records   = []

    for _, stu in student_df.iterrows():
        for _, hos in hostel_df.iterrows():
            # Merge student + hostel into one flat dict (mimics survey rows)
            room_types = ["single", "double", "triple"]
            room_type  = stu.get("room_type_pref", "double")
            if room_type not in room_types:
                room_type = "double"

            rec = {
                "student_id":          stu["student_id"],
                "hostel_id":           hos["hostel_id"],
                "room_type":           room_type,
                "gender":              stu["gender"],
                "year_of_study":       stu["year_of_study"],
                "budget_inr":          stu["budget_inr"],
                "ac_preference":       stu["ac_preference"],
                "sleep_schedule":      stu.get("sleep_schedule", "normal"),
                "room_type_pref":      stu.get("room_type_pref", "double"),
                "home_distance_km":    stu["home_distance_km"],
                "disability":          stu["disability"],
                "scholarship":         stu["scholarship"],
                "first_choice_hostel": stu["first_choice_hostel"],
                "ac":                  hos["ac"],
                "distance_km":         hos["distance_km"],
                "mess_rating":         hos["mess_rating"],
                "wifi_rating":         hos["wifi_rating"],
                "annual_fee_inr":      hos["annual_fee_inr"],
            }
            records.append(rec)

    combo_df  = pd.DataFrame(records)
    combo_feat= build_preference_features(combo_df)
    X         = combo_feat[feat_cols].values.astype(float)
    preds     = pipeline.predict(X)
    combo_df["predicted_satisfaction"] = np.clip(preds, 1.0, 10.0)

    # Pivot: students x hostels
    pivot = combo_df.pivot_table(
        index="student_id", columns="hostel_id",
        values="predicted_satisfaction", aggfunc="mean",
    )
    return pivot


def load_preference_model():
    """Load and return the saved preference model pipeline."""
    return joblib.load(PREF_MODEL_PATH)
