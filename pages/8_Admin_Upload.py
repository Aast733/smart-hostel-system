"""
pages/8_Admin_Upload.py  –  Admin Upload & Retrain page.

Allows uploading real CSV files (same columns as synthetic data) to:
  1. Replace synthetic data
  2. Retrain demand and preference models on real data
  3. Re-run allocation with the new models

The column schema is shown clearly for each expected file.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import io

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import DATA_DIR, HOSTEL_CSV, DEMAND_CSV, STUDENTS_CSV, SURVEY_CSV

st.set_page_config(page_title="Admin Upload", page_icon="📤", layout="wide")
st.markdown("# 📤 Admin Upload & Retrain")
st.caption(
    "Upload your own CSV files to replace synthetic data and retrain the models. "
    "Files must match the expected column schema."
)

st.warning(
    "⚠️ Uploading will overwrite the current data files and trigger a full retrain. "
    "Make sure your CSV columns match the schema exactly."
)

# ── Schema reference ───────────────────────────────────────────────────────────
with st.expander("📋 Required Column Schemas"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**hostels.csv**")
        st.code("""hostel_id, name, gender (M/F),
single_rooms, double_rooms, triple_rooms,
total_beds, ac (0/1), distance_km,
mess_rating (1-5), wifi_rating (1-5),
annual_fee_inr, floors""")

        st.markdown("**historical_demand.csv**")
        st.code("""year, semester (1/2), hostel_id,
hostel_name, gender, total_beds, ac,
distance_km, mess_rating, wifi_rating,
annual_fee_inr, floors,
cohort_size, rent_index,
applications [TARGET], occupancy, occupancy_rate""")

    with col2:
        st.markdown("**students.csv**")
        st.code("""student_id, gender (M/F), year_of_study (1-4),
budget_inr, ac_preference (0/1),
sleep_schedule (early/normal/night),
room_type_pref (single/double/triple),
home_distance_km, disability (0/1),
scholarship (0/1),
first_choice_hostel, ranked_preferences,
far_away (0/1), first_year (0/1)""")

        st.markdown("**satisfaction_survey.csv**")
        st.code("""student_id, hostel_id, room_type,
gender, year_of_study, budget_inr,
ac_preference, sleep_schedule, room_type_pref,
home_distance_km, disability, scholarship,
first_choice_hostel, hostel_gender,
ac, distance_km, mess_rating, wifi_rating,
annual_fee_inr,
satisfaction_rating [TARGET]""")

st.divider()

# ── Upload widgets ─────────────────────────────────────────────────────────────
file_configs = [
    ("🏨 Hostel Master Data", "hostels.csv", HOSTEL_CSV,
     ["hostel_id", "name", "gender", "single_rooms", "double_rooms", "triple_rooms",
      "total_beds", "ac", "distance_km", "mess_rating", "wifi_rating", "annual_fee_inr", "floors"]),
    ("📈 Historical Demand", "historical_demand.csv", DEMAND_CSV,
     ["year", "semester", "hostel_id", "applications"]),
    ("👥 Student Applicants", "students.csv", STUDENTS_CSV,
     ["student_id", "gender", "year_of_study", "budget_inr", "ac_preference"]),
    ("⭐ Satisfaction Survey", "satisfaction_survey.csv", SURVEY_CSV,
     ["student_id", "hostel_id", "satisfaction_rating"]),
]

uploaded_files = {}
for label, expected_name, save_path, required_cols in file_configs:
    st.subheader(label)
    col_upload, col_preview = st.columns([1, 2])

    with col_upload:
        uploaded = st.file_uploader(
            f"Upload {expected_name}",
            type=["csv"],
            key=f"upload_{expected_name}",
        )

        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    st.success(f"✅ {len(df)} rows, all required columns present")
                    uploaded_files[save_path] = df
                    with col_preview:
                        st.dataframe(df.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Error reading file: {e}")
        else:
            with col_preview:
                # Show current file info
                if save_path.exists():
                    current = pd.read_csv(save_path)
                    st.caption(f"Current file: {len(current)} rows")
                    st.dataframe(current.head(3), use_container_width=True)
                else:
                    st.caption("No file loaded (synthetic data will be used)")

    st.divider()

# ── Save and retrain button ────────────────────────────────────────────────────
if uploaded_files:
    st.subheader("🚀 Save Uploads & Retrain")
    st.markdown(f"Ready to save **{len(uploaded_files)}** file(s) and retrain all models.")

    if st.button("✅ Confirm Upload & Retrain", type="primary"):
        # Save uploaded files
        with st.spinner("Saving uploaded files …"):
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            for save_path, df in uploaded_files.items():
                df.to_csv(save_path, index=False)
                st.success(f"Saved {save_path.name}")

        # Retrain
        with st.spinner("Retraining models on new data … (may take ~30s)"):
            try:
                from src.data.load import load_all
                data = load_all()
                from src.prediction.demand_model import train_demand_model, forecast_next_year
                dr = train_demand_model(data["demand"])
                forecast_next_year(data["demand"], dr["best_pipeline"])
                from src.prediction.preference_model import train_preference_model
                train_preference_model(data["survey"])
                st.success("✅ Models retrained successfully!")
                st.info("Now go to the **Room Allocation** page and click **Run Allocation** to apply new models.")
                st.cache_data.clear()
                st.cache_resource.clear()
            except Exception as e:
                st.error(f"Retraining failed: {e}")
                st.exception(e)
else:
    st.info("Upload one or more files above to enable retraining.")

st.divider()
# ── Reset to synthetic data ────────────────────────────────────────────────────
with st.expander("🔄 Reset to Synthetic Data"):
    st.markdown("This will delete the current data files and regenerate synthetic data.")
    if st.button("⚠️ Reset to Synthetic Data", type="secondary"):
        with st.spinner("Regenerating synthetic data …"):
            import subprocess, sys as _sys
            result = subprocess.run(
                [_sys.executable, str(ROOT / "run_pipeline.py")],
                capture_output=True, text=True, cwd=str(ROOT),
            )
        if result.returncode == 0:
            st.success("Synthetic data regenerated and pipeline re-run successfully!")
            st.cache_data.clear()
        else:
            st.error("Pipeline failed.")
            st.code(result.stderr[-2000:])
