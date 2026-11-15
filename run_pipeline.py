"""
run_pipeline.py  -  One-command end-to-end pipeline runner.

Run this file once to:
  1. Generate all synthetic data (data/*.csv)
  2. Train the demand forecasting model (Part A)
  3. Train the preference / satisfaction model (Part A)
  4. Run all allocation methods (Part B): Random, Greedy, Hungarian + Oracle
  5. Evaluate all methods using true satisfaction (not model predictions)
  6. Save all outputs (models/*.joblib, outputs/*.csv, outputs/report.txt)

Usage (Windows):
    cd smart-hostel-system
    python -m venv venv
    venv\\Scripts\\activate
    pip install -r requirements.txt
    python run_pipeline.py
"""

import sys
import io
import time
from pathlib import Path

# Fix Windows cp1252 terminal encoding (prevents UnicodeEncodeError on emoji/arrows)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# Make sure src/ is on the Python path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Friendly dependency check for beginners running via VS Code
try:
    import pandas
    import numpy
    import sklearn
    import scipy
    import joblib
except ModuleNotFoundError as e:
    print("\n" + "=" * 65)
    print(f"  [ERROR] Missing required library: {e.name}")
    print("  To fix this in VS Code:")
    print("    1. Select the project virtual environment (venv):")
    print("       Press Ctrl+Shift+P -> 'Python: Select Interpreter' -> select './venv'")
    print("    2. Or run setup_and_run.bat by double-clicking it")
    print("    3. Or install dependencies:")
    print("       pip install -r requirements.txt")
    print("=" * 65 + "\n")
    sys.exit(1)

# ── Ensure all output folders exist ───────────────────────────────────────────
for folder in ["data", "models", "outputs"]:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    print("=" * 60)
    print("  SMART HOSTEL SYSTEM - FULL PIPELINE")
    print("=" * 60)

    # ── STEP 1: Generate synthetic data ───────────────────────────────────────
    from src.data.generate import run_all as generate_data
    data = generate_data()
    hostel_df  = data["hostels"]
    demand_df  = data["demand"]
    student_df = data["students"]
    survey_df  = data["survey"]

    # ── STEP 2: Train demand model (Part A) ────────────────────────────────────
    from src.prediction.demand_model import train_demand_model, forecast_next_year
    demand_result = train_demand_model(demand_df)
    forecast_df   = forecast_next_year(demand_df, demand_result["best_pipeline"])
    print(f"\n  Demand forecast:")
    print(forecast_df[["hostel_id", "predicted_applications", "total_beds", "status"]].to_string(index=False))

    # ── STEP 3: Train preference model (Part A) ────────────────────────────────
    from src.prediction.preference_model import (
        train_preference_model,
        predict_satisfaction_matrix,
    )
    pref_result = train_preference_model(survey_df)

    # ── STEP 4: Build predicted satisfaction matrix (used by optimizer) ────────
    print("\n=== BUILDING SATISFACTION MATRIX (for optimizer) ===")
    sat_matrix = predict_satisfaction_matrix(
        student_df, hostel_df, pref_result["best_pipeline"]
    )
    print(f"  Satisfaction matrix: {sat_matrix.shape[0]} students x {sat_matrix.shape[1]} hostels")

    # ── STEP 5: Run all allocation methods (Part B) ───────────────────────────
    print("\n=== PART B: ROOM ALLOCATION ===")
    from src.optimization.allocators import (
        allocate_random,
        allocate_greedy,
        allocate_hungarian,
        allocate_oracle,
        save_allocation,
    )

    # Random baseline
    print("\n  [1/4] Random allocation …")
    alloc_rand, wl_rand = allocate_random(student_df, hostel_df)
    save_allocation(alloc_rand, wl_rand, "random")

    # Greedy FCFS
    print("\n  [2/4] Greedy FCFS allocation …")
    alloc_greedy, wl_greedy = allocate_greedy(student_df, hostel_df, sat_matrix)
    save_allocation(alloc_greedy, wl_greedy, "greedy")

    # Hungarian (our proposed method)
    print("\n  [3/4] Hungarian (optimal) allocation …")
    alloc_hung, wl_hung = allocate_hungarian(student_df, hostel_df, sat_matrix)
    save_allocation(alloc_hung, wl_hung, "hungarian")

    # Oracle upper bound (uses hidden true satisfaction)
    print("\n  [4/4] Oracle allocation (upper bound) …")
    from src.data.generate import _true_satisfaction
    alloc_oracle, wl_oracle = allocate_oracle(
        student_df, hostel_df,
        true_sat_fn=_true_satisfaction,
    )
    save_allocation(alloc_oracle, wl_oracle, "oracle")

    # ── STEP 6: Evaluate all methods ──────────────────────────────────────────
    print("\n=== EVALUATION (using TRUE hidden satisfaction) ===")
    from src.evaluation.metrics import evaluate_all_methods

    results = {
        "Random":    (alloc_rand,   wl_rand),
        "Greedy":    (alloc_greedy, wl_greedy),
        "Hungarian": (alloc_hung,   wl_hung),
        "Oracle":    (alloc_oracle, wl_oracle),
    }
    eval_df = evaluate_all_methods(results, student_df, hostel_df)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"{'='*60}")
    print("\nKey outputs:")
    print(f"  data/               -> {len(list((ROOT/'data').glob('*.csv')))} CSV files")
    print(f"  models/             -> {len(list((ROOT/'models').glob('*.joblib')))} .joblib models")
    print(f"  outputs/            -> {len(list((ROOT/'outputs').glob('*.*')))} result files")
    print(f"\nEvaluation summary:")
    cols = ["method", "allocation_rate_%", "mean_satisfaction", "happy_%",
            "constraint_violations"]
    available = [c for c in cols if c in eval_df.columns]
    print(eval_df[available].to_string(index=False))
    print("\nTo launch the dashboard:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
