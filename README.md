# Smart Hostel Demand Prediction & Intelligent Room Allocation System

> **University Supervised Learning Course Project**  
> A complete ML + Optimization pipeline with a Streamlit dashboard.

---

## 📋 Table of Contents
1. [Quick Start (Windows)](#quick-start-windows)
2. [Project Structure](#project-structure)
3. [What This System Does](#what-this-system-does)
4. [Part A – Prediction](#part-a--prediction)
5. [Part B – Optimization](#part-b--optimization)
6. [Evaluation Results Summary](#evaluation-results-summary)
7. [Dashboard Pages](#dashboard-pages)
8. [Troubleshooting](#troubleshooting)
9. [Limitations & Future Scope](#limitations--future-scope)

---

## Quick Start (Windows)

> **Prerequisites:** Python 3.10 to 3.13, VS Code, internet connection for `pip install`

### Option 1: 1-Click Launch (Easiest for Beginners)
Simply double-click **`setup_and_run.bat`** in the folder. It will automatically:
1. Create the `venv` virtual environment.
2. Install all dependencies from `requirements.txt`.
3. Check and load the pre-trained models.
4. Launch the Streamlit dashboard in your browser.

---

### Option 2: Running Inside VS Code
1. Open VS Code: `File -> Open Folder...` and select the **`smart-hostel-system`** folder.
2. Open the built-in terminal (`Ctrl + ~` or `Terminal -> New Terminal`).
3. Set up the environment once:
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. **To run the app directly in VS Code:**
   - Either open **`app.py`** and simply click the **Play** button (▶) in the top-right corner.
   - Or press **`F5`** (it will use the bundled `.vscode/launch.json` profile: *🏠 Launch Streamlit Dashboard*).
   - Or run in terminal: `streamlit run app.py`

---

### Option 3: Command Line / PowerShell Manual Steps

```bat
:: 1. Open PowerShell or Command Prompt, navigate to the project folder
cd smart-hostel-system

:: 2. Create a virtual environment
python -m venv venv

:: 3. Activate the virtual environment
venv\Scripts\activate

:: 4. Install all dependencies
pip install -r requirements.txt

:: 5. Run the full pipeline (generates data, trains models, runs allocation, evaluates)
python run_pipeline.py

:: 6. Launch the Streamlit dashboard
streamlit run app.py
```

The browser should automatically open at `http://localhost:8501`.  
If not, paste that URL into your browser manually.

---

## Project Structure

```
smart-hostel-system/
├── app.py                        ← Streamlit home page (run this)
├── run_pipeline.py               ← End-to-end pipeline (run first)
├── requirements.txt
├── README.md
├── pages/
│   ├── 1_Data_Explorer.py
│   ├── 2_Demand_Prediction.py
│   ├── 3_Satisfaction_Model.py
│   ├── 4_WhatIf_Simulator.py
│   ├── 5_Room_Allocation.py
│   ├── 6_Evaluation.py
│   ├── 7_Student_Lookup.py
│   └── 8_Admin_Upload.py
├── src/
│   ├── config.py                 ← All constants (hostels, paths, seed)
│   ├── data/
│   │   ├── generate.py           ← Synthetic data generator
│   │   └── load.py               ← CSV loaders
│   ├── prediction/               ← PART A: Supervised Learning
│   │   ├── features.py
│   │   ├── demand_model.py
│   │   └── preference_model.py
│   ├── optimization/             ← PART B: Allocation
│   │   ├── constraints.py
│   │   ├── allocators.py
│   │   └── roommate.py
│   └── evaluation/
│       └── metrics.py
├── data/                         ← Generated CSVs (auto-created)
├── models/                       ← Saved .joblib models (auto-created)
├── outputs/                      ← Results, metrics, report (auto-created)
└── docs/
    ├── PROJECT_EXPLANATION.md
    └── VIVA_QUESTIONS.md
```

---

## What This System Does

This project solves two problems every university hostel office faces each semester:

| Problem | Our Solution |
|---|---|
| **How many students will apply to each hostel?** | Part A: Demand forecasting (regression) |
| **Who gets which room?** | Part B: Constrained assignment optimization |

**Data flow:**
```
Historical demand → Feature engineering → Regression model → Demand forecast
Survey data       → Feature engineering → Preference model → Satisfaction scores
Satisfaction scores → Feasibility matrix → Hungarian algorithm → Allocation
Allocation → True satisfaction oracle → Evaluation metrics
```

---

## Part A – Prediction

### A1: Demand Forecasting

- **Target:** `applications` (number of students applying per hostel per semester)
- **Validation:** Time-based split (train 2010-2023, test 2024-2025) + TimeSeriesSplit CV
- **Models trained:**
  - Naive Baseline (predicts training mean)
  - Ridge Regression (L2 regularised linear model)
  - Random Forest (200 trees, depth 8)
  - Gradient Boosting (200 estimators, LR=0.08)
- **Key features:** lag applications, rolling mean, cohort size, fee ratio, semester flag
- **Output:** Best model → `models/demand_model.joblib`, next-year forecast per hostel

### A2: Student-Room Satisfaction Model

- **Target:** `satisfaction_rating` (1-10, from historical surveys)
- **Validation:** 80/20 split + 5-fold CV on training portion
- **Models:** Ridge, Random Forest, HistGradientBoosting
- **Key features:** budget gap, AC match, distance score, mess rating, first-choice match
- **Output:** Best model → `models/preference_model.joblib`, predicted satisfaction matrix

---

## Part B – Optimization

### Hard Constraints (NEVER violated)
1. **Gender match** – male students → boys' hostels only
2. **Budget** – annual fee ≤ student budget + ₹5,000 tolerance
3. **Disability** – disabled students → ground floor only (floor 1)
4. **Capacity** – beds not double-assigned

### Allocation Methods

| Method | Description |
|---|---|
| **Random** | Randomly assigns feasible students to beds (baseline) |
| **Greedy FCFS** | Priority-ordered, best-satisfaction hostel first |
| **Hungarian** | Global optimum via Linear Sum Assignment (scipy) |
| **Oracle** | Hungarian but uses TRUE hidden satisfaction (upper bound) |

### Roommate Compatibility
After Hungarian assignment, a **swap-based local search** reduces sleep-schedule conflicts:
- Night-owls and early-risers in the same room = mismatch
- Algorithm swaps students between rooms until no improving swap exists

---

## Evaluation Results Summary

*(Exact numbers vary by random seed. These are representative results with seed=42.)*

| Method | Allocation Rate | Mean Satisfaction | Happy % | Violations |
|---|---|---|---|---|
| Random | ~95% | ~6.0 | ~45% | 0 |
| Greedy | ~95% | ~7.2 | ~65% | 0 |
| **Hungarian** | **~95%** | **~7.6** | **~72%** | **0** |
| Oracle | ~95% | ~8.1 | ~85% | 0 |

**Conclusion:** Hungarian outperforms baselines because it globally optimises the assignment
rather than making greedy local decisions. The gap to Oracle shows the cost of
using model predictions instead of true satisfaction.

---

## Dashboard Pages

| Page | What you'll find |
|---|---|
| 🏠 Home | KPI cards, architecture diagram, pipeline runner |
| 🔍 Data Explorer | Browse all 4 datasets with filters and charts |
| 📈 Demand Prediction | Model comparison, actual-vs-predicted, forecast |
| ⭐ Satisfaction Model | Scatter plot, feature importance, residuals |
| 🎛️ What-If Simulator | Live demand re-prediction with parameter sliders |
| 🏨 Room Allocation | Run allocation, view tables, waitlist, occupancy |
| 📊 Evaluation | Side-by-side comparison with charts and conclusion |
| 👤 Student Lookup | Enter student ID → see allocation + satisfaction |
| 📤 Admin Upload | Upload real CSVs to retrain |

---

## Troubleshooting

### "File not found" or "No module named src"
**Cause:** You are not running from the `smart-hostel-system/` folder.  
**Fix:** Make sure you `cd smart-hostel-system` before running any command.

### "activate is not recognized" or execution policy error
**Cause:** PowerShell execution policy blocks scripts.  
**Fix:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\activate
```

### "ModuleNotFoundError: No module named 'streamlit'"
**Cause:** Virtual environment not activated.  
**Fix:**
```bat
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### App shows "Run pipeline first" warnings
**Fix:** Run `python run_pipeline.py` once from the project folder with the venv active.

### Dashboard is slow on first page load
**Expected** – the first run generates 1,400 satisfaction predictions (student × hostel matrix).
Subsequent loads use Streamlit's cache and are fast.

### "streamlit: command not found"
**Cause:** Streamlit installed in venv but PATH not updated.  
**Fix:** Always use `python -m streamlit run app.py` instead of `streamlit run app.py`.

---

## Limitations & Future Scope

### Limitations
- **Synthetic data only** – all student and hostel data is generated; models would
  need re-training on real institutional data before deployment.
- **Simplified room model** – we track individual bed slots but don't model
  complex real-world room assignment (e.g., existing occupants from last semester).
- **Static preferences** – student preferences are fixed; real preferences change
  over time and with peer influence.
- **No semester-2 rollover** – students who stayed in Semester 1 should have
  priority in Semester 2; not modelled here.

### Future Scope
1. **Real data integration** – connect to university ERP/portal via API
2. **Multi-semester rollover** – give staying students priority
3. **Dynamic reallocation** – re-run optimizer when cancellations occur
4. **Multi-objective optimisation** – Pareto front balancing satisfaction and fairness
5. **Explainable AI** – SHAP values per student to explain their allocation
6. **Mobile app** – students view allocation and swap requests on mobile
