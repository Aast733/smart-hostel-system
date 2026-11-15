"""
app.py  –  Streamlit Entry Point (Home / Overview page).

This is the file you run with:  streamlit run app.py

It shows:
  - Project title and overview
  - Architecture diagram (Prediction → Optimization → Evaluation)
  - KPI cards: total beds, applicants, allocation rate, mean satisfaction
  - Navigation hint to other pages

On FIRST LAUNCH, if models or data are missing, the pipeline is run automatically.
"""

import sys
import io
from pathlib import Path

# Fix Windows cp1252 terminal encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Ensure src/ is importable ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Friendly check for beginners running via VS Code before installing dependencies
try:
    import streamlit as st
except ModuleNotFoundError:
    print("\n" + "=" * 65)
    print("  [ERROR] Streamlit is not installed in the active Python environment.")
    print("  To fix this:")
    print("    1. Select the project virtual environment (venv) in VS Code:")
    print("       Press Ctrl+Shift+P -> 'Python: Select Interpreter' -> select './venv'")
    print("    2. Or run setup_and_run.bat by double-clicking it")
    print("    3. Or install dependencies:")
    print("       pip install -r requirements.txt")
    print("=" * 65 + "\n")
    sys.exit(1)

# ── If run directly via VS Code "Play" button or 'python app.py', automatically launch Streamlit
if not st.runtime.exists():
    from streamlit.web import cli as stcli
    print("[INFO] Starting Streamlit server...")
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), "--global.developmentMode=false"]
    sys.exit(stcli.main())

import pandas as pd

from src.config import (
    HOSTEL_CSV, DEMAND_CSV, STUDENTS_CSV, SURVEY_CSV,
    DEMAND_MODEL_PATH, PREF_MODEL_PATH,
    ALLOCATION_CSV, EVAL_CSV, TOTAL_BEDS, OUTPUTS_DIR, DATA_DIR,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Hostel System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/dormitory.png", width=80)
    st.title("Smart Hostel System")
    st.caption("Supervised Learning & Optimization")
    st.divider()
    st.markdown("**Navigation**")
    st.markdown("""
    Use the **Pages** menu (☰ top-left) to navigate:
    1. 🏠 Home *(this page)*
    2. 🔍 Data Explorer
    3. 📈 Demand Prediction
    4. ⭐ Satisfaction Model
    5. 🎛️ What-If Simulator
    6. 🏨 Room Allocation
    7. 📊 Evaluation
    8. 👤 Student Lookup
    9. 📤 Admin Upload
    """)
    st.divider()
    seed_val = st.number_input("Random Seed", value=42, min_value=0, max_value=9999, step=1)
    st.caption(f"Current seed: {seed_val}")
    st.divider()

    if st.button("🚀 Run Full Pipeline", use_container_width=True, type="primary"):
        with st.spinner("Running pipeline … (may take ~30 seconds)"):
            import subprocess, sys as _sys
            result = subprocess.run(
                [_sys.executable, str(ROOT / "run_pipeline.py")],
                capture_output=True, text=True, cwd=str(ROOT),
            )
        if result.returncode == 0:
            st.success("Pipeline complete!")
            st.cache_data.clear()
        else:
            st.error("Pipeline failed – see terminal for details.")
            st.code(result.stderr[-2000:])

# ── Auto-run pipeline on first launch if data/models are missing ──────────────
@st.cache_resource(show_spinner=False)
def _ensure_pipeline_run():
    required = [HOSTEL_CSV, STUDENTS_CSV, DEMAND_MODEL_PATH, PREF_MODEL_PATH,
                OUTPUTS_DIR / "allocation_hungarian.csv", EVAL_CSV]
    if not all(p.exists() for p in required):
        import subprocess, sys as _sys
        subprocess.run(
            [_sys.executable, str(ROOT / "run_pipeline.py")],
            cwd=str(ROOT),
        )

_ensure_pipeline_run()

# ── Load KPI data ─────────────────────────────────────────────────────────────
@st.cache_data
def _load_kpis():
    kpis = {"beds": TOTAL_BEDS, "applicants": 0, "alloc_rate": 0.0, "mean_sat": 0.0}
    try:
        stu = pd.read_csv(STUDENTS_CSV)
        kpis["applicants"] = len(stu)
    except Exception:
        pass
    try:
        ev  = pd.read_csv(EVAL_CSV)
        hung_row = ev[ev["method"] == "Hungarian"]
        if not hung_row.empty:
            kpis["alloc_rate"] = float(hung_row.iloc[0]["allocation_rate_%"])
            kpis["mean_sat"]   = float(hung_row.iloc[0]["mean_satisfaction"])
    except Exception:
        pass
    return kpis

kpis = _load_kpis()

# ── Hero section ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1.1rem;
    color: #94a3b8;
    margin-bottom: 1.5rem;
}
.kpi-card {
    background: linear-gradient(135deg, #1e1b4b, #312e81);
    border: 1px solid #4f46e5;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(99,102,241,0.15);
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #a5b4fc;
}
.kpi-label {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-top: 0.3rem;
}
.arch-box {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin: 0.5rem 0;
    font-size: 0.95rem;
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="hero-title">🏠 Smart Hostel System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Demand Prediction & Intelligent Room Allocation — '
    'A Supervised Learning + Optimization Project</p>',
    unsafe_allow_html=True
)

# ── KPI cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-value">{kpis["beds"]:,}</div>'
        '<div class="kpi-label">Total Beds Available</div></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-value">{kpis["applicants"]:,}</div>'
        '<div class="kpi-label">Student Applicants</div></div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-value">{kpis["alloc_rate"]:.1f}%</div>'
        '<div class="kpi-label">Allocation Rate (Hungarian)</div></div>',
        unsafe_allow_html=True,
    )
with c4:
    sat_color = "#6ee7b7" if kpis["mean_sat"] >= 7 else "#fca5a5"
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-value" style="color:{sat_color}">'
        f'{kpis["mean_sat"]:.2f}/10</div>'
        '<div class="kpi-label">Mean True Satisfaction</div></div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Architecture diagram ──────────────────────────────────────────────────────
st.subheader("🏗️ System Architecture")
col_a, col_b, col_c = st.columns([1, 0.15, 1])

with col_a:
    st.markdown("""
    <div class="arch-box">
    <b style="color:#818cf8">📊 PART A — PREDICTION</b><br>
    <small>
    ↳ Historical demand data (2010-2025)<br>
    ↳ Feature engineering: lags, cohort, fee ratio<br>
    ↳ Models: Ridge, Random Forest, Gradient Boosting<br>
    ↳ Best model → demand forecast per hostel<br><br>
    ↳ Survey data → satisfaction features<br>
    ↳ Models: Ridge, RF, HistGradientBoosting<br>
    ↳ Output: predicted satisfaction score (1-10)<br>
    </small>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("<br><br><br><h2 style='text-align:center;color:#6366f1'>→</h2>",
                unsafe_allow_html=True)

with col_c:
    st.markdown("""
    <div class="arch-box">
    <b style="color:#34d399">🏨 PART B — OPTIMIZATION</b><br>
    <small>
    ↳ Feasibility matrix (hard constraints)<br>
    &nbsp;&nbsp;&nbsp;• Gender match, budget, disability floor<br>
    ↳ Cost matrix: –(priority × predicted_sat)<br>
    ↳ Hungarian algorithm (scipy LSAP)<br>
    ↳ Roommate swap local search<br>
    ↳ Baselines: Random, Greedy FCFS<br>
    ↳ Upper bound: Oracle (true satisfaction)<br><br>
    ↳ Evaluation: allocation rate, Gini, happy %<br>
    </small>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Problem statement ─────────────────────────────────────────────────────────
st.subheader("📋 Problem Statement")
st.markdown("""
Universities manage **limited hostel beds** for a growing student population.
Two sub-problems must be solved each semester:

| Sub-problem | Approach | Outputs |
|---|---|---|
| **How many students will apply?** | Supervised regression (demand model) | Per-hostel forecast with OVER/UNDER status |
| **Who gets which room?** | Constrained assignment optimization | Allocation table + waitlist |

The system must respect **hard constraints** (gender, budget, disability) while
maximising **priority-weighted satisfaction** — where first-years, disabled, scholarship
and far-away students get higher priority.

> 🔑 **Key insight**: Prediction (Part A) feeds Optimization (Part B). The ML model's
> predicted satisfaction score becomes the benefit in the assignment cost matrix.
""")

st.info("💡 Use the sidebar pages to explore each component of the system in detail.")
