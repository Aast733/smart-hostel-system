"""
pages/1_Data_Explorer.py  –  Data Explorer page.

Lets users browse and filter the four datasets:
  - Hostel master table
  - Historical demand panel with trend charts
  - Student applicants
  - Satisfaction survey
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import HOSTEL_CSV, DEMAND_CSV, STUDENTS_CSV, SURVEY_CSV

st.set_page_config(page_title="Data Explorer", page_icon="🔍", layout="wide")

st.markdown("# 🔍 Data Explorer")
st.caption("Browse, filter, and visualise all datasets used in the system.")

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_hostels():
    return pd.read_csv(HOSTEL_CSV)

@st.cache_data
def load_demand():
    return pd.read_csv(DEMAND_CSV)

@st.cache_data
def load_students():
    return pd.read_csv(STUDENTS_CSV)

@st.cache_data
def load_survey():
    return pd.read_csv(SURVEY_CSV)

tab1, tab2, tab3, tab4 = st.tabs(["🏨 Hostels", "📈 Historical Demand", "👥 Students", "⭐ Survey"])

# ── TAB 1: Hostels ─────────────────────────────────────────────────────────────
with tab1:
    df = load_hostels()
    st.subheader(f"Hostel Master Table  ({len(df)} hostels)")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.dataframe(df, use_container_width=True, height=300)
        st.download_button("⬇ Download CSV", df.to_csv(index=False),
                           "hostels.csv", "text/csv", key="dl_hostels")

    with col2:
        fig = px.bar(
            df, x="name", y="total_beds", color="gender",
            color_discrete_map={"M": "#6366f1", "F": "#ec4899"},
            title="Bed Capacity per Hostel",
            labels={"name": "Hostel", "total_beds": "Total Beds", "gender": "Gender"},
        )
        fig.update_layout(xaxis_tickangle=-30, height=320)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig2 = px.scatter(
            df, x="distance_km", y="annual_fee_inr",
            size="total_beds", color="ac",
            color_discrete_map={0: "#64748b", 1: "#0ea5e9"},
            hover_name="name",
            title="Fee vs Distance (size=capacity, color=AC)",
            labels={"distance_km": "Distance (km)", "annual_fee_inr": "Annual Fee (₹)"},
        )
        st.plotly_chart(fig2, use_container_width=True)
    with col4:
        fig3 = px.bar(
            df, x="name", y=["single_rooms", "double_rooms", "triple_rooms"],
            barmode="stack",
            title="Room Type Breakdown",
            labels={"value": "Rooms", "variable": "Room Type", "name": "Hostel"},
            color_discrete_map={
                "single_rooms": "#6366f1",
                "double_rooms": "#8b5cf6",
                "triple_rooms": "#ec4899",
            },
        )
        fig3.update_layout(xaxis_tickangle=-30, height=350)
        st.plotly_chart(fig3, use_container_width=True)


# ── TAB 2: Historical Demand ───────────────────────────────────────────────────
with tab2:
    df = load_demand()
    st.subheader(f"Historical Demand Panel  ({len(df)} records, 2010-2025)")

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        hostel_filter = st.multiselect(
            "Filter by hostel", df["hostel_id"].unique().tolist(),
            default=df["hostel_id"].unique().tolist(), key="dem_h_filter",
        )
    with col_f2:
        sem_filter = st.multiselect(
            "Semester", [1, 2], default=[1, 2], key="dem_s_filter",
        )
    with col_f3:
        year_range = st.slider(
            "Year range", 2010, 2025, (2010, 2025), key="dem_yr_filter",
        )

    mask = (
        df["hostel_id"].isin(hostel_filter) &
        df["semester"].isin(sem_filter) &
        df["year"].between(*year_range)
    )
    filtered = df[mask]

    fig = px.line(
        filtered, x="year", y="applications",
        color="hostel_id", facet_col="semester",
        markers=True,
        title="Applications Over Time (by Hostel and Semester)",
        labels={"applications": "Applications", "year": "Year", "hostel_id": "Hostel"},
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(
        filtered, x="year", y="occupancy_rate",
        color="hostel_id", facet_col="semester",
        title="Occupancy Rate Over Time",
    )
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(filtered.head(200), use_container_width=True, height=200)
    st.download_button("⬇ Download filtered CSV", filtered.to_csv(index=False),
                       "demand_filtered.csv", "text/csv", key="dl_demand")


# ── TAB 3: Students ────────────────────────────────────────────────────────────
with tab3:
    df = load_students()
    st.subheader(f"Student Applicants  ({len(df)} students)")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df, x="budget_inr", color="gender",
            color_discrete_map={"M": "#6366f1", "F": "#ec4899"},
            nbins=30, barmode="overlay",
            title="Budget Distribution by Gender",
            labels={"budget_inr": "Budget (₹)", "gender": "Gender"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.pie(
            df, names="year_of_study", title="Applicants by Year of Study",
            color_discrete_sequence=px.colors.sequential.Plasma_r,
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.histogram(
            df, x="home_distance_km", nbins=40,
            title="Home Distance Distribution",
            labels={"home_distance_km": "Distance from Home (km)"},
            color_discrete_sequence=["#8b5cf6"],
        )
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        flag_counts = pd.DataFrame({
            "Category": ["First Year", "Disabled", "Scholarship", "Far Away"],
            "Count":    [df["first_year"].sum(), df["disability"].sum(),
                         df["scholarship"].sum(), df["far_away"].sum()],
        })
        fig4 = px.bar(flag_counts, x="Category", y="Count",
                      color="Category",
                      title="Priority Group Sizes",
                      color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig4, use_container_width=True)

    st.dataframe(df.head(100), use_container_width=True, height=200)
    st.download_button("⬇ Download CSV", df.to_csv(index=False),
                       "students.csv", "text/csv", key="dl_students")


# ── TAB 4: Survey ─────────────────────────────────────────────────────────────
with tab4:
    df = load_survey()
    st.subheader(f"Historical Satisfaction Survey  ({len(df)} entries)")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df, x="satisfaction_rating",
            color="hostel_id", nbins=20, barmode="overlay",
            title="Satisfaction Rating Distribution by Hostel",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        avg = df.groupby("hostel_id")["satisfaction_rating"].mean().reset_index()
        fig2 = px.bar(avg, x="hostel_id", y="satisfaction_rating",
                      color="hostel_id",
                      title="Mean Satisfaction per Hostel",
                      labels={"satisfaction_rating": "Avg Rating (1-10)"},
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(df.head(100), use_container_width=True, height=200)
    st.download_button("⬇ Download CSV", df.to_csv(index=False),
                       "survey.csv", "text/csv", key="dl_survey")
