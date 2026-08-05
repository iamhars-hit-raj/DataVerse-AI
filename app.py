import streamlit as st
import pandas as pd

# ==========================================
# Frontend Imports
# ==========================================
from frontend.sidebar import navigation
from frontend.dashboard import show_dashboard
from frontend.theme import apply_theme
from frontend.feature_engineering import feature_engineering_page
from frontend.machine_learning import machine_learning_page

# ==========================================
# Backend Imports
# ==========================================
from backend.overview import show_dataset_overview
from backend.cleaning import data_cleaning
from backend.eda import exploratory_data_analysis

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="DataVerse AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_theme()

# ==========================================
# Sidebar Navigation
# ==========================================
page = navigation()

# ==========================================
# Session State Initialization
# ==========================================
if "df" not in st.session_state:
    st.session_state.df = None

if "filename" not in st.session_state:
    st.session_state.filename = None

# Used to detect a newly uploaded dataset
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None

# ==========================================
# Sidebar Dataset Upload
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    # Load only when a NEW file is uploaded
    if uploaded_file.name != st.session_state.last_uploaded:

        try:

            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state.df = df
            st.session_state.filename = uploaded_file.name
            st.session_state.last_uploaded = uploaded_file.name

            st.sidebar.success("✅ Dataset Loaded Successfully")

        except Exception as e:

            st.sidebar.error(f"❌ {e}")

# ==========================================
# Dashboard
# ==========================================
if page == "🏠 Dashboard":

    show_dashboard(
        st.session_state.df,
        st.session_state.filename
    )

# ==========================================
# About
# ==========================================
elif page == "ℹ️ About":

    st.title("ℹ️ About DataVerse AI")

    st.markdown("""
# 📊 DataVerse AI

An End-to-End AI Powered Data Science Platform.

---

## Features

- 📂 Upload Dataset
- 📈 Dataset Overview
- 🧹 Data Cleaning
- 📊 Exploratory Data Analysis
- 🧪 Feature Engineering
- 🤖 Machine Learning
- 📄 Report Generation

---

## Tech Stack

- Python
- Streamlit
- Pandas
- Plotly
- Scikit-Learn

---

Developed as a Professional Data Science Portfolio Project.
""")

# ==========================================
# Remaining Pages
# ==========================================
else:

    if st.session_state.df is None:

        st.warning("👈 Please upload a dataset from the sidebar.")

        st.stop()

    # Always work on the latest dataset
    df = st.session_state.df

    st.title(page)

    st.divider()

    # ======================================
    # Upload Dataset
    # ======================================
    if page == "📂 Upload Dataset":

        st.success(f"✅ {st.session_state.filename}")

        st.write("### Dataset Preview")

        st.dataframe(df.head(), use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

    # ======================================
    # Dataset Overview
    # ======================================
    elif page == "📈 Dataset Overview":

        show_dataset_overview(df)

    # ======================================
    # Data Cleaning
    # ======================================
    elif page == "🧹 Data Cleaning":

        st.session_state.df = data_cleaning(df)

    # ======================================
    # Exploratory Data Analysis
    # ======================================
    elif page == "📊 Exploratory Data Analysis":

        exploratory_data_analysis(df)

    # ======================================
    # Feature Engineering
    # ======================================
    elif page == "🧪 Feature Engineering":

        feature_engineering_page(df)

    # ======================================
    # Machine Learning
    # ======================================
    elif page == "⚙️ Machine Learning":
        machine_learning_page()

    # ======================================
    # Reports
    # ======================================
    elif page == "📄 Reports":

        st.info("🚧 Automated Reports Coming Soon")